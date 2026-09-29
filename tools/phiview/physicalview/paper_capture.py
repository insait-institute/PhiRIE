"""Reproducible PhiView paper captures from real server renders and physics.

Every frame has a camera/state sidecar. Rendering and technical success are
separate from a human's publication-quality review. Never embellishes a render.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from physicalview.phiview import Demo, save_json

FEATURES = {
 'a_original': 'Full original Gaussian scene',
 'b_highlight_all': 'Highlight existing discovered object proposals',
 'c_mouse_pick': 'Select an object through its rendered pixel',
 'd_simulatable': 'Enable the selected rigid body',
 'e_physical_parameters': 'Inspect active physical parameters',
 'f_generated_choices': 'Switch registered generated object variants',
 'g_clean_selected': 'Remove the selected object',
 'f2_clean_all': 'Remove all simulatable objects',
 'h_prompt_inpaint': 'Prompt-conditioned scene inpainting',
 'i_fall_friction': 'Fall and friction simulation',
 'j_throw': 'Throw a selected object',
 'k_shoot': 'Camera-ray projectile simulation',
 'l_robot': 'Place an arm and execute a scripted command',
 'm_navigation': 'WASDQE and mouse-look navigation',
}
SURVEY_VERSION = 2


def quality(image):
    import cv2
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    return {'black_fraction':float(np.mean(gray<5)),
            'white_fraction':float(np.mean(gray>250)),
            'laplacian_variance':float(cv2.Laplacian(gray,cv2.CV_32F).var()),
            'std':float(gray.std())}


class Capture:
    def __init__(self, args):
        self.d=Demo(args); self.root=self.d.out
        previous=self.root/'features.json'
        self.rows=json.loads(previous.read_text()) if args.features and previous.exists() else {}
        self.frames=[]
        self.paper_wh=(args.width,args.height)
        self.capture_code={
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(set(Path(__file__).parent.glob('paper*.py')) | set(Path(__file__).parent.glob('phiview*.py')) | set(Path(__file__).parent.glob('inpaint_surface*.py')))},
            'features':FEATURES,'source_native_resolution':self.d.native_wh,
            'export_resolution':self.paper_wh,'png_lossless':True,
            'rasterize_mode':self.d.scene.rasterize_mode,
            'image_enhancement':'none','publication_review':'pending'}
        save_json(self.root/'capture-code.json',self.capture_code)

    def shot(self, feature, name, extra=None):
        from PIL import Image
        d=self.d; image=d.render()
        p=self.root/feature/(name+'.png');p.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(image).save(p,compress_level=4)
        mask,depth,w2c,K=d.frames[d.frame_id]
        ys,xs=np.where(mask==d.scene.ids.get(d.selected,-1))
        selected_bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None
        rec={'file':str(p.relative_to(self.root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
             'camera_name':d.camera_name,'w2c':w2c,'K':K,'resolution':d.wh,
             'quality':quality(image),'state':d.status(),'qpos':d.physics.data.qpos.copy(),
             'qvel':d.physics.data.qvel.copy(),'mask_source':d.scene.mask_sources,
             'selected_bbox_xyxy':selected_bbox,'selected_visible_pixels':len(xs),
             'publication_review':'pending','extra':extra or {},'capture_source_sha256':self.capture_code['source_sha256']}
        rec['rasterize_mode']=d.scene.rasterize_mode
        save_json(p.with_suffix('.json'),rec)
        self.frames.append(rec['file']);return image

    def run_feature(self,key,fn):
        t=time.monotonic(); before=len(self.frames)
        try:
            info=fn() or {}
            self.rows[key]={'status':'captured','evidence':info,'frames':self.frames[before:],
                            'publication_review':'pending'}
        except Exception as e:
            self.rows[key]={'status':'failed','error':str(e),'traceback':traceback.format_exc(),
                            'frames':self.frames[before:],'publication_review':'not_eligible'}
        self.rows[key]['seconds']=time.monotonic()-t
        save_json(self.root/'features.json',self.rows)
        print(key,self.rows[key]['status'],self.rows[key].get('error',''),flush=True)

    def survey(self):
        from PIL import Image,ImageDraw
        d=self.d; d.wh=(768,512);d.highlight=False
        candidates=[n for n,r in d.state.objects.items() if n in d.physics.available]
        if not candidates:candidates=list(d.state.objects)
        if not candidates:raise ValueError('No selectable object proposals')
        tabletop=[n for n in candidates
                  if any(w in d.state.objects[n].label.lower()
                         for w in ('mug','cup','bottle','kettle','stapler','telephone'))
                  and max(d.state.objects[n].meta.get('extent',[99])) < .6]
        if tabletop:
            candidates=tabletop
        # Selection is based on visibility/composition, never on downstream outcomes.
        names=set()
        for n in candidates:
            f=d.state.objects[n].meta.get('frame')
            if f in d.state.cameras:names.add(f)
            v=d.state.result_set.out_dir/'inpaint'/n/'views.json'
            if v.exists():
                for row in json.loads(v.read_text()):
                    f=row.get('frame',row.get('name'))
                    if f in d.state.cameras:names.add(f)
        for i in np.linspace(0,len(d.camera_names)-1,min(24,len(d.camera_names))).astype(int):names.add(d.camera_names[i])
        scores=[];tiles=[];best=None
        for camera in sorted(names):
            d.execute({'op':'camera','name':camera});im=d.render()
            mask=d.frames[d.frame_id][0];q=quality(im)
            for n in candidates:
                ys,xs=np.where(mask==d.scene.ids[n]); area=len(xs)/mask.size
                if len(xs)<80:continue
                center=np.array([xs.mean()/mask.shape[1],ys.mean()/mask.shape[0]])
                edge=float(np.mean((xs<8)|(xs>=mask.shape[1]-8)|(ys<8)|(ys>=mask.shape[0]-8)))
                r=d.state.objects[n]
                variants=sum(bool(p.gs_ply and p.aligned and p.aligned.get('T') and not p.aligned.get('rejected')) for p in r.proposals.values())
                compact=min(area/.04,1.)*np.exp(-max(0.,area-.15)*10)
                score=compact*(1-min(1.,np.linalg.norm(center-[.5,.54])))*(1-edge)*(1-q['black_fraction'])
                score*=1.2 if variants>=2 else 1.
                score*=1.1 if any(w in r.label.lower() for w in ('mug','cup','kettle','bottle','stapler')) else 1.
                row={'camera':camera,'object':n,'label':r.label,'score':float(score),'fraction':area,'variants':variants,'quality':q,
                     'pose':{'position':d.camera.position.copy(),'yaw':d.camera.yaw,'pitch':d.camera.pitch,'fov':d.camera.fov},'view_kind':'source_anchor'}
                scores.append(row)
                if best is None or score>best['score']:best=row
            thumb=Image.fromarray(im);thumb.thumbnail((384,256)); tile=Image.new('RGB',(384,280),'white');tile.paste(thumb,(0,0))
            ImageDraw.Draw(tile).text((8,261),camera,fill='black');tiles.append(tile)
        # Refine composition through actual camera motion toward a visible target.
        # This is a new full-Gaussian render, not a crop/upscale or altered image.
        from physicalview.phiview_scene import FlyCamera
        from physicalview.render import look_at_w2c
        initial_scores=list(scores)
        for n in candidates:
            anchors=sorted([r for r in initial_scores if r['object']==n],key=lambda r:r['score'],reverse=True)[:2]
            for anchor in anchors:
                rec=d.state.objects[n];center=np.asarray(rec.meta['centroid'])
                position=np.asarray(anchor['pose']['position']);v=position-center;distance=np.linalg.norm(v)
                desired=max(.55,min(1.2,max(rec.meta.get('extent',[.2]))*4.5))
                position=center+v*min(1.,desired/max(distance,1e-6))
                d.camera=FlyCamera.from_w2c(look_at_w2c(position,center),45.)
                d.camera_name=anchor['camera'];im=d.render();mask=d.frames[d.frame_id][0]
                ys,xs=np.where(mask==d.scene.ids[n]);area=len(xs)/mask.size
                if len(xs)<80:continue
                q=quality(im);edge=float(np.mean((xs<8)|(xs>=mask.shape[1]-8)|(ys<8)|(ys>=mask.shape[0]-8)))
                compact=min(area/.04,1.)*np.exp(-max(0.,area-.15)*10)
                score=compact*(1-edge)*(1-q['black_fraction'])
                if q['black_fraction']>.08:score*=.2
                row={**anchor,'score':float(score),'fraction':area,'quality':q,'view_kind':'target_framing',
                     'pose':{'position':d.camera.position.copy(),'yaw':d.camera.yaw,'pitch':d.camera.pitch,'fov':d.camera.fov}}
                scores.append(row)
                if best is None or score>best['score']:best=row
        if best is None:raise ValueError('No object has enough visible pixels in the camera survey')
        canvas=Image.new('RGB',(384*4,280*((len(tiles)+3)//4)),'#ddd')
        for i,t in enumerate(tiles):canvas.paste(t,((i%4)*384,(i//4)*280))
        canvas.save(self.root/'camera-survey.jpg',quality=90)
        save_json(self.root/'camera-survey.json',{'version':SURVEY_VERSION,'selected':best,'candidates':scores})
        self.chosen=best; d.wh=self.paper_wh
        d.execute({'op':'camera','name':best['camera']});d.execute({'op':'select','object':best['object']})
        d.camera=FlyCamera(np.asarray(best['pose']['position']),best['pose']['yaw'],best['pose']['pitch'],best['pose']['fov'])
        self.base_camera=copy.deepcopy(d.camera)
        return best

    def restore(self,mode='original',highlight=False):
        d=self.d
        if d.physics.robot is not None or d.physics.projectile:
            from physicalview.phiview_sim import DemoPhysics
            if d.physics.renderer is not None:d.physics.renderer.close()
            d.physics=DemoPhysics(d.state,self.root)
        d.physics.reset();d.physics.enabled=set();d.camera=copy.deepcopy(self.base_camera)
        d.wh=self.paper_wh;d.mode=mode;d.highlight=highlight
        d.selected=self.chosen['object']

    def sequence(self,key,action,seconds=2.,extra=None):
        import imageio.v2 as imageio
        d=self.d;self.restore('simulation')
        self.shot(key,action+'-before',extra)
        d.execute({'op':action,'strength':1.3})
        initial=d.physics.data.xpos[d.physics.model.body(d.selected).id].copy()
        times=[0.,.15,.4,.8,1.4,seconds]
        self.shot(key,action+'-t000',extra)
        writer=imageio.get_writer(self.root/key/(action+'.mp4'),fps=15,codec='libx264',macro_block_size=1,quality=9)
        next_key=1; trajectory=[]
        try:
            for i in range(round(seconds*30)):
                d.physics.step(1/30)
                t=(i+1)/30
                trajectory.append({'t':t,'position':d.physics.data.xpos[d.physics.model.body(d.selected).id].copy()})
                if next_key<len(times) and t>=times[next_key]-1e-6:
                    d.wh=self.paper_wh;self.shot(key,f'{action}-t{round(t*1000):04d}',extra);next_key+=1
                if i%2==0:
                    d.wh=(1440,960);writer.append_data(d.render())
        finally:writer.close();d.wh=self.paper_wh
        save_json(self.root/key/(action+'-trajectory.json'),trajectory)
        displacement=float(np.linalg.norm(trajectory[-1]['position']-initial))
        if not np.isfinite(d.physics.data.qpos).all():raise ValueError('Non-finite physics state')
        if displacement<.01:raise ValueError(f'{action}: displacement below 1 cm ({displacement:g} m)')
        return {'displacement_m':displacement,'simulated_seconds':seconds}

    def capture(self):
        d=self.d
        actions={
            'a_original':self.original,
            'b_highlight_all':self.highlights,
            'c_mouse_pick':self.pick,
            'd_simulatable':self.simulatable,
            'e_physical_parameters':self.parameters,
            'g_clean_selected':lambda:self.clean('g_clean_selected','clean_selected'),
            'f2_clean_all':lambda:self.clean('f2_clean_all','clean_all'),
            'i_fall_friction':lambda:{a:self.sequence('i_fall_friction',a,2.) for a in ('fall','friction')},
            'j_throw':lambda:self.sequence('j_throw','throw',2.),
            'k_shoot':self.shoot,
            'f_generated_choices':self.variants,
            'm_navigation':self.navigation,
            'l_robot':self.robot,
            'h_prompt_inpaint':self.prompt,
        }
        requested=set(d.args.features or FEATURES)
        for key,action in actions.items():
            if key in requested:self.run_feature(key,action)
        self.contact_sheet()
        save_json(self.root/'capture-complete.json',{'features':self.rows,'scene':d.args.scene,'chosen':self.chosen,
                  'publication_review':'pending','counts':{'captured':sum(r['status']=='captured' for r in self.rows.values()),'total':len(FEATURES)}})

    def original(self):
        self.restore();self.d.execute({'op':'camera','name':self.context_camera()});self.shot('a_original','overview')
        self.d.camera=copy.deepcopy(self.base_camera);self.shot('a_original','target-detail')
        return {'gaussians':self.d.scene.count,'downsampling':False}
    def highlights(self):
        self.restore(highlight=True);self.d.selected=None
        self.d.execute({'op':'camera','name':self.context_camera()});self.shot('b_highlight_all','all-proposals-overview')
        self.d.camera=copy.deepcopy(self.base_camera);self.shot('b_highlight_all','all-proposals-detail')
        return {'proposals':len(self.d.scene.names),'provenance':self.d.manifest['discovery_provenance']}
    def context_camera(self):
        """Choose an observed room view independently of the interaction close-up."""
        if hasattr(self,'context_name'):return self.context_name
        d=self.d;previous=(d.wh,d.highlight,d.selected,copy.deepcopy(d.camera),d.camera_name)
        names={self.chosen['camera']}
        for i in np.linspace(0,len(d.camera_names)-1,min(48,len(d.camera_names))).astype(int):
            names.add(d.camera_names[i])
        rows=[]
        try:
            d.wh=(768,512);d.highlight=False;d.selected=None
            for name in sorted(names):
                d.execute({'op':'camera','name':name});im=d.render();mask=d.frames[d.frame_id][0]
                ids,counts=np.unique(mask,return_counts=True)
                visible=int(np.count_nonzero((ids>0)&(counts>=80)))
                q=quality(im)
                # Favor multiple visible proposals and a level room view. No generated
                # image, inpainting outcome or simulation outcome enters this choice.
                level=float(np.exp(-max(0.,abs(d.camera.pitch)-.55)*2))
                score=(visible+min(float(np.mean(mask>0))/.08,1.))*level*(1-q['black_fraction'])**3
                rows.append({'camera':name,'score':score,'visible_proposals':visible,
                             'pitch':d.camera.pitch,'quality':q})
            best=max(rows,key=lambda r:r['score']);self.context_name=best['camera']
            save_json(self.root/'context-survey.json',{'selected':best,'candidates':rows,
                      'selection_basis':'Observed-camera proposal visibility and room framing only'})
        finally:
            d.wh,d.highlight,d.selected,d.camera,d.camera_name=previous
        return self.context_name
    def pick(self):
        self.restore();d=self.d;d.selected=None;self.shot('c_mouse_pick','before')
        import cv2
        mask=d.frames[d.frame_id][0]==d.scene.ids[self.chosen['object']]
        dist=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
        y,x=np.unravel_index(np.argmax(dist),dist.shape)
        command={'op':'pick','frame':d.frame_id,'x':(x+.5)/d.wh[0],'y':(y+.5)/d.wh[1]}
        d.execute(command);assert d.selected==self.chosen['object'];d.highlight=True
        self.shot('c_mouse_pick','after',{'click':command});return command
    def simulatable(self):
        self.restore('original',True);d=self.d;self.shot('d_simulatable','before')
        d.execute({'op':'enable'});self.shot('d_simulatable','enabled',{'physical_parameters':d.physics.parameters(d.selected)})
        return {'enabled':sorted(d.physics.enabled)}
    def parameters(self):
        self.restore(highlight=True);d=self.d;p=d.physics.parameters(d.selected)
        if not p:raise ValueError('No physical parameters')
        self.shot('e_physical_parameters','selected',p)
        save_json(self.root/'e_physical_parameters'/'parameters.json',p);return p
    def clean(self,key,mode):
        self.restore();self.shot(key,'before');self.d.execute({'op':'view','mode':mode})
        self.shot(key,'after');return {'mode':mode,'removed':sorted(self.d.physics.available) if mode=='clean_all' else [self.d.selected]}
    def variants(self):
        self.restore('simulation');d=self.d;self.shot('f_generated_choices','original')
        options=[s for s,p in d.state.objects[d.selected].proposals.items() if p.gs_ply and p.aligned and p.aligned.get('T') and not p.aligned.get('rejected')]
        used=[]
        try:
            for s in options:
                d.execute({'op':'variant','source':s});self.shot('f_generated_choices',s);used.append(s)
        finally:d.execute({'op':'variant','source':'original'})
        if not used:raise ValueError('No registered generated alternatives for selected object')
        return {'variants':used,'collision_rebuilt':True}
    def navigation(self):
        self.restore();d=self.d;self.shot('m_navigation','start')
        poses=[]
        for label,keys,look in [('w',['w'],[0,0]),('s',['s'],[0,0]),('a',['a'],[0,0]),('d',['d'],[0,0]),('q',['q'],[0,0]),('e',['e'],[0,0]),('rotate',[],[50,-12])]:
            d.camera=copy.deepcopy(self.base_camera);d.camera.update(keys,.1,look)
            self.shot('m_navigation',label,{'keys':keys,'look':look});poses.append(label)
        return {'inputs':poses,'interaction_latency_not_measured':True}
    def shoot(self):
        self.restore('simulation');d=self.d;d.physics.enable([d.selected]);self.shot('k_shoot','before')
        mask=d.frames[d.frame_id][0];ys,xs=np.where(mask==d.scene.ids[d.selected]);i=len(xs)//2
        cmd={'op':'shoot','frame':d.frame_id,'x':(xs[i]+.5)/d.wh[0],'y':(ys[i]+.5)/d.wh[1],'speed':8.}
        d.execute(cmd);events=[]
        for i in range(90):
            d.physics.step(1/60)
            if i in (3,8,14,23,44,89):self.shot('k_shoot',f't{round((i+1)/60*1000):04d}',{'command':cmd})
        events=d.physics.events.copy()
        save_json(self.root/'k_shoot'/'contacts.json',events)
        if not events:raise ValueError('No projectile contact in this camera-ray trial')
        return {'contacts':events,'command':cmd}
    def robot(self):
        import imageio.v2 as imageio
        self.restore('simulation');d=self.d
        d.camera.position-=d.camera.forward()*1.0;d.camera.fov=max(70.,d.camera.fov)
        self.shot('l_robot','before')
        command=d.args.robot_command;d.execute({'op':'robot_command','command':command});self.shot('l_robot','placed',{'command':command})
        q0=d.physics.data.qpos[d.physics.robot['qadr']].copy();visible=0
        writer=imageio.get_writer(self.root/'l_robot'/'reach.mp4',fps=15,codec='libx264',macro_block_size=1,quality=9)
        try:
            for i in range(120):
                d.physics.step(1/30)
                if i in (14,29,59,89,119):
                    d.wh=self.paper_wh;self.shot('l_robot',f't{round((i+1)/30*1000):04d}',{'command':command})
                    visible=max(visible,int(np.sum(d.physics.overlay_mask)))
                if i%2==0:d.wh=(1440,960);writer.append_data(d.render())
        finally:writer.close();d.wh=self.paper_wh
        delta=float(np.linalg.norm(d.physics.data.qpos[d.physics.robot['qadr']]-q0))
        if visible<200 or delta<.01:raise ValueError(f'Robot visibility/motion insufficient: pixels={visible}, motion={delta}')
        return {'command':command,'robot':d.physics.robot_status,'visible_pixels':visible,'joint_motion_norm':delta,'task_success_claimed':False}
    def prompt(self):
        self.restore();d=self.d;before=self.shot('h_prompt_inpaint','before')
        manifest=self.root/'prompt-source.json'
        if not manifest.exists():raise ValueError('Fresh prompt build has not completed')
        data=json.loads(manifest.read_text());from agents.core.common import load_gaussians
        d.scene.prompt_backgrounds[frozenset(data['objects'])]=load_gaussians(data['path'],device='cuda')
        d.selected=data['objects'][0];d.mode='clean_selected'
        after=self.shot('h_prompt_inpaint','after',data)
        delta=np.abs(after.astype(np.int16)-before.astype(np.int16))
        data['changed_pixels_gt10']=int(np.sum(np.max(delta,axis=-1)>10))
        data['mean_absolute_pixel_difference']=float(delta.mean())
        if data['changed_pixels_gt10']<100:raise ValueError('Prompt result is not visibly different in the selected camera')
        return data
    def contact_sheet(self):
        from PIL import Image,ImageDraw
        rows=[]
        for key in FEATURES:
            paths=sorted((self.root/key).glob('*.png'))
            if not paths:continue
            picks=paths if len(paths)<=3 else [paths[0],paths[len(paths)//2],paths[-1]]
            row=Image.new('RGB',(1440,355),'white');draw=ImageDraw.Draw(row);draw.text((8,6),key,fill='black')
            for i,p in enumerate(picks):
                im=Image.open(p);im.thumbnail((480,320));row.paste(im,(i*480,30));draw.text((i*480+8,337),p.stem,fill='black')
            rows.append(row)
        canvas=Image.new('RGB',(1440,355*len(rows)),'white')
        for i,row in enumerate(rows):canvas.paste(row,(0,i*355))
        canvas.save(self.root/'contact-sheet.jpg',quality=92)


def prepare_feature_retry(root, requested):
    """Archive exactly the groups being replaced, retaining unrelated evidence."""
    import shutil
    root=Path(root)
    if not requested or not (root/'features.json').exists():return None
    history=root/'feature-attempts'/str(time.time_ns());history.mkdir(parents=True)
    for filename in ('features.json','capture-complete.json','capture-code.json','manifest.json','actions.jsonl'):
        if (root/filename).exists():shutil.copy2(root/filename,history/filename)
    for key in requested:
        if (root/key).exists():shutil.move(str(root/key),history/key)
    return history


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/phiview.yaml')
    ap.add_argument('--scene',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--width',type=int,default=2880);ap.add_argument('--height',type=int,default=1920)
    ap.add_argument('--robot-command',default='reach the object')
    ap.add_argument('--features',nargs='+',choices=list(FEATURES),help='Bounded feature retry; total remains 14 for coverage')
    ap.add_argument('--survey-only',action='store_true');args=ap.parse_args()
    prepare_feature_retry(args.out,args.features)
    cap=Capture(args)
    saved=cap.root/'camera-survey.json'
    if not args.survey_only and saved.exists() and json.loads(saved.read_text()).get('version')==SURVEY_VERSION:
        from physicalview.phiview_scene import FlyCamera
        cap.chosen=json.loads(saved.read_text())['selected'];best=cap.chosen
        cap.d.execute({'op':'camera','name':best['camera']});cap.d.execute({'op':'select','object':best['object']})
        p=best['pose'];cap.d.camera=FlyCamera(np.asarray(p['position']),p['yaw'],p['pitch'],p['fov'])
        cap.base_camera=copy.deepcopy(cap.d.camera)
    else:
        cap.survey()
    if not args.survey_only:cap.capture()

if __name__=='__main__':main()
