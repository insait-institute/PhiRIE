"""Known-embodiment GS observation replay through the maintained CompositeObs.

Preparation reads native runtime state only on the evaluator side. Rendering
receives named robot joints and generated-object poses, never room geometry.
"""
from pathlib import Path
import json
import gzip
import shutil
import time
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from robo.roundtrip.gaussian_build import sha,save,capture_rows
from robo.roundtrip.gaussian_motion import body_matrix,asset_world_transform


def named_joint_state(model,qpos,names):
    import mujoco
    qpos=np.asarray(qpos,dtype=float)
    if qpos.shape!=(model.nq,) or not np.isfinite(qpos).all():raise ValueError('native qpos shape or values changed')
    result={}
    for name in names:
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if j<0:raise ValueError('known embodiment joint absent')
        size={0:7,1:4}.get(int(model.jnt_type[j]),1);start=model.jnt_qposadr[j]
        result[name]=qpos[start:start+size].tolist()
    return result


def prepare(config,out):
    import mujoco
    episode=Path(config['episode']);capsule=Path(config['capsule']);scene=Path(config['runtime_scene_xml'])
    for key,path in [('result',episode/'result.json'),('trace',episode/'trace.json.gz'),('capsule',capsule/'robot_manifest.json'),('runtime_scene_xml',scene)]:
        if sha(path)!=config[key+'_sha256']:raise ValueError(key+' identity changed')
    result=json.loads((episode/'result.json').read_text());initial=json.loads((episode/'initial_state.json').read_text())
    from robo.manifest.hash import canonical_hash
    if canonical_hash(initial)!=result['initial_state_sha256'] or result['error'] is not None or not result['executed']:
        raise ValueError('complete native runtime trace required')
    metadata=json.loads((capsule/'robot_manifest.json').read_text());model=mujoco.MjModel.from_xml_path(str(scene))
    joints=named_joint_state(model,initial['qpos'],metadata['joint_qpos'])
    if joints!=metadata['joint_qpos']:raise ValueError('runtime joint mapping differs from sealed capture embodiment')
    trace=json.load(gzip.open(episode/'trace.json.gz','rt'))
    if len(trace)!=result['ticks'] or [x['tick'] for x in trace]!=list(range(len(trace))):raise ValueError('incomplete native trace')
    samples=[{'tick':0,'joints':joints,'target_pose':initial['object_states']['obj']}]
    for i in np.linspace(0,len(trace)-1,5,dtype=int):
        row=trace[i];samples.append({'tick':int(i)+1,'joints':named_joint_state(model,row['qpos'],joints),'target_pose':row['objects']['obj']})
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    receipt={'schema_version':1,'scope':'recorded_native_runtime_known_robot_and_generated_target_only',
        'source_hashes':{k:v for k,v in config.items() if k.endswith('_sha256')},'canonical_instance_id':result['canonical_instance_id'],
        'samples':samples,'runtime_model_used_for_named_joint_addresses_only':True,'native_room_geometry_exported':False,
        'native_segmentation_exported':False,'new_policy_episode':False,'sample_rule':'initial plus five uniformly spaced original trace states'}
    save(out/'runtime_observation_states.json',receipt);return receipt


def render(config,out):
    import mujoco
    from PIL import Image,ImageDraw
    from robo.rendering.pi05_render import CompositeObs
    state_path=Path(config['states']);capsule=Path(config['capsule']);background=Path(config['background']);obj=Path(config['object'])
    for key,path in [('states',state_path),('capsule',capsule/'robot_manifest.json'),('background',background),('object',obj/'trellis_gs.ply'),('alignment',obj/'aligned.json')]:
        if sha(path)!=config[key+'_sha256']:raise ValueError(key+' identity changed')
    state=json.loads(state_path.read_text());meta=json.loads((capsule/'robot_manifest.json').read_text())
    if sha(capsule/'robot.xml')!=meta['robot_xml_sha256']:raise ValueError('robot XML changed')
    for name,digest in meta['files'].items():
        if sha(capsule/name)!=digest:raise ValueError('known robot mesh changed')
    public,rows=capture_rows(config['capture']);rows=rows[:2];w,h=public['width'],public['height']
    root=ET.fromstring((capsule/'robot.xml').read_text());visual=root.find('visual')
    if visual is None:visual=ET.SubElement(root,'visual')
    glob=visual.find('global')
    if glob is None:glob=ET.SubElement(visual,'global')
    glob.set('offwidth',str(w));glob.set('offheight',str(h))
    for i,row in enumerate(rows):
        K=np.asarray(row['K']);T=np.asarray(row['T_world_from_camera'])@np.diag([1,-1,-1,1])
        if not np.allclose(K,[[K[0,0],0,w/2],[0,K[0,0],h/2],[0,0,1]],atol=1e-9,rtol=0):raise ValueError('centered square-pixel public camera required')
        ET.SubElement(root.find('worldbody'),'camera',name=f'train_{i}',pos=' '.join(map(str,T[:3,3])),
            quat=' '.join(map(str,Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]])),fovy=str(np.degrees(2*np.arctan(h/(2*K[0,0])))))
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mujoco.MjData(model)
    alignment=np.asarray(json.loads((obj/'aligned.json').read_text())['T']);initial=state['samples'][0]['target_pose']
    class RuntimeView:
        free_bodies={'obj_00':True}
        def body_pose(self,name):
            T=asset_world_transform(alignment,initial,self.sample['target_pose']);scale=np.linalg.norm(T[:3,0])
            return T[:3,3],Rotation.from_matrix(T[:3,:3]/scale).as_quat()[[3,0,1,2]]
    env=RuntimeView();env.model=model;env.data=data
    out=Path(out);out.mkdir(parents=True,exist_ok=False);factory=out/'factory';(factory/'inpaint').mkdir(parents=True);(factory/'objects/obj_00').mkdir(parents=True)
    shutil.copyfile(background,factory/'inpaint/clean_background.ply')
    for name in ['trellis_gs.ply','aligned.json']:shutil.copyfile(obj/name,factory/'objects/obj_00'/name)
    save(factory/'objects/objects.json',[{'index':0}]);option=mujoco.MjvOption();option.geomgroup[0]=0;option.geomgroup[1]=1
    renderer=CompositeObs(env,factory,w,h,robot_compositing='expected_depth_v1',scene_option=option)
    records=[];sheet=Image.new('RGB',(1280,6*400),'white');draw=ImageDraw.Draw(sheet)
    try:
        for si,sample in enumerate(state['samples']):
            env.sample=sample
            for name,q in sample['joints'].items():
                j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
                if j<0:raise ValueError('runtime robot joint absent in capsule')
                start=model.jnt_qposadr[j];data.qpos[start:start+len(q)]=q
            mujoco.mj_forward(model,data)
            for ci,row in enumerate(rows):
                before=(data.qpos.copy(),data.qvel.copy(),data.time);start=time.monotonic();rgb=renderer.render(f'train_{ci}');elapsed=time.monotonic()-start
                unchanged=np.array_equal(before[0],data.qpos) and np.array_equal(before[1],data.qvel) and before[2]==data.time
                if not unchanged:raise ValueError('observer mutated runtime state')
                layers=renderer.last_layers;e=renderer.last_visibility;stem=f'tick{sample["tick"]:04}_camera{ci}'
                Image.fromarray(rgb).save(out/(stem+'.png'));np.savez_compressed(out/(stem+'_layers.npz'),**layers,**{k:v for k,v in e.items() if isinstance(v,np.ndarray)})
                visible=e['robot_visible'];occluded=e['robot_occluded'];unknown=e['unknown_coverage'];mask=layers['robot_mask']
                exact=np.array_equal(rgb[visible],layers['robot_rgb'][visible]);no_paste=np.array_equal(rgb[~visible],layers['world_rgb'][~visible])
                if not exact or not no_paste:raise ValueError('depth-tested composition byte invariant failed')
                record={'native_tick':sample['tick'],'camera':row['frame_id'],'robot_pixels':int(mask.sum()),'visible_robot_pixels':int(visible.sum()),
                    'occluded_robot_pixels':int(occluded.sum()),'unknown_robot_pixels':int((unknown&mask).sum()),'unknown_scene_pixels':int(unknown.sum()),
                    'visible_core_byte_exact':exact,'nonvisible_not_pasted':no_paste,'state_unchanged':unchanged,'render_s':elapsed,
                    'rgb':stem+'.png','rgb_sha256':sha(out/(stem+'.png')),'layers':stem+'_layers.npz','layers_sha256':sha(out/(stem+'_layers.npz'))}
                records.append(record);sheet.paste(Image.fromarray(rgb).resize((640,360)),(ci*640,si*400+30));draw.text((ci*640+8,si*400+8),f'Native replay tick {sample["tick"]}; public TRAIN camera{ci}; GS expected depth',fill='black')
    finally:renderer.mj_renderer.close()
    sheet.save(out/'contact_sheet.png')
    receipt={'schema_version':1,'scope':'recorded_native_states_factorized_GS_known_robot_depth_diagnostic','config':config,'frames':records,
        'native_room_geometry_read':False,'heldout_read':False,'policy_input_emitted':False,'new_physics_episode':False,
        'camera_scope':'first two declared TRAIN cameras, not native policy camera replacement',
        'robot_rendering':'known embodiment capsule only; pinned visual groups; capsule headlight, no native room lights',
        'depth_semantics':'approximate GS expected metric camera-z, opacity>=0.95; unknown pixels never authorize robot overlay',
        'background_limitation':'single-plane completion may retain ghosts and has no hidden-surface guarantee',
        'renderer':'robo.rendering.pi05_render.CompositeObs','mujoco_capsule_renderer_version':mujoco.__version__,
        'passed':all(r['state_unchanged'] and r['visible_core_byte_exact'] and r['nonvisible_not_pasted'] for r in records)}
    save(out/'observation_invariance.json',receipt);return receipt
