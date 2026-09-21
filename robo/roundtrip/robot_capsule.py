"""Known-embodiment-only robot export for static TRAIN capture decontamination.

The privileged acquisition side supplies the official robot model, never a
filtered room model. Only robot-local assets, base pose and named joint state
leave this boundary. Public RGB-D, not native room visibility, tests occlusion.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import hashlib
import json
import shutil
import numpy as np


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def export_robot(official_robot_xml,native_model,native_data,out,*,asset_roots):
    import mujoco
    root=ET.fromstring(official_robot_xml)
    if root.find('worldbody') is None:
        raise ValueError('self-contained official robot model required')
    if list(root.iter('include')):raise ValueError('robot capsule cannot import external XML')
    world=root.find('worldbody');roots=world.findall('body')
    if not roots or world.findall('geom'):raise ValueError('robot-only body tree required; no world geometry')
    for body in roots:
        name=body.get('name')
        if not name:raise ValueError('named robot base required')
        bid=mujoco.mj_name2id(native_model,mujoco.mjtObj.mjOBJ_BODY,name)
        if bid<0:raise ValueError('known robot base absent in native model')
        if native_model.body_parentid[bid]!=0:raise ValueError('robot base must attach to world')
        body.set('pos',' '.join(repr(float(x)) for x in native_model.body_pos[bid]))
        body.set('quat',' '.join(repr(float(x)) for x in native_model.body_quat[bid]))
        for key in ('euler','axisangle','xyaxes','zaxis'):body.attrib.pop(key,None)
    # The capsule starts from the robot's own upstream MJCF. Never admit assets
    # merely because they happened to occur in a canonical room XML.
    assets=[];allowed=[Path(p).resolve() for p in asset_roots]
    for node in root.iter():
        if 'file' not in node.attrib:continue
        p=Path(node.attrib['file']).resolve()
        if node.tag not in ('mesh','texture') or not p.is_file() or not any(p.is_relative_to(a) for a in allowed):
            raise ValueError('robot asset outside explicit embodiment asset roots')
        assets.append((node,p,_sha(p)))
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    data=mujoco.MjData(model);joints={}
    for j in range(model.njnt):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j)
        nj=mujoco.mj_name2id(native_model,mujoco.mjtObj.mjOBJ_JOINT,name) if name else -1
        if nj<0 or model.jnt_type[j]!=native_model.jnt_type[nj]:raise ValueError('known robot joint identity mismatch')
        size={int(mujoco.mjtJoint.mjJNT_FREE):7,int(mujoco.mjtJoint.mjJNT_BALL):4}.get(int(model.jnt_type[j]),1)
        q=native_data.qpos[native_model.jnt_qposadr[nj]:native_model.jnt_qposadr[nj]+size].copy()
        data.qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j]+size]=q;joints[name]=q.tolist()
    mujoco.mj_forward(model,data)
    errors=[]
    for g in range(model.ngeom):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g)
        ng=mujoco.mj_name2id(native_model,mujoco.mjtObj.mjOBJ_GEOM,name) if name else -1
        if ng<0 or model.geom_type[g]!=native_model.geom_type[ng] or not np.array_equal(model.geom_size[g],native_model.geom_size[ng]):
            raise ValueError('robot geometry differs from declared embodiment')
        errors.append(max(float(np.max(np.abs(data.geom_xpos[g]-native_data.geom_xpos[ng]))),float(np.max(np.abs(data.geom_xmat[g]-native_data.geom_xmat[ng])))))
    if not errors or max(errors)>1e-12:raise ValueError('robot capsule static pose differs from canonical robot')
    out=Path(out);out.mkdir(parents=True,exist_ok=False);(out/'assets').mkdir()
    for node,p,digest in assets:
        dest=out/'assets'/(digest+p.suffix)
        if not dest.exists():shutil.copyfile(p,dest)
        node.set('file',str(dest.resolve()))
    (out/'robot.xml').write_text(ET.tostring(root,encoding='unicode'))
    receipt={'schema_version':1,'scope':'known_robot_embodiment_only','source':'official robot_model.get_xml after base/gripper composition',
        'official_robot_xml_sha256':hashlib.sha256(official_robot_xml.encode()).hexdigest(),'robot_xml_sha256':_sha(out/'robot.xml'),
        'files':{str(p.relative_to(out)):_sha(p) for p in sorted((out/'assets').iterdir())},'joint_qpos':joints,
        'robot_geoms':model.ngeom,'max_static_geometry_pose_error':max(errors),'predeclared_pose_tolerance':1e-12,
        'native_room_geometry_exported':False,'native_object_geometry_exported':False,'native_segmentation_exported':False,
        'visibility_source':'public TRAIN metric depth only; visibility evaluation not yet run'}
    (out/'robot_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    import argparse
    import mujoco
    import robosuite
    from robosuite.robots import ROBOT_CLASS_MAPPING
    p=argparse.ArgumentParser();p.add_argument('--canonical',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    robot_config=json.loads((a.capture/'robot_config.json').read_text())
    robot=ROBOT_CLASS_MAPPING[robot_config['robot']](robot_type=robot_config['robot'],idn=0)
    robot.load_model();official_xml=robot.robot_model.get_xml()
    model=mujoco.MjModel.from_xml_path(str(a.canonical/'scene.xml'));data=mujoco.MjData(model)
    state=json.loads((a.canonical/'canonical_state.json').read_text())
    qpos=np.asarray(state['qpos'],dtype=float)
    if qpos.shape!=(model.nq,) or not np.isfinite(qpos).all():raise ValueError('sealed canonical qpos incompatible')
    data.qpos[:]=qpos;mujoco.mj_forward(model,data)
    receipt=export_robot(official_xml,model,data,a.out/'public_robot',
        asset_roots=[Path(robosuite.__file__).resolve().parent/'models/assets'])
    provenance={'privileged_acquisition':True,'known_embodiment_assistance':'official robot model plus robot-only named joint qpos/base pose',
        'canonical_xml_sha256':_sha(a.canonical/'scene.xml'),'canonical_state_sha256':_sha(a.canonical/'canonical_state.json'),
        'public_capture_manifest_sha256':_sha(a.capture/'capture_manifest.json'),
        'robot_config_sha256':_sha(a.capture/'robot_config.json'),'capsule_manifest_sha256':_sha(a.out/'public_robot/robot_manifest.json'),
        'native_room_rendered':False,'native_segmentation_read':False,'policy_invoked':False}
    (a.out/'private_export_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()


def public_depth_visibility(robot_mask,robot_depth,observed_depth,*,tolerance_m=0.001):
    mask=np.asarray(robot_mask,dtype=bool);rz=np.asarray(robot_depth);z=np.asarray(observed_depth)
    if mask.shape!=rz.shape or z.shape!=mask.shape or not np.isfinite(tolerance_m) or tolerance_m<0:
        raise ValueError('same calibrated depth grid and fixed nonnegative tolerance required')
    valid=np.isfinite(rz)&(rz>0)&np.isfinite(z)&(z>0)
    visible=mask&valid&(np.abs(rz-z)<=tolerance_m)
    occluded=mask&valid&(rz>z+tolerance_m)
    unknown=mask&~(visible|occluded)
    return visible,occluded,unknown


def render_capture_masks(capsule,capture,out):
    """Only sealed known robot and public TRAIN depth; no native room access."""
    import mujoco
    from types import SimpleNamespace
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from robo.roundtrip.gaussian_build import capture_rows
    from robo.rendering.mujoco_masks import render_robot_mask
    capsule=Path(capsule);capture=Path(capture);out=Path(out)
    metadata=json.loads((capsule/'robot_manifest.json').read_text())
    if metadata['scope']!='known_robot_embodiment_only' or _sha(capsule/'robot.xml')!=metadata['robot_xml_sha256']:
        raise ValueError('robot capsule identity mismatch')
    for name,digest in metadata['files'].items():
        if _sha(capsule/name)!=digest:raise ValueError('robot asset identity mismatch')
    public,rows=capture_rows(capture);width=public['width'];height=public['height']
    root=ET.fromstring((capsule/'robot.xml').read_text())
    visual=root.find('visual')
    if visual is None:visual=ET.SubElement(root,'visual')
    glob=visual.find('global')
    if glob is None:glob=ET.SubElement(visual,'global')
    glob.set('offwidth',str(width));glob.set('offheight',str(height))
    for i,row in enumerate(rows):
        k=np.asarray(row['K']);cv=np.asarray(row['T_world_from_camera']);gl=cv@np.diag([1,-1,-1,1])
        if not np.allclose(k,[[k[0,0],0,width/2],[0,k[0,0],height/2],[0,0,1]],rtol=0,atol=1e-9):
            raise ValueError('robot camera renderer supports centered square pixels only')
        quat=Rotation.from_matrix(gl[:3,:3]).as_quat()[[3,0,1,2]]
        ET.SubElement(root.find('worldbody'),'camera',name=f'public_train_{i}',pos=' '.join(map(str,gl[:3,3])),
            quat=' '.join(map(str,quat)),fovy=str(np.rad2deg(2*np.arctan(height/(2*k[1,1])))))
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mujoco.MjData(model)
    for name,q in metadata['joint_qpos'].items():
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if j<0:raise ValueError('capsule joint absent')
        data.qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j]+len(q)]=q
    mujoco.mj_forward(model,data);out.mkdir(parents=True,exist_ok=False);results=[]
    # Same pinned RoboSuite capture convention: visible meshes enabled,
    # collision meshes disabled (environments/base.py _reset_internal).
    option=mujoco.MjvOption();option.geomgroup[0]=0;option.geomgroup[1]=1
    renderer=mujoco.Renderer(model,height=height,width=width)
    try:
        renderer.enable_depth_rendering()
        for i,row in enumerate(rows):
            camera=f'public_train_{i}'
            mask=render_robot_mask(SimpleNamespace(model=model,data=data),camera=camera,width=width,height=height,scene_option=option)>0
            renderer.update_scene(data,camera=camera,scene_option=option);depth=renderer.render().copy()
            observed=np.load(capture/row['depth_m'],allow_pickle=False)
            visible,occluded,unknown=public_depth_visibility(mask,depth,observed)
            frame=row['frame_id'];files={}
            for kind,array in [('silhouette',mask),('visible',visible),('occluded',occluded),('unknown',unknown)]:
                path=out/f'{frame}_{kind}.png';Image.fromarray(array.astype(np.uint8)*255).save(path);files[path.name]=_sha(path)
            results.append({'frame_id':frame,'camera':row,'files':files,'silhouette_pixels':int(mask.sum()),'visible_pixels':int(visible.sum()),'occluded_pixels':int(occluded.sum()),'unknown_pixels':int(unknown.sum())})
    finally:renderer.close()
    receipt={'schema_version':1,'scope':'known_embodiment_public_TRAIN_robot_masks','capsule_sha256':_sha(capsule/'robot_manifest.json'),
        'capture_manifest_sha256':_sha(capture/'capture_manifest.json'),'frames':results,'depth_tolerance_m':0.001,
        'visual_geom_groups':option.geomgroup.tolist(),'renderer_contract':'pinned RoboSuite render_collision_mesh=False, render_visual_mesh=True',
        'tolerance_definition':'predeclared absolute agreement of known robot metric z with public observed metric z',
        'native_room_geometry_read':False,'native_segmentation_read':False,'heldout_read':False,'policy_invoked':False,
        'unknown_background_rule':'covered-by-robot background is unobserved, not recovered ground truth'}
    (out/'robot_train_masks.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def erase_and_carve(capture,masks,masks_sha256,background,background_sha256,
                    checkpoint,checkpoint_sha256,out):
    """Existing eraser and removal primitive; unobserved holes stay explicit."""
    import os
    import torch
    from PIL import Image
    from scipy.spatial import cKDTree
    from plyfile import PlyData,PlyElement
    from agents.edit.inpaint_prepare import surface_removal_indices,RADIUS
    from agents.edit.inpaint_masks import _merge_mask,PUBLIC_ALGORITHM as MASK_RECIPE
    from agents.edit.inpaint_qwen import load_lama,erase_masked_view,PUBLIC_ALGORITHM as ERASE_RECIPE
    from robo.roundtrip.gaussian_build import capture_rows
    capture,masks,background,checkpoint,out=map(Path,(capture,masks,background,checkpoint,out))
    public,rows=capture_rows(capture)
    if _sha(masks/'robot_train_masks.json')!=masks_sha256 or _sha(checkpoint)!=checkpoint_sha256:
        raise ValueError('robot masks or eraser checkpoint identity changed')
    if _sha(background)!=background_sha256:raise ValueError('frozen background changed')
    record=json.loads((masks/'robot_train_masks.json').read_text())
    if (record['scope']!='known_embodiment_public_TRAIN_robot_masks' or record['capture_manifest_sha256']!=_sha(capture/'capture_manifest.json')
            or record.get('visual_geom_groups',[1])[0]!=0 or [r['frame_id'] for r in record['frames']]!=[r['frame_id'] for r in rows]):
        raise ValueError('same public TRAIN capture and native visual-mesh mask convention required')
    out.mkdir(parents=True,exist_ok=False);os.environ['LAMA_MODEL']=str(checkpoint)
    torch.set_num_threads(4);torch.manual_seed(0);model=load_lama();erased=[];points=[]
    for row,mask_record in zip(rows,record['frames']):
        for name,digest in mask_record['files'].items():
            if _sha(masks/name)!=digest:raise ValueError('robot mask pixels changed')
        visible=np.asarray(Image.open(masks/(row['frame_id']+'_visible.png')))>0
        z=np.load(capture/row['depth_m'],allow_pickle=False);v,u=np.nonzero(visible)
        k=np.asarray(row['K']);camera=np.stack([(u-k[0,2])*z[v,u]/k[0,0],(v-k[1,2])*z[v,u]/k[1,1],z[v,u]],axis=1)
        t=np.asarray(row['T_world_from_camera']);points.append(camera@t[:3,:3].T+t[:3,3])
        mask,_=_merge_mask(visible,np.zeros((0,*visible.shape),bool));rgb=np.asarray(Image.open(capture/row['rgb']).convert('RGB'))
        mp=out/(row['frame_id']+'_erasure_mask.png');Image.fromarray(mask.astype(np.uint8)*255).save(mp)
        result={'frame_id':row['frame_id'],'mask':mp.name,'mask_sha256':_sha(mp),'status':'NOT_RUN'}
        try:
            image=erase_masked_view(rgb,mask,model)
            if image is None or not np.array_equal(image[~mask],rgb[~mask]):raise ValueError('erasure did not preserve outside-mask TRAIN bytes')
            path=out/(row['frame_id']+'_erased.png');Image.fromarray(image).save(path)
            result.update(status='ERASED',rgb=path.name,rgb_sha256=_sha(path),outside_mask_equal=True)
        except Exception as error:result.update(status='ENHANCER_FAILED',error={'type':type(error).__name__,'reason':str(error)})
        erased.append(result)
    points=np.concatenate(points);_,idx=np.unique(np.floor(points/.005).astype(np.int64),axis=0,return_index=True);points=points[np.sort(idx)]
    np.save(out/'public_observed_robot_points.npy',points)
    ply=PlyData.read(str(background));xyz=np.stack([ply['vertex'][x] for x in 'xyz'],axis=1)
    removed=surface_removal_indices(cKDTree(xyz),points);keep=np.ones(len(xyz),bool);keep[removed]=False
    np.save(out/'removed_gaussian_indices.npy',removed)
    dest=out/'observed_robot_carved.ply';PlyData([PlyElement.describe(ply['vertex'].data[keep],'vertex')],text=False).write(str(dest))
    receipt={'schema_version':1,'scope':'known_robot_TRAIN_erasure_and_observed_surface_carve','capture_manifest_sha256':_sha(capture/'capture_manifest.json'),
        'mask_receipt_sha256':masks_sha256,'background_sha256':background_sha256,'checkpoint_sha256':checkpoint_sha256,
        'planned_train_views':len(rows),'erased_views':sum(r['status']=='ERASED' for r in erased),'rows':erased,
        'mask_recipe':MASK_RECIPE,'erasure_recipe':ERASE_RECIPE,'observed_voxel_m':.005,'removal_radius_m':RADIUS,
        'observed_robot_points':len(points),'source_gaussians':len(xyz),'removed_gaussians':len(removed),'remaining_gaussians':int(keep.sum()),
        'carved_gaussians_sha256':_sha(dest),'background_completion':'NOT_RUN; removed/occluded surface holes are unknown',
        'complete_robot_removal_guaranteed':False,'hidden_room_geometry_read':False,'heldout_read':False,'policy_arm':False}
    (out/'robot_erasure_carve.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
