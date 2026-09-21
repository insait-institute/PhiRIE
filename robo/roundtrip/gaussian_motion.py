"""Render-only object GS replay from an existing native B0 body-pose trace."""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robo.manifest.hash import canonical_hash
from robo.roundtrip.gaussian_build import sha,save,capture_rows


def body_matrix(pose):
    value=np.asarray(pose,dtype=float)
    if value.shape!=(7,) or not np.isfinite(value).all() or abs(np.linalg.norm(value[3:])-1)>1e-6:
        raise ValueError('native body pose requires metric XYZ and unit quaternion WXYZ')
    out=np.eye(4);out[:3,3]=value[:3]
    out[:3,:3]=Rotation.from_quat(value[[4,5,6,3]]).as_matrix()
    return out


def asset_world_transform(alignment,initial_body,current_body):
    alignment=np.asarray(alignment,dtype=float)
    if alignment.shape!=(4,4) or not np.isfinite(alignment).all() or not np.allclose(alignment[3],[0,0,0,1]):
        raise ValueError('finite canonical-to-world similarity required')
    scale=np.linalg.norm(alignment[:3,:3],axis=0)
    if (not (scale>0).all() or not np.allclose(scale,scale[0],rtol=1e-7)
            or not np.allclose((alignment[:3,:3]/scale[0]).T@(alignment[:3,:3]/scale[0]),np.eye(3),atol=1e-7)
            or np.linalg.det(alignment[:3,:3])<=0):
        raise ValueError('asset scale must be applied exactly once as a proper similarity')
    initial=body_matrix(initial_body);current=body_matrix(current_body)
    if np.array_equal(np.asarray(initial_body),np.asarray(current_body)):return alignment.copy()
    # Body origin, never center of mass. Initial body-to-asset offset remains
    # fixed, even for an object whose centroid/COM differs from its body origin.
    return current@np.linalg.inv(initial)@alignment


def extract(config,out):
    episode=Path(config['episode']);obj=Path(config['source_object']);background=Path(config['background'])
    result=json.loads((episode/'result.json').read_text())
    initial=json.loads((episode/'initial_state.json').read_text())
    if sha(episode/'result.json')!=config['episode_result_sha256'] or sha(episode/'trace.json.gz')!=config['trace_sha256']:
        raise ValueError('recorded native episode bytes changed')
    if canonical_hash(initial)!=result['initial_state_sha256']:
        raise ValueError('initial native runtime state differs')
    build=json.loads((obj/'build_manifest.json').read_text())
    if sha(obj/'build_manifest.json')!=config['source_object_manifest_sha256']:
        raise ValueError('source constructor manifest changed')
    if (result['controller_method']!='B0_FIXED_NATIVE' or not result['executed'] or result['error'] is not None
            or result['canonical_instance_id']!=build['canonical_instance_id']):
        raise ValueError('same canonical completed fixed B0 trace required')
    aligned_path=obj/'construction/objects/obj_00/aligned.json'
    if sha(aligned_path)!=build['source_hashes']['construction/objects/obj_00/aligned.json']:
        raise ValueError('frozen alignment bytes changed')
    if sha(background/'native_background.json')!=config['background_manifest_sha256']:
        raise ValueError('background receipt changed')
    bg=json.loads((background/'native_background.json').read_text())
    if bg['status']!='BUILT' or sha(background/'background/clean_background.ply')!=bg['clean_background_sha256']:
        raise ValueError('genuine completed clean background required')
    if bg['output_files']['factory/objects/obj_1000/trellis_gs.ply']!=build['source_hashes']['construction/objects/obj_00/trellis_gs.ply']:
        raise ValueError('background and movable asset identities differ')
    bg_alignment=background/'factory/objects/obj_1000/aligned.json'
    if sha(bg_alignment)!=bg['output_files']['factory/objects/obj_1000/aligned.json'] or json.loads(bg_alignment.read_text())['T']!=json.loads(aligned_path.read_text())['T']:
        raise ValueError('background and movable asset registrations differ')
    _,cameras=capture_rows(config['capture'])
    if bg['capture_manifest_sha256']!=build['capture_manifest_sha256'] or bg['capture_manifest_sha256']!=sha(Path(config['capture'])/'capture_manifest.json'):
        raise ValueError('capture/source background identity differs')
    trace=json.load(gzip.open(episode/'trace.json.gz','rt'))
    if len(trace)<5 or len(trace)!=result['ticks'] or [r['tick'] for r in trace]!=list(range(result['ticks'])):
        raise ValueError('incomplete or reordered native trace')
    indices=np.linspace(0,len(trace)-1,5,dtype=int).tolist()
    aligned=json.loads(aligned_path.read_text())['T'];initial_pose=initial['object_states']['obj']
    samples=[{'native_tick':0,'body_pose':initial_pose}]+[
        {'native_tick':trace[i]['tick']+1,'body_pose':trace[i]['objects']['obj']} for i in indices]
    for sample in samples:sample['T_world_from_asset']=asset_world_transform(aligned,initial_pose,sample['body_pose']).tolist()
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    receipt=dict(schema_version=1,scope='recorded_native_body_GS_render_replay',config=config,
        canonical_instance_id=result['canonical_instance_id'],source_object_id=build['object_id'],
        body_role='obj',pose_frame='native body origin; quaternion wxyz; metric meters',
        initial_body_to_asset=(np.linalg.inv(body_matrix(initial_pose))@np.asarray(aligned)).tolist(),
        sampling='initial state plus five uniformly spaced trace entries; no outcome selection',samples=samples,
        camera=cameras[0],camera_rule='first declared TRAIN camera',original_native_success=result['success'],
        physics_executed=False,new_policy_episode=False,static_parked_robot_in_background=True,
        live_robot_occlusion='NOT_RUN',paper_ready=False)
    save(out/'native_pose_to_gs.json',receipt)
    return receipt


def render(config,out):
    from agents.core import common as C
    from PIL import Image,ImageDraw
    import torch
    out=Path(out);receipt=extract(config,out)
    obj=Path(config['source_object'])/'construction/objects/obj_00/trellis_gs.ply'
    build=json.loads((Path(config['source_object'])/'build_manifest.json').read_text())
    if sha(obj)!=build['source_hashes']['construction/objects/obj_00/trellis_gs.ply']:raise ValueError('object GS bytes changed')
    bg=C.load_gaussians(Path(config['background'])/'background/clean_background.ply');asset=C.load_gaussians(obj)
    if asset['sh_degree']!=0:raise ValueError('existing transform has no directional-SH rotation; this diagnostic requires SH0 asset')
    camera=receipt['camera'];w2c=np.linalg.inv(camera['T_world_from_camera']);K=np.asarray(camera['K'])
    public,_=capture_rows(config['capture']);width,height=public['width'],public['height'];frames=[]
    for sample in receipt['samples']:
        moved=C.transform_gaussians(asset,np.asarray(sample['T_world_from_asset']))
        rgb,_,_=C.render_view(C.cat_gaussians([bg,moved]),w2c,K,width,height,scale=1.)
        pixels=np.uint8(np.clip(rgb,0,1)*255);name=f"native_tick_{sample['native_tick']:04}.png"
        Image.fromarray(pixels).save(out/name);frames.append(pixels)
        sample.update(rgb=name,rgb_sha256=sha(out/name))
    repeated=C.transform_gaussians(asset,np.asarray(receipt['samples'][0]['T_world_from_asset']))
    rgb,_,_=C.render_view(C.cat_gaussians([bg,repeated]),w2c,K,width,height,scale=1.)
    same=np.array_equal(frames[0],np.uint8(np.clip(rgb,0,1)*255))
    sheet=Image.new('RGB',(1920,800),'white');draw=ImageDraw.Draw(sheet)
    for index,(sample,pixels) in enumerate(zip(receipt['samples'],frames)):
        x=(index%3)*640;y=(index//3)*400
        draw.text((x+8,y+8),f"Recorded native tick {sample['native_tick']} | render replay; parked robot fixed",fill='black')
        sheet.paste(Image.fromarray(pixels).resize((640,360)),(x,y+32))
    sheet.save(out/'contact_sheet.png')
    receipt.update(static_repeat_rgb_equal=same,source_object_gs_sha256=sha(obj),renderer='agents.core.common',
        renderer_source_sha256=sha(C.__file__),gpu=torch.cuda.get_device_name(0),rendered_frames=len(frames),
        contact_sheet_sha256=sha(out/'contact_sheet.png'))
    save(out/'render_receipt.json',receipt)
    if not same:raise ValueError('static repeated GS render differed')
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--extract-only',action='store_true');a=p.parse_args();config=json.loads(a.config.read_text())
    print(json.dumps((extract if a.extract_only else render)(config,a.out)))


if __name__=='__main__':main()
