"""Adapt observed robot-hole evidence to the maintained planar Gaussian filler."""
from pathlib import Path
import json
import numpy as np
from robo.roundtrip.gaussian_build import sha,save,capture_rows


def prepare(config,out):
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    from PIL import Image
    from agents.edit.inpaint_prepare import fit_plane,RADIUS,RING
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    erasure=Path(config['robot_erasure']);root=json.loads((erasure/'robot_erasure_carve.json').read_text())
    if sha(erasure/'robot_erasure_carve.json')!=config['robot_erasure_sha256'] or root['erased_views']!=root['planned_train_views']:
        raise ValueError('complete sealed robot erasure required')
    for name,key in [('public_observed_robot_points.npy','observed_points_sha256'),('removed_gaussian_indices.npy','removed_indices_sha256')]:
        if sha(erasure/name)!=config[key]:raise ValueError('observed points/removal indices changed')
    scene=Path(config['observed_scene_mesh'])
    if sha(scene)!=config['observed_scene_mesh_sha256']:raise ValueError('observed-only scene mesh changed')
    capture=Path(config['capture']);public,rows=capture_rows(capture)
    if root['capture_manifest_sha256']!=sha(capture/'capture_manifest.json'):raise ValueError('capture identity mismatch')
    ply=PlyData.read(str(scene));verts=np.stack([ply['vertex'][k] for k in 'xyz'],axis=1).astype(float)
    robot=np.load(erasure/'public_observed_robot_points.npy',allow_pickle=False)
    distance,_=cKDTree(robot).query(verts,workers=1);zmin=robot[:,2].min();center=robot.mean(axis=0)
    d_xy=np.linalg.norm(verts[:,:2]-center[:2],axis=1);radius=np.linalg.norm(robot[:,:2]-center[:2],axis=1).max()
    ring=(d_xy>radius)&(d_xy<radius+RING)&(np.abs(verts[:,2]-zmin)<RADIUS)&(distance>RADIUS)
    receipt={'schema_version':1,'scope':'observed_robot_hole_existing_planar_fill','config':config,'ring_points':int(ring.sum()),
        'ring_m':RING,'vertical_band_m':RADIUS,'minimum_ring_points':100,'status':'NOT_RUN','hidden_geometry_read':False,
        'coverage_rule':'only fitted support-plane patch; other occluded surfaces remain unknown','policy_arm':False}
    if ring.sum()<100:
        receipt.update(status='NO_PLANE',reason='existing support-ring minimum not met');save(out/'robot_fill_preparation.json',receipt);return receipt
    origin,normal,u,v,trim_ok=fit_plane(verts[ring])
    if not trim_ok:
        receipt.update(status='NO_PLANE',reason='existing robust plane fit did not converge');save(out/'robot_fill_preparation.json',receipt);return receipt
    projected=robot-origin;uv=np.stack([projected@u,projected@v],axis=1)
    rng=np.random.RandomState(0);uv=uv[rng.choice(len(uv),min(len(uv),800),replace=False)]
    plane={'origin':origin.tolist(),'normal':normal.tolist(),'u':u.tolist(),'v':v.tolist(),'trim_ok':True,'footprint_uv':uv.tolist()}
    # Only masked robot-hole supervision. Target-erased frame is composed first;
    # target-mask pixels never receive a target-present robot-erasure patch.
    target=Path(config['target_erasure']);target_manifest=json.loads((target/'native_erasure.json').read_text())
    if sha(target/'native_erasure.json')!=config['target_erasure_sha256']:raise ValueError('prior target erasure changed')
    targets={r['frame'].removesuffix('.png'):r for r in target_manifest['rows'] if r['status']=='ERASED'}
    if rows[0]['frame_id'] not in targets:raise ValueError('first declared primary TRAIN view requires prior target erasure')
    factory=out/'factory';prep=out/'preparation';slot=prep/'obj_2000';slot.mkdir(parents=True);(factory/'objects').mkdir(parents=True)
    save(factory/'objects/objects.json',[{'index':2000,'source':'known_robot_public_depth','aabb':[robot.min(0).tolist(),robot.max(0).tolist()]}])
    save(slot/'plane.json',plane);np.save(prep/'removal_union_idx.npy',np.load(erasure/'removed_gaussian_indices.npy',allow_pickle=False))
    views=[];masks={};images={};overlaps={}
    for i,(row,erased) in enumerate(zip(rows,root['rows'])):
        if row['frame_id']!=erased['frame_id']:raise ValueError('erasure view order changed')
        rm=np.asarray(Image.open(erasure/erased['mask']))>0;rgb=np.asarray(Image.open(erasure/erased['rgb']).convert('RGB')).copy()
        if sha(erasure/erased['mask'])!=erased['mask_sha256'] or sha(erasure/erased['rgb'])!=erased['rgb_sha256']:raise ValueError('robot erasure bytes changed')
        mask=rm.copy();overlaps[row['frame_id']]=0
        if row['frame_id'] in targets:
            tr=targets[row['frame_id']];tm=np.asarray(Image.open(target/tr['mask']))>0
            if sha(target/tr['mask'])!=tr['mask_sha256'] or sha(target/tr['erased_rgb'])!=tr['erased_rgb_sha256']:raise ValueError('target erasure bytes changed')
            # Overlap is removed from supervision rather than assigning either
            # incompatible single-object erasure as a known joint completion.
            overlap=rm&tm;mask[overlap]=False;overlaps[row['frame_id']]=int(overlap.sum())
            rgb[tm]=np.asarray(Image.open(target/tr['erased_rgb']).convert('RGB'))[tm]
        mp=out/f'robot_loss_mask_{i}.png';ip=out/f'joint_source_{i}.png';Image.fromarray(mask.astype(np.uint8)*255).save(mp);Image.fromarray(rgb).save(ip)
        masks[str(i)]={'path':str(mp),'sha256':sha(mp)};images[str(i)]={'path':str(ip),'sha256':sha(ip)}
        views.append({'frame':row['frame_id']+'.png','w2c':np.linalg.inv(row['T_world_from_camera']).tolist()})
    save(slot/'views.json',views)
    k=rows[0]['K'];intrinsics={'w':public['width'],'h':public['height'],'fl_x':k[0][0],'fl_y':k[1][1],'cx':k[0][2],'cy':k[1][2],'frames':[]}
    save(out/'intrinsics.json',intrinsics)
    receipt.update(status='PREPARED',plane=plane,primary_frame=rows[0]['frame_id'],train_views=len(rows),
        unknown_overlap_pixels=overlaps,supervision='robot mask only; overlaps with prior target erasure excluded, no entire-frame loss',
        masks=masks,erasures=images,train_images={r['frame_id']+'.png':{'path':str(capture/r['rgb']),'sha256':r['rgb_sha256']} for r in rows},
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    save(out/'robot_fill_preparation.json',receipt);return receipt


def fill(config,out):
    """Existing filler; one primary masked view, all six diagnostics retained."""
    prep=Path(config['preparation']);receipt_path=prep/'robot_fill_preparation.json'
    if sha(receipt_path)!=config['preparation_sha256']:raise ValueError('robot fill preparation changed')
    prepared=json.loads(receipt_path.read_text())
    if prepared['status']!='PREPARED':raise ValueError('robot fill requires an admissible observed plane')
    for name,digest in prepared['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(prep/name)!=digest:raise ValueError('prepared robot-fill artifact changed')
    background=Path(config['background'])
    if sha(background)!=config['background_sha256']:raise ValueError('target-clean source background changed')
    public,rows=capture_rows(config['capture'])
    if rows[0]['frame_id']!=prepared['primary_frame']:raise ValueError('primary supervision camera changed')
    from agents.edit.inpaint_fill import _fill,PUBLIC_ALGORITHM
    import torch
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.manual_seed(0)
    masks={};erasures={}
    for i in range(len(rows)):
        masks[('obj_2000',i)]={'path':str(prep/f'robot_loss_mask_{i}.png')}
        erasures[('obj_2000',i)]={'path':str(prep/f'joint_source_{i}.png')}
    result=_fill({'factory':prep/'factory','preparation':prep/'preparation','output':out,
        'masks':masks,'erasures':erasures,'intrinsics':prep/'intrinsics.json','splat':background,
        'train_images':{r['frame_id']+'.png':{'path':str(Path(config['capture'])/r['rgb'])} for r in rows}})
    if result['primary_frames']!=[prepared['primary_frame']+'.png'] or len(result['diagnostic_frames'])!=6:
        raise ValueError('existing filler changed declared supervision/diagnostic roster')
    summary={'schema_version':1,'scope':'first_DEV_known_robot_hole_planar_GS_fill','status':'BUILT','config':config,
        'preparation_sha256':sha(receipt_path),'recipe':PUBLIC_ALGORITHM,'result':result,
        'supervision':prepared['supervision'],'unknown_overlap_pixels':prepared['unknown_overlap_pixels'],
        'plane':{k:v for k,v in prepared['plane'].items() if k!='footprint_uv'},
        'coverage_limit':'one observed support-plane patch; unseen wall/counter/front surfaces not guaranteed recovered',
        'heldout_read':False,'native_room_geometry_read':False,'policy_arm':False,
        'files':{str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()}}
    save(out/'robot_background.json',summary);return summary
