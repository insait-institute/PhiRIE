"""TRAIN-only sampled workspace visibility and residual occupied inventory.

Grid samples are diagnostic: free sample centers do not certify free voxels or
the absence of thin/occluded obstacles. No evaluator identities are accepted.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from PIL import Image
from robo.roundtrip.build import read_train, _sha, _write, _config_sha

RECIPE=dict(schema_version=1,tier='DEV',voxel_m=.04,depth_band_m=.02,
    connectivity=6,component_minimum_voxels=1,role_mask_policy='all_eligible_candidates',
    spatial_extent='entire_predeclared_workspace_including_approach_and_transport',
    robot_exclusion='NONE; unclassified residual may include known robot',
    absence_claim=False)


def sample_depth(points, depth, K, T, band):
    """Camera-Z convention; missing/behind/off-camera points stay unknown."""
    p=(points-np.asarray(T)[:3,3])@np.asarray(T)[:3,:3]
    projected=p@np.asarray(K).T
    uv=np.zeros((len(p),2),dtype=int);front=p[:,2]>0
    uv[front]=np.rint(projected[front,:2]/projected[front,2,None]).astype(int)
    inside=front&(uv[:,0]>=0)&(uv[:,1]>=0)&(uv[:,0]<depth.shape[1])&(uv[:,1]<depth.shape[0])
    z=np.full(len(p),np.nan);z[inside]=depth[uv[inside,1],uv[inside,0]]
    good=inside&np.isfinite(z)&(z>0)
    surface=good&(np.abs(z-p[:,2])<=band)
    free=good&(p[:,2]<z-band)
    return free,surface,uv


def inspect(capture, inventory, out, config):
    if config['recipe']!=RECIPE:raise ValueError('workspace recipe must be prospectively frozen')
    capture,inventory,out=map(Path,(capture,inventory,out))
    for n,h in config['inventory_files'].items():
        if Path(n).is_absolute() or '..' in Path(n).parts or _sha(inventory/n)!=h:raise ValueError('inventory closure differs')
    report=json.loads((inventory/'workspace_inventory.json').read_text())
    candidates=json.loads((inventory/'mask_candidates.json').read_text())
    manifest,frames=read_train(capture)
    if _sha(capture/'capture_manifest.json')!=report['capture_manifest_sha256']:raise ValueError('TRAIN capture identity differs')
    bounds=np.asarray(report['workspace_bounds_world_m'],float)
    if bounds.shape!=(2,3) or not np.isfinite(bounds).all() or np.any(bounds[1]<=bounds[0]):raise ValueError('invalid workspace bounds')
    axes=[np.arange(bounds[0,i]+RECIPE['voxel_m']/2,bounds[1,i],RECIPE['voxel_m']) for i in range(3)]
    shape=tuple(map(len,axes))
    points=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
    if len(points)>2000000:raise ValueError('bounded grid exceeded')
    surface_count=np.zeros(len(points),np.uint16);free_count=surface_count.copy();assigned=np.zeros(len(points),bool)
    view_rows=[]
    for f in frames:
        depth=np.load(capture/f['depth_m'],allow_pickle=False)
        free,surface,uv=sample_depth(points,depth,f['K'],f['T_world_from_camera'],RECIPE['depth_band_m'])
        role_mask=np.zeros(depth.shape,bool)
        for c in candidates:
            if c['frame_id']!=f['frame_id'] or c['status']!='eligible':continue
            if c['mask_file'] not in config['inventory_files'] or _sha(inventory/c['mask_file'])!=c['mask_sha256']:raise ValueError('role mask not sealed')
            mask=np.asarray(Image.open(inventory/c['mask_file']))>0
            if mask.shape!=depth.shape:raise ValueError('role mask camera shape differs')
            role_mask|=mask
        ids=np.flatnonzero(surface)
        assigned[ids]|=role_mask[uv[ids,1],uv[ids,0]]
        free_count+=free;surface_count+=surface
        view_rows.append(dict(frame_id=f['frame_id'],camera_sha256=_config_sha(f),free_samples=int(free.sum()),surface_samples=int(surface.sum())))
    residual=(surface_count>0)&~assigned
    components,count=label(residual.reshape(shape))
    out.mkdir(parents=True,exist_ok=False);rows=[]
    for i in range(1,count+1):
        ids=np.flatnonzero(components.ravel()==i);path=out/f'residual-{i:06d}.npy';np.save(path,points[ids],allow_pickle=False)
        rows.append(dict(component_id=f'residual-{i:06d}',sample_count=len(ids),points_path=path.name,points_sha256=_sha(path),
            state='UNCLASSIFIED_OCCUPIED_TRAIN_SAMPLES',minimum_confirmed_views=int(surface_count[ids].min()),
            bounds_world_m=[points[ids].min(0).tolist(),points[ids].max(0).tolist()],discarded=False))
    np.savez_compressed(out/'workspace_samples.npz',points_world_m=points,free_views=free_count,surface_views=surface_count,assigned_role=assigned)
    unknown=(free_count==0)&(surface_count==0)
    result=dict(schema_version=1,kind='TRAIN_sampled_workspace_coverage',recipe=RECIPE,config_sha256=_config_sha(config),
        capture_manifest_sha256=report['capture_manifest_sha256'],inventory_manifest_sha256=_sha(inventory/'workspace_inventory.json'),
        workspace_bounds_world_m=bounds.tolist(),sample_count=len(points),unknown_samples=int(unknown.sum()),
        observed_surface_samples=int((surface_count>0).sum()),unclassified_surface_samples=int(residual.sum()),
        free_surface_conflict_samples=int(((free_count>0)&(surface_count>0)).sum()),
        view_rows=view_rows,residual_components=rows,discarded_components=0,heldout_access=False,native_asset_access=False,
        approach_transport='same complete declared extent sampled; no policy trajectory outcome selection',
        thin_obstacle_absence_certified=False,occluded_obstacle_absence_certified=False,L2_READY=False,
        admission='COMPLETENESS_NOT_ESTABLISHED',controller_decision=None,
        unknown_voxels_are_not_a_controller_gate=True,sample_file_sha256=_sha(out/'workspace_samples.npz'))
    _write(out/'workspace_coverage.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('capture','inventory','out','config'):p.add_argument('--'+n,required=True)
    a=p.parse_args();inspect(a.capture,a.inventory,a.out,json.loads(Path(a.config).read_text()))
if __name__=='__main__':main()
