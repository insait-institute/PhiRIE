"""Prepare a declared task workspace from public TRAIN RGB-D, not native assets.

Geometric crops support future role discovery. They are explicitly not semantic
object masks or a certificate that all task obstacles were recovered.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from robo.roundtrip.build import read_train
from robo.roundtrip.capture import backproject_camera_z
from robo.roundtrip.build import _sha as file_hash, _config_sha as canonical_hash


ROLES={'target','receptacle','source_support','destination_support','obstacle'}


def validate_workspace(config):
    if config.get('schema_version')!=1 or config.get('scope')!='L2_task_workspace':
        raise ValueError('workspace requires explicit prospective L2 scope')
    if config.get('source_kind')!='public_train_declaration' or config.get('geometry_stride')!=4:
        raise ValueError('workspace source/geometry stride must be declared before observation processing')
    bounds=np.asarray(config.get('workspace_bounds_world_m'),dtype=float)
    if bounds.shape!=(2,3) or not np.isfinite(bounds).all() or np.any(bounds[1]<=bounds[0]):
        raise ValueError('workspace bounds must be ordered finite world-meter min/max')
    if config.get('coverage_claim')!='unverified_approach_and_transport_extent':
        raise ValueError('existing capture cannot certify approach/transport completeness')
    rows=config.get('planned_roles',[]);ids=[r.get('role_id') for r in rows]
    if not rows or not all(isinstance(x,str) and re.fullmatch(r'[A-Za-z0-9_-]+',x) for x in ids) or len(ids)!=len(set(ids)):
        raise ValueError('nonempty unique planned role IDs required')
    roles=[r.get('role') for r in rows]
    if roles.count('target')!=1 or roles.count('receptacle')!=1 or not {'source_support','destination_support'}.issubset(roles):
        raise ValueError('workspace roster must include target, destination and both supports')
    for row in rows:
        if set(row)!={'role_id','role','public_phrase'} or row['role'] not in ROLES or not isinstance(row['public_phrase'],str) or not row['public_phrase'].strip():
            raise ValueError('role declaration accepts only public phrase and role IDs, no native body/mask mapping')
    if config.get('obstacle_inventory_complete') is not False:
        raise ValueError('unsegmented workspace cannot certify obstacle inventory')
    return bounds


def prepare_workspace(capture,config,out):
    bounds=validate_workspace(config);capture=Path(capture);out=Path(out)
    manifest,frames=read_train(capture)
    capture_hash=file_hash(capture/'capture_manifest.json')
    if config.get('capture_manifest_sha256')!=capture_hash:raise ValueError('workspace capture differs from frozen declaration')
    out.mkdir(parents=True,exist_ok=False);rows=[]
    for frame in frames:
        if not frame['depth_m'].startswith('train/') or frame['depth_m'] not in manifest['files']:
            raise ValueError('workspace input must be a sealed TRAIN observation')
        depth=np.load(capture/frame['depth_m'],allow_pickle=False)
        points,valid=backproject_camera_z(depth,frame['K'],frame['T_world_from_camera'])
        stride=config['geometry_stride'];points=points[::stride,::stride];valid=valid[::stride,::stride]
        mask=valid & np.all(points>=bounds[0],axis=-1)&np.all(points<=bounds[1],axis=-1)
        path=out/(frame['frame_id']+'_observed_points.npy');np.save(path,points[mask],allow_pickle=False)
        rows.append(dict(frame_id=frame['frame_id'],rgb=frame['rgb'],rgb_sha256=manifest['files'][frame['rgb']],
            depth=frame['depth_m'],depth_sha256=manifest['files'][frame['depth_m']],
            camera_row_sha256=canonical_hash(frame),sampled_valid_depth_count=int(valid.sum()),
            observed_workspace_point_count=int(mask.sum()),points_file=path.name,points_sha256=file_hash(path)))
    roles=[dict(**row,state='NOT_RUN',semantic_mask_available=False,reconstructed_artifact=None,
        blocker='semantic role discovery and source-bound construction not yet executed') for row in config['planned_roles']]
    report=dict(schema_version=1,kind='public_train_workspace_observation_manifest',config_sha256=canonical_hash(config),
        capture_manifest_sha256=capture_hash,workspace_bounds_world_m=bounds.tolist(),geometry_stride=config['geometry_stride'],
        scope='L2_task_workspace',construction_completed=False,L2_READY=False,frames=rows,planned_roles=roles,
        observed_frames=sum(r['observed_workspace_point_count']>0 for r in rows),planned_frames=len(frames),
        data_status='AVAILABLE_PUBLIC_TRAIN_DEPTH' if any(r['observed_workspace_point_count'] for r in rows) else 'NO_OBSERVED_POINTS_IN_DECLARED_WORKSPACE',
        heldout_access=False,native_asset_access=False,semantic_segmentation=False,obstacle_inventory_complete=False,
        approach_transport_coverage_certified=False,retained_context_contacts_recorded=False,
        implementation_blockers=['semantic source/destination-support and obstacle discovery','support/obstacle construction adapters',
            'native static fixture binding','L2 complete inventory and retained/outside-region contact accounting'],
        limitation='observed surface samples and planned role denominator only; unobserved or occluded geometry remains unknown')
    with (out/'workspace_manifest.json').open('x') as f:json.dump(report,f,indent=2)
    with (out/'workspace_config.json').open('x') as f:json.dump(config,f,indent=2)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','config','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args(argv);r=prepare_workspace(a.capture,json.loads(Path(a.config).read_text()),a.out)
    print(json.dumps(r,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
