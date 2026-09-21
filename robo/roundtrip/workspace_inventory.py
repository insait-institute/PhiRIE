"""Automatic observed L2 role inventory using the existing SAM3/RGB-D producer.

All masks and rejected clusters remain visible. This is not proof of occluded
obstacle absence, complete task-space coverage, or a completed L2 reconstruction.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from robo.roundtrip.build import read_train,_sha,_write,_config_sha
from robo.roundtrip.capture import backproject_camera_z
from robo.roundtrip.workspace import validate_workspace


def associate_candidates(candidates,config):
    """Repeat the established anchor/one-mask-per-view association for all clusters."""
    unused=sorted([r for r in candidates if r['status']=='eligible'],key=lambda r:(-r['score'],-len(r['points']),r['candidate_id']))
    groups=[]
    while unused:
        anchor=unused.pop(0);tree=cKDTree(anchor['points']);group=[anchor];frames={anchor['frame_id']};remaining=[]
        for candidate in unused:
            fraction=float(np.mean(tree.query(candidate['points'])[0]<config['matching_distance_m']))
            if candidate['frame_id'] not in frames and fraction>=config['matching_fraction']:
                candidate['fraction_near_cluster_anchor']=fraction;group.append(candidate);frames.add(candidate['frame_id'])
            else:remaining.append(candidate)
        groups.append(group);unused=remaining
    return groups


def discover_inventory(capture,output,config,predictor):
    declaration=config['workspace_inventory'];bounds=validate_workspace(declaration)
    capture=Path(capture);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest,frames=read_train(capture);capture_sha=_sha(capture/'capture_manifest.json')
    if capture_sha!=declaration['capture_manifest_sha256'] or capture_sha!=config['capture_manifest_sha256']:raise ValueError('workspace TRAIN identity differs')
    role_rows=[];all_clusters=[];candidates=[];maskdir=output/'masks';maskdir.mkdir(exist_ok=False)
    for role in declaration['planned_roles']:
        role_candidates=[]
        for frame in frames:
            depth=np.load(capture/frame['depth_m'],allow_pickle=False);rgb=np.asarray(Image.open(capture/frame['rgb']).convert('RGB'))
            points,valid=backproject_camera_z(depth,frame['K'],frame['T_world_from_camera'])
            in_bounds=valid&np.all(points>=bounds[0],axis=-1)&np.all(points<=bounds[1],axis=-1)
            for index,(mask,score) in enumerate(predictor(rgb,role['public_phrase'])):
                mask=np.asarray(mask)
                if mask.dtype!=np.bool_ or mask.shape!=depth.shape or not np.isfinite(score):raise ValueError('invalid SAM3 mask/score')
                cid=f"{role['role_id']}__{frame['frame_id']}__{index:04d}";path=maskdir/(cid+'.png')
                Image.fromarray(mask.astype(np.uint8)*255).save(path);observed=points[mask&in_bounds]
                if len(observed)>config['maximum_points_per_mask']:observed=observed[np.linspace(0,len(observed)-1,config['maximum_points_per_mask'],dtype=int)]
                row={'candidate_id':cid,'frame_id':frame['frame_id'],'role_id':role['role_id'],'score':float(score),
                    'mask_pixels':int(mask.sum()),'workspace_depth_points_before_subsample':int((mask&in_bounds).sum()),
                    'outside_workspace_depth_points':int((mask&valid&~in_bounds).sum()),'mask_file':str(path.relative_to(output)),
                    'mask_sha256':_sha(path),'rgb_sha256':manifest['files'][frame['rgb']],'depth_sha256':manifest['files'][frame['depth_m']],
                    'camera_sha256':_config_sha(frame),'points':observed,
                    'status':'eligible' if score>=config['score_min'] and len(observed)>=config['minimum_mask_pixels'] else 'insufficient_automatic_mask_evidence'}
                role_candidates.append(row)
        groups=associate_candidates(role_candidates,config);clusters=[]
        for index,group in enumerate(groups):
            cid=role['role_id']+f'-cluster-{index:04d}';point=np.concatenate([r['points'] for r in group])
            if len(point)>config['maximum_observation_points']:point=point[np.linspace(0,len(point)-1,config['maximum_observation_points'],dtype=int)]
            directory=output/'clusters'/cid;directory.mkdir(parents=True)
            np.save(directory/'observation_points.npy',point,allow_pickle=False)
            state='OBSERVED_MULTIVIEW' if len(group)>=config['minimum_views'] else 'INSUFFICIENT_VIEWS'
            row={'cluster_id':cid,'role_id':role['role_id'],'role':role['role'],'public_phrase':role['public_phrase'],
                 'candidate_ids':[r['candidate_id'] for r in group],'confirmed_views':len(group),'state':state,
                 'point_count':len(point),'points_file':str((directory/'observation_points.npy').relative_to(output)),
                 'points_sha256':_sha(directory/'observation_points.npy'),'observed_bounds_world_m':[point.min(0).tolist(),point.max(0).tolist()],
                 'geometry_source':'automatic TRAIN mask and metric depth','mesh_reconstructed':False,'native_role_binding':None}
            _write(directory/'cluster.json',row);clusters.append(row);all_clusters.append(row)
        role_rows.append({**role,'candidate_count':len(role_candidates),'observed_clusters':sum(r['state']=='OBSERVED_MULTIVIEW' for r in clusters),
            'state':'OBSERVED_MULTIVIEW' if any(r['state']=='OBSERVED_MULTIVIEW' for r in clusters) else 'UNRESOLVED_FROM_TRAIN',
            'cluster_ids':[r['cluster_id'] for r in clusters],'absence_certified':False})
        candidates.extend([{k:v for k,v in r.items() if k!='points'} for r in role_candidates])
    _write(output/'mask_candidates.json',candidates)
    report={'schema_version':1,'kind':'automatic_observed_workspace_role_inventory','scope':'L2_task_workspace',
        'capture_manifest_sha256':capture_sha,'config_sha256':_config_sha(config),'workspace_bounds_world_m':bounds.tolist(),
        'planned_roles':role_rows,'observed_clusters':all_clusters,'semantic_model':'existing pinned SAM3 predictor',
        'mask_threshold':config['score_min'],'association_distance_m':config['matching_distance_m'],
        'association_fraction':config['matching_fraction'],'minimum_confirmed_views':config['minimum_views'],
        'read_splits':['train'],'heldout_access':False,'native_asset_access':False,'policy_outcome_access':False,
        'cross_role_instance_aliasing':'UNRESOLVED; overlapping semantic groups are not separate certified objects',
        'obstacle_inventory_complete':False,'approach_transport_coverage_certified':False,'L2_READY':False,
        'unknown_space':'unobserved and occluded surfaces remain unknown; zero masks is not proof of no obstacle',
        'remaining_steps':['role asset construction','resolve semantic aliases','bind every retained/reconstructed native entity after construction freeze',
                           'declared workspace coverage and actual contact audit','complete L2 import and native engineering gate']}
    _write(output/'workspace_inventory.json',report);return report
