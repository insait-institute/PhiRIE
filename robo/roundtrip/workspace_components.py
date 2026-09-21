"""Build explicitly partial observed workspace supports with the existing TSDF core."""
import argparse
import json
from pathlib import Path
import numpy as np
from robo.roundtrip.build import _sha,_write,read_train
from robo.roundtrip.shared_candidates import fuse_masked_observations


def residual_inventory(coverage_root,capture,out,config):
    """Convert every frozen residual voxel component to actual TRAIN depth masks.

    This creates geometric candidates, not semantic obstacle identities. No
    native entity assignment or unseen-space completion is manufactured.
    """
    from PIL import Image
    from robo.roundtrip.capture import backproject_camera_z
    from robo.roundtrip.build import _config_sha
    root,capture,out=map(Path,(coverage_root,capture,out))
    expected=dict(minimum_mask_pixels=64,minimum_views=2,maximum_points_per_mask=10000,maximum_observation_points=20000)
    if config.get('residual_conversion')!=expected or config.get('tier')!='DEV':raise ValueError('residual conversion must retain fixed DEV workspace evidence settings')
    for name,digest in config['inventory_files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or (root/name).is_symlink() or _sha(root/name)!=digest:raise ValueError('residual coverage closure differs')
    if 'workspace_coverage.json' not in config['inventory_files']:raise ValueError('coverage receipt not sealed')
    cov=json.loads((root/'workspace_coverage.json').read_text());manifest,frames=read_train(capture)
    if cov.get('kind')!='TRAIN_sampled_workspace_coverage' or cov.get('native_asset_access') is not False or cov.get('heldout_access') is not False or cov['recipe']!=config['coverage_recipe'] or cov['recipe']['voxel_m']!=.04:
        raise ValueError('only the sealed TRAIN-only residual recipe is allowed')
    if cov['capture_manifest_sha256']!=_sha(capture/'capture_manifest.json'):raise ValueError('residual capture differs')
    components=cov['residual_components'];ids=[r['component_id'] for r in components]
    if ids!=config['cluster_ids'] or len(ids)!=len(set(ids)):raise ValueError('all fixed residual components required in original order')
    bounds=np.asarray(cov['workspace_bounds_world_m'],float);voxel=cov['recipe']['voxel_m']
    if bounds.shape!=(2,3) or not np.isfinite(bounds).all() or np.any(bounds[1]<=bounds[0]):raise ValueError('invalid coverage bounds')
    shape=tuple(len(np.arange(bounds[0,i]+voxel/2,bounds[1,i],voxel)) for i in range(3))
    if np.prod(shape)>2000000:raise ValueError('bounded residual grid exceeded')
    labels=np.zeros(shape,np.int32)
    for i,row in enumerate(components,1):
        name=row['points_path']
        if name not in config['inventory_files'] or config['inventory_files'][name]!=row['points_sha256']:raise ValueError('residual points not sealed')
        p=np.load(root/name,allow_pickle=False)
        if p.ndim!=2 or p.shape[1]!=3 or not len(p) or not np.isfinite(p).all():raise ValueError('invalid residual sample coordinates')
        ix=np.floor((p-bounds[0])/voxel).astype(int)
        if np.any(ix<0) or np.any(ix>=shape) or not np.allclose(p,bounds[0]+(ix+.5)*voxel,rtol=0,atol=1e-10):raise ValueError('residual samples differ from frozen voxel lattice')
        if np.any(labels[tuple(ix.T)]):raise ValueError('overlapping residual component IDs')
        labels[tuple(ix.T)]=i
    out.mkdir(parents=True,exist_ok=False);(out/'masks').mkdir();candidates=[];observations={cid:[] for cid in ids};selected={cid:[] for cid in ids}
    for frame in frames:
        depth=np.load(capture/frame['depth_m'],allow_pickle=False);points,valid=backproject_camera_z(depth,frame['K'],frame['T_world_from_camera'])
        inside=valid&np.all(points>=bounds[0],axis=-1)&np.all(points<bounds[1],axis=-1)
        ix=np.zeros((*depth.shape,3),int);ix[inside]=np.floor((points[inside]-bounds[0])/voxel).astype(int)
        inside&=np.all(ix<shape,axis=-1);pixel_labels=np.zeros(depth.shape,np.int32);pixel_labels[inside]=labels[tuple(ix[inside].T)]
        for i,cid in enumerate(ids,1):
            mask=pixel_labels==i;observed=points[mask];count=len(observed);candidate_id=cid+'__'+frame['frame_id'];path=out/'masks'/(candidate_id+'.png');Image.fromarray(mask.astype(np.uint8)*255).save(path)
            if len(observed)>expected['maximum_points_per_mask']:observed=observed[np.linspace(0,len(observed)-1,expected['maximum_points_per_mask'],dtype=int)]
            eligible=count>=expected['minimum_mask_pixels']
            row=dict(candidate_id=candidate_id,frame_id=frame['frame_id'],role_id=cid,mask_file=str(path.relative_to(out)),mask_sha256=_sha(path),rgb_sha256=manifest['files'][frame['rgb']],depth_sha256=manifest['files'][frame['depth_m']],camera_sha256=_config_sha(frame),mask_pixels=count,status='eligible' if eligible else 'insufficient_pixel_support',semantic_confidence=None,source='frozen residual voxel membership of TRAIN depth pixels')
            candidates.append(row)
            if eligible:selected[cid].append(candidate_id);observations[cid].append(observed)
    clusters=[];roles=[]
    for cid in ids:
        points=np.concatenate(observations[cid]) if observations[cid] else np.empty((0,3))
        if len(points)>expected['maximum_observation_points']:points=points[np.linspace(0,len(points)-1,expected['maximum_observation_points'],dtype=int)]
        directory=out/'clusters'/cid;directory.mkdir(parents=True);pointfile=directory/'observation_points.npy';np.save(pointfile,points,allow_pickle=False)
        state='OBSERVED_MULTIVIEW' if len(selected[cid])>=expected['minimum_views'] else 'INSUFFICIENT_VIEWS'
        row=dict(cluster_id=cid,role_id=cid,role='obstacle',candidate_ids=selected[cid],state=state,confirmed_views=len(selected[cid]),point_count=len(points),points_file=str(pointfile.relative_to(out)),points_sha256=_sha(pointfile),semantic_identity='UNCLASSIFIED',native_role_binding=None)
        _write(directory/'cluster.json',row);clusters.append(row);roles.append(dict(role_id=cid,role='obstacle',state=state,semantic_identity='UNCLASSIFIED',absence_certified=False))
    _write(out/'mask_candidates.json',candidates)
    report=dict(schema_version=1,kind='residual_geometric_candidate_inventory',canonical_instance_id=config['canonical_instance_id'],capture_manifest_sha256=cov['capture_manifest_sha256'],workspace_bounds_world_m=bounds.tolist(),planned_roles=roles,observed_clusters=clusters,source_coverage_sha256=_sha(root/'workspace_coverage.json'),inherited_unknown_samples=cov['unknown_samples'],discarded_components=0,heldout_access=False,native_asset_access=False,L2_READY=False,native_role_mapping='UNRESOLVED: residual components have no predeclared semantic instance correspondence')
    _write(out/'workspace_inventory.json',report)
    derived=dict(config,method='OBSERVED_RESIDUAL_OBSTACLE_TSDF',inventory_files={str(p.relative_to(out)):_sha(p) for p in out.rglob('*') if p.is_file()})
    _write(out/'derived_component_config.json',derived)
    return derived


def build_components(inventory,capture,physics,out,config):
    inventory=Path(inventory);capture=Path(capture);out=Path(out)
    if config.get('tier')!='DEV' or config.get('method') not in ('OBSERVED_WORKSPACE_COMPONENT_TSDF','OBSERVED_RESIDUAL_OBSTACLE_TSDF'):raise ValueError('separate observed DEV component method required')
    paths=config['inventory_files']
    for name,digest in paths.items():
        p=Path(name);file=inventory/p
        if p.is_absolute() or '..' in p.parts or file.is_symlink() or _sha(file)!=digest:raise ValueError('observed inventory closure changed')
    if not {'workspace_inventory.json','mask_candidates.json'}<=set(paths):raise ValueError('inventory manifest files not sealed')
    report=json.loads((inventory/'workspace_inventory.json').read_text());manifest,frames=read_train(capture)
    if _sha(capture/'capture_manifest.json')!=report['capture_manifest_sha256'] or _sha(physics)!=config['physics_prior_sha256']:
        raise ValueError('observed component capture or physical prior differs')
    ids=config['cluster_ids']
    if not ids or len(ids)!=len(set(ids)):raise ValueError('unique fixed observed component roster required')
    clusters={r['cluster_id']:r for r in report['observed_clusters']};candidates={r['candidate_id']:r for r in json.loads((inventory/'mask_candidates.json').read_text())};cameras={r['frame_id']:r for r in frames}
    if any(cid not in clusters or clusters[cid]['role'] not in ('source_support','destination_support','obstacle') for cid in ids):raise ValueError('unresolved workspace role cannot become a component')
    required={clusters[cid]['points_file'] for cid in ids}
    required.update(candidates[mid]['mask_file'] for cid in ids for mid in clusters[cid]['candidate_ids'])
    if not required<=set(paths):raise ValueError('component input file not sealed in inventory closure')
    out.mkdir(parents=True,exist_ok=False);rows=[]
    for cid in ids:
        cluster=clusters[cid];dest=out/cid
        try:
            if cluster['state']!='OBSERVED_MULTIVIEW':raise ValueError('component lacks confirmed TRAIN views')
            views=[]
            for mid in cluster['candidate_ids']:
                row=candidates[mid];camera=cameras[row['frame_id']]
                if row['role_id']!=cluster['role_id'] or not camera['rgb'].startswith('train/') or not camera['depth_m'].startswith('train/'):
                    raise ValueError('role or TRAIN view association differs')
                if manifest['files'][camera['rgb']]!=row['rgb_sha256'] or manifest['files'][camera['depth_m']]!=row['depth_sha256']:
                    raise ValueError('source RGB-D hash differs')
                views.append(dict(camera=camera,mask_path=inventory/row['mask_file'],mask_sha256=row['mask_sha256'],candidate_id=mid))
            pointfile=inventory/cluster['points_file']
            if _sha(pointfile)!=cluster['points_sha256']:raise ValueError('observed point center changed')
            summary=fuse_masked_observations(capture,views,np.load(pointfile,allow_pickle=False).mean(0),physics,dest,
                config['fusion'],workspace_bounds_world_m=report['workspace_bounds_world_m'])
            row=dict(cluster_id=cid,role=cluster['role'],status='BUILT',object_dir=str(dest.resolve()),
                method=config['method'],native_import='NOT_RUN',geometry_source='automatic TRAIN masks and metric depth',
                completion='CoACD convex collision approximation of partial observed TSDF',**summary)
            if config['method']=='OBSERVED_RESIDUAL_OBSTACLE_TSDF':
                from robo.roundtrip.build import _config_sha
                row.update(canonical_component_id='residual-'+_config_sha(dict(canonical_instance_id=config['canonical_instance_id'],capture_manifest_sha256=report['capture_manifest_sha256'],cluster_id=cid)),
                    canonical_instance_id=config['canonical_instance_id'],capture_manifest_sha256=report['capture_manifest_sha256'],semantic_identity='UNCLASSIFIED',native_role_binding=None)
            _write(dest/'workspace_component_receipt.json',row)
        except Exception as exc:
            row=dict(cluster_id=cid,role=cluster['role'],status='BUILD_FAILED',error=f'{type(exc).__name__}: {exc}',partial_output=str(dest.resolve()))
        rows.append(row)
    result=dict(schema_version=1,canonical_instance_id=config['canonical_instance_id'],cohort_id=config['cohort_id'],method=config['method'],tier='DEV',planned_components=len(ids),built_components=sum(r['status']=='BUILT' for r in rows),
        rows=rows,inventory_manifest_sha256=_sha(inventory/'workspace_inventory.json'),capture_manifest_sha256=report['capture_manifest_sha256'],
        physics_prior_sha256=config['physics_prior_sha256'],physical_prior='fixed category-independent source bytes; no native mass/friction fit',
        inherited_unresolved_roles=[r for r in report['planned_roles'] if r['state']!='OBSERVED_MULTIVIEW'],
        cross_role_aliasing='UNRESOLVED',obstacle_inventory_complete=False,L2_READY=False,heldout_access=False,native_asset_access=False,
        policy_outcomes='NOT_RUN',native_role_mapping='NOT_RUN; only allowed after component freeze')
    _write(out/'workspace_components.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['inventory','capture','physics','out','config']:p.add_argument('--'+name,required=True)
    a=p.parse_args();config=json.loads(Path(a.config).read_text());inventory=a.inventory
    if config.get('input_kind')=='frozen_residual_coverage':
        inventory=Path(a.out).parent/'derived_residual_inventory'
        config=residual_inventory(a.inventory,a.capture,inventory,config)
    build_components(inventory,a.capture,a.physics,a.out,config)
if __name__=='__main__':main()
