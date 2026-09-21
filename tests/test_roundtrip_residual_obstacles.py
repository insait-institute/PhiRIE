import json
from pathlib import Path
import numpy as np
import pytest
from robo.roundtrip.build import _sha
from robo.roundtrip.workspace_components import residual_inventory,build_components


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    root=tmp_path/'coverage';root.mkdir();capture=tmp_path/'capture';(capture/'train').mkdir(parents=True)
    (capture/'capture_manifest.json').write_text('{}');frames=[];files={}
    for i in range(2):
        name=f'train/d{i}.npy';np.save(capture/name,np.ones((8,8)));files[name]=_sha(capture/name);files[f'train/rgb{i}.png']='rgb'
        frames.append(dict(frame_id=f'f{i}',depth_m=name,rgb=f'train/rgb{i}.png',K=[[1000,0,3.5],[0,1000,3.5],[0,0,1]],T_world_from_camera=np.eye(4).tolist()))
    components=[]
    for i,p in enumerate(([0,0,1],[.04,0,1])):
        name=f'residual-{i:06d}.npy';np.save(root/name,np.array([p]));components.append(dict(component_id=f'residual-{i:06d}',points_path=name,points_sha256=_sha(root/name)))
    cov=dict(kind='TRAIN_sampled_workspace_coverage',native_asset_access=False,heldout_access=False,recipe=dict(voxel_m=.04),capture_manifest_sha256=_sha(capture/'capture_manifest.json'),workspace_bounds_world_m=[[-.02,-.02,.98],[.06,.02,1.02]],residual_components=components,unknown_samples=9)
    (root/'workspace_coverage.json').write_text(json.dumps(cov));physics=tmp_path/'physics.json';physics.write_text('{}')
    config=dict(input_kind='frozen_residual_coverage',method='OBSERVED_RESIDUAL_OBSTACLE_TSDF',tier='DEV',canonical_instance_id='native-dev',cohort_id='dev',coverage_recipe=cov['recipe'],cluster_ids=[r['component_id'] for r in components],inventory_files={p.name:_sha(p) for p in root.iterdir()},physics_prior_sha256=_sha(physics),fusion={},residual_conversion=dict(minimum_mask_pixels=64,minimum_views=2,maximum_points_per_mask=10000,maximum_observation_points=20000))
    monkeypatch.setattr('robo.roundtrip.workspace_components.read_train',lambda p:({'files':files},frames))
    return root,capture,tmp_path/'derived',config,physics,tmp_path/'built'


def test_all_residuals_preserved_and_existing_fusion_used(fixture,monkeypatch):
    root,capture,out,config,physics,built=fixture
    c=residual_inventory(root,capture,out,config);report=json.loads((out/'workspace_inventory.json').read_text())
    assert len(report['observed_clusters'])==2 and report['discarded_components']==0
    assert report['observed_clusters'][0]['state']=='OBSERVED_MULTIVIEW'
    assert report['observed_clusters'][1]['state']=='INSUFFICIENT_VIEWS'
    def fuse(capture,views,center,physics,dest,config,**kw):
        assert len(views)==2 and np.allclose(center,[0,0,1]);dest.mkdir();return dict(collision_parts=1,artifact_hashes={})
    monkeypatch.setattr('robo.roundtrip.workspace_components.fuse_masked_observations',fuse)
    result=build_components(out,capture,physics,built,c)
    assert result['planned_components']==2 and result['built_components']==1
    assert result['rows'][1]['status']=='BUILD_FAILED'
    assert result['rows'][0]['semantic_identity']=='UNCLASSIFIED' and result['rows'][0]['native_role_binding'] is None


@pytest.mark.parametrize('mutation',['drop_component','changed_points','threshold','native_access'])
def test_residual_inputs_fail_closed(fixture,mutation):
    root,capture,out,config,*_=fixture
    if mutation=='drop_component':config['cluster_ids']=config['cluster_ids'][:1]
    if mutation=='changed_points':np.save(root/'residual-000000.npy',[[1,2,3]])
    if mutation=='threshold':config['residual_conversion']['minimum_views']=1
    if mutation=='native_access':
        p=root/'workspace_coverage.json';r=json.loads(p.read_text());r['native_asset_access']=True;p.write_text(json.dumps(r));config['inventory_files'][p.name]=_sha(p)
    with pytest.raises(ValueError):residual_inventory(root,capture,out,config)
    assert not out.exists()
