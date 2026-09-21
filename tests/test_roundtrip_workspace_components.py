import json
import numpy as np
import pytest
from robo.roundtrip.build import _sha
from robo.roundtrip.workspace_components import build_components


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    root=tmp_path/'inventory';root.mkdir();capture=tmp_path/'capture';capture.mkdir()
    (capture/'capture_manifest.json').write_text('{}');physics=tmp_path/'physics.json';physics.write_text('{}')
    np.save(root/'points.npy',np.array([[1.,2,3],[3,4,5]]));(root/'mask.png').write_bytes(b'mask')
    cluster=dict(cluster_id='support0',role='source_support',role_id='s',state='OBSERVED_MULTIVIEW',candidate_ids=['m'],points_file='points.npy',points_sha256=_sha(root/'points.npy'))
    report=dict(capture_manifest_sha256=_sha(capture/'capture_manifest.json'),observed_clusters=[cluster],workspace_bounds_world_m=[[0,0,0],[5,5,5]],planned_roles=[dict(role='obstacle',state='UNRESOLVED_FROM_TRAIN')])
    (root/'workspace_inventory.json').write_text(json.dumps(report))
    (root/'mask_candidates.json').write_text(json.dumps([dict(candidate_id='m',role_id='s',frame_id='f',rgb_sha256='rgb',depth_sha256='depth',mask_file='mask.png',mask_sha256=_sha(root/'mask.png'))]))
    monkeypatch.setattr('robo.roundtrip.workspace_components.read_train',lambda p:({'files':{'train/rgb':'rgb','train/depth':'depth'}},[dict(frame_id='f',rgb='train/rgb',depth_m='train/depth')]))
    config=dict(tier='DEV',method='OBSERVED_WORKSPACE_COMPONENT_TSDF',canonical_instance_id='native-dev',cohort_id='dev',cluster_ids=['support0'],fusion={},physics_prior_sha256=_sha(physics),inventory_files={p.name:_sha(p) for p in root.iterdir()})
    return root,capture,physics,tmp_path/'out',config


def test_components_use_observed_center_and_preserve_unresolved_inventory(fixture,monkeypatch):
    def fuse(capture,views,center,physics,dest,config,**kw):
        assert np.array_equal(center,[2,3,4]);assert kw['workspace_bounds_world_m']==[[0,0,0],[5,5,5]]
        dest.mkdir();return dict(collision_parts=2,artifact_hashes={})
    monkeypatch.setattr('robo.roundtrip.workspace_components.fuse_masked_observations',fuse)
    r=build_components(*fixture)
    assert r['built_components']==1 and not r['L2_READY'] and not r['obstacle_inventory_complete']
    assert r['inherited_unresolved_roles'][0]['role']=='obstacle'
    with pytest.raises(FileExistsError):build_components(*fixture)


@pytest.mark.parametrize('mutation',['unsealed','tampered','unknown','duplicate'])
def test_component_closure_and_declared_roster_fail_before_output(fixture,mutation):
    root,capture,physics,out,config=fixture
    if mutation=='unsealed':config['inventory_files'].pop('mask.png')
    if mutation=='tampered':(root/'mask.png').write_bytes(b'changed')
    if mutation=='unknown':config['cluster_ids']=['unknown']
    if mutation=='duplicate':config['cluster_ids']*=2
    with pytest.raises(ValueError):build_components(*fixture)
    assert not out.exists()


def test_actual_build_failure_stays_in_component_denominator(fixture,monkeypatch):
    def fail(*a,**kw):raise RuntimeError('CoACD failed')
    monkeypatch.setattr('robo.roundtrip.workspace_components.fuse_masked_observations',fail)
    r=build_components(*fixture)
    assert r['planned_components']==1 and r['built_components']==0
    assert r['rows'][0]['status']=='BUILD_FAILED' and 'CoACD failed' in r['rows'][0]['error']
