import copy
import json
import numpy as np
import pytest
from robo.roundtrip.workspace import prepare_workspace,validate_workspace
from robo.roundtrip.identity import file_hash


def config():
    return dict(schema_version=1,scope='L2_task_workspace',source_kind='public_train_declaration',geometry_stride=4,
        workspace_bounds_world_m=[[-1,-1,.1],[1,1,2]],coverage_claim='unverified_approach_and_transport_extent',
        obstacle_inventory_complete=False,planned_roles=[dict(role_id=role,role=role,public_phrase=phrase) for role,phrase in
            [('target','lemon'),('receptacle','plate'),('source_support','sink bottom'),('destination_support','countertop')]])


def test_public_workspace_points_do_not_become_semantic_recovery(tmp_path,monkeypatch):
    root=tmp_path/'public';(root/'train').mkdir(parents=True)
    (root/'capture_manifest.json').write_text('{}');np.save(root/'train/depth.npy',np.ones((8,8)))
    row=dict(frame_id='f000000',depth_m='train/depth.npy',rgb='train/rgb.png',K=[[4,0,4],[0,4,4],[0,0,1]],T_world_from_camera=np.eye(4).tolist())
    manifest={'files':{'train/depth.npy':file_hash(root/'train/depth.npy'),'train/rgb.png':'frozen-rgb'}}
    monkeypatch.setattr('robo.roundtrip.workspace.read_train',lambda p:(manifest,[row]))
    c=config();c['capture_manifest_sha256']=file_hash(root/'capture_manifest.json')
    r=prepare_workspace(root,c,tmp_path/'output')
    assert r['data_status']=='AVAILABLE_PUBLIC_TRAIN_DEPTH' and r['observed_frames']==1
    assert not r['L2_READY'] and not r['semantic_segmentation'] and not r['obstacle_inventory_complete']
    assert all(x['state']=='NOT_RUN' for x in r['planned_roles'])
    assert np.load(tmp_path/'output/f000000_observed_points.npy').shape==(4,3)
    with pytest.raises(FileExistsError):prepare_workspace(root,c,tmp_path/'output')
    c['capture_manifest_sha256']='changed'
    with pytest.raises(ValueError,match='frozen'):prepare_workspace(root,c,tmp_path/'changed')


@pytest.mark.parametrize('kind',['native_role','complete','missing_support','bounds'])
def test_workspace_cannot_claim_unavailable_role_or_coverage(kind):
    c=config()
    if kind=='native_role':c['planned_roles'][0]['body_name']='native_obj'
    if kind=='complete':c['obstacle_inventory_complete']=True
    if kind=='missing_support':c['planned_roles'].pop()
    if kind=='bounds':c['workspace_bounds_world_m']=[[1,1,1],[0,0,0]]
    with pytest.raises(ValueError):validate_workspace(c)
