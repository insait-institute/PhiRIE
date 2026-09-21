import json
import numpy as np
import pytest
from PIL import Image
from robo.roundtrip.workspace_inventory import associate_candidates,discover_inventory
from robo.roundtrip.build import _sha


def test_all_clusters_keep_negative_views_and_empty_obstacle(tmp_path,monkeypatch):
 root=tmp_path/'capture';(root/'train').mkdir(parents=True);(root/'capture_manifest.json').write_text('{}')
 rows=[];files={}
 for i in range(2):
  rgb=f'train/{i}.png';depth=f'train/{i}.npy';Image.fromarray(np.zeros((8,8,3),np.uint8)).save(root/rgb);np.save(root/depth,np.ones((8,8)))
  files[rgb]=_sha(root/rgb);files[depth]=_sha(root/depth)
  rows.append(dict(frame_id=f'f{i}',rgb=rgb,depth_m=depth,K=[[8,0,4],[0,8,4],[0,0,1]],T_world_from_camera=np.eye(4).tolist()))
 monkeypatch.setattr('robo.roundtrip.workspace_inventory.read_train',lambda p:({'files':files},rows))
 declaration=dict(schema_version=1,scope='L2_task_workspace',source_kind='public_train_declaration',geometry_stride=4,workspace_bounds_world_m=[[-1,-1,.1],[1,1,2]],coverage_claim='unverified_approach_and_transport_extent',obstacle_inventory_complete=False,capture_manifest_sha256=_sha(root/'capture_manifest.json'),planned_roles=[dict(role_id=r,role=r,public_phrase=r) for r in ['target','receptacle','source_support','destination_support','obstacle']])
 cfg=dict(workspace_inventory=declaration,capture_manifest_sha256=declaration['capture_manifest_sha256'],score_min=.45,minimum_mask_pixels=4,maximum_points_per_mask=100,maximum_observation_points=200,matching_distance_m=.02,matching_fraction=.2,minimum_views=2)
 def predictor(rgb,phrase):
  if phrase=='obstacle':return []
  a=np.zeros((8,8),bool);a[:2,:2]=True;b=np.zeros_like(a);b[-2:,-2:]=True
  return [(a,.9),(b,.8)]
 report=discover_inventory(root,tmp_path/'out',cfg,predictor)
 assert len(report['observed_clusters'])==8
 assert report['planned_roles'][-1]['state']=='UNRESOLVED_FROM_TRAIN'
 assert not report['obstacle_inventory_complete'] and not report['L2_READY']
 assert all(r['confirmed_views']==2 for r in report['observed_clusters'])
 assert report['read_splits']==['train'] and not report['native_asset_access']
 with pytest.raises(FileExistsError):discover_inventory(root,tmp_path/'out',cfg,predictor)


def test_association_never_uses_two_masks_of_one_view_for_confirmation():
 config=dict(matching_distance_m=.02,matching_fraction=.2)
 rows=[dict(candidate_id=str(i),frame_id='same',status='eligible',score=.9,points=np.zeros((4,3))) for i in range(2)]
 groups=associate_candidates(rows,config)
 assert [len(g) for g in groups]==[1,1]
