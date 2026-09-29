import json
import pytest
from physicalview.paper_require_complete import require_complete
from physicalview.paper_capture import FEATURES


def test_dataset_gate_rejects_incomplete_images_even_if_feature_says_captured(tmp_path):
    rows=[{'scene':f'scene{i}'} for i in range(10)]
    (tmp_path/'roster.json').write_text(json.dumps({'datasets':{'scannetpp':rows}}))
    for row in rows:
        folder=tmp_path/'scannetpp'/row['scene'];folder.mkdir(parents=True)
        features={}
        for key in FEATURES:
            image=folder/key/'frame.png';image.parent.mkdir();image.write_bytes(b'fixture')
            image.with_suffix('.json').write_text('{}')
            features[key]={'status':'captured','frames':[f'{key}/frame.png']}
        (folder/'capture-complete.json').write_text(json.dumps({'features':features}))
    result=require_complete(tmp_path,'scannetpp')
    assert result['captured_groups']==140
    assert result['publication_quality_verified'] is False
    (tmp_path/'scannetpp/scene9/l_robot/frame.json').unlink()
    with pytest.raises(RuntimeError,match='scene9/l_robot:missing'):
        require_complete(tmp_path,'scannetpp')


def test_bounded_retry_archives_replaced_group_and_retains_unrelated_evidence(tmp_path):
    from physicalview.paper_capture import prepare_feature_retry
    records={'a_original':{'status':'captured'},'h_prompt_inpaint':{'status':'captured'}}
    (tmp_path/'features.json').write_text(json.dumps(records))
    for key in records:
        folder=tmp_path/key;folder.mkdir();(folder/'frame.png').write_bytes(b'original pixels')
        (folder/'review.json').write_text('{"status":"approved"}')
    archive=prepare_feature_retry(tmp_path,['h_prompt_inpaint'])
    assert (archive/'h_prompt_inpaint/frame.png').read_bytes()==b'original pixels'
    assert not (tmp_path/'h_prompt_inpaint').exists()
    assert (tmp_path/'a_original/frame.png').read_bytes()==b'original pixels'
    assert json.loads((tmp_path/'features.json').read_text())==records


def test_behavior_capture_does_not_fuse_different_object_states_or_episodes():
    import numpy as np
    from physicalview.paper_behavior import select_consistent_clips
    def pose(v):
        T=np.eye(4);T[:3,3]=v[:3];return T
    def clip(ep,index,cup_x):
        return {'ep':ep,'clip_id':f'{ep}-{index}','D':np.eye(4),'cam_frames':[{},{}],
            'traj':{name:np.array([[x,0,0,0,0,0,1]]) for name,x in [('floor',0),('wall',1),('table',2),('cup',cup_x)]}}
    clips=[clip('a',0,3),clip('a',1,3.001),clip('a',2,3.2),clip('b',0,4)]
    kept,report=select_consistent_clips(clips,pose)
    assert [c['clip_id'] for c in kept]==['a-0','a-1']
    assert report['selected_episode']=='a'
    assert report['selected_camera_frames']==4
    assert report['episodes'][0]['audit'][2]['kept'] is False


def test_behavior_reference_uses_numeric_clip_time():
    import numpy as np
    from physicalview.paper_behavior import select_consistent_clips
    clips=[{'ep':'episode','clip_id':f'episode-{t}:{t+11}','D':np.eye(4),'cam_frames':[{}],
            'traj':{n:np.array([[0,0,0,0,0,0,1]]) for n in ('floor','wall','table')}} for t in (1000,200)]
    _,report=select_consistent_clips(clips,lambda _:np.eye(4))
    assert report['reference_clip']=='episode-200:211'


def test_gallery_does_not_count_captured_flag_without_image_sidecar(tmp_path):
    from PIL import Image
    from physicalview.paper_pack import main
    scene=tmp_path/'scannetpp'/'scene';feature=scene/'a_original';feature.mkdir(parents=True)
    Image.new('RGB',(32,24)).save(feature/'frame.png')
    (scene/'features.json').write_text(json.dumps({'a_original':{'status':'captured','frames':['a_original/frame.png']}}))
    main(['--root',str(tmp_path)])
    rows=json.loads((tmp_path/'coverage.json').read_text())
    assert rows[0]['status']=='missing_artifacts'
    assert json.loads((tmp_path/'progress-summary.json').read_text())['datasets']['scannetpp']['captured_groups']==0


def test_native_libero_freezes_objects_omitted_by_discovery(tmp_path):
    import mujoco
    from physicalview.paper_libero import export_native_physics
    scene=tmp_path/'source';scene.mkdir();(scene/'scans').mkdir()
    (scene/'native-scene.xml').write_text('''<mujoco><worldbody>
      <body name="detected" pos="0 0 1"><freejoint/><geom name="a" type="box" size=".1 .1 .1"/></body>
      <body name="omitted" pos="1 0 1"><joint type="free"/><geom name="b" type="box" size=".1 .1 .1"/></body>
    </worldbody></mujoco>''')
    (scene/'scans/segments_anno.json').write_text(json.dumps({'segGroups':[
        {'objectId':1,'native_body':'detected'},{'objectId':2,'native_body':'omitted'}]}))
    receipt=export_native_physics(scene,tmp_path/'asset',[{'index':0,'gt_object_id':1}],{'obj_00'})
    model=mujoco.MjModel.from_xml_path(str(tmp_path/'asset/sim_export/scene.xml'))
    assert model.nv==6
    assert model.body('omitted').jntnum==0
    assert receipt['frozen']==['omitted']
    assert receipt['maximum_geom_position_error_m']==0
