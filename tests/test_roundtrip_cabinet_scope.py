import copy
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from robo.roundtrip.fixture_scope import component_roster,import_fixture_component,import_fixture_identity,check_native_sink_scorer_identity
from robo.roundtrip.scope_capture import cabinet_view_plan
from robo.roundtrip.capture_native import NATIVE_CAMERAS
from robo.roundtrip.build import task_role_prompt
from tests.test_roundtrip_scope import factory


XML='''<mujoco><worldbody><body name="cab_main" pos="1 2 1">
<geom name="cab_bottom" type="box" size=".3 .2 .01"/>
<geom name="cab_bottom_visual" type="box" size=".3 .2 .01" group="1" contype="0" conaffinity="0"/>
<geom name="cab_wall" type="box" size=".01 .2 .3" pos=".3 0 .3"/>
<geom name="cab_reg_level0" type="box" size=".3 .2 .2" pos="0 0 .2" rgba="0 1 0 0" group="1" contype="0" conaffinity="0"/>
<body name="cab_door"><joint name="cab_hinge"/><geom name="cab_door_geom" type="box" size=".3 .01 .3"/></body>
<body name="cab_upper"><geom name="cab_upper_shelf" type="box" size=".3 .2 .01" pos="0 0 .4"/></body>
</body></worldbody></mujoco>'''


def test_bottom_shelf_preserves_wall_doors_upper_shelf_and_goals(tmp_path):
    pytest.importorskip("robocasa", reason="requires the separate native robot runtime")
    import mujoco
    output,r=import_fixture_component(XML,body_name='cab_main',object_dir=factory(tmp_path),object_id='shelf',role='support',component_kind='cabinet_bottom_shelf')
    old,new=ET.fromstring(XML),ET.fromstring(output)
    for name in ['cab_wall','cab_door_geom','cab_upper_shelf','cab_reg_level0']:
        assert ET.tostring(old.find('.//geom[@name="'+name+'"]'))==ET.tostring(new.find('.//geom[@name="'+name+'"]'))
    assert r['removed_original_geoms']==['cab_bottom','cab_bottom_visual']
    assert r['component_roster']['retained_direct_physical_geoms']==['cab_wall']
    assert new.find('.//geom[@name="cab_bottom"]') is None
    mujoco.MjModel.from_xml_string(output)
    report=check_native_sink_scorer_identity(XML,output,body_name='cab_main',task_id='PickPlaceCounterToCabinet')
    assert report['status']=='PASS' and report['contains_positive_and_negative']


def test_cabinet_identity_and_missing_separable_surface():
    output,r=import_fixture_identity(XML,body_name='cab_main',component_kind='cabinet_bottom_shelf')
    assert r['body_xml_equal']
    with pytest.raises(ValueError,match='separate named'):
        component_roster(XML.replace('cab_bottom_visual','merged_wall_mesh'),'cab_main','cabinet_bottom_shelf')


def test_public_cabinet_support_requires_explicit_component_phrase():
    s='Pick the mug from the counter and place it in the cabinet.'
    assert task_role_prompt(s,'support','bottom cabinet shelf','cabinet_bottom_shelf')=='bottom cabinet shelf'
    for instruction,prompt in [(s,'native_cab_12'),(s.replace('cabinet','sink'),'bottom cabinet shelf')]:
        with pytest.raises(ValueError):task_role_prompt(instruction,'support',prompt,'cabinet_bottom_shelf')


def test_fixed_camera_only_trajectory_and_separated_splits():
    poses={n:np.eye(4) for n in NATIVE_CAMERAS};poses[NATIVE_CAMERAS[1]][0,3]=.5
    original=copy.deepcopy(poses);frames=cabinet_view_plan(poses)
    assert len(frames)==14 and sum(f['split']=='train' for f in frames)==12
    for n in poses:np.testing.assert_array_equal(poses[n],original[n])
    for test in frames[12:]:
        assert not any(np.allclose(test['camera']['T_world_from_camera'],train['camera']['T_world_from_camera']) for train in frames[:12])
    with pytest.raises(ValueError):cabinet_view_plan({'target_pose':np.eye(4)})


def test_custom_capture_refuses_test_before_any_adapter_or_output(tmp_path):
    from robo.roundtrip.matrix import acquire_instance
    with pytest.raises(ValueError,match='DEV-only'):
        acquire_instance({'split':'test'},'slot',tmp_path/'out',capture_view_plan=cabinet_view_plan)
    assert not (tmp_path/'out').exists()
