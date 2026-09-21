import pytest
from robo.roundtrip.scope_trace_diagnostics import summarize_trace


def fixture():
    initial={'time':1.,'object_states':{'obj':[10.,0.,1.,1.,0.,0.,0.]}}
    rows=[{'tick':i,'simulation_time_s':1+(i+1)*.05,'objects':{'obj':[10.+x,0.,1.+z,1.,0.,0.,0.]},'native_predicates':{'native_task_success':False,'object_origin_inside_sink':False,'gripper_distance_m':.2}} for i,(x,z) in enumerate([(0.,-.1),(.3,.2)])]
    return rows,initial


def test_uses_saved_initial_not_first_tick_and_retains_prefix():
    rows,initial=fixture();r=summarize_trace(rows,initial)
    assert r['objects']['obj']['max_z_above_initial_m']==pytest.approx(.2)
    assert r['objects']['obj']['first_step_displacement_m']==pytest.approx(.1)
    assert r['duration_s']==pytest.approx(.1)
    assert r['lift_success'] is None and r['contact_pairs']=='NOT_RECORDED'
    assert summarize_trace(rows[:1],initial)['recorded_ticks']==1


def test_missing_tick_fails():
    rows,initial=fixture();rows[1]['tick']=2
    with pytest.raises(ValueError,match='contiguous'):summarize_trace(rows,initial)


def test_empty_fails():
    with pytest.raises(ValueError,match='empty'):summarize_trace([],fixture()[1])
