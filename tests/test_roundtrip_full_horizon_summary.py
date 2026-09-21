import json
from robo.eval.episode_log import write_timeseries
from robo.roundtrip.full_horizon import summarize_episode


def test_full_horizon_keeps_first_success_and_final_failure_distinct(tmp_path):
    r=dict(horizon=3,success=False,error=None,execution_kind='closed_loop_visual_policy',native_predicates={'native_task_success':False})
    (tmp_path/'result.json').write_text(json.dumps(r));(tmp_path/'actions.json').write_text('[[0],[0],[0]]')
    write_timeseries(tmp_path/'trace.json.gz',[{'tick':i,'native_predicates':{'native_task_success':i==1}} for i in range(3)])
    out=summarize_episode(tmp_path)
    assert out['first_success_step']==2 and out['success_ever'] and out['success_at_horizon'] is False
    r['error']='resource crash';(tmp_path/'result.json').write_text(json.dumps(r))
    assert summarize_episode(tmp_path)['success_at_horizon'] is None


def test_aggregate_rejects_duplicate_or_incomplete_unit(tmp_path):
    import pytest
    from robo.roundtrip.full_horizon import aggregate_units
    with pytest.raises(ValueError,match='unique'):aggregate_units([tmp_path,tmp_path])
    (tmp_path/'progress.json').write_text(json.dumps(dict(reset_seed=0,complete=False,rows=[])))
    with pytest.raises(ValueError,match='incomplete'):aggregate_units([tmp_path])
