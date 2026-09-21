import json
from pathlib import Path

import pytest

from robo.eval import native_final_release as release
from robo.eval.native_scale_tables import CORE


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')
    return path


def primary(tmp_path):
    rows = []
    for method in CORE:
        for i in range(48):
            for reset in range(10):
                executed = method == 'REF_NATIVE'
                rows.append(dict(unit_id=f'{method}-{i}-{reset}', cohort_id='test',
                    canonical_instance_id=f'i{i}', reset_id=f'r{reset}', policy_rng_seed=reset,
                    policy_id='frozen', controller_method=method, scope='L0_target_only',
                    sensor_regime='ideal', renderer='native', execution_protocol='primary_native',
                    layout_id=i // 16, style_id=i // 8, task_id='sink', split='TEST', native_horizon=600,
                    terminal_status='RECORDED' if executed else 'BUILD_FAILED', executed=executed,
                    success=True if executed else None))
    planned = tmp_path/'planned.jsonl'
    planned.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    ledger = tmp_path/'ledger.jsonl'
    ledger.write_bytes(planned.read_bytes())
    status = put(tmp_path/'status.json', dict(planned=2400, terminal=2400, groups=[{'unmeasured': 0}]))
    receipt = put(tmp_path/'completion.json', dict(state='COLLECTED', collector_returncode=0,
        planned=2400, terminal=2400, missing_units=0, unmeasured_units=0, all_units_measured=True,
        status_path=str(status), status_sha256=release._sha(status)))
    return dict(completion_receipt=str(receipt), planned_units=release.bind(planned), final_ledger=str(ledger)), rows


def test_complete_primary_accepts_legitimate_null_native_nonrollouts(tmp_path):
    plan, _ = primary(tmp_path)
    result = release.validate_primary(plan)
    assert result['planned'] == result['measured'] == 2400
    assert result['executed'] == 480 and result['nonexecuted'] == 1920


@pytest.mark.parametrize('failure', ['missing', 'pending', 'duplicate', 'receipt', 'roster'])
def test_incomplete_or_changed_primary_never_qualifies(tmp_path, failure):
    plan, rows = primary(tmp_path)
    ledger = Path(plan['final_ledger'])
    if failure == 'missing':
        rows.pop()
    elif failure == 'pending':
        rows[-1].update(terminal_status='RESOURCE_FAILED', executed=False, success=None)
    elif failure == 'duplicate':
        rows[-1] = rows[-2]
    elif failure == 'receipt':
        path = Path(plan['completion_receipt']); j = json.loads(path.read_text());j['all_units_measured'] = False;put(path,j)
    else:
        Path(plan['planned_units']['path']).write_text('changed')
    ledger.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):
        release.validate_primary(plan)


def test_absent_completion_waits_without_reading_outcomes(tmp_path):
    assert release.validate_primary({'completion_receipt': str(tmp_path/'absent')}) is None


def appearance(tmp_path, monkeypatch):
    monkeypatch.setattr(release, 'validate_primary', lambda plan: {'planned':2400,'measured':2400})
    from robo.eval import native_scale_scope
    monkeypatch.setattr(native_scale_scope, 'scope_tables', lambda rows: ({'T4_scope': [
        {'scope_run_id':'phase','planned':1200,'measured':400,'unmeasured':800,'executed':0}], 'T4_native_controls': []}, {}))
    baseline = put(tmp_path/'base.json', {'native_scale_up': {'appearance_records': ['cold.json']}})
    render = put(tmp_path/'render.json', {'render_protocol':'capture_train_prelude_v1',
        'native_import_render_identity':[{'byte_exact':True}]})
    metric = put(tmp_path/'metric.json', {'render_protocol':'capture_train_prelude_v1','source':release.bind(render)})
    failed = put(tmp_path/'failure.json', {'error':'unchanged-native heldout rendering differs'})
    manifest = put(tmp_path/'manifest.json', {'rows':[{},{}]})
    warm = put(tmp_path/'warm/001/collection.json', {'manifest_sha256': release._sha(manifest),
        'protocol':'capture_train_prelude_v1','planned_render_units':3,
        'metrics':{'good':release.bind(metric)},'ready':{'good':release.bind(render)},
        'failed':{'bad':release.bind(failed)},'metric_failures':{},'unattempted':['pending']})
    put(tmp_path/'scope/001/status.json', {'planned':1200,'terminal':400,'groups':[{'unmeasured':800}]})
    (tmp_path/'scope/001/episode_ledger.jsonl').write_text('{}\n')
    put(tmp_path/'scope/001/collection.json', dict(scope_run_id='phase',terminal=400,executed=0,
        episode_ledger_sha256=release._sha(tmp_path/'scope/001/episode_ledger.jsonl'),planned_units_sha256='roster'))
    plan=dict(baseline_config=release.bind(baseline),planned_units={'path':'planned'},final_ledger='final',
        warm_collections=str(tmp_path/'warm'),warm_manifest=release.bind(manifest),
        scope_collections=str(tmp_path/'scope'),scope_source={'planned_units_sha256':'roster'})
    return plan,warm,render,metric


def test_warm_only_boundary_retains_failed_and_unattempted_views(tmp_path, monkeypatch):
    plan,_,_,metric=appearance(tmp_path,monkeypatch)
    config,boundary=release.prepare_inputs(plan)
    assert config['native_scale_up']['appearance_records']==[str(metric)]
    assert config['native_scale_up']['native_paper_section'] is False
    assert boundary['warm_unattempted']==['pending'] and len(boundary['warm_render_failures'])==1
    assert boundary['scope_unmeasured']==800


@pytest.mark.parametrize('failure',['cold','identity','hash'])
def test_invalid_render_never_enters_final_quality(tmp_path,monkeypatch,failure):
    plan,warm,render,metric=appearance(tmp_path,monkeypatch)
    if failure=='cold':
        j=json.loads(metric.read_text());j['render_protocol']='cold';put(metric,j)
    elif failure=='identity':
        j=json.loads(render.read_text());j['native_import_render_identity'][0]['byte_exact']=False;put(render,j)
        j=json.loads(metric.read_text());j['source']=release.bind(render);put(metric,j)
    else:
        render.write_text('{}')
    if failure!='hash':
        j=json.loads(warm.read_text());j['metrics']['good']=release.bind(metric);j['ready']['good']=release.bind(render);put(warm,j)
    with pytest.raises(ValueError):release.prepare_inputs(plan)
