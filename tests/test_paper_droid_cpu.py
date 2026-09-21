import json
from pathlib import Path

import pytest
import yaml

from robo.eval import paper_pipeline as p
from robo.eval.real_world_metrics import generate
from robo.manifest.hash import canonical_hash
from tests.test_paper_pipeline_audit import fixture_config
from tests.test_paper_engineering_appendix import put


def fixture(tmp_path, *, all_failed=False):
    root = tmp_path / 'cpu-freeze'
    directory = root / 'full_cpu_summary'
    rows, captures = [], []
    labs = ['iprl', 'autolab', 'tri', 'rail', 'rpl', 'clvr', 'iris', 'pennpal', 'real', 'iprl']
    for i, lab in enumerate(labs):
        wid = f'droid_{lab}_{i}'
        plan = put(root / f'plans/{wid}.json', {'split': {'held_out_video_indices': [10, 11, 12],
            'held_out_fk_indices': [20, 21, 22], 'train_fk_indices': [0, 1, 2]}})
        captures.append(dict(workspace_id=wid, capture_id=f'capture{i}', plan=plan))
        success = i % 2 == 0 and not all_failed
        evaluation = dict(held_out_metrics=dict(center_rms_m=.02, rotation_residual_deg=dict(median=3.)),
            evaluated_reference_frames=2, gate=dict(passed=True),
            fit=put(root / f'evidence/{wid}/fit.json', {}),
            reference=put(root / f'evidence/{wid}/ref.json', {}), plan=plan) if success else None
        result = dict(status='COMPLETE' if success else 'FAILED', alignment_evaluation=evaluation,
                      runtime_seconds=12., failure=None if success else dict(reason='frozen guard rejected'))
        identity = put(root / 'real_world/workspaces' / wid / 'result.json', result)
        rows.append(dict(source='droid', workspace_id=wid, capture_id=f'capture{i}', freeze_id=root.name,
            source_commit='a'*40, execution_status=result['status'], reconstruction_success=False,
            full_build_success=False, accepted_objects=None, runtime_minutes=None, cpu_stage_runtime_seconds=12.,
            translation_cm=2. if success else None, rotation_deg=3. if success else None,
            alignment_pass=True if success else None, failure_reason=None if success else 'frozen guard rejected',
            planned_reference_frames=3, evaluated_reference_frames=2 if success else 0,
            independent_alignment=dict(held_out_ids=[20, 21, 22], construction_ids=[0, 1, 2], selection_ids=[0, 1, 2],
                                       reference_sha256=evaluation['reference']['sha256'] if success else None),
            result_identity=identity))
    config = put(root / 'config.json', dict(captures=captures))
    contract = dict(freeze_id=root.name, code=dict(commit='a'*40, dirty=False),
        configs=[dict(field='real_world_config', resolved_path=config['path'], source_content_sha256=config['sha256'])])
    contract['contract_sha256'] = canonical_hash(contract)
    put(root / 'contract/freeze_manifest.json', contract)
    spec = put(directory / 'workspaces.json', dict(scope='prospective_cpu_alignment', paper_ready=False, rows=rows))
    (directory / 'workspaces.csv').write_text('fixture row inventory\n')
    generate(directory / 'workspaces.json', None, directory)
    audit = dict(schema_version=1, scope='complete_original_droid_cpu_cohort_integrity', status='PASS',
        freeze_id=root.name, source_commit='a'*40, contract_sha256=contract['contract_sha256'],
        planned_workspaces=10, result_records=10, completed_cpu_units=sum(r['execution_status']=='COMPLETE' for r in rows),
        failed_cpu_units=sum(r['execution_status']=='FAILED' for r in rows), not_run_units=0,
        artifacts={f.name:dict(path=str(f),sha256=p._sha256(f),size_bytes=f.stat().st_size) for f in directory.iterdir()},
        script=put(root / 'audit_driver.json', {}), full_e7_complete=False, paper_ready=False)
    spec['completion_audit'] = put(root / 'full_cpu_completion_audit.json', audit)
    return spec


def republish(spec, payload):
    path = Path(spec['path']); spec.update(put(path, payload))
    audit_path = Path(spec['completion_audit']['path']); audit=json.loads(audit_path.read_text())
    audit['artifacts']['workspaces.json'] = {k:spec[k] for k in ('path','sha256','size_bytes')}
    spec['completion_audit'] = put(audit_path, audit)


def test_droid_table_uses_canonical_rows_without_refit_and_retains_failures(tmp_path, monkeypatch):
    spec=fixture(tmp_path)
    from agents.eval import droid_alignment_eval
    monkeypatch.setattr(droid_alignment_eval, 'evaluate_sealed_fit', lambda *a,**k: pytest.fail('formatter refit'))
    path, config=fixture_config(tmp_path)
    config['engineering_appendix']={'droid_cpu_alignment':spec};path.write_text(yaml.safe_dump(config))
    out=tmp_path/'out';report=p.generate(path,out)
    text=(out/'generated_tables/droid_cpu_alignment_engineering.tex').read_text()
    assert 'IPRL (1)' in text and 'IPRL (2)' in text
    assert '2/3 & 2.00 & 3.00' in text and '0/3 & -- & --' in text
    assert 'not Gaussian reconstruction' in text and report['paper_ready'] is False
    assert json.loads((out/'sources/droid_cpu_alignment.json').read_text())['rows']==json.loads(Path(spec['path']).read_text())['rows']
    assert 'rows[9].planned_reference_frames' in (out/'claim_ledger.csv').read_text()


def test_all_failed_cpu_cohort_is_publishable_integrity_not_success(tmp_path):
    spec=fixture(tmp_path,all_failed=True)
    payloads,_=p._engineering_sources({'droid_cpu_alignment':spec})
    assert len(payloads['droid_cpu_alignment']['rows'])==10
    assert all(r['translation_cm'] is None for r in payloads['droid_cpu_alignment']['rows'])


@pytest.mark.parametrize('kind', ['omitted','duplicate','order','fullbuild','reconstruction','invented_metric',
    'ref_count','planned_count','split','failure','runtime','partial','claim','aggregate','input_bytes','config_bytes','source'])
def test_droid_cpu_tamper_negatives(tmp_path,kind):
    spec=fixture(tmp_path);path=Path(spec['path']);payload=json.loads(path.read_text());rows=payload['rows']
    if kind=='omitted':rows.pop()
    elif kind=='duplicate':rows[1]=rows[0]
    elif kind=='order':rows.reverse()
    elif kind=='fullbuild':rows[0]['full_build_success']=True
    elif kind=='reconstruction':rows[0]['reconstruction_success']=True
    elif kind=='invented_metric':rows[1]['translation_cm']=0.
    elif kind=='ref_count':rows[0]['evaluated_reference_frames']=3
    elif kind=='planned_count':rows[0]['planned_reference_frames']=2
    elif kind=='split':rows[0]['independent_alignment']['held_out_ids']=[0]
    elif kind=='failure':rows[1]['failure_reason']=None
    elif kind=='runtime':rows[0]['cpu_stage_runtime_seconds']=1.
    elif kind=='claim':payload['paper_ready']=True
    elif kind in {'partial','source'}:
        ap=Path(spec['completion_audit']['path']);audit=json.loads(ap.read_text())
        audit['result_records' if kind=='partial' else 'source_commit']=9 if kind=='partial' else 'b'*40
        spec['completion_audit']=put(ap,audit)
    elif kind=='aggregate':(path.parent/'real_world_table.json').write_text('{}')
    elif kind=='input_bytes':Path(rows[0]['result_identity']['path']).write_text('{}')
    elif kind=='config_bytes':(path.parent.parent/'config.json').write_text('{}')
    if kind not in {'partial','source','aggregate','input_bytes','config_bytes'}:republish(spec,payload)
    with pytest.raises(ValueError):p._engineering_sources({'droid_cpu_alignment':spec})
