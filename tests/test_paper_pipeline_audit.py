"""Audit rendering must preserve denominators, missing rows, and source identity."""
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from robo.eval.paper_pipeline import generate, CONSTRUCTION_ROSTER, FIDELITY_ROSTER


def fixture_config(tmp_path):
    construction = {"rows": [{"regime": CONSTRUCTION_ROSTER[i], "instances": 10,
        "planned_scenes": 2, "accepted_instances": 3, "yield": .3,
        "f1_20": None, "stability": None, "runtime_minutes": None}
        for i in range(5)]}
    fidelity = {"validation": {"valid": True}, "rows": [
        {"method": FIDELITY_ROSTER[i], "unit": "room" if i < 4 else "object",
         "n_images": 8 if i < 3 else 0, "n_objects": 0,
         "n_scenes": 1 if i < 3 else 0, "psnr": 25 if i < 3 else None,
         "ssim": .9 if i < 3 else None, "lpips": .1 if i < 3 else None}
        for i in range(8)]}
    agentic = {"rows": [{"policy_id": f"A{i}", "planned_jobs": 10,
        "accepted_jobs": 2 if i == 4 else 10, "build_coverage": .2 if i == 4 else 1.,
        "f1_20": .6, "cd_cm": 3., "catastrophic_collapses": 0,
        "stable_fraction": .5, "retry_count": 4 if i >= 3 else 0,
        "abstain_count": 8 if i == 4 else 0, "runtime_minutes_per_scene": 2.}
        for i in range(5)]}
    config = {"mode": "preliminary_audit", "freeze_id": "test-audit",
              "audited_artifacts": {}}
    for name, payload in [("construction", construction), ("fidelity", fidelity),
                          ("agentic", agentic)]:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload))
        config["audited_artifacts"][name] = {
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, config


def test_preliminary_tables_preserve_missing_methods_and_denominators(tmp_path):
    path, _ = fixture_config(tmp_path)
    out, paper = tmp_path / "out", tmp_path / "paper"
    result = generate(path, out, paper)
    assert result["paper_ready"] is False
    tex = (paper / "tables/agentic_ablation_main.tex").read_text()
    assert "2/10 & 20.0\\%" in tex
    assert "Preliminary audit" in tex
    fidelity = (paper / "tables/object_factorization_main.tex").read_text()
    assert all(FIDELITY_ROSTER[i] in fidelity for i in range(8))
    assert "--" in fidelity
    assert (out / "claim_ledger.csv").exists()
    assert len(list((out / "sources").glob("*.csv"))) == 3
    with pytest.raises(FileExistsError, match="overwrite"):
        generate(path, out, paper)


def test_hash_drift_rejected_before_paper_or_output_mutation(tmp_path):
    path, config = fixture_config(tmp_path)
    Path(config["audited_artifacts"]["agentic"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        generate(path, tmp_path / "out", tmp_path / "paper")
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "paper").exists()


@pytest.mark.parametrize("bad", ["denominator", "missing_policy", "nonfinite"])
def test_invalid_aggregate_cannot_render(tmp_path, bad):
    path, config = fixture_config(tmp_path)
    source = Path(config["audited_artifacts"]["agentic"]["path"])
    payload = json.loads(source.read_text())
    if bad == "denominator":
        payload["rows"][4]["planned_jobs"] = 9
    elif bad == "missing_policy":
        payload["rows"].pop()
    else:
        payload["rows"][0]["f1_20"] = float("nan")
    source.write_text(json.dumps(payload))
    config["audited_artifacts"]["agentic"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        generate(path, tmp_path / "out", tmp_path / "paper")
    assert not (tmp_path / "paper").exists()


def test_audit_cannot_promote_to_submission_mode(tmp_path):
    path, config = fixture_config(tmp_path)
    config["mode"] = "paper"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="preliminary_audit"):
        generate(path, tmp_path / "out")


def test_failed_write_does_not_publish_output_or_paper(tmp_path, monkeypatch):
    path, _ = fixture_config(tmp_path)
    def fail(*args, **kwargs):
        raise OSError("injected write failure")
    monkeypatch.setattr("robo.eval.paper_pipeline.write_json", fail)
    with pytest.raises(OSError, match="injected"):
        generate(path, tmp_path / "out", tmp_path / "paper")
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "paper").exists()
    assert not list(tmp_path.glob(".paper-audit-*"))


def test_display_order_preserves_original_source_indices(tmp_path):
    path, config = fixture_config(tmp_path)
    source = Path(config['audited_artifacts']['fidelity']['path'])
    payload = json.loads(source.read_text())
    payload['rows'].reverse()
    source.write_text(json.dumps(payload))
    config['audited_artifacts']['fidelity']['sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_text(yaml.safe_dump(config))
    out = tmp_path / 'out'
    generate(path, out)
    tex = (out / 'generated_tables/object_factorization_main.tex').read_text()
    positions = [tex.index(name) for name in FIDELITY_ROSTER]
    assert positions == sorted(positions)
    import csv
    with (out / 'claim_ledger.csv').open() as handle:
        claims = list(csv.DictReader(handle))
    first_psnr = next(c for c in claims if c['claim_id'].startswith('fidelity') and c['source_field'].endswith('.psnr'))
    assert first_psnr['source_field'] == 'rows[7].psnr'


@pytest.mark.parametrize('name,field', [('construction', 'regime'), ('fidelity', 'method')])
def test_exact_roster_names_reject_same_size_drift(tmp_path, name, field):
    path, config = fixture_config(tmp_path)
    source = Path(config['audited_artifacts'][name]['path'])
    payload = json.loads(source.read_text())
    payload['rows'][0][field] = 'undeclared treatment'
    source.write_text(json.dumps(payload))
    config['audited_artifacts'][name]['sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='exact'):
        generate(path, tmp_path / 'out')


def fresh_fixture(tmp_path):
    path,config=fixture_config(tmp_path)
    spec=config['audited_artifacts']['agentic'];source=Path(spec['path'])
    payload=json.loads(source.read_text())
    payload.update(schema_version=1,study_scope='automatic_training_only_engineering',freeze_id='evaluation',
        construction_freeze_id='construction',paper_ready=False,headline_eligible=False,retry_claim_status='case_study_only',
        counts=dict(scenes=1,jobs_per_policy=10,policy_object_rows=50,genuine_retry_jobs=4))
    for row in payload['rows']:
        row.update(geometry_evaluated_jobs=1 if row['policy_id']=='A4' else 3,
            physical_tested_jobs=row['accepted_jobs'],physical_stable_jobs=row['accepted_jobs']//2)
    source.write_text(json.dumps(payload));spec['sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    keys=('construction_train_only_discovery_binding','control_frozen_before_evaluation',
        'independent_GT_input_hashes','matching_and_surface_replay_exact','unmatched_geometry_null',
        'aggregate_replay_exact','nonheadline_gates','retry_new_proposal_transform_and_different_action')
    audit=dict(status='PASS',paper_ready=False,headline_eligible=False,claim_gate='NOT_RUN',freeze_id='evaluation',
        construction_freeze_id='construction',aggregate_rows_replayed=payload['rows'],planned_jobs=10,
        terminal_policy_rows=50,unique_retry_actions=4,matched_jobs=3,unmatched_jobs=7,source_commit='a'*40,
        checks={k:'PASS' for k in keys},evidence_hashes={str(source):dict(sha256=spec['sha256'],size_bytes=source.stat().st_size)})
    audit_path=tmp_path/'completion.json';audit_path.write_text(json.dumps(audit))
    spec['completion_audit']=dict(path=str(audit_path),sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest())
    config['claim_decisions']={'AGENTIC_TITLE_CLAIM':dict(status='NOT_RUN',decision='narrow',
        required_experiment='E3',required_gate='complete matched full-cohort comparison',
        observed_result='Only an independently evaluated case study is available.',
        enabled_sentence='The automatic-discovery case study does not establish room-scale gains.',
        removed_sentence='Its legacy object population uses GT-assisted preparation.')}
    path.write_text(yaml.safe_dump(config));return path,config


def test_fresh_case_study_has_matched_denominator_no_legacy_caption_or_total_runtime(tmp_path):
    path,config=fresh_fixture(tmp_path);out=tmp_path/'out';report=generate(path,out)
    tex=(out/'generated_tables/agentic_ablation_main.tex').read_text()
    assert 'automatic-discovery controller case study' in tex and 'geom. n' in tex
    assert 'stable/tested' in tex and '2/10 & 20.0\\% & 1 &' in tex
    assert 'GT-assisted legacy crops' not in tex and 'min/scene' not in tex
    assert 'Runtime is omitted' in tex
    assert report['runtime_display']=='omitted_incomplete_generation_accounting'
    assert report['sources']['agentic']['completion_audit']['matched_jobs']==3
    assert report['prose_decisions_applied'] is False
    import csv
    rows=list(csv.DictReader((out/'claim_ledger.csv').open()))
    assert any(r['source_field']=='rows[4].geometry_evaluated_jobs' for r in rows)
    assert not any(r['source_field'].endswith('runtime_minutes_per_scene') for r in rows)
    claim=next(r for r in rows if r['claim_id']=='AGENTIC_TITLE_CLAIM')
    assert claim['enabled_sentence']==config['claim_decisions']['AGENTIC_TITLE_CLAIM']['enabled_sentence']
    assert claim['removed_sentence']==config['claim_decisions']['AGENTIC_TITLE_CLAIM']['removed_sentence']
    assert claim['claim_gate']=='NOT_RUN'


@pytest.mark.parametrize('mutation',['missing_audit','scope','audit_hash','source_binding','audit_rows',
    'population','geometry_count','physical_count','independence','headline'])
def test_fresh_scope_integrity_failures_do_not_publish(tmp_path,mutation):
    path,config=fresh_fixture(tmp_path);spec=config['audited_artifacts']['agentic']
    source=Path(spec['path']);payload=json.loads(source.read_text())
    audit_path=Path(spec['completion_audit']['path']);audit=json.loads(audit_path.read_text())
    if mutation=='missing_audit':del spec['completion_audit']
    elif mutation=='scope':payload['study_scope']='new_unsupported_scope'
    elif mutation=='audit_hash':audit_path.write_text(audit_path.read_text()+' ')
    elif mutation=='source_binding':audit['evidence_hashes'][str(source)]['sha256']='f'*64
    elif mutation=='audit_rows':audit['aggregate_rows_replayed'][0]['accepted_jobs']=1
    elif mutation=='population':audit['terminal_policy_rows']=49
    elif mutation=='geometry_count':payload['rows'][4]['geometry_evaluated_jobs']=3
    elif mutation=='physical_count':payload['rows'][4]['physical_stable_jobs']=3
    elif mutation=='independence':audit['checks']['control_frozen_before_evaluation']='FAIL'
    elif mutation=='headline':payload['headline_eligible']=True
    if mutation in {'scope','geometry_count','physical_count','headline'}:
        source.write_text(json.dumps(payload));spec['sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
        audit['aggregate_rows_replayed']=payload['rows'];audit['evidence_hashes'][str(source)]=dict(sha256=spec['sha256'],size_bytes=source.stat().st_size)
    if mutation not in {'missing_audit','audit_hash'}:
        audit_path.write_text(json.dumps(audit));spec['completion_audit']['sha256']=hashlib.sha256(audit_path.read_bytes()).hexdigest()
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):generate(path,tmp_path/'out',tmp_path/'paper')
    assert not (tmp_path/'out').exists() and not (tmp_path/'paper').exists()


def raw_fidelity_fixture(tmp_path):
    path,config=fresh_fixture(tmp_path);spec=config['audited_artifacts']['fidelity'];source=Path(spec['path'])
    payload=json.loads(source.read_text());payload.update(freeze_id='raw',provenance=dict(code_commit='c'*40))
    for i,row in enumerate(payload['rows']):
        row.update(n_images=400 if i==0 else 0,n_scenes=50 if i==0 else 0,
            psnr=25 if i==0 else None,ssim=.9 if i==0 else None,lpips=.1 if i==0 else None)
        if i==0:row['metric_samples']=dict(psnr=400,ssim=400,lpips=400)
    manifest=tmp_path/'metric_manifest.json';manifest.write_text('{}');payload['manifest_path']=str(manifest)
    payload['manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest();source.write_text(json.dumps(payload))
    spec['sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    coverage=tmp_path/'coverage.json';coverage.write_text(json.dumps(dict(planned_scenes=50,completed_scenes=50,planned_views=400,
        rows=[dict(scene_id=str(i),status='PASS') for i in range(50)])))
    receipt=dict(status='PASS',paper_ready=False,code_commit='c'*40,scenes=[str(i) for i in range(50)],n_images=400,
        table=dict(path=str(source),sha256=spec['sha256']),contract_sha256='d'*64,
        coverage=dict(path=str(coverage),sha256=hashlib.sha256(coverage.read_bytes()).hexdigest()))
    receipt_path=tmp_path/'fidelity_receipt.json';receipt_path.write_text(json.dumps(receipt))
    spec['completion_receipt']=dict(path=str(receipt_path),sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest())
    path.write_text(yaml.safe_dump(config));return path,config


def test_raw_only_fidelity_caption_and_full_denominator(tmp_path):
    path,config=raw_fidelity_fixture(tmp_path);out=tmp_path/'out';generate(path,out)
    tex=(out/'generated_tables/object_factorization_main.tex').read_text()
    assert 'Only the input Gaussian row is populated' in tex
    assert '50 & 400 & 25.00 & 0.900 & 0.100' in tex
    assert 'separate controller case study reports geometry only for matched jobs' in tex
    assert 'No independent masked-view/surface evaluation is available' not in tex


@pytest.mark.parametrize('mutation',['coverage','receipt_source','metric_manifest','metric_count'])
def test_raw_fidelity_rejects_stale_completion_or_coverage(tmp_path,mutation):
    path,config=raw_fidelity_fixture(tmp_path);spec=config['audited_artifacts']['fidelity']
    receipt_path=Path(spec['completion_receipt']['path']);receipt=json.loads(receipt_path.read_text())
    if mutation=='coverage':receipt['scenes'].pop()
    elif mutation=='receipt_source':receipt['code_commit']='other'
    elif mutation=='metric_manifest':
        payload=json.loads(Path(spec['path']).read_text());Path(payload['manifest_path']).write_text('changed')
    else:
        source=Path(spec['path']);payload=json.loads(source.read_text());payload['rows'][0]['metric_samples']['lpips']=399
        source.write_text(json.dumps(payload));spec['sha256']=hashlib.sha256(source.read_bytes()).hexdigest();receipt['table']['sha256']=spec['sha256']
    receipt_path.write_text(json.dumps(receipt));spec['completion_receipt']['sha256']=hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):generate(path,tmp_path/'out')


def full_fixture(tmp_path):
    path,config=fresh_fixture(tmp_path);spec=config['audited_artifacts']['agentic']
    source=Path(spec['path']);payload=json.loads(source.read_text())
    payload['counts']=dict(scenes=50,jobs_per_policy=1871,policy_object_rows=9355,genuine_retry_jobs=4)
    payload['runtime_scope']='attributed_initial_generation_plus_canonical_registration_physics_and_retries'
    payload['retry_claim_status']='retry_count_threshold_met_preliminary_only'
    for row in payload['rows']:
        row.update(planned_jobs=1871,accepted_jobs=100,build_coverage=100/1871,
            geometry_evaluated_jobs=80,physical_tested_jobs=100,physical_stable_jobs=50,
            runtime_accounting_status='PASS',canonical_runtime_minutes_per_scene=.5,
            generation_process_minutes_per_scene=1.5,runtime_minutes_per_scene=2.)
    audit_path=Path(spec['completion_audit']['path']);audit=json.loads(audit_path.read_text())
    audit.update(schema_version=1,scope='complete_cohort_independent_evaluation_integrity',
        planned_scenes=50,planned_jobs=1871,terminal_policy_rows=9355,matched_jobs=1500,unmatched_jobs=371,
        runtime_scope=payload['runtime_scope'],runtime_accounting_status={f'A{i}':'PASS' for i in range(5)})
    del audit['checks']['matching_and_surface_replay_exact']
    audit['checks'].update({k:'PASS' for k in ('complete_frozen_roster','evaluation_metric_shards_authenticated','runtime_accounting_replay_exact')})
    _refresh_full(path,config,payload,audit)
    return path,config,payload,audit


def _refresh_full(path,config,payload,audit):
    spec=config['audited_artifacts']['agentic'];source=Path(spec['path'])
    source.write_text(json.dumps(payload));spec['sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    audit['aggregate_rows_replayed']=payload['rows']
    audit['evidence_hashes'][str(source)]=dict(sha256=spec['sha256'],size_bytes=source.stat().st_size)
    audit_path=Path(spec['completion_audit']['path']);audit_path.write_text(json.dumps(audit))
    spec['completion_audit']['sha256']=hashlib.sha256(audit_path.read_bytes()).hexdigest()
    path.write_text(yaml.safe_dump(config))


def test_full_engineering_cohort_has_conditional_denominators_and_scoped_runtime(tmp_path):
    path,config,payload,audit=full_fixture(tmp_path)
    report=generate(path,tmp_path/'out')
    tex=(tmp_path/'out/generated_tables/agentic_ablation_main.tex').read_text()
    assert 'complete declared 50-scene engineering cohort' in tex
    assert 'attrib. min/scene' in tex and '100/1871' in tex and '50/100' in tex
    assert 'failures and recoveries' in tex
    assert 'neither capture-to-sim nor fleet elapsed time' in tex
    assert 'one predeclared scene' not in tex and 'GT-assisted legacy crops' not in tex
    assert 'source Gaussian training' in report['runtime_scope_note']
    assert 'does not rerun the matcher' in report['runtime_scope_note']
    assert 'do not establish manipulation gains' in tex
    assert report['runtime_display']=='attributed_algorithm_phase' and not report['paper_ready']
    import csv
    claims=list(csv.DictReader((tmp_path/'out/claim_ledger.csv').open()))
    assert any(r['source_field']=='rows[0].runtime_minutes_per_scene' for r in claims)
    assert next(r for r in claims if r['claim_id']=='AGENTIC_TITLE_CLAIM')['claim_gate']=='NOT_RUN'


def test_full_runtime_missing_evidence_remains_dash(tmp_path):
    path,config,payload,audit=full_fixture(tmp_path)
    payload['rows'][0].update(runtime_accounting_status='NOT_RUN',runtime_minutes_per_scene=None,
                              generation_process_minutes_per_scene=None)
    audit['runtime_accounting_status']['A0']='NOT_RUN'
    _refresh_full(path,config,payload,audit);generate(path,tmp_path/'out')
    tex=(tmp_path/'out/generated_tables/agentic_ablation_main.tex').read_text()
    a0=next(line for line in tex.splitlines() if 'A0: fixed TRELLIS' in line)
    assert a0.endswith(r'& -- \\') and 'incomplete timing evidence' in tex


@pytest.mark.parametrize('mutation',['unknown_scope','schema','scenes','jobs','matching_population',
    'shard_gate','runtime_gate','runtime_scope','runtime_status','negative_runtime','mismatched_sum','unknown_runtime_value',
    'headline','independence'])
def test_full_audit_and_runtime_failures_cannot_publish(tmp_path,mutation):
    path,config,payload,audit=full_fixture(tmp_path)
    if mutation=='unknown_scope':audit['scope']='unrecognized_full_audit'
    elif mutation=='schema':audit['schema_version']=2
    elif mutation=='scenes':audit['planned_scenes']=49
    elif mutation=='jobs':audit['planned_jobs']=1870
    elif mutation=='matching_population':audit['unmatched_jobs']=370
    elif mutation=='shard_gate':audit['checks']['evaluation_metric_shards_authenticated']='NOT_RUN'
    elif mutation=='runtime_gate':audit['checks']['runtime_accounting_replay_exact']='FAIL'
    elif mutation=='runtime_scope':payload['runtime_scope']=audit['runtime_scope']='capture_to_sim_wall_time'
    elif mutation=='runtime_status':audit['runtime_accounting_status']['A0']='NOT_RUN'
    elif mutation=='negative_runtime':payload['rows'][0]['generation_process_minutes_per_scene']=-1
    elif mutation=='mismatched_sum':payload['rows'][0]['runtime_minutes_per_scene']=2.5
    elif mutation=='unknown_runtime_value':
        payload['rows'][0]['runtime_accounting_status']='NOT_RUN';audit['runtime_accounting_status']['A0']='NOT_RUN'
    elif mutation=='headline':audit['headline_eligible']=True
    elif mutation=='independence':audit['checks']['control_frozen_before_evaluation']='FAIL'
    _refresh_full(path,config,payload,audit)
    with pytest.raises(ValueError):generate(path,tmp_path/'out',tmp_path/'paper')
    assert not (tmp_path/'out').exists() and not (tmp_path/'paper').exists()


def test_omit_only_unmeasured_object_table_preserves_sources_and_skeleton(tmp_path):
    path,config=fixture_config(tmp_path)
    config['omit_unmeasured_object_table']=True
    path.write_text(yaml.safe_dump(config))
    out=tmp_path/'out';result=generate(path,out)
    main=(out/'generated_tables/object_factorization_main.tex').read_text()
    skeleton=(out/'generated_tables/object_fidelity_unmeasured_skeleton.tex').read_text()
    assert 'tab:fidelity' in main and 'tab:object-fidelity' not in main
    assert 'tab:object-fidelity' in skeleton
    assert result['omitted_unmeasured_object_table'] is True
    assert result['object_method_source_rows_preserved']==4
    assert len(json.loads((out/'sources/fidelity.json').read_text())['rows'])==8
    assert all(name in (out/'sources/fidelity.csv').read_text() for name in FIDELITY_ROSTER)


@pytest.mark.parametrize('metric',['psnr','ssim','lpips','cd_cm','f1_20','collapses'])
def test_object_table_omission_rejects_any_measured_value_including_zero(tmp_path,metric):
    path,config=fixture_config(tmp_path)
    config['omit_unmeasured_object_table']=True
    source=Path(config['audited_artifacts']['fidelity']['path']);payload=json.loads(source.read_text())
    payload['rows'][4][metric]=0
    source.write_text(json.dumps(payload));config['audited_artifacts']['fidelity']['sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='measured metrics'):generate(path,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_object_table_omission_flag_requires_boolean(tmp_path):
    path,config=fixture_config(tmp_path);config['omit_unmeasured_object_table']='true';path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='boolean'):generate(path,tmp_path/'out')
