import json

import pytest
import yaml

from robo.eval.metric_utils import paired_hierarchical_bootstrap
from robo.eval.native_scale_tables import load_units, tables
from robo.eval.paper_pipeline import generate


def row(method='REF_NATIVE', reset='r0', status='RECORDED', success=True, instance='i0', layout=0):
    return dict(unit_id=f'{method}-{instance}-{reset}', cohort_id='dev', canonical_instance_id=instance,
                reset_id=reset, policy_rng_seed=0, policy_id='frozen', controller_method=method,
                scope='L0_target_only', sensor_regime='ideal_rgbd', renderer='native',
                execution_protocol='primary_native', layout_id=layout, style_id=0,
                task_id='sink', split='DEV', native_horizon=600, terminal_status=status,
                executed=None if status == 'NOT_SCHEDULED' else status != 'BUILD_FAILED',
                success=success)


def files(tmp_path, plans, outcomes):
    paths = [tmp_path / name for name in ('planned.jsonl', 'ledger.jsonl')]
    for path, rows in zip(paths, (plans, outcomes)):
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return paths


def test_missing_and_pending_are_null_but_build_failure_is_service_zero(tmp_path):
    plans = [row(), row('B0', status='BUILD_FAILED', success=None),
             row(reset='r1'), row('B0', reset='r1', status='NOT_SCHEDULED', success=None)]
    units = load_units(*files(tmp_path, plans, plans[:2] + plans[3:]))
    main, comparison, _ = tables(units)
    method = next(r for r in main if r['method'] == 'B0')
    assert method['planned'] == 2 and method['measured'] == 1 and method['build_failed'] == 1
    assert method['success_per_planned'] is None and method['executed'] == 0
    assert comparison[0]['delta_pp'] is None
    assert comparison[0]['observed_pairs_descriptive']['delta'] == -1


def test_complete_negative_native_pair_and_failed_reference_remain(tmp_path):
    plans = [row(), row('B0', success=False), row(reset='r1', success=False),
             row('B0', reset='r1', success=False)]
    main, comparisons, tasks = tables(load_units(*files(tmp_path, plans, plans)))
    assert comparisons[0]['delta_pp'] == -50
    assert comparisons[0]['observed_pairs_descriptive']['both_failure'] == 1
    assert len(tasks) == 2 and sum(r['planned'] for r in main) == 4


@pytest.mark.parametrize('mutation', ['duplicate', 'extra', 'identity', 'false_pending', 'unknown'])
def test_rejects_corrupt_ledger(tmp_path, mutation):
    plans = [row()]
    outcomes = [row()]
    if mutation == 'duplicate':
        outcomes *= 2
    elif mutation == 'extra':
        outcomes.append(row(reset='r2'))
    elif mutation == 'identity':
        outcomes[0]['layout_id'] = 3
    elif mutation == 'false_pending':
        outcomes[0]['terminal_status'] = 'NOT_SCHEDULED'
    else:
        outcomes[0]['terminal_status'] = 'SOMETHING'
    with pytest.raises(ValueError):
        load_units(*files(tmp_path, plans, outcomes))


def test_hierarchical_instance_macro_and_pairing():
    rows = [dict(layout_id='l0', canonical_instance_id='a', reset_id=str(i), a=0, b=1) for i in range(9)]
    rows.append(dict(layout_id='l1', canonical_instance_id='b', reset_id='0', a=1, b=0))
    stats = paired_hierarchical_bootstrap(rows)
    assert stats['delta'] == 0  # episode micro would incorrectly be +0.8
    assert stats['instances'] == 2 and stats['layouts'] == 2 and stats['matched_resets'] == 10
    assert stats == paired_hierarchical_bootstrap(rows)
    with pytest.raises(ValueError, match='duplicate'):
        paired_hierarchical_bootstrap(rows + rows[:1])


def test_pipeline_generates_null_planning_release_and_never_overwrites(tmp_path):
    plans = [row(status='NOT_SCHEDULED', success=None), row('B0', status='NOT_SCHEDULED', success=None)]
    planned, ledger = files(tmp_path, plans, plans)
    config = tmp_path / 'pipeline.yaml'
    config.write_text(yaml.safe_dump({'native_scale_up': {'planned_units': str(planned), 'episode_ledger': str(ledger)}}))
    out = tmp_path / 'tables'
    report = generate(config, out)
    assert report['measured_units'] == 0 and report['state'] == 'INCOMPLETE'
    assert len(json.loads((out / 'T1a_appearance.json').read_text())) == 4
    assert all(r['success_per_planned'] is None for r in json.loads((out / 'T2_manipulation.json').read_text()))
    assert (out / 'claim_ledger.csv').exists() and (out / 'T2_manipulation.tex').exists()
    with pytest.raises(FileExistsError):
        generate(config, out)


def test_unresolved_planning_has_no_invented_instances(tmp_path):
    planned = row(status='NOT_SCHEDULED', success=None)
    planned.pop('unit_id')
    planned.update(plan_unit_id='plan-1', instance_slot_id='slot-1', canonical_instance_id=None,
                   terminal_status='UNRESOLVED_INSTANCE')
    path = tmp_path / 'unresolved.jsonl'
    path.write_text(json.dumps(planned) + '\n')
    config = tmp_path / 'planning.yaml'
    config.write_text(yaml.safe_dump({'native_scale_up': {'mode': 'planning', 'planned_units': str(path)}}))
    report = generate(config, tmp_path / 'tables')
    table = json.loads((tmp_path / 'tables/T2_manipulation.json').read_text())[0]
    assert table['planned'] == 1 and table['planned_instance_slots'] == 1
    assert table['instances'] == 0 and table['instance_macro_success'] is None
    assert report['measured_units'] == 0


def test_source_tamper_stops_before_output_directory(tmp_path):
    plan = row()
    plan.update(result_path=str(tmp_path / 'result.json'), result_sha256='0' * 64)
    (tmp_path / 'result.json').write_text('{}')
    planned, ledger = files(tmp_path, [plan], [plan])
    config = tmp_path / 'pipeline.yaml'
    config.write_text(yaml.safe_dump({'native_scale_up': {'planned_units': str(planned), 'episode_ledger': str(ledger)}}))
    with pytest.raises(ValueError, match='bytes changed'):
        generate(config, tmp_path / 'tables')
    assert not (tmp_path / 'tables').exists()


def test_scope_hook_preserves_primary_population_and_source_closure(tmp_path, monkeypatch):
    from robo.eval import native_scale_scope
    from robo.eval.native_scale_tables import _sha
    plans = [row(), row('B0', success=False)]
    planned, ledger = files(tmp_path, plans, plans)
    receipt = tmp_path / 'scope_source.json'
    receipt.write_text('{}')
    scope_row = dict(scope_run_id='fresh-scope', method='B3_AGENT_NATIVE',
                     scope='L1_target_destination', instances=24, layouts=3,
                     executed=0, planned=240, success_per_planned=None,
                     success_per_executed=None, rollout_coverage_observed=0,
                     absolute_replay_position_error_cm=None,
                     retained_context={'annotation': 'machine-only context metadata'})
    calls = []
    baseline_row = dict(scope_row, scope='L0_target_only')
    scope_row['additional_scope_nonexecution_observed'] = 20
    def authenticated_scope(records):
        calls.append(records)
        return {'T4_scope': [baseline_row, scope_row], 'T4_scope_comparisons': [],
                'T4_native_controls': [], 'T4_scope_failures': []}, {
                    'scope_fixture': {'path': str(receipt), 'sha256': _sha(receipt)}}
    monkeypatch.setattr(native_scale_scope, 'scope_tables', authenticated_scope)
    config = tmp_path / 'pipeline.yaml'
    config.write_text(yaml.safe_dump({'native_scale_up': {
        'planned_units': str(planned), 'episode_ledger': str(ledger),
        'scope_sources': [{'manifest': str(receipt)}]}}))
    out = tmp_path / 'tables'
    release = generate(config, out)
    assert calls == [[{'manifest': str(receipt)}]]
    assert release['planned_units'] == release['measured_units'] == 2
    assert sum(r['planned'] for r in json.loads((out / 'T2_manipulation.json').read_text())) == 2
    assert json.loads((out / 'T4_scope.json').read_text()) == [baseline_row, scope_row]
    import csv
    csv_rows = list(csv.DictReader((out / 'T4_scope.csv').open()))
    assert csv_rows[0]['additional_scope_nonexecution_observed'] == ''
    assert csv_rows[1]['additional_scope_nonexecution_observed'] == '20'
    assert 'machine-only context metadata' not in (out / 'T4_scope.tex').read_text()
    assert (out / 'T4_scope_comparisons.csv').exists()
    assert release['source_lineage']['scope_fixture']['sha256'] == _sha(receipt)
    assert release['source_lineage']['scope_implementation']['sha256'] == _sha(native_scale_scope.__file__)


def test_invalid_scope_source_fails_before_release(tmp_path, monkeypatch):
    from robo.eval import native_scale_scope
    def invalid_scope(records):
        raise ValueError('scope source bytes changed')
    monkeypatch.setattr(native_scale_scope, 'scope_tables', invalid_scope)
    planned, ledger = files(tmp_path, [row()], [row()])
    config = tmp_path / 'pipeline.yaml'
    config.write_text(yaml.safe_dump({'native_scale_up': {
        'planned_units': str(planned), 'episode_ledger': str(ledger),
        'scope_sources': [{'manifest': 'changed.json'}]}}))
    out = tmp_path / 'tables'
    with pytest.raises(ValueError, match='scope source bytes changed'):
        generate(config, out)
    assert not out.exists()


def test_changed_release_boundary_fails_before_output(tmp_path):
    planned, ledger = files(tmp_path, [row()], [row()])
    boundary = tmp_path / 'boundary.json'
    boundary.write_text('{}')
    config = tmp_path / 'pipeline.yaml'
    config.write_text(yaml.safe_dump({'native_scale_up': {
        'planned_units': str(planned), 'episode_ledger': str(ledger),
        'input_boundary': {'path': str(boundary), 'sha256': '0' * 64}}}))
    with pytest.raises(ValueError, match='input boundary bytes changed'):
        generate(config, tmp_path / 'tables')
    assert not (tmp_path / 'tables').exists()


@pytest.mark.parametrize('tier',['DEV','TEST'])
def test_partial_paper_transfer_fails_without_touching_paper(tmp_path,tier):
    plans=[dict(row(),split=tier),dict(row('B0_FIXED_NATIVE',status='NOT_SCHEDULED',success=None),split=tier)]
    planned,ledger=files(tmp_path,plans,plans)
    config=tmp_path/'pipeline.yaml'
    config.write_text(yaml.safe_dump({'native_scale_up':dict(planned_units=str(planned),episode_ledger=str(ledger),native_paper_section=True)}))
    paper=tmp_path/'paper';paper.mkdir();(paper/'existing.txt').write_text('preserve')
    with pytest.raises(ValueError,match='every primary cohort outcome terminal'):
        generate(config,tmp_path/'tables',paper_root=paper)
    assert list(paper.iterdir())==[paper/'existing.txt']


def test_complete_test_transfer_preserves_dev_and_names_accurate_audit(tmp_path):
    plans=[dict(row(),split='TEST'),dict(row('B0_FIXED_NATIVE',success=False),split='TEST')]
    planned,ledger=files(tmp_path,plans,plans)
    config=tmp_path/'pipeline.yaml'
    config.write_text(yaml.safe_dump({'native_scale_up':dict(planned_units=str(planned),episode_ledger=str(ledger),native_paper_section=True)}))
    paper=tmp_path/'paper';(paper/'paper_sections').mkdir(parents=True)
    (paper/'paper_sections/06_native_dev.tex').write_text('unchanged DEV')
    generate(config,tmp_path/'tables',paper_root=paper)
    assert (paper/'paper_sections/06_native_dev.tex').read_text()=='unchanged DEV'
    assert (paper/'paper_sections/06_native_test.tex').exists()
    receipt=json.loads((paper/'audit/native_test_transfer.json').read_text())
    assert receipt['primary_complete'] is True and 'TEST' in receipt['scope']
    assert not (paper/'audit/native_dev_transfer.json').exists()


def media_unit(tmp_path, *, missing=False):
    from robo.eval.native_scale_tables import _sha
    unit=row()
    video=tmp_path/'continuous.mp4'
    if not missing:video.write_bytes(b'original continuous worker video')
    action=tmp_path/'actions.json';action.write_text('[1,2]')
    result=tmp_path/'result.json'
    result.write_text(json.dumps(dict(executed=True,success=True,video_path=str(video),video_error=None,
        actions_path=str(action),actions_sha256=_sha(action))))
    unit.update(result_path=str(result),result_sha256=_sha(result))
    return unit,video,action


def test_release_seals_worker_video_and_preserves_acquisition_boundary(tmp_path):
    from robo.eval.native_scale_tables import _sha
    unit,video,action=media_unit(tmp_path)
    planned,ledger=files(tmp_path,[unit],[unit]);config=tmp_path/'pipeline.yaml'
    config.write_text(yaml.safe_dump({'native_scale_up':dict(planned_units=str(planned),episode_ledger=str(ledger))}))
    output=tmp_path/'release';report=generate(config,output)
    seal=json.loads((output/'episode_media_seal.json').read_text())
    assert seal['seal_boundary']=='table_release_creation'
    assert seal['original_acquisition_seal_established'] is False
    assert seal['executed_units']==1
    media=seal['records'][0]['media']['video_path']
    assert media['state']=='SEALED_AT_RELEASE' and media['sha256']==_sha(video)
    assert media['prior_result_digest_present'] is False
    assert report['source_lineage']['unit_0_video_path']['sha256']==_sha(video)
    before=report['source_lineage']['unit_0_video_path']['sha256']
    video.write_bytes(b'substituted video')
    assert _sha(video)!=before  # Renderer must compare to this sealed digest.


def test_missing_video_does_not_delete_executed_outcome(tmp_path):
    from robo.eval.native_scale_tables import episode_media_sources,finish_media_seal,_sha
    unit,_,_=media_unit(tmp_path,missing=True)
    sources,records=episode_media_sources([unit]);seal=finish_media_seal(records,{k:dict(path=str(p),sha256=_sha(p)) for k,p in sources.items()})
    assert seal['records'][0]['media']['video_path']['state']=='MISSING'
    assert unit['executed'] is True and unit['success'] is True
    assert 'unit_0_video_path' not in sources


def test_existing_action_digest_is_not_recertified_after_mutation(tmp_path):
    from robo.eval.native_scale_tables import episode_media_sources,finish_media_seal,_sha
    unit,_,action=media_unit(tmp_path);action.write_text('[999]')
    sources,records=episode_media_sources([unit])
    with pytest.raises(ValueError,match='existing result digest'):
        finish_media_seal(records,{k:dict(path=str(p),sha256=_sha(p)) for k,p in sources.items()})


def test_retry_action_sidecars_are_release_bound_without_inventing_missing_proof(tmp_path):
    from robo.eval.native_scale_tables import seal_retry_action_sources,_sha
    pool=tmp_path/'candidate_pool.json';pool.write_text(json.dumps({'retry_candidate':{'proposal_id':'new-transform'}}))
    side=tmp_path/'registration_retry';side.mkdir()
    (side/'registration.json').write_text('{"transform":"genuine new action"}')
    lineage={str(pool):dict(path=str(pool),sha256=_sha(pool))}
    receipt=seal_retry_action_sources(lineage)
    assert str(side/'registration.json') in lineage
    assert receipt['records'][0]['sidecars']['evidence.json']['state']=='MISSING'
    assert str(side/'evidence.json') not in lineage
    pool.write_text('{}')
    with pytest.raises(ValueError,match='pool changed'):seal_retry_action_sources(lineage)


def test_adjudication_disclosure_counts_original_failures_and_execution_separately(tmp_path):
    from robo.eval.native_scale_tables import adjudication_tables,_sha
    audit=tmp_path/'audit.json';audit.write_text(json.dumps(dict(part_properties=dict(is_watertight=True,is_convex=False),
        geometry_repaired=False,threshold_changed=False,new_policy_outcomes=0,
        limitations=['Near-threshold local nonconvexity: keep nuance.'])))
    proof={'independent_audit':dict(path=str(audit),sha256=_sha(audit))}
    failed=[dict(row('B0_FIXED_NATIVE',reset='r'+str(i),status='BUILD_FAILED',success=None),measured=True,
        original_terminal_status='CODE_FAILED',failure_classification='invalid_supplied_collision_volume',classification_adjudication=proof) for i in range(2)]
    executed=dict(row('B0_FIXED_NATIVE',instance='i1'),measured=True)
    geometry=[dict(method='B0_FIXED_NATIVE',cohort_id='dev',scope='L0_target_only',split='DEV',accepted_objects=2)]
    records,summary=adjudication_tables(failed+[executed],geometry)
    assert len(records)==2 and all(r['native_success'] is None for r in records)
    assert summary[0]['classified_units']==2 and summary[0]['affected_instances']==1
    assert summary[0]['producer_accepted_objects']==2 and summary[0]['native_executed_instances']==1
    failed[0]['executed']=True
    with pytest.raises(ValueError,match='prepolicy'):adjudication_tables(failed,geometry)
    failed[0]['executed']=False;audit.write_text('{}')
    with pytest.raises(ValueError,match='bytes changed'):adjudication_tables(failed,geometry)
