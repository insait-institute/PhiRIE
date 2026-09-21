import copy
import json
from pathlib import Path

import pytest

from robo.eval.native_scale_scope import scope_tables
from robo.eval.native_scale_tables import _sha
from robo.manifest.hash import canonical_hash
from robo.roundtrip.local_policy_instance import SCOPE_BLOCKS
from robo.roundtrip.spec import load_spec


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def phase(tmp_path, scope_status='RECORDED'):
    base = load_spec(Path(__file__).parents[1] / 'configs/experiments/sim_recon_sim/reference.yaml')
    base.update(schema_version=2, cohort_id='dev', canonical_manifest_sha256='a' * 64,
                reset_contract_sha256='b' * 64, execution_protocol='primary_native', horizon=900,
                scope_run_id='scope-phase', scope_engine_protocol='l1_b3_v1',
                policy_engine_protocol='per_canonical_engine_v1')
    base['instance']['task_id'] = 'PickPlaceSinkToCounter'
    base['policy']['checkpoint_receipt_sha256'] = 'c' * 64
    plans, outcomes = [], []
    for instance in range(2):
        iid = f'native-{instance}'
        engine = dict(engine_protocol='per_canonical_engine_v1', canonical_instance_id=iid,
                      engine_id=iid, process_uuid='process-' + iid, runtime={'device': 'synthetic'})
        engine['runtime_fingerprint'] = canonical_hash(engine['runtime'])
        for reset in range(2):
            baseline = None
            for method, scope in SCOPE_BLOCKS['l1_b3_v1']:
                uid = f'{instance}-{reset}-{method}-{scope}'
                config = copy.deepcopy(base)
                config.update(canonical_instance_id=iid, reset_id=f'r{reset}', policy_rng_seed=reset,
                              controller_method=method, scope=scope,
                              replacement_scope='target_only' if scope == 'L0_target_only' else 'target_destination')
                config['instance']['layout_id'] = instance
                path = tmp_path / 'configs' / (uid + '.json')
                write(path, config)
                unit = dict(unit_id=uid, cohort_id='dev', canonical_instance_id=iid,
                    reset_id=f'r{reset}', policy_rng_seed=reset, policy_id='policy', controller_method=method,
                    scope=scope, sensor_regime=base['sensor_regime'], renderer='native', execution_protocol='primary_native',
                    layout_id=instance, style_id=config['instance']['style_id'], task_id='PickPlaceSinkToCounter',
                    split='development', native_horizon=900, scope_run_id='scope-phase',
                    config_path=str(path), config_sha256=canonical_hash(config), terminal_status='NOT_SCHEDULED',
                    executed=None, success=None)
                plans.append(copy.deepcopy(unit))
                status = scope_status if scope == 'L1_target_destination' else 'RECORDED'
                if status in ('ABSTAINED', 'BUILD_FAILED'):
                    evidence = tmp_path / 'construction' / (uid + '.json')
                    write(evidence, {'status': status, 'policy_invoked': False})
                    unit.update(terminal_status=status, executed=False, evidence_path=str(evidence), evidence_sha256=_sha(evidence))
                elif status == 'RECORDED':
                    result_path = tmp_path / 'workers' / uid / 'runner/episode/result.json'
                    result = {k: unit[k] for k in ('canonical_instance_id', 'controller_method', 'scope', 'reset_id', 'policy_rng_seed', 'config_sha256')}
                    # Reference includes a real failure in this synthetic complete roster.
                    result.update(executed=True, error=None, success=(reset == 0 if method == 'REF_NATIVE' else scope == 'L0_target_only'),
                                  native_schema_version=2, execution_kind='closed_loop_visual_policy',
                                  policy_engine=engine, policy_identity={'checkpoint_receipt_sha256': 'c' * 64, 'policy_engine': engine},
                                  reset_contract_sha256='b' * 64)
                    result['policy_identity_sha256'] = canonical_hash(result['policy_identity'])
                    if scope != 'L0_target_only':
                        bundle = dict(scope=scope, canonical_instance_id=iid, baseline_result_sha256=baseline['result_sha256'],
                                      treatment_axis='replacement_scope', system_variant='B3_target_B3_destination_L1',
                                      entities=[{'role': role, 'constructor_method': method} for role in ('target', 'receptacle')],
                                      claim_scope='native room and support retained; binding-specific')
                        bundle_path = result_path.parents[2] / 'scope_bundle.json'
                        write(bundle_path, bundle)
                        result['scope_treatment'] = dict(treatment_axis='replacement_scope', baseline_scope='L0_target_only',
                            system_variant='B3_target_B3_destination_L1',
                            reference_result_sha256=baseline['result_sha256'], scope_bundle_sha256=_sha(bundle_path))
                    write(result_path, result)
                    unit.update(terminal_status=status, executed=True, success=result['success'], result=result,
                                result_path=str(result_path), result_sha256=_sha(result_path))
                    if method == 'B3_AGENT_NATIVE' and scope == 'L0_target_only':
                        baseline = unit
                outcomes.append(unit)
    record = dict(expected_instances=2, expected_resets_per_instance=2)
    for name, rows in [('planned_units', plans), ('episode_ledger', outcomes)]:
        path = tmp_path / (name + '.jsonl')
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        record[name], record[name + '_sha256'] = str(path), _sha(path)
    manifest = tmp_path / 'scope_execution_manifest.json'
    write(manifest, dict(kind='canonical_scope_matrix_binding', scope_run_id='scope-phase', protocol='l1_b3_v1',
                         planned_units=12, controls=8, scope_measurements=4, primary_results_reused=False))
    record.update(manifest=str(manifest), manifest_sha256=_sha(manifest))
    return record


def amend_ledger(record, fn):
    path = Path(record['episode_ledger'])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    fn(rows)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    record['episode_ledger_sha256'] = _sha(path)


def test_complete_negative_scope_pair_preserves_controls_and_clusters(tmp_path):
    outputs, sources = scope_tables([phase(tmp_path)])
    rows = outputs['T4_scope']
    assert len(rows) == 2 and all(r['planned'] == 4 for r in rows)
    l1 = next(r for r in rows if r['scope'] == 'L1_target_destination')
    assert l1['success_per_planned'] == 0 and l1['executed'] == 4
    assert l1['absolute_replay_position_error_cm'] is None
    comparison = outputs['T4_scope_comparisons'][0]
    assert comparison['delta_pp'] == -100 and comparison['ci95_pp'] == [-100, -100]
    assert comparison['observed_pairs_descriptive']['instances'] == 2
    assert comparison['observed_pairs_descriptive']['layouts'] == 2
    assert comparison['preservation_claim'] == 'NOT_RUN'
    assert outputs['T4_native_controls'][0]['successes_observed'] == 2
    assert sources


@pytest.mark.parametrize('status', ['ABSTAINED', 'BUILD_FAILED', 'NOT_SCHEDULED'])
def test_failed_and_unmeasured_scopes_keep_full_denominator(tmp_path, status):
    outputs, _ = scope_tables([phase(tmp_path, status)])
    l1 = next(r for r in outputs['T4_scope'] if r['scope'] == 'L1_target_destination')
    assert l1['planned'] == 4 and l1['executed'] == 0
    assert l1['success_per_executed'] is None and l1['reconstructed_context'] is None
    assert l1['success_per_planned'] == (None if status == 'NOT_SCHEDULED' else 0)
    assert len([r for r in outputs['T4_scope_failures'] if r['scope'] == 'L1_target_destination']) == 4


@pytest.mark.parametrize('mutation', ['process', 'baseline', 'policy', 'reset', 'bundle', 'result_scope'])
def test_rejects_cross_process_or_changed_scope_evidence(tmp_path, mutation):
    record = phase(tmp_path)
    def change(rows):
        row = next(r for r in rows if r['scope'] == 'L1_target_destination')
        path = Path(row['result_path']); result = json.loads(path.read_text())
        if mutation == 'process':
            result['policy_engine']['process_uuid'] = 'different'
        elif mutation == 'baseline':
            result['scope_treatment']['reference_result_sha256'] = '0' * 64
        elif mutation == 'policy':
            result['policy_identity']['checkpoint_receipt_sha256'] = 'different'
        elif mutation == 'reset':
            result['reset_contract_sha256'] = 'd' * 64
        elif mutation == 'bundle':
            result['scope_treatment']['scope_bundle_sha256'] = '0' * 64
        else:
            result['scope'] = 'L2_workspace'
        write(path, result); row['result'] = result; row['result_sha256'] = _sha(path)
    amend_ledger(record, change)
    with pytest.raises(ValueError):
        scope_tables([record])


@pytest.mark.parametrize('mutation', ['instances', 'resets', 'phase', 'source', 'duplicate'])
def test_rejects_narrowed_or_duplicate_scope_populations(tmp_path, mutation):
    record = phase(tmp_path)
    if mutation == 'instances':
        record['expected_instances'] = 24
    elif mutation == 'resets':
        record['expected_resets_per_instance'] = 10
    elif mutation == 'phase':
        amend_ledger(record, lambda rows: rows[0].update(scope_run_id='another-phase'))
    elif mutation == 'source':
        record['planned_units_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        scope_tables([record, record] if mutation == 'duplicate' else [record])


def test_missing_construction_proof_cannot_create_zero_success(tmp_path):
    record = phase(tmp_path, 'ABSTAINED')
    amend_ledger(record, lambda rows: next(r for r in rows if r['scope'] == 'L1_target_destination').pop('evidence_sha256'))
    with pytest.raises(ValueError, match='construction evidence'):
        scope_tables([record])


def test_zero_gap_between_failed_methods_enables_no_preservation_claim(tmp_path):
    record = phase(tmp_path, 'ABSTAINED')
    def fail_baseline(rows):
        for row in rows:
            if row['controller_method'] == 'B3_AGENT_NATIVE' and row['scope'] == 'L0_target_only':
                path = Path(row['result_path']); result = json.loads(path.read_text())
                result['success'] = False; write(path, result)
                row.update(success=False, result=result, result_sha256=_sha(path))
    amend_ledger(record, fail_baseline)
    tables, _ = scope_tables([record])
    comparison = tables['T4_scope_comparisons'][0]
    assert comparison['delta_pp'] == 0
    assert comparison['observed_pairs_descriptive']['both_failure'] == 4
    assert comparison['preservation_claim'] == comparison['scope_benefit_claim'] == 'NOT_RUN'


def test_inherited_target_losses_are_separate_from_new_scope_losses(tmp_path):
    record = phase(tmp_path, 'BUILD_FAILED')
    def inherit_one_instance(rows):
        for row in rows:
            if row['canonical_instance_id'] == 'native-0' and row['controller_method'] == 'B3_AGENT_NATIVE' and row['scope'] == 'L0_target_only':
                scope = next(r for r in rows if r['canonical_instance_id'] == row['canonical_instance_id']
                             and r['reset_id'] == row['reset_id'] and r['scope'] == 'L1_target_destination')
                row.update(terminal_status='BUILD_FAILED', executed=False, success=None,
                           evidence_path=scope['evidence_path'], evidence_sha256=scope['evidence_sha256'])
                for k in ('result', 'result_path', 'result_sha256'):
                    row.pop(k)
    amend_ledger(record, inherit_one_instance)
    outputs, _ = scope_tables([record])
    scope = next(r for r in outputs['T4_scope'] if r['scope'] == 'L1_target_destination')
    assert scope['inherited_target_nonexecution_observed'] == 2
    assert scope['additional_scope_nonexecution_observed'] == 2
    assert scope['inherited_target_instances_observed'] == scope['additional_scope_instances_observed'] == 1
    def forged_inheritance(rows):
        r = next(r for r in rows if r['canonical_instance_id'] == 'native-0' and r['scope'] == 'L0_target_only'
                 and r['controller_method'] == 'B3_AGENT_NATIVE')
        r['terminal_status'] = 'ABSTAINED'
    amend_ledger(record, forged_inheritance)
    with pytest.raises(ValueError, match='inherited target'):
        scope_tables([record])
