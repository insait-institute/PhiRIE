"""T4 accounting from separate canonical scope ledgers and frozen full rosters.

Fresh scope controls never enter the primary T2 population. Each phase is
independent: paired statistics require the same method, instance, reset and
policy process, and use the existing hierarchical bootstrap.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from robo.eval.metric_utils import paired_hierarchical_bootstrap
from robo.eval.native_scale_evidence import Sources
from robo.eval.native_scale_tables import BLOCK, _aggregate, load_units
from robo.manifest.hash import canonical_hash

METHODS = ('B3_AGENT_NATIVE', 'B4_ROOM_REPAIR_NATIVE')
BASE = 'L0_target_only'


def _pair_key(row):
    return (row['canonical_instance_id'], row['reset_id'], row['policy_rng_seed'])


def _failure_origin(unit, baseline):
    if unit['terminal_status'] not in ('BUILD_FAILED', 'ABSTAINED'):
        return None
    if baseline['terminal_status'] in ('BUILD_FAILED', 'ABSTAINED'):
        if any(unit.get(k) != baseline.get(k) for k in ('terminal_status', 'evidence_path', 'evidence_sha256')):
            raise ValueError('scope inherited target failure changes its construction evidence')
        return 'inherited_target'
    return 'additional_scope'


def _load_phase(record, src):
    from robo.roundtrip.local_policy_instance import SCOPE_BLOCKS
    from robo.roundtrip.spec import validate_spec

    manifest = src.read(record['manifest'], record['manifest_sha256'])
    run = manifest.get('scope_run_id')
    if (manifest.get('kind') != 'canonical_scope_matrix_binding' or not run
            or manifest.get('protocol') not in SCOPE_BLOCKS
            or manifest.get('primary_results_reused') is not False):
        raise ValueError('scope phase needs a declared fresh-control protocol')
    planned = src.bind(record['planned_units'], record['planned_units_sha256'])
    ledger = src.bind(record['episode_ledger'], record['episode_ledger_sha256'])
    units = load_units(planned, ledger)
    counts = (len(units), sum(u['scope'] == BASE for u in units),
              sum(u['scope'] != BASE for u in units))
    if counts != tuple(manifest[k] for k in ('planned_units', 'controls', 'scope_measurements')):
        raise ValueError('scope population differs from frozen phase manifest')
    n, resets = record['expected_instances'], record['expected_resets_per_instance']
    if type(n) is not int or type(resets) is not int or min(n, resets) < 1:
        raise ValueError('positive declared instance/reset counts required')
    arms = tuple(SCOPE_BLOCKS[manifest['protocol']])
    grouped = defaultdict(dict)
    configs, results, engines = {}, {}, {}
    for unit in units:
        if unit.get('scope_run_id') != run:
            raise ValueError('scope run identity missing or changed in ledger')
        arm = unit['controller_method'], unit['scope']
        key = _pair_key(unit)
        if arm not in arms or key in grouped[arm]:
            raise ValueError('unexpected scope arm or duplicate paired reset')
        grouped[arm][key] = unit
        config = src.read(unit['config_path'])
        if (canonical_hash(config) != unit['config_sha256']
                or config.get('scope_run_id') != run
                or config.get('scope_engine_protocol') != manifest['protocol']):
            raise ValueError('frozen scope config or phase identity changed')
        validate_spec(config)
        for field in ('controller_method', 'scope', 'canonical_instance_id', 'reset_id', 'policy_rng_seed'):
            if config[field] != unit[field]:
                raise ValueError('scope config disagrees with canonical roster')
        configs[unit['unit_id']] = config
        if unit['terminal_status'] in ('BUILD_FAILED', 'ABSTAINED'):
            if not unit.get('evidence_path') or not unit.get('evidence_sha256'):
                raise ValueError('scope nonexecution requires immutable construction evidence')
            src.bind(unit['evidence_path'], unit['evidence_sha256'])
        if unit['executed'] is not True:
            continue
        result = src.read(unit['result_path'], unit['result_sha256'])
        if (result.get('executed') is not True or result.get('success') != unit['success']
                or result.get('error') is not None or result.get('config_sha256') != unit['config_sha256']):
            raise ValueError('scope result outcome/config differs from canonical ledger')
        for field in ('controller_method', 'scope', 'canonical_instance_id', 'reset_id', 'policy_rng_seed'):
            if result.get(field) != unit[field]:
                raise ValueError('scope result identity differs from planned unit')
        identity = result.get('policy_identity')
        if (not isinstance(identity, dict) or not identity
                or result.get('policy_identity_sha256') != canonical_hash(identity)
                or identity.get('checkpoint_receipt_sha256') != config['policy']['checkpoint_receipt_sha256']
                or result.get('reset_contract_sha256') != config['reset_contract_sha256']):
            raise ValueError('scope policy/checkpoint or reset evidence changed')
        engine = result.get('policy_engine', {})
        if (engine.get('engine_protocol') != 'per_canonical_engine_v1'
                or engine.get('canonical_instance_id') != key[0]
                or not engine.get('process_uuid') or not engine.get('engine_id')
                or engine.get('runtime_fingerprint') != canonical_hash(engine.get('runtime', {}))
                or identity.get('policy_engine') != engine):
            raise ValueError('scope policy engine identity is missing or corrupt')
        if key[0] in engines and engines[key[0]] != engine:
            raise ValueError('scope controls cross policy processes or hardware')
        engines[key[0]] = engine
        results[unit['unit_id']] = result
    keys = set(grouped[arms[0]])
    if set(grouped) != set(arms) or any(set(grouped[a]) != keys for a in arms):
        raise ValueError('scope roster is not rectangular across methods and scopes')
    by_instance = defaultdict(set)
    for iid, reset, rng in keys:
        by_instance[iid].add((reset, rng))
    if len(by_instance) != n or any(len(v) != resets for v in by_instance.values()):
        raise ValueError('scope roster narrows declared instances or resets')
    declared = manifest.get('canonical_instance_ids')
    if declared is not None and (len(declared) != n or set(declared) != set(by_instance)):
        raise ValueError('scope roster changes predeclared canonical instances')
    # All scope rows belong to one scientific block apart from the treatment scope.
    block_fields = tuple(k for k in BLOCK if k != 'scope')
    blocks = {tuple(u[k] for k in block_fields) for u in units}
    if len(blocks) != 1:
        raise ValueError('scope phase mixes cohorts, policies or observation protocols')
    block = dict(zip(block_fields, next(iter(blocks))))
    return run, block, grouped, configs, results


def scope_tables(records):
    """Return T4 tables and source lineage for paper_pipeline's existing writer.

    Each record binds one full phase manifest/roster and its canonical merger
    snapshot, with explicit expected instance and reset counts. Incomplete
    phases retain null population rates. Repeated phases are never pooled.
    """
    from robo.roundtrip.scope_bundle import validate_scope_pair

    src = Sources()
    src.bind(__file__)
    tables, comparisons, controls, failures = [], [], [], []
    seen = set()
    for record in records:
        run, block, groups, configs, results = _load_phase(record, src)
        if run in seen:
            raise ValueError('duplicate scope phase; cannot count repeated controls twice')
        seen.add(run)
        for (method, scope), indexed in sorted(groups.items()):
            arm = list(indexed.values())
            summary = dict(scope_run_id=run, **block, method=method, scope=scope, **_aggregate(arm))
            summary.update(absolute_replay_position_error_cm=None,
                           scorer_interpretation='binding-specific native task predicate',
                           reconstructed_context=None, retained_context=None)
            if method == 'REF_NATIVE':
                controls.append(summary)
                continue
            if method not in METHODS:
                raise ValueError('unplanned scope method')
            scope_receipts = []
            if scope == BASE:
                if any(u['executed'] is True for u in arm):
                    summary.update(reconstructed_context='target', retained_context='native destination and room')
                tables.append(summary)
                continue
            baseline = groups[(method, BASE)]
            pairs = []
            origins = defaultdict(list)
            for key, unit in indexed.items():
                ref = baseline[key]
                origin = _failure_origin(unit, ref)
                if origin:
                    origins[origin].append(unit['canonical_instance_id'])
                c, b = configs[unit['unit_id']], configs[ref['unit_id']]
                allowed = {'scope', 'replacement_scope'}
                if canonical_hash({k: v for k, v in c.items() if k not in allowed}) != canonical_hash({k: v for k, v in b.items() if k not in allowed}):
                    raise ValueError('scope pair changes fields beyond replacement scope')
                if unit['executed'] is True:
                    if ref['executed'] is not True:
                        raise ValueError('executed scope arm lacks executed same-method baseline')
                    actual, base = results[unit['unit_id']], results[ref['unit_id']]
                    validate_scope_pair(c, b, base)
                    treatment = actual.get('scope_treatment', {})
                    if (treatment.get('treatment_axis') != 'replacement_scope'
                            or treatment.get('baseline_scope') != BASE
                            or treatment.get('reference_result_sha256') != ref['result_sha256']
                            or actual.get('policy_identity') != base.get('policy_identity')
                            or actual.get('reset_contract_sha256') != base.get('reset_contract_sha256')):
                        raise ValueError('scope pair does not bind the actual same-method control')
                    bundle_path = Path(unit['result_path']).parents[2] / 'scope_bundle.json'
                    bundle = src.read(bundle_path, treatment['scope_bundle_sha256'])
                    if (bundle.get('baseline_result_sha256') != ref['result_sha256']
                            or bundle.get('canonical_instance_id') != unit['canonical_instance_id']
                            or bundle.get('scope') != scope or bundle.get('treatment_axis') != 'replacement_scope'
                            or bundle.get('system_variant') != treatment.get('system_variant')
                            or not bundle.get('entities')
                            or any(e.get('constructor_method') != method for e in bundle['entities'])):
                        raise ValueError('scope bundle describes another baseline or replacement')
                    scope_receipts.append(bundle)
                if ref['measured'] and unit['measured']:
                    pairs.append(dict(layout_id=unit['layout_id'], canonical_instance_id=key[0], reset_id=key[1],
                                      a=int(ref['service_success']), b=int(unit['service_success'])))
            stats = paired_hierarchical_bootstrap(pairs)
            complete = len(pairs) == len(indexed)
            comparisons.append(dict(scope_run_id=run, **block, method=method, baseline_scope=BASE, scope=scope,
                planned_pairs=len(indexed), measured_pairs=len(pairs), paired_population_complete=complete,
                delta_pp=100 * stats['delta'] if complete else None,
                ci95_pp=[100 * v for v in stats['ci95']] if complete else [None, None],
                observed_pairs_descriptive=stats, preservation_claim='NOT_RUN', scope_benefit_claim='NOT_RUN'))
            if scope_receipts:
                summary['reconstructed_context'] = '; '.join(sorted({', '.join(sorted(e['role'] for e in b['entities']))
                    + (': ' + b['static_destination_scope'] if b.get('static_destination_scope') else '') for b in scope_receipts}))
                retained = [b.get('retained_fixture_context', b['claim_scope']) for b in scope_receipts]
                summary['retained_context'] = '; '.join(sorted({'; '.join(v) if isinstance(v, list) else v for v in retained}))
            for origin in ('inherited_target', 'additional_scope'):
                summary[origin + '_nonexecution_observed'] = len(origins[origin])
                summary[origin + '_instances_observed'] = len(set(origins[origin]))
            tables.append(summary)
        for arm in groups.values():
            for u in arm.values():
                if u['success'] is not True:
                    failures.append(dict(scope_run_id=run, unit_id=u['unit_id'], canonical_instance_id=u['canonical_instance_id'],
                        method=u['controller_method'], scope=u['scope'], reset_id=u['reset_id'], terminal_status=u['terminal_status'],
                        measured=u['measured'], executed=u['executed'], success=u['success'],
                        evidence_path=u.get('evidence_path'), evidence_sha256=u.get('evidence_sha256'),
                        construction_failure_origin=_failure_origin(u, groups[(u['controller_method'], BASE)][_pair_key(u)])
                        if u['scope'] != BASE else None))
    return {'T4_scope': tables, 'T4_scope_comparisons': comparisons,
            'T4_native_controls': controls, 'T4_scope_failures': failures}, src.lineage
