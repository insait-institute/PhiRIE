"""Authenticate existing native context/cost/diagnostic receipts for paper tables.

No geometry, physical or rollout metrics are recomputed here. Construction
self-checks never become independent physics validation or policy outcomes.
"""
from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path
from robo.eval.construction_metrics import _number

METHODS={'B3':'B3_AGENT_NATIVE','B4':'B4_ROOM_REPAIR_NATIVE','BM':'BM_BUDGET_MATCHED_NATIVE','V1':'V1_VERIFY_ABSTAIN_NATIVE'}


class Sources:
    def __init__(self):self.lineage={}
    def bind(self,path,digest=None):
        from robo.eval.native_scale_tables import _sha
        path=Path(path).resolve();actual=_sha(path)
        if digest is not None and actual!=digest:raise ValueError(f'evidence source bytes changed: {path}')
        self.lineage[str(path)]={'path':str(path),'sha256':actual}
        return path
    def read(self,path,digest=None):
        return json.loads(self.bind(path,digest).read_text())
    def seconds(self,row,key,path):
        value=_number(row,key,record=str(path),nullable=True)
        if value is not None and value<0:raise ValueError('negative stage seconds')
        return value


def context_tables(units,main,roots,src):
    populations={r['canonical_instance_id'] for r in units if r['canonical_instance_id'] is not None}
    found={};details=[]
    for root in map(Path,roots):
        receipt=src.read(root/'execution_receipt.json');summary=src.read(root/'context_results.json')
        indexed={}
        for row in summary:
            key=(row['instance_slot_id'],row['method'])
            if key in indexed:raise ValueError('duplicate context summary row')
            indexed[key]=row
        for row in receipt:
            iid,method=row['canonical_instance_id'],METHODS[row['method']];key=(iid,method)
            if iid not in populations:raise ValueError('unplanned context instance')
            if key in found:raise ValueError('duplicate context instance/method')
            if row['state']=='BUILD_FAILED':found[key]={'state':'BUILD_FAILED'};continue
            if row['state']!='COMPLETE':found[key]={'state':'UNMEASURED'};continue
            path=Path(row['build_manifest']);manifest=src.read(path,row['build_manifest_sha256'])
            if manifest['source_code']['dirty']:raise ValueError('context source was dirty')
            if manifest['method']!=row['method'] or manifest['accepted']!=row['accepted']:
                raise ValueError('context decision mismatch')
            before=src.read(path.parent/'verification_before.json');after=src.read(path.parent/'verification_after.json')
            report=indexed.pop((row['instance_slot_id'],row['method']),None)
            if report is None:raise ValueError('missing measured context summary row')
            checks={'accepted':manifest['accepted'],'actual_calls':manifest['actual_calls'],
                    'before_drift_m':before['settle_drift_m'],'after_drift_m':after['settle_drift_m'],
                    'before_rotation_deg':before['settle_rotation_deg'],'after_rotation_deg':after['settle_rotation_deg'],
                    'before_penetration_m':before['initial_penetration_m'],'after_penetration_m':after['initial_penetration_m'],
                    'after_public_residual_m':after['public_object_surface_guard']['residual_m'],
                    'total_wall_s':manifest['timing']['total_wall_s'],
                    'extra_actions_wall_s':manifest['timing']['extra_actions_wall_s']}
            if any(report[k]!=v for k,v in checks.items()):raise ValueError('context aggregate differs from raw producer')
            if report['native_policy_success'] is not None:raise ValueError('construction check is not policy outcome')
            if type(manifest['accepted']) is not bool or not 0<=manifest['actual_calls']<=2:
                raise ValueError('invalid context decision/action budget')
            for k in ('total_wall_s','extra_actions_wall_s'):src.seconds(checks,k,path)
            # Explicit canonical identity/capture binding, not only slot names.
            binding=row['binding']
            if binding['canonical_instance_id']!=iid or manifest['capture_manifest_sha256']!=binding['capture_manifest_sha256']:
                raise ValueError('context canonical capture differs')
            found[key]={'state':'COMPLETE','accepted':manifest['accepted'],'own_check_passed':after['passed'],'scope':binding.get('scope','L0_target_only'),'cohort_id':binding.get('cohort_id'),**checks}
            details.append(dict(report,canonical_instance_id=iid,method=method,
                                own_check_passed=after['passed'],independent_room_stability=None))
        if indexed:raise ValueError('unplanned context summary rows')
    return _context_summary(units,main,found),details


def _context_summary(units,main,found):
    rows=[]
    for arm in main:
        if arm['method'] not in METHODS.values():continue
        selected=[u for u in units if u['controller_method']==arm['method'] and all(u[k]==arm[k] for k in ('cohort_id','scope','sensor_regime','renderer','execution_protocol','split','policy_id'))]
        ids={u['canonical_instance_id'] for u in selected}
        known=[found[(iid,arm['method'])] for iid in ids if (iid,arm['method']) in found and arm['scope']=='L0_target_only']
        if any(r.get('cohort_id') not in (None,arm['cohort_id']) or r.get('scope',arm['scope'])!=arm['scope'] for r in known):raise ValueError('context block differs from native population')
        measured=[r for r in known if r['state']=='COMPLETE'];terminal=sum(r['state'] in ('COMPLETE','BUILD_FAILED') for r in known)
        accepted_ids={iid for iid in ids if found.get((iid,arm['method']),{}).get('accepted') is True}
        if any(u.get('executed') is True and found.get((u['canonical_instance_id'],arm['method']),{}).get('accepted') is False for u in selected):raise ValueError('abstained context build was executed')
        rows.append(dict(method=arm['method'],cohort_id=arm['cohort_id'],scope=arm['scope'],split=arm['split'],
            instances_planned=len(ids),instances_probed=len(measured),instance_decisions_available=terminal,
            instances_accepted=sum(r['accepted'] for r in measured) if terminal==len(ids) else None,
            build_failed_instances=sum(r['state']=='BUILD_FAILED' for r in known),
            upstream_build_failed=sum(r['state']=='BUILD_FAILED' and r.get('upstream',True) for r in known),
            context_artifact_failed=sum(r['state']=='BUILD_FAILED' and not r.get('upstream',True) for r in known),
            own_check_passed_instances=sum(r['own_check_passed'] for r in measured) if measured else None,
            tasks_planned=arm['planned'],tasks_accepted=sum(u['canonical_instance_id'] in accepted_ids for u in selected) if terminal==len(ids) else None,tasks_executed=arm['executed'],native_outcomes_available=arm['measured'],
            success_per_planned=arm['success_per_planned'],success_per_executed=arm['success_per_executed'],
            independent_room_stability=None,extra_tool_calls=sum(r['actual_calls'] for r in measured) if measured else None,
            measured_context_seconds=sum(r['total_wall_s'] for r in measured) if measured else None,
            measured_extra_action_seconds=sum(r['extra_actions_wall_s'] for r in measured) if measured else None,
            whole_method_seconds=None,interpretation=f"{arm['split']} construction self-check; independent native policy outcomes from canonical ledger"))
    return rows


def context_binding_tables(units,main,paths,src):
    """Read final per-instance decisions through the canonical constructor join."""
    from robo.eval.native_scale_construction import construction_decisions
    decisions,lineage=construction_decisions(units,paths);src.lineage.update(lineage)
    found={};details=[]
    for (iid,method),row in decisions.items():
        if method not in METHODS.values():continue
        if row['terminal_status']=='NOT_READY':found[iid,method]={'state':'UNMEASURED'};continue
        if row['terminal_status']=='BUILD_FAILED':found[iid,method]={'state':'BUILD_FAILED','upstream':bool(row.get('propagation_rule'))};continue
        path=Path(row['build_manifest']);m=src.read(path,row['build_manifest_sha256'])
        if METHODS.get(m['method'])!=method or m['source_code']['dirty'] or m['accepted']!=row['accepted']:
            raise ValueError('context binding method/source/decision differs')
        plan=src.read(row['plan_path'],row['plan_sha256']);binding=plan['binding']
        if binding['canonical_instance_id']!=iid or m['capture_manifest_sha256']!=binding['capture_manifest_sha256']:
            raise ValueError('context canonical capture differs')
        if type(m['actual_calls']) is not int or not 0<=m['actual_calls']<=2 or m['maximum_calls']!=2:
            raise ValueError('context action budget differs')
        before=src.read(path.parent/'verification_before.json');after=src.read(path.parent/'verification_after.json')
        checks=dict(accepted=m['accepted'],actual_calls=m['actual_calls'],
            before_drift_m=before['settle_drift_m'],after_drift_m=after['settle_drift_m'],
            before_rotation_deg=before['settle_rotation_deg'],after_rotation_deg=after['settle_rotation_deg'],
            before_penetration_m=before['initial_penetration_m'],after_penetration_m=after['initial_penetration_m'],
            after_public_residual_m=after['public_object_surface_guard']['residual_m'],
            total_wall_s=src.seconds(m['timing'],'total_wall_s',path),
            extra_actions_wall_s=src.seconds(m['timing'],'extra_actions_wall_s',path))
        found[iid,method]=dict(state='COMPLETE',own_check_passed=after['passed'],scope=binding['scope'],cohort_id=binding['cohort_id'],**checks)
        details.append(dict(canonical_instance_id=iid,method=method,build_manifest=str(path),build_manifest_sha256=row['build_manifest_sha256'],
            own_check_passed=after['passed'],independent_room_stability=None,native_policy_success=None,
            algorithm_required_context=method in ('B4_ROOM_REPAIR_NATIVE','BM_BUDGET_MATCHED_NATIVE'),
            action_bank_contract=m['action_bank_contract'],**checks))
    by_instance=defaultdict(dict)
    for r in details:by_instance[r['canonical_instance_id']][r['method']]=r
    for pair in by_instance.values():
        a,b=pair.get('B4_ROOM_REPAIR_NATIVE'),pair.get('BM_BUDGET_MATCHED_NATIVE')
        if a and b and a['action_bank_contract']!=b['action_bank_contract']:raise ValueError('B4/BM action bank differs')
    return _context_summary(units,main,found),details


def runtime_tables(units,records,src,construction_bindings=()):
    ids={u['canonical_instance_id'] for u in units if u['canonical_instance_id'] is not None}
    decisions={}
    if construction_bindings:
        from robo.eval.native_scale_construction import construction_decisions
        decisions,lineage=construction_decisions(units,construction_bindings);src.lineage.update(lineage)
    seen=set();details=[]
    for record in records:
        root=Path(record['b0_build']);failed=not (root/'build_manifest.json').exists();decision={}
        if failed:
            iid=record['canonical_instance_id'];decision=decisions.get((iid,'B0_FIXED_NATIVE'),{})
            if decision.get('terminal_status')!='BUILD_FAILED' or Path(decision.get('build_manifest','')).resolve()!=(root/'build_failure.json').resolve():
                raise ValueError('failed runtime requires bound constructor failure')
            manifest=src.read(root/'build_failure.json',decision['build_manifest_sha256'])
        else:
            manifest=src.read(root/'build_manifest.json');iid=manifest['canonical_instance_id']
        if iid not in ids or iid in seen:raise ValueError('duplicate or unplanned runtime instance')
        seen.add(iid)
        components={}
        for phase in ('segment','generate','align'):
            p=root/f'{phase}_runtime.json'
            if p.exists():
                r=src.read(p,manifest.get('source_hashes',{}).get(p.name) or decision.get('source_evidence',{}).get(str(p)))
                if not failed and r['exit_code']!=0:raise ValueError('successful-build runtime has failed phase')
                components['B0_'+phase+'_wrapper']=src.seconds(r,'wall_s',p)
            else:components['B0_'+phase+'_wrapper']=None
        components.update(shared_RVG_wrapper=None,shared_RVG_inference_nested=None,shared_selection=None,separate_CoACD=None)
        if record.get('shared_root') or record.get('candidate_pool'):
            shared=Path(record.get('shared_root','.'))
            pool_path=Path(record['candidate_pool']) if record.get('candidate_pool') else shared/'selection/candidate_pool.json'
            rvg_path=Path(record['rvg_receipt']) if record.get('rvg_receipt') else shared/'rvg/rvg_receipt.json'
            pool=src.read(pool_path)
            from robo.eval.native_scale_tables import _sha
            if pool['b0_manifest_sha256']!=_sha(root/'build_manifest.json'):raise ValueError('runtime pool B0 binding differs')
            r=src.read(rvg_path,pool['rvg_receipt_sha256'])
            components['shared_RVG_wrapper']=src.seconds(r,'wall_s',shared)
            components['shared_RVG_inference_nested']=src.seconds(r,'producer_wall_s',shared)
            if 'selection_wall_s' in pool:components['shared_selection']=src.seconds(pool,'selection_wall_s',shared)
        details.extend(dict(canonical_instance_id=iid,stage=k,seconds=v,construction_status='BUILD_FAILED' if failed else 'BUILT',whole_method_seconds=None,
            accounting_basis='original producer wall time; reused prerequisites remain charged',
            incremental_construction_compute_for_table_release_seconds=0.,reuse_discount_applied=False) for k,v in components.items())
    groups=defaultdict(list)
    for r in details:groups[r['stage']].append(r)
    table=[]
    for stage,rows in sorted(groups.items()):
        values=[r['seconds'] for r in rows if r['seconds'] is not None]
        table.append(dict(stage=stage,instances_planned=len(ids),instances_with_build_receipt=len(seen),
            instances_timed=len(values),timing_coverage=len(values)/len(ids),
            conditional_seconds_per_timed_instance=math.fsum(values)/len(values) if values else None,
            measured_seconds_sum=math.fsum(values) if values else None,whole_method_seconds=None,
            accounting_basis='original producer wall time; reused B0 prerequisites remain charged to B3/B4/BM',
            incremental_construction_compute_for_table_release_seconds=0.,reuse_discount_applied=False,
            required_by_methods=['B0_FIXED_NATIVE','B3_AGENT_NATIVE','B4_ROOM_REPAIR_NATIVE','BM_BUDGET_MATCHED_NATIVE'] if stage.startswith('B0_') else ['B3_AGENT_NATIVE','B4_ROOM_REPAIR_NATIVE','BM_BUDGET_MATCHED_NATIVE'],
            failed_builds_with_timing=sum(r['construction_status']=='BUILD_FAILED' and r['seconds'] is not None for r in rows),
            exclusions='capture, failures without bound timing receipts, scheduler waiting; separate CoACD/selection unmeasured unless explicitly timed; RVG inference nested within wrapper, never add twice'))
    return table,details


def _full_horizon_v2(payload,src):
    """Normalize a sealed one-canonical six-arm diagnostic, without recomputing metrics."""
    mapping={'REF_NATIVE':'REF_NATIVE','REF_IMPORT_CONTROL':'REF_IMPORT_CONTROL',
             'B0_REPLAY':'B0_FIXED_NATIVE_REPLAY','B0_LEARNED':'B0_FIXED_NATIVE',
             'OBSERVED_REPLAY':'OBSERVED_SURFACE_NATIVE_REPLAY','OBSERVED_LEARNED':'OBSERVED_SURFACE_NATIVE'}
    if payload['planned_canonical_instances']!=1 or payload['planned_episodes']!=6 or len(payload['rows'])!=6:
        raise ValueError('schema2 diagnostic requires complete predeclared six-arm roster')
    if {r['label'] for r in payload['rows']}!=set(mapping):raise ValueError('schema2 diagnostic method roster differs')
    src.read(payload['source_progress'],payload['source_progress_sha256'])
    group=payload['canonical_instance_id']+'/'+payload['reset_id'];rows=[];counts={}
    for raw in payload['rows']:
        row=dict(raw,method=mapping[raw['label']],seed=group,canonical_instance_id=payload['canonical_instance_id'],reset_id=payload['reset_id'])
        result=src.read(Path(row['directory'])/'result.json',row['result_sha256'])
        if result['canonical_instance_id']!=payload['canonical_instance_id'] or result['reset_id']!=payload['reset_id'] or result['execution_protocol']!='full_horizon_feedback_diagnostic':
            raise ValueError('schema2 canonical diagnostic identity differs')
        if result['comparison_contract_sha256']!=payload['config_comparison_sha256'] or (row['execution_kind']=='closed_loop_visual_policy' and result['policy_identity']['checkpoint_receipt_sha256']!=payload['checkpoint_receipt_sha256']):
            raise ValueError('schema2 frozen comparison differs')
        if raw.get('metric_path'):
            replay=src.read(raw['metric_path'],raw['metric_sha256'])
            row.update(replay_metrics_path=raw['metric_path'],replay_metrics_sha256=raw['metric_sha256'],matched_steps=replay['matched_steps'])
        rows.append(row)
        counts[row['method']]=dict(planned=1,executed=1,completed_full_horizon=int(row['full_horizon_completed']),success_ever=int(row['success_ever']),success_at_horizon=int(row['success_at_horizon']))
    if payload['executed_episodes']!=len(rows) or payload['completed_full_horizon']!=sum(r['full_horizon_completed'] for r in rows):
        raise ValueError('schema2 diagnostic total differs')
    return dict(payload,rows=rows,counts=counts,protocol='full_horizon_feedback_diagnostic',source_receipts=[])


def full_horizon_tables(paths,src):
    rows=[];counts=[];seen=set();directories=set()
    for path in paths:
        payload=src.read(path)
        if payload.get('schema_version')==2 and 'protocol' not in payload:payload=_full_horizon_v2(payload,src)
        if payload['protocol']!='full_horizon_feedback_diagnostic':raise ValueError('mixed replay protocol')
        for receipt in payload['source_receipts']:src.read(receipt['path'],receipt['sha256'])
        methods=defaultdict(list);references={};native_results={}
        for row in payload['rows']:
            key=(str(Path(path).resolve()),row['seed'],row['method'])
            if key in seen:raise ValueError('duplicate full-horizon row')
            seen.add(key);root=Path(row['directory'])
            if str(root.resolve()) in directories:raise ValueError('duplicate diagnostic episode across summaries')
            directories.add(str(root.resolve()));result=src.read(root/'result.json',row['result_sha256'])
            src.read(root/'actions.json',row['actions_sha256'])
            if result['horizon']!=row['planned_steps'] or row['recorded_steps']>row['planned_steps'] or result['ticks']!=row['recorded_steps']:
                raise ValueError('full horizon budget differs')
            if row['success_at_horizon']!=result['native_predicates']['native_task_success']:
                raise ValueError('native full-horizon outcome differs')
            if row.get('position_rmse_cm') is not None or row.get('rotation_error_deg') is not None:
                raise ValueError('absolute correspondence unestablished')
            if row.get('replay_metrics_path'):
                replay=src.read(row['replay_metrics_path'],row['replay_metrics_sha256'])
                for field in ('relative_displacement_rmse_cm','relative_rotation_increment_mean_deg','matched_steps'):
                    if row[field]!=replay[field]:raise ValueError('relative replay metric differs')
            if row['method']=='REF_NATIVE':references[row['seed']]=row
            methods[row['method']].append(row);native_results[(row['seed'],row['method'])]=result
            rows.append(dict(row,protocol=payload['protocol'],source_summary=str(path),position_rmse_cm=None,rotation_error_deg=None))
        for row in payload['rows']:
            ref=native_results[(row['seed'],'REF_NATIVE')];actual=native_results[(row['seed'],row['method'])]
            frozen=('horizon','reset_state_id','reset_seed','scene_id','task_id','sensor_regime')
            if actual.get('native_schema_version')==2:
                frozen+=('canonical_instance_id','canonical_manifest_sha256','reset_contract_sha256','comparison_contract_sha256','policy_rng_seed')
                if row['execution_kind']=='closed_loop_visual_policy':frozen+=('policy_identity',)
            elif row['execution_kind']=='closed_loop_visual_policy':frozen+=('policy_identity','config_sha256')
            if any(actual[k]!=ref[k] for k in frozen):raise ValueError('full-horizon native frozen fields differ')
            if row['execution_kind']=='fixed_action_replay' and row['actions_sha256']!=references[row['seed']]['actions_sha256']:
                raise ValueError('replay does not reuse exact reference actions')
        for method,group in methods.items():
            declared=payload['counts'][method]
            observed=dict(executed=len(group),completed_full_horizon=sum(r['full_horizon_completed'] for r in group),success_ever=sum(r['success_ever'] for r in group),success_at_horizon=sum(r['success_at_horizon'] for r in group))
            if any(declared[k]!=v for k,v in observed.items()):raise ValueError('full horizon count mismatch')
            counts.append(dict(protocol=payload['protocol'],method=method,source_summary=str(path),**declared,
                coverage=declared['executed']/declared['planned'],absolute_position_error_cm=None,
                scope=payload['claim_scope'],population_feedback_benefit='NOT_RUN'))
        if len(payload['rows'])>payload['planned_episodes']:raise ValueError('unplanned diagnostic episode')
    fields=sorted({key for row in rows for key in row})
    rows=[{key:row.get(key) for key in fields} for row in rows]
    return counts,rows


def conclusions(main,comparisons):
    rows=[]
    for arm in main:
        tier={'DEV':'DEV','development':'DEV','TEST':'TEST','test':'TEST'}.get(arm['split'])
        if tier is None:raise ValueError('unrecognized conclusion split')
        text=(f"In the declared {tier} {arm['scope']} block, {arm['method']} executed {arm['executed']}/{arm['planned']} planned episodes; "
              f"{arm['successes_observed']} observed successes; {arm['unmeasured']} outcomes remain unmeasured.")
        rows.append(dict(claim_id=tier.lower()+'_counts_'+arm['method'],gate='PASS',scope=f'descriptive {tier} accounting only',enabled_sentence=text,
                         cohort_complete=arm.get('complete',False),required_gate='validated immutable planned and terminal ledger',
                         outcome_benefit='NOT_RUN',test_generalization='NOT_RUN',preservation_claim='NOT_RUN',independent_physics_validation='NOT_RUN'))
    for pair in comparisons:
        tier={'DEV':'DEV','development':'DEV','TEST':'TEST','test':'TEST'}.get(pair['split'])
        if tier is None:raise ValueError('unrecognized comparison split')
        complete=pair['paired_population_complete'];competent=pair.get('reference_has_success',False)
        text=(f"The complete paired {tier} {pair['method']} minus {pair['reference']} difference is {pair['delta_pp']:.2f} percentage points "
              f"(descriptive hierarchical 95% interval {pair['ci95_pp']}); this does not establish noninferiority or preservation.") if complete else ''
        direction='UNMEASURED'
        if complete:
            low,high=pair['ci95_pp'];direction='POSITIVE' if low>0 else ('NEGATIVE' if high<0 else 'INCONCLUSIVE')
        rows.append(dict(claim_id=tier.lower()+'_pair_'+pair['method']+'_'+pair['reference'],gate='PASS' if text else 'NOT_RUN',
                         scope=f'descriptive {tier} paired comparison',enabled_sentence=text,cohort_complete=complete,
                         required_gate='complete paired canonical/reset population; no partial-pair population inference',
                         reference_has_observed_success=competent,paired_direction=direction,
                         outcome_benefit='NOT_RUN',test_generalization='NOT_RUN',preservation_claim='NOT_RUN',
                         preservation_reason='no predeclared noninferiority margin; nonsignificance and all-failure pairs are not preservation',
                         independent_physics_validation='NOT_RUN'))
    fields=sorted({k for row in rows for k in row})
    return [{k:row.get(k) for k in fields} for row in rows]


def evidence_tables(units,main,comparisons,config):
    src=Sources();tables={}
    if config.get('appearance_records'):
        from robo.eval.native_scale_appearance import appearance_table
        tables['T1a_appearance']=appearance_table(units,config['appearance_records'],config['appearance_methods'],src)
        module=Path(__file__).with_name('native_scale_appearance.py');src.bind(module)
    if config.get('context_records'):
        tables['T3_verification'],tables['context_diagnostics']=context_tables(units,main,config['context_records'],src)
    if config.get('context_construction_bindings'):
        if config.get('context_records'):raise ValueError('choose one context source representation')
        tables['T3_verification'],tables['context_diagnostics']=context_binding_tables(units,main,config['context_construction_bindings'],src)
    if config.get('runtime_records'):
        tables['construction_stage_costs'],tables['construction_stage_cost_details']=runtime_tables(units,config['runtime_records'],src,config.get('construction_bindings',()))
    if config.get('full_horizon_summaries'):
        tables['full_horizon_diagnostic'],tables['full_horizon_episodes']=full_horizon_tables(config['full_horizon_summaries'],src)
    all_conclusions=conclusions(main,comparisons)
    tables['dev_conclusion_ledger']=[r for r in all_conclusions if r['claim_id'].startswith('dev_')]
    tables['test_conclusion_ledger']=[r for r in all_conclusions if r['claim_id'].startswith('test_')]
    from robo.eval.native_scale_tables import _sha
    p=Path(__file__).resolve();src.lineage[str(p)]={'path':str(p),'sha256':_sha(p)}
    return tables,src.lineage
