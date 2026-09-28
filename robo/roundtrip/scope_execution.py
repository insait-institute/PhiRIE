"""Compose scope assets into canonical matrix workers and one admitted engine.

Each scope method gets its own same-engine L0 control using exactly that target.
The source primary roster is read only; its REF results are never reused.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from robo.roundtrip.matrix import NATIVE_PYTHON,IDENTITY_FIELDS,read_rows,save_new,sha,canonical_hash,validate_terminal
from robo.roundtrip.local_policy_instance import SCOPE_BLOCKS


def emit(primary,targets,destinations,out,*,worker_root,protocol,run_id,plan_only=False,frozen_plan=None):
    from robo.roundtrip.spec import validate_spec
    if plan_only and frozen_plan is not None:raise ValueError('planning and binding are separate transactions')
    if protocol not in SCOPE_BLOCKS or not run_id:raise ValueError('explicit admitted scope protocol and unique run ID required')
    out=Path(out);out.mkdir(parents=True,exist_ok=False);worker_root=Path(worker_root).resolve()
    dst={r['canonical_instance_id']:r for r in destinations}
    if len(dst)!=len(destinations):raise ValueError('duplicate destination canonical')
    target={(r['canonical_instance_id'],r['controller_method']):r for r in targets}
    if len(target)!=len(targets):raise ValueError('duplicate target method binding')
    refs=[r for r in primary if r['canonical_instance_id'] in dst and r['controller_method']=='REF_NATIVE' and r['scope']=='L0_target_only']
    if {r['canonical_instance_id'] for r in refs}!=set(dst):raise ValueError('destination roster has unbound canonical')
    if len({(r['canonical_instance_id'],r['reset_id']) for r in refs})!=len(refs):raise ValueError('duplicate reference reset')
    rows=[];configs={}
    if frozen_plan is not None:
        rows=[copy.deepcopy(u) for u in frozen_plan if u['canonical_instance_id'] in dst]
        expected={(r['canonical_instance_id'],r['reset_id'],method,scope) for r in refs for method,scope in SCOPE_BLOCKS[protocol]}
        if len(rows)!=len(expected) or {(u['canonical_instance_id'],u['reset_id'],u['controller_method'],u['scope']) for u in rows}!=expected:raise ValueError('frozen scope denominator differs')
        for u in rows:
            c=json.loads(Path(u['config_path']).read_text())
            if canonical_hash(c)!=u['config_sha256'] or c.get('scope_run_id')!=run_id or c.get('scope_engine_protocol')!=protocol or u.get('scope_run_id')!=run_id:raise ValueError('frozen scope plan/config changed')
            if Path(u['result_path_planned']).resolve()!=worker_root/u['unit_id']/'runner/episode/result.json':raise ValueError('frozen worker path differs')
    for ref in ([] if frozen_plan is not None else refs):
        for method,scope in SCOPE_BLOCKS[protocol]:
            u=copy.deepcopy(ref);u.update(controller_method=method,scope=scope,terminal_status='NOT_SCHEDULED',executed=None,success=None)
            key={k:u[k] for k in IDENTITY_FIELDS};key['scope_run_id']=run_id
            u['unit_id']='unit-'+canonical_hash(key)[:32];u['plan_unit_id']='plan-'+canonical_hash(key)[:24]
            c=json.loads(Path(ref['config_path']).read_text());c.update(controller_method=method,scope=scope,
                replacement_scope='target_only' if scope=='L0_target_only' else 'target_destination',
                scope_engine_protocol=protocol,scope_run_id=run_id,policy_engine_protocol='per_canonical_engine_v1')
            c.pop('policy_engine_id',None);validate_spec(c)
            path=out/'configs'/(u['unit_id']+'.json');save_new(path,c)
            u.update(config_path=str(path.resolve()),config_sha256=canonical_hash(c),result_path_planned=str(worker_root/u['unit_id']/'runner/episode/result.json'),
                scope_run_id=run_id,source_primary_unit_id=ref['unit_id'])
            rows.append(u);configs[u['unit_id']]=c
    if plan_only:
        save_new(out/'planned_units.jsonl',rows,jsonl=True)
        save_new(out/'scope_execution_manifest.json',{'schema_version':2,'kind':'canonical_scope_matrix_binding','complete_phase_plan':True,'canonical_instance_ids':sorted(dst),'resets_per_instance':len({r['reset_id'] for r in refs}),'scope_run_id':run_id,'protocol':protocol,'planned_units':len(rows),'controls':sum(u['scope']=='L0_target_only' for u in rows),'scope_measurements':sum(u['scope']=='L1_target_destination' for u in rows),'planned_units_sha256':sha(out/'planned_units.jsonl'),'unresolved_construction_is_failure':False,'primary_results_reused':False,'commands':'bound separately after typed construction outcomes; frozen plan rows never change'})
        return rows,{}
    units={(u['canonical_instance_id'],u['reset_id'],u['controller_method'],u['scope']):u for u in rows}
    commands={};failures=[]
    for u in rows:
        cid,method,scope=u['canonical_instance_id'],u['controller_method'],u['scope']
        target_binding=target.get((cid,method));dest=dst[cid];failure=None
        if method!='REF_NATIVE':
            if target_binding is None:raise ValueError('target build unresolved; cannot start a partial process')
            if target_binding.get('terminal_status') in ('BUILD_FAILED','ABSTAINED'):failure=target_binding
            elif not target_binding.get('object_dir'):raise ValueError('executable target artifact missing')
        if scope=='L1_target_destination' and failure is None:
            role=dest.get('methods',{}).get(method)
            if role is None:raise ValueError('destination method unresolved; no hybrid fallback')
            if role.get('terminal_status') in ('BUILD_FAILED','ABSTAINED'):failure=role
            elif not role.get('candidate_pool') or (method=='B4_ROOM_REPAIR_NATIVE' and not role.get('repair_receipt')):
                raise ValueError('scope destination requires matching B3 pool and explicit B4 repair receipt')
        if failure is not None:
            evidence=failure.get('evidence_path') or failure.get('binding_source') or failure.get('build_manifest')
            if not evidence or not Path(evidence).is_file():raise ValueError('failed build requires immutable evidence')
            terminal={**{k:u[k] for k in IDENTITY_FIELDS},'unit_id':u['unit_id'],'planned_unit_sha256':canonical_hash(u),
                'terminal_status':failure['terminal_status'],'executed':False,'success':None,
                'failure':'scope dependency unavailable; no policy outcome inferred','evidence_path':str(evidence),'evidence_sha256':sha(evidence)}
            validate_terminal(u,terminal);save_new(worker_root/u['unit_id']/'terminal.json',terminal);failures.append(u['unit_id']);continue
        input_hashes={}
        if method!='REF_NATIVE':
            from robo.roundtrip.scope_bundle import artifact_hashes
            obj=Path(target_binding['object_dir']).resolve()
            input_hashes.update({str(obj/k):v for k,v in artifact_hashes(obj).items()})
        if scope=='L1_target_destination':
            for path in [dest['build_manifest'],*[role[k] for k in ('candidate_pool','rvg_receipt','repair_receipt') if role.get(k)]]:
                input_hashes[str(Path(path).resolve())]=sha(path)
        if frozen_plan is None:u['scope_input_hashes']=input_hashes
        base=[NATIVE_PYTHON,'-m',
              'robo.roundtrip.scope_worker' if scope=='L1_target_destination' else 'robo.roundtrip.paired',
              '--config',u['config_path'],'--canonical-reference',u['bundle_dir'],'--canonical-manifest',u['canonical_manifest'],
              '--reset-bank',u['reset_bank'],'--out',str(Path(u['result_path_planned']).parent.parent),'--host','localhost','--port','8017']
        requires=[]
        if method!='REF_NATIVE':
            baseline=units[(cid,u['reset_id'],method if scope=='L1_target_destination' else 'REF_NATIVE','L0_target_only')]
            requires.append(baseline['result_path_planned'])
            if scope=='L0_target_only':base+=['--object-dir',target_binding['object_dir'],'--reference-episode',str(Path(baseline['result_path_planned']).parent)]
            else:
                role=dest['methods'][method]
                base+=['--baseline-config',baseline['config_path'],'--baseline-episode',str(Path(baseline['result_path_planned']).parent),
                    '--target-dir',target_binding['object_dir'],'--destination-build-manifest',dest['build_manifest'],'--destination-candidate-pool',role['candidate_pool']]
                requires.extend([dest['build_manifest'],role['candidate_pool']])
                for field,flag in [('rvg_receipt','--destination-rvg-receipt'),('repair_receipt','--destination-repair-receipt')]:
                    if role.get(field):base+=[flag,role[field]];requires.append(role[field])
        commands[u['unit_id']]={'argv':base,'planned_unit_sha256':canonical_hash(u),'requires_files':requires,'result_relative_path':'runner/episode/result.json','scope_input_hashes':input_hashes}
    save_new(out/'planned_units.jsonl',rows,jsonl=True);save_new(out/'commands.json',commands)
    save_new(out/'scope_execution_manifest.json',{'kind':'canonical_scope_matrix_binding','scope_run_id':run_id,'protocol':protocol,
        'planned_units':len(rows),'controls':sum(u['scope']=='L0_target_only' for u in rows),'scope_measurements':sum(u['scope']!='L0_target_only' for u in rows),
        'nonexecuted_build_failures':failures,'primary_results_reused':False,'source_frozen_plan_sha256':None if frozen_plan is None else canonical_hash(frozen_plan),'target_bindings_sha256':canonical_hash(targets),'destination_bindings_sha256':canonical_hash(destinations),
        'runner':'robo.roundtrip.local_policy_instance --scope-protocol '+protocol,'ledger':'canonical matrix.worker and robo.eval.harness_runner',
        'admission_rule':'same-process gates and actual sealed DEV engine admission before any episode'})
    return rows,commands


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['primary','targets','destinations','out','worker-root','protocol','run-id']:p.add_argument('--'+name,required=True)
    a=p.parse_args();emit(read_rows(a.primary),read_rows(a.targets),read_rows(a.destinations),a.out,worker_root=a.worker_root,protocol=a.protocol,run_id=a.run_id)
if __name__=='__main__':main()
