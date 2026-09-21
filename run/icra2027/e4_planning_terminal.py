"""Authenticate planning applicability failures without inventing reset/rollout data."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess

from robo.eval import e4_candidate_screen as screen
from run.icra2027 import e4_compact_materialization as api

SCOPE='automatic_compact_planning_applicability_audit'
REASON='no_prepared_size_admissible_target'


def _exact_code(root, expected):
    root=Path(root)
    if root.resolve(strict=True)!=root:
        raise ValueError('terminal code root is not canonical')
    top=subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=root,text=True).strip()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'],cwd=root,text=True)
    if Path(top)!=root or head!=expected or dirty.strip():
        raise ValueError('terminal audit code is not exact and clean')
    return {'code_root':str(root),'commit':head,'dirty':False}


def source_contract(config_path, stage_root, expected_source_commit):
    stage=Path(stage_root).resolve(strict=True)
    identity=api.identity(config_path)
    e0_path=stage/'contract/freeze_manifest.json';contract=api.read(e0_path)
    api.contract_digest(contract)
    if (contract['freeze_id']!=stage.name or contract['code']['commit']!=expected_source_commit
            or contract['code']['dirty'] is not False
            or api.resource(contract,'e4_compact_qualification_config')!=identity):
        raise ValueError('terminal source E0/config binding differs')
    code=_exact_code(contract['code']['repository'],expected_source_commit)
    config=api.read(config_path)
    if config['freeze_id']!=stage.name:
        raise ValueError('terminal source freeze differs')
    runtime=api.resource(contract,'qualification_python')
    if api.identity(runtime['path'])!=runtime:
        raise ValueError('terminal source Python changed')
    return dict(config=identity,e0=api.identity(e0_path),code_root=code['code_root'],
        code_commit=expected_source_commit,freeze_id=stage.name,runtime=runtime)


# Run the original canonical APIs at their actual source. Validation compiles
# existing exports; pure planning reads discovery AABBs, never reset telemetry.
SCRIPT='''import json,sys
from pathlib import Path
from robo.eval import e4_candidate_screen as s, e4_task_freeze as t
from run.icra2027.e4_compact_qualification import validate_stage
from run.icra2027 import e4_compact_materialization as api
r=json.load(sys.stdin); root=s.evidence_root(); stage=Path(r['e0']['path']).parent.parent
config,review=validate_stage(r['config']['path'],stage,r['code_commit'])
if review['status']!='READY':raise ValueError('terminal source not ready')
scenes={}
for scene in sorted(config['export_reuse']):
    d=stage/'automatic_candidates'/scene
    b=s._validate_bundle(d/'population',root=root,expected_kind='e4_automatic_candidate_population')
    g=json.loads((d/'population/gate.json').read_text())
    if g['code']!={'code_root':r['code_root'],'commit':r['code_commit'],'dirty':False}:raise ValueError('population source differs')
    fs={p:d/'materialized'/p for p in s.POLICIES}
    reports={p:t._factory_report(fs[p],scene_id=scene,policy=p,root=root) for p in s.POLICIES}
    t._automatic_population_tasks(d/'population/gate.json',root=root,scene_id=scene,factories=fs,reports=reports)
    dimensions={slot:[float(hi-lo) for lo,hi in zip(o['aabb'][0],o['aabb'][1],strict=True)] for slot,o in g['objects'].items()}
    try:
        plan=s.plan_automatic_population_tasks(g)
    except s.CandidateScreenError as exc:
        if str(exc)!='automatic task planning has no prepared size-admissible target; all declared pairs remain unqualified':raise
        status='FAIL';reason='no_prepared_size_admissible_target'
    else:
        if plan['candidate_count']!=g['counts']['semantic_pairs']:raise ValueError('planner dropped population')
        status='PASS';reason=None
    qualifier=None
    q=stage/'scene_qualifiers'/scene
    if q.exists():
        checked=s._validate_qualifier_output(root=root,screen_id=r['freeze_id'],scene_id=scene,expected_commit=r['code_commit'])
        rows=checked['metric_rows']; selected=[x for x in rows if x['task_id'] in api.FIXED_TASKS]
        qualifier={'gate':s._identity(q/'gate.json',root=root),'manifest':s._identity(q/'manifest.json',root=root),
          'metrics':s._identity(q/'metrics.jsonl',root=root),'seal':s._identity(q/'seal.json',root=root),
          'planned_cells':len(rows),'passed_cells':sum(x['passed'] for x in rows),
          'fixed_selected_cells':len(selected),'fixed_passed_cells':sum(x['passed'] for x in selected)}
    scenes[scene]={'semantic_pairs':g['counts']['semantic_pairs'],'planning_status':status,'reason':reason,
      'observed_dimensions':dimensions,'population':s._identity(d/'population/gate.json',root=root),
      'population_manifest_sha256':b['manifest_sha256'],'population_seal_sha256':b['seal_sha256'],
      'exports':g['exports'],'qualifier':qualifier,'task_definition':None,'reset_bank':None,'camera':None,'rollout':None}
protocol=config['source']['canonical']['protocol']
print(json.dumps({'protocol':{k:protocol[k] for k in ('path','sha256')},'scenes':scenes}))
'''


def _replay(source):
    current=source_contract(source['config']['path'],Path(source['e0']['path']).parent.parent,source['code_commit'])
    if current!=source:
        raise ValueError('terminal original source closure changed')
    environment=dict(os.environ,SIMANY_EVIDENCE_ROOT=str(screen.evidence_root()))
    environment.pop('PYTHONPATH',None)
    completed=subprocess.run([source['runtime']['path'],'-c',SCRIPT],cwd=source['code_root'],
        input=json.dumps(source),text=True,capture_output=True,check=True,env=environment)
    return json.loads(completed.stdout)


def _payload(source, producer_code):
    observed=_replay(source)
    from run.icra2027 import e4_compact_harness as compact
    compact.checked_protocol(observed['protocol']['path'],observed['protocol']['sha256'])
    scenes=observed['scenes']
    if set(scenes)!=set(compact.SCENES) or sum(r['semantic_pairs'] for r in scenes.values())!=28:
        raise ValueError('terminal audit lost the fixed semantic population')
    if not any(r['planning_status']=='FAIL' for r in scenes.values()):
        raise ValueError('terminal audit requires an observed planning failure')
    return dict(schema_version=1,scope=SCOPE,status='PASS',applicability_status='FAIL',
        execution_status='NOT_RUN',paper_ready=False,headline_eligible=False,claim_gate='FAIL',
        producer_code=producer_code,source=source,protocol=observed['protocol'],
        planned_semantic_pairs=28,planned_episodes=40,scenes=scenes,
        limitation='Applicability audit only; no task/reset/camera/policy execution is asserted.')


def produce(*, config_path, stage_root, expected_source_commit, out):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=screen.CODE_ROOT,text=True).strip()
    code=screen._code_snapshot(commit)
    source=source_contract(config_path,stage_root,expected_source_commit)
    payload=_payload(source,code)
    if screen._code_snapshot(code['commit'])!=code:
        raise ValueError('terminal producer changed during audit')
    with Path(out).open('x') as stream:
        json.dump(payload,stream,indent=2,sort_keys=True);stream.write('\n')
    return payload


def validate_terminal_receipt(path, *, expected_producer_commit):
    payload=api.read(path)
    code=payload['producer_code']
    if code!=_exact_code(code['code_root'],expected_producer_commit):
        raise ValueError('terminal receipt producer differs')
    if payload!=_payload(payload['source'],code):
        raise ValueError('terminal receipt failed exact source replay')
    return payload



FULL_SCOPE = 'full_discovery_qualification_terminal_coverage'
FULL_CONFIG_SCOPE = 'full_discovery_terminal_audit_config'
FULL_STATES = {'SOURCE_NO_QUERIES','PLANNING_UNAVAILABLE','EXPORT_REJECTED','QUALIFICATION_COMPLETE'}
ISA_MODE = 'AVX512F,AVX512CD,AVX512_KNL,AVX512_KNM,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL'


def full_source_contract(config_path, stage_root, expected_source_commit):
    stage=Path(stage_root).resolve(strict=True);config_id=api.identity(config_path)
    e0=stage/'contract/freeze_manifest.json';contract=api.read(e0);api.contract_digest(contract)
    if (contract['freeze_id']!=stage.name or contract['code']['commit']!=expected_source_commit
            or contract['code']['dirty'] is not False
            or api.resource(contract,'e4_full_qualification_config')!=config_id):
        raise ValueError('full terminal original E0/config/source differs')
    code=_exact_code(contract['code']['repository'],expected_source_commit)
    config=api.read(config_path)
    if config['freeze_id']!=stage.name or config['scope']!='e4_full_discovery_budget_qualification':
        raise ValueError('full terminal original scope/freeze differs')
    return dict(config=config_id,e0=api.identity(e0),code_root=code['code_root'],
        code_commit=expected_source_commit,freeze_id=stage.name,
        runtime=api.resource(contract,'qualification_python'),
        environment=api.resource(contract,'qualification_environment'),
        osmesa_environment=api.resource(contract,'osmesa_environment'))


def prepare_full_config(*, source_config, source_root, expected_source_commit,
                        job_ids, compatibility_path, freeze_id, out, prior_attempts=None):
    source=full_source_contract(source_config,source_root,expected_source_commit)
    config=api.read(source_config)
    if set(job_ids)!=set(config['source']['scenes']) or len(set(job_ids.values()))!=50:
        raise ValueError('full terminal job IDs must retain all50 original scenes')
    if any(not isinstance(x,str) or not x.isdigit() for x in job_ids.values()):
        raise ValueError('terminal job IDs must be ordinary numeric IDs')
    prior_attempts=prior_attempts or {}
    if not set(prior_attempts)<=set(job_ids):raise ValueError('prior attempt outside frozen scene roster')
    all_jobs=list(job_ids.values())+[j for values in prior_attempts.values() for j in values]
    if len(set(all_jobs))!=len(all_jobs) or any(not isinstance(j,str) or not j.isdigit() for j in all_jobs):
        raise ValueError('prior attempts duplicate a job or are not ordinary IDs')
    raw=subprocess.check_output(['sacct','-X','-j',','.join(all_jobs),'-nP',
        '--format=JobIDRaw,State,ExitCode,ElapsedRaw,NodeList,JobName%128,WorkDir%512'],text=True)
    records={}
    for line in raw.splitlines():
        fields=line.split('|')
        if fields[0] in all_jobs:
            if fields[0] in records or len(fields) not in (7,8):raise ValueError('invalid scheduler record')
            records[fields[0]]=dict(job_id=fields[0],state=fields[1],exit_code=fields[2],
                elapsed_seconds=int(fields[3]),hostname=fields[4],job_name=fields[5],work_directory=fields[6])
    if set(records)!=set(all_jobs) or any(records[j]['state'] not in {'COMPLETED','FAILED'} for j in job_ids.values()):
        raise ValueError('full terminal audit waits for every original scene; resource failures need missing-unit recovery')
    stage=Path(source_root)
    history={}
    for sid,jid in job_ids.items():
        for attempt in [jid,*prior_attempts.get(sid,[])]:
            record=records[attempt]
            expected_name='e4-full-pilot' if sid==config['pilot_scene'] else 'e4-full-'+sid
            if (record['work_directory']!=source['code_root'] or record['job_name']!=expected_name
                    or record['elapsed_seconds']<0):
                raise ValueError('scheduler attempt does not belong to original scene/source')
            if attempt!=jid and record['state'] not in {'OUT_OF_MEMORY','TIMEOUT','FAILED'}:
                raise ValueError('prior attempt is not a preserved terminal failure')
        history[sid]=[{**records[j],'logs':{ext:api.identity(stage/'logs'/f'{sid}-{j}.{ext}')
            for ext in ('err','out')}} for j in prior_attempts.get(sid,[])]
    compatibility=api.read(compatibility_path)
    if (compatibility.get('scope')!='fixed_resource_CPU_ISA_compatibility_envelope'
            or compatibility.get('source_commit')!=expected_source_commit
            or compatibility.get('freeze_id')!=source['freeze_id']
            or compatibility.get('E0')!=source['e0']
            or compatibility.get('execution_config')!=source['config']
            or compatibility.get('disabled_features')!=ISA_MODE
            or compatibility.get('acceptance_tolerances_changed') is not False):
        raise ValueError('terminal hardware compatibility envelope differs')
    payload=dict(schema_version=1,scope=FULL_CONFIG_SCOPE,freeze_id=freeze_id,source=source,
        compatibility=api.identity(compatibility_path),cpu_feature_mode=ISA_MODE,
        jobs={sid:records[jid] for sid,jid in sorted(job_ids.items())},prior_attempts=history,paper_ready=False)
    with Path(out).open('x') as f:json.dump(payload,f,indent=2,sort_keys=True);f.write('\n')
    return payload


def _full_worker_command(config, worker):
    import shlex
    source=config['source']
    command='source '+shlex.quote(source['environment']['path'])+'\nexport NPY_DISABLE_CPU_FEATURES='+shlex.quote(ISA_MODE)+'\nexec '+shlex.quote(source['runtime']['path'])+' -c '+shlex.quote(worker.read_text())
    return ['bash','-c',command]


def _full_replay(config, *, scene_ids=None, process_receipt=None):
    source=config['source'];stage=Path(source['e0']['path']).parent.parent
    if full_source_contract(source['config']['path'],stage,source['code_commit'])!=source:
        raise ValueError('original full source changed')
    if api.identity(config['compatibility']['path'])!=config['compatibility'] or config['cpu_feature_mode']!=ISA_MODE:
        raise ValueError('full terminal compatibility identity changed')
    for sid,attempts in config.get('prior_attempts',{}).items():
        if sid not in config['jobs']:raise ValueError('prior attempt scene differs')
        for attempt in attempts:
            for identity in attempt['logs'].values():
                if api.identity(identity['path'])!=identity:raise ValueError('prior attempt log changed')
    worker=Path(__file__).with_name('e4_full_terminal_worker.py')
    env=dict(os.environ,SIMANY_EVIDENCE_ROOT=str(screen.evidence_root()),PYTHONNOUSERSITE='1')
    for key in ('PYTHONHOME','PYTHONPATH','SIMANY_SCENE','SIMANY_OUT','SIMANY_SCENE_DIR','NPY_DISABLE_CPU_FEATURES'):
        env.pop(key,None)
    command=_full_worker_command(config,worker)
    request=dict(config)
    if scene_ids is not None:request['_scene_ids']=list(scene_ids)
    import hashlib,time
    request_text=json.dumps(request);started=time.monotonic()
    completed=subprocess.run(command,cwd=source['code_root'],env=env,
        input=request_text,capture_output=True,text=True,check=process_receipt is None)
    if process_receipt is not None:
        process_receipt.update(command=command,cwd=source['code_root'],
            request_sha256=hashlib.sha256(request_text.encode()).hexdigest(),
            worker=api.identity(worker),returncode=completed.returncode,
            stdout=completed.stdout,stderr=completed.stderr,runtime_seconds=time.monotonic()-started)
        completed.check_returncode()
    if full_source_contract(source['config']['path'],stage,source['code_commit'])!=source:
        raise ValueError('original source changed during terminal replay')
    return json.loads(completed.stdout)


def _full_payload(config, code):
    from collections import Counter
    if config.get('replay_mode')=='sealed_scene_cache_v1':
        from run.icra2027.e4_terminal_cache import compose
        observed=compose(config,code)
    elif 'replay_mode' in config:raise ValueError('unsupported terminal replay mode')
    else:observed=_full_replay(config)
    source=observed['source_config'];protocol=observed['protocol']
    scenes=observed['scenes'];cells=observed['cells']
    if (len(scenes)!=50 or set(scenes)!=set(config['jobs']) or set(scenes)!=set(source['source']['scenes'])
            or any(r['status'] not in FULL_STATES for r in scenes.values())):
        raise ValueError('full terminal scene population/status differs')
    tasks={q['task_id']:q['scene_id'] for q in protocol['qualification_tasks']}
    if len(tasks)!=269 or len(protocol['qualification_tasks'])!=269:
        raise ValueError('full terminal task population differs')
    expected={(sid,tid,p,e) for tid,sid in tasks.items() for p in ('A0','A4') for e in range(5)}
    actual=[(r['scene_id'],r['task_id'],r['policy_id'],r['episode']) for r in cells]
    if len(cells)!=2690 or len(set(actual))!=2690 or set(actual)!=expected:
        raise ValueError('full terminal dropped or replaced a planned qualification cell')
    from robo.eval.e4_candidate_screen import BASE_SEED
    for row in cells:
        if row['cell_id']!=f"{row['policy_id'].lower()}__{row['task_id']}__seed{BASE_SEED}__ep{row['episode']}":
            raise ValueError('terminal cell identity differs')
        if row['policy_executed'] is not False or row['policy_success'] is not None:
            raise ValueError('terminal audit cannot invent policy telemetry')
        if row['source_kind']!='canonical_qualifier':
            if row['qualification_state']!='NOT_RUN' or any(row.get(k) is not None for k in ('qualification_evidence','reset_definition','camera','physics')):
                raise ValueError('unavailable source contains fabricated measurements')
        elif row['qualification_evidence']['cell_id']!=row['cell_id'] or row['qualification_state']!=('PASS' if row['qualification_evidence']['passed'] else 'FAIL'):
            raise ValueError('canonical prerequisite evidence differs')
    for sid,scene in scenes.items():
        scene_cells=[r for r in cells if r['scene_id']==sid]
        canonical=[r for r in scene_cells if r['source_kind']=='canonical_qualifier']
        if (scene['qualified_cells']!=sum(r['qualification_state']=='PASS' for r in canonical)
                or type(scene['actual_900_step_cells']) is not int
                or not 0<=scene['actual_900_step_cells']<=len(canonical)
                or scene.get('policy_executed')!=0 or scene.get('policy_success') is not None
                or (bool(canonical) != (scene['status']=='QUALIFICATION_COMPLETE'))
                or any(r['source_kind']!=scene['status'].lower() for r in scene_cells if r not in canonical)):
            raise ValueError('terminal scene summaries differ from canonical prerequisite cells')
        selected=sorted(t for t,s in tasks.items() if s==sid)
        if scene['selected_task_ids']!=selected or scene['planned_cells']!=10*len(selected):
            raise ValueError('terminal scene task roster differs')
        if (len(set(scene['strict_pass_task_ids']))!=len(scene['strict_pass_task_ids'])
                or not set(scene['strict_pass_task_ids'])<=set(selected)
                or any(sum(r['task_id']==task and r['qualification_state']=='PASS' for r in canonical)!=10
                       for task in scene['strict_pass_task_ids'])):
            raise ValueError('terminal strict task membership differs')
        if scene['status']=='SOURCE_NO_QUERIES' and selected:
            raise ValueError('nonempty selected population labeled source-only')
    fixed=dict(planned_scenes=50,planned_objects=1871,input_semantic_queries=6155,
        budget_exclusions=5886,planned_semantic_queries=269,planned_qualification_cells=2690)
    if any(source.get(k)!=v for k,v in fixed.items()):raise ValueError('full source denominator differs')
    gate=dict(schema_version=1,scope=FULL_SCOPE,status='PASS',producer_code=code,
        freeze_id=config['freeze_id'],source=config['source'],**fixed,
        scene_status_counts=dict(Counter(s['status'] for s in scenes.values())),
        cell_state_counts=dict(Counter(r['qualification_state'] for r in cells)),
        cell_source_counts=dict(Counter(r['source_kind'] for r in cells)),
        prerequisite_checked_cells=sum(r['source_kind']=='canonical_qualifier' for r in cells),
        actual_900_step_cells=sum(s['actual_900_step_cells'] for s in scenes.values()),
        qualified_cells=sum(s['qualified_cells'] for s in scenes.values()),
        strict_pass_task_ids=sorted(t for s in scenes.values() for t in s['strict_pass_task_ids']),
        prior_attempts=config.get('prior_attempts',{}),
        scheduler_elapsed_seconds=sum(r.get('elapsed_seconds',0) for r in config['jobs'].values())+sum(r['elapsed_seconds'] for rows in config.get('prior_attempts',{}).values() for r in rows),
        runtime_scope='Sum of per-job scheduler elapsed times, including declared failed attempts; hardware-specific, not fleet elapsed or capture-to-sim runtime.',
        policy_target_episodes=160,instantiated_policy_episodes=0,policy_executed=0,policy_success=None,
        camera_status='NOT_RUN',rollout_ledger=None,claim_gate='NOT_RUN',paper_ready=False,headline_eligible=False,
        scope_note='Full prerequisite coverage only. Source-unavailable logical cells are not simulator tests. Policy160 is the protocol target, not an instantiated reset bank or executed matrix.')
    if 'validation_cache' in observed:gate['validation_cache']=observed['validation_cache']
    return gate,scenes,cells


def _validate_full_stage(config_path,stage_root,expected_code_commit,out,code):
    config=api.read(config_path)
    stage=Path(stage_root).resolve(strict=True);e0=api.read(stage/'contract/freeze_manifest.json');api.contract_digest(e0)
    if (config.get('scope')!=FULL_CONFIG_SCOPE or config.get('schema_version')!=1
            or config.get('paper_ready') is not False or config['freeze_id']!=stage.name
            or e0['freeze_id']!=stage.name or e0['code']['commit']!=expected_code_commit
            or e0['code']['dirty'] is not False
            or _exact_code(e0['code']['repository'],expected_code_commit)!=code
            or api.resource(e0,'e4_full_terminal_config')!=api.identity(config_path)):
        raise ValueError('full terminal stage E0/config/source differs')
    if Path(out).resolve()!=stage/'full_qualification':raise ValueError('terminal output must belong to new stage')
    return config,stage


def produce_full(*,config_path,stage_root,expected_code_commit,out):
    code=screen._code_snapshot(expected_code_commit)
    config,stage=_validate_full_stage(config_path,stage_root,expected_code_commit,out,code)
    if Path(out).exists():raise FileExistsError('sealed full terminal coverage cannot be overwritten')
    gate,scenes,cells=_full_payload(config,code)
    if screen._code_snapshot(expected_code_commit)!=code:raise ValueError('terminal producer changed')
    gate['audit_config']=api.identity(config_path);gate['audit_E0']=api.identity(stage/'contract/freeze_manifest.json')
    screen._publish_bundle(Path(out),manifest_kind=FULL_SCOPE,payloads={
        'gate.json':screen._json_bytes(gate),'scene_receipts.json':screen._json_bytes(scenes),
        'planned_qualification_cells.jsonl':''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in cells).encode()},
        manifest_fields={'code':code,'freeze_id':stage.name})
    return gate


def validate_full_coverage(path,*,expected_producer_commit):
    out=Path(path);bundle=screen._validate_bundle(out,root=screen.evidence_root(),expected_kind=FULL_SCOPE)
    gate=api.read(out/'gate.json');code=gate['producer_code']
    if code!=_exact_code(code['code_root'],expected_producer_commit):raise ValueError('full terminal producer differs')
    if api.identity(gate['audit_config']['path'])!=gate['audit_config'] or api.identity(gate['audit_E0']['path'])!=gate['audit_E0']:
        raise ValueError('full terminal stage inputs changed')
    stage=Path(gate['audit_E0']['path']).parent.parent
    config,_=_validate_full_stage(gate['audit_config']['path'],stage,expected_producer_commit,out,code)
    if bundle['manifest'].get('code')!=code or bundle['manifest'].get('freeze_id')!=stage.name:
        raise ValueError('terminal bundle producer/freeze differs')
    expected,scenes,cells=_full_payload(config,code)
    expected.update(audit_config=gate['audit_config'],audit_E0=gate['audit_E0'])
    actual_cells=[json.loads(x) for x in (out/'planned_qualification_cells.jsonl').read_text().splitlines()]
    if gate!=expected or api.read(out/'scene_receipts.json')!=scenes or actual_cells!=cells:
        raise ValueError('full terminal exact original-source replay differs')
    return {**gate,'manifest_sha256':bundle['manifest_sha256'],'seal_sha256':bundle['seal_sha256']}

def main():
    import sys
    if len(sys.argv)>1 and sys.argv[1] in ('prepare-full','produce-full','validate-full'):
        command=sys.argv[1];p=argparse.ArgumentParser(description=__doc__)
        for name in ('config','source-root','expected-source-commit','freeze-root','expected-code-commit','out','job-ids','compatibility','freeze-id','prior-attempts'):
            p.add_argument('--'+name)
        a=p.parse_args(sys.argv[2:])
        if command=='prepare-full':
            r=prepare_full_config(source_config=a.config,source_root=a.source_root,expected_source_commit=a.expected_source_commit,job_ids=api.read(a.job_ids),compatibility_path=a.compatibility,freeze_id=a.freeze_id,out=a.out,prior_attempts=api.read(a.prior_attempts) if a.prior_attempts else None)
        elif command=='produce-full':r=produce_full(config_path=a.config,stage_root=a.freeze_root,expected_code_commit=a.expected_code_commit,out=a.out)
        else:r=validate_full_coverage(a.out,expected_producer_commit=a.expected_code_commit)
        print(json.dumps({'scope':r['scope'],'freeze_id':r['freeze_id'],'out':a.out}));return
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--source-root',required=True)
    p.add_argument('--expected-source-commit',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();r=produce(config_path=a.config,stage_root=a.source_root,
        expected_source_commit=a.expected_source_commit,out=a.out)
    print(json.dumps({'status':r['status'],'applicability_status':r['applicability_status'],
                      'execution_status':r['execution_status'],'out':a.out}))

if __name__=='__main__':main()
