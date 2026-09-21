"""Cohort admission/publication around immutable e7 public grounding producers.

This module never renders, fuses, segments, labels, or writes feature rows.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import socket
from run.icra2027 import e6_public_reconstruction as shared

CODE=Path(__file__).resolve().parents[2]
PRODUCER_SHA='e7c5265e8c481686c93429a3fe4f6be809464ea8'
KEYS={'schema_version','scope','freeze_id','source_commit','measurement_producer','fresh_config',
      'reference_configs','roster','recipe_sha256','reuse','native_replay_audits','validation_host'}
MEASUREMENT_KEYS={'schema_version','scope','freeze_id','source_commit','construction','discovery_runtime','rgb_runtime','protocol'}
NATIVE_AUDIT_SCRIPT_SHA='5f4a4614b4114f36ac3da38c23f34afb654b2e8891e8a779a0c2c773f949b296'
REUSED=[('behavior_task0023','clean'),*[('behavior_task0020',c) for c in shared.CONDITIONS]]


def source(ref):
    if set(ref)!={'path','commit'} or ref['commit']!=PRODUCER_SHA:
        raise ValueError('grounding producer must be exact e7')
    p=Path(ref['path']).resolve(strict=True)
    shared.sealed.sealed_cpu._inside(p,root=shared.ROOT.parent,label='grounding producer source')
    top=subprocess.check_output(['git','-C',str(p),'rev-parse','--show-toplevel'],text=True).strip()
    snap=shared.git_snapshot(p)
    if Path(top)!=p or snap['dirty'] or snap['commit']!=PRODUCER_SHA:
        raise ValueError('original grounding source changed')
    return p


def recipe(config):
    """Only publication freeze identity is excluded; every other field is retained."""
    if (set(config)!=MEASUREMENT_KEYS or config['schema_version']!=1
        or config['source_commit']!=PRODUCER_SHA or config['scope']!='e6_public_grounding_v1'):
        raise ValueError('unknown or changed original measurement configuration')
    return dict(schema_version=1,measurement_config={k:v for k,v in config.items() if k!='freeze_id'},
        excluded_fields={'freeze_id':'publication identity only; original configurations and manifests retained'},
        recipe_scope='exact e7 source, inputs, discovered object IDs, checkpoint, predicted cameras, roles, queries and association implementation; no controller or policy invoked')


def replay(cfg,ref,scene=None,condition=None):
    code=source(cfg['measurement_producer']);config=shared.read(ref['path'])
    if shared.identity(ref['path'])!=ref:raise ValueError('original config bytes changed')
    rgb=shared.read(config['rgb_runtime']['path']);stage=Path(ref['path']).parent
    if scene=='behavior_task0020' and ref==cfg['reference_configs'][1]:
        return native_replay(cfg,ref,scene,condition,code,rgb)
    command=[rgb['python'],'-m','run.icra2027.e6_public_grounding','--config',ref['path'],'--stage-root',str(stage)]
    command+=['--validate'] if scene is None else ['--scene',scene,'--condition',condition,'--validate-output']
    env=shared.environment(rgb,stage,'validation');env.update(PYTHONPATH=str(code),SIMANY_ROOT=str(code))
    result=subprocess.run(command,cwd=code,env=env,text=True,capture_output=True)
    allowed={0} if scene is None else {0,3}
    if result.returncode not in allowed:raise ValueError('original e7 validation failed: '+result.stderr[-2000:]+result.stdout[-2000:])
    if scene is None:
        if result.stdout.strip()!='E6_GROUNDING_PREFLIGHT=PASS':raise ValueError('original preflight marker missing')
        gate=None
    else:
        gate=json.loads(result.stdout)
        if (result.returncode==0)!=(gate['stage_status']=='PASS'):raise ValueError('original exit/status differ')
    return gate,dict(command=command,cwd=str(code),returncode=result.returncode,
                     stdout_sha256=shared.hashlib.sha256(result.stdout.encode()).hexdigest())


def native_replay(cfg,ref,scene,condition,code,rgb):
    """Authenticate exact native-host result without recomputing its floats here."""
    audit_ref=cfg['native_replay_audits'][condition]
    if shared.identity(audit_ref['path'])!=audit_ref:raise ValueError('native audit changed')
    audit=shared.read(audit_ref['path']);stage=Path(ref['path']).parent
    command=[rgb['python'],'-m','run.icra2027.e6_public_grounding','--config',ref['path'],'--stage-root',str(stage),
             '--scene',scene,'--condition',condition,'--validate-output']
    if (audit.get('schema_version')!=1 or audit.get('scope')!='exact_original_host_e7_validation'
        or audit.get('status')!='PASS' or audit.get('returncode')!=0 or audit.get('condition')!=condition
        or audit.get('source',{}).get('commit')!=PRODUCER_SHA or audit.get('source',{}).get('dirty') is not False
        or audit.get('code_root')!=str(code) or audit.get('config')!=ref or audit.get('command')!=command
        or audit.get('metric_tolerance_changed') is not False or audit.get('model_rerun') is not False
        or audit.get('paper_ready') is not False or audit.get('audit_script',{}).get('sha256')!=NATIVE_AUDIT_SCRIPT_SHA):
        raise ValueError('native audit source/command/verdict differs')
    for identity in [audit['stdout'],audit['stderr'],audit['audit_script'],audit['original_allocation'],*audit['terminal'].values()]:
        if shared.identity(identity['path'])!=identity:raise ValueError('native audit evidence changed')
    job=audit['original_finalizer_job_id']
    original={'clean':('sof1-h200-3','833155'),'mild':('hala','833147'),'severe':('sof1-h200-3','833156')}
    if (audit['hostname'],job)!=original[condition]:raise ValueError('fixed original host/allocation differs')
    if condition!='mild' and not str(audit.get('job_id','')).isdigit():raise ValueError('native audit execution job missing')
    if (audit['original_allocation']['path']!=str(stage/'launchers'/('allocation_'+job+'.json'))
        or shared.read(audit['original_allocation']['path'])!={'condition':condition,'job_id':job,'node':audit['hostname'],'phase':'finalize','resource_class':'cpu'}
        or audit['hostname'] not in {'hala','sof1-h200-3'}):raise ValueError('original finalizer host differs')
    origin=stage/'audit/public_grounding'/(scene+'_'+condition)
    expected_terminal={f:shared.identity(origin/'terminal'/f) for f in ('gate.json','manifest.json','seal.json')}
    if audit['terminal']!=expected_terminal:raise ValueError('native audit unit differs')
    bundle=shared.sealed._validate_bundle(origin/'terminal',root=shared.ROOT,expected_kind='e6_public_grounding')
    actual={str(p.relative_to(origin)):shared.file_identity(p) for p in sorted(origin.rglob('*'))
            if p.is_file() and 'terminal' not in p.relative_to(origin).parts}
    if actual!=bundle['manifest']['output_members']:raise ValueError('native original output bytes changed')
    gate=shared.read(origin/'terminal/gate.json')
    if (audit['stdout']['sha256']!=expected_terminal['gate.json']['sha256']
        or audit['stdout']['size_bytes']!=expected_terminal['gate.json']['size_bytes']
        or shared.read(audit['stdout']['path'])!=gate):raise ValueError('native stdout differs from original gate bytes')
    # Reauthenticate all construction inputs through the original host-independent
    # public API. Only the already-audited association arithmetic is not rerun.
    probe=("import json;from run.icra2027 import e6_public_grounding as d;"
           "cfg,rt,rgb,_=d.validate("+repr(ref['path'])+","+repr(str(stage))+");"
           "g,o,r=d.original_unit(cfg,rgb,"+repr(scene)+","+repr(condition)+");print(json.dumps(r))")
    env=shared.environment(rgb,stage,'validation');env.update(PYTHONPATH=str(code),SIMANY_ROOT=str(code))
    result=subprocess.run([rgb['python'],'-c',probe],cwd=code,env=env,text=True,capture_output=True)
    if result.returncode or json.loads(result.stdout)!=gate['original_construction']:
        raise ValueError('native audit original construction inputs changed')
    return gate,dict(method='exact_original_host_audit_plus_current_artifact_and_construction_revalidation',
        native_audit=audit_ref,original_host=audit['hostname'],command=command,cwd=str(code),returncode=0,
        stdout_sha256=audit['stdout']['sha256'],input_probe_stdout_sha256=shared.hashlib.sha256(result.stdout.encode()).hexdigest())


def original_roster(construction,ref):
    """Use the existing exact-7c loader so query identities retain their source root."""
    from run.icra2027.e6_public_reconstruction_full import producer
    if shared.identity(ref['path'])!=ref or construction['roster']!=ref:
        raise ValueError('original public roster identity changed')
    api=producer(construction['measurement_producer'])
    return api.validate_roster(shared.read(ref['path']))


def validate(config_path,stage,*,require_e0=True):
    config_path=Path(config_path).resolve(strict=True);stage=Path(stage).resolve(strict=True)
    cfg=shared.read(config_path);snap=shared.git_snapshot(CODE)
    if (set(cfg)!=KEYS or cfg['schema_version']!=1 or cfg['scope']!='e6_public_grounding_full18'
        or snap['dirty'] or snap['commit']!=cfg['source_commit']):raise ValueError('cohort schema or clean source differs')
    if (stage.parent!=shared.ROOT/'outputs/icra2027' or cfg['freeze_id']!=stage.name
        or not shared.CANONICAL_FREEZE_ID.fullmatch(stage.name) or config_path.parent!=stage):raise ValueError('cohort freeze path differs')
    if cfg['validation_host']!='hala' or socket.gethostname().split('.')[0]!=cfg['validation_host']:
        raise ValueError('cohort exact local validation requires declared Hala host')
    if set(cfg['native_replay_audits'])!=set(shared.CONDITIONS):raise ValueError('all three native pilot audits required')
    source(cfg['measurement_producer'])
    refs=[cfg['fresh_config'],*cfg['reference_configs']]
    if len(refs)!=3 or len({r['path'] for r in refs})!=3:raise ValueError('separate fresh, original smoke and pilot configs required')
    for ref in [cfg['roster'],*refs,*cfg['native_replay_audits'].values()]:
        if shared.identity(ref['path'])!=ref:raise ValueError('cohort input changed')
    measurement=[shared.read(r['path']) for r in refs]
    for old in measurement:
        if shared.canonical_hash(recipe(old))!=cfg['recipe_sha256']:raise ValueError('measurement recipe differs')
    construction_ref=measurement[0]['construction']
    if shared.identity(construction_ref['path'])!=construction_ref:raise ValueError('original construction config changed')
    construction=shared.read(construction_ref['path'])
    if construction['roster']!=cfg['roster']:raise ValueError('complete original RGB/query roster differs')
    roster=original_roster(construction,cfg['roster'])
    if [(r['scene_id'],r['condition_id']) for r in cfg['reuse']]!=REUSED:raise ValueError('predeclared four reuse units differ')
    for row in cfg['reuse']:
        if set(row)!={'scene_id','condition_id','config','terminal_manifest','terminal_seal'}:raise ValueError('reuse schema differs')
        expected=refs[1 if row['scene_id']=='behavior_task0023' else 2]
        if row['config']!=expected:raise ValueError('reuse scene/config binding differs')
        unit=Path(expected['path']).parent/'audit/public_grounding'/(row['scene_id']+'_'+row['condition_id'])
        for key,filename in [('terminal_manifest','manifest.json'),('terminal_seal','seal.json')]:
            if row[key]!=shared.identity(unit/'terminal'/filename):raise ValueError('original reuse seal changed')
    # Source/runtime E0 replay is done once per context, not once per original unit.
    for ref in refs:replay(cfg,ref)
    # Full admission requires actual original smoke plus all three pilot gates.
    for row in cfg['reuse']:
        gate,_=replay(cfg,row['config'],row['scene_id'],row['condition_id'])
        if gate['stage_status']!='PASS':raise ValueError('original smoke/pilot integrity gate failed')
    if require_e0:
        contract=shared.read(stage/'contract/freeze_manifest.json')
        payload={k:v for k,v in contract.items() if k not in {'created_utc','environment','contract_sha256'}}
        if (shared.canonical_hash(payload)!=contract['contract_sha256'] or contract['code']['commit']!=cfg['source_commit']
            or contract['code']['dirty'] or contract['freeze_id']!=stage.name):raise ValueError('cohort E0 source/digest differs')
        bindings={'config':shared.identity(config_path),'fresh_config':refs[0],'smoke_config':refs[1],
                  'pilot_config':refs[2],'roster':cfg['roster'],**{'native_'+c:r for c,r in cfg['native_replay_audits'].items()}}
        scripts=[shared.read(r['path'])['audit_script'] for r in cfg['native_replay_audits'].values()]
        if any(r!=scripts[0] for r in scripts):raise ValueError('native audit script binding differs')
        bindings['native_audit_script']=scripts[0]
        for name,ref in bindings.items():
            hits=[r for r in contract['resource_inventory'] if r['id']=='e6_grounding_cohort_'+name]
            if len(hits)!=1 or hits[0]['sha256']!=ref['sha256'] or hits[0]['resolved_path']!=ref['path']:
                raise ValueError('cohort E0 input binding differs: '+name)
    return cfg,roster


def unit_config(cfg,roster,scene,condition):
    rows=[r for r in roster['rows'] if (r['scene_id'],r['condition_id'])==(scene,condition)]
    if len(rows)!=1:raise ValueError('unit outside full18 roster')
    reused=next((r for r in cfg['reuse'] if (r['scene_id'],r['condition_id'])==(scene,condition)),None)
    return rows[0],reused['config'] if reused else cfg['fresh_config'],bool(reused)


def result(cfg,roster,scene,condition):
    row,ref,reused=unit_config(cfg,roster,scene,condition)
    gate,receipt=replay(cfg,ref,scene,condition)
    if (gate['scene_id']!=scene or gate['condition_id']!=condition or gate['planned_queries']!=4
        or gate['planned_cohort_queries']!=72 or gate['feature_rows_written']!=0 or gate['paper_ready'] is not False
        or {q['task_id']:q['query_sha256'] for q in gate['queries']}!=row['query_hashes']):
        raise ValueError('original per-unit query/denominator identity differs')
    if reused and gate['stage_status']!='PASS':raise ValueError('only validated completed original units may be reused')
    origin=Path(ref['path']).parent/'audit/public_grounding'/(scene+'_'+condition)
    shared.sealed.sealed_cpu._inside(origin,root=shared.ROOT,label='original grounding evidence')
    return dict(schema_version=1,scope=cfg['scope'],source_commit=cfg['source_commit'],freeze_id=cfg['freeze_id'],
        measurement_producer=cfg['measurement_producer'],recipe_sha256=cfg['recipe_sha256'],
        scene_id=scene,condition_id=condition,mode='reused_original' if reused else 'fresh_pinned_producer',
        original_config=ref,original_terminal={f:shared.identity(origin/'terminal'/f) for f in ('gate.json','manifest.json','seal.json')},
        original_validation=receipt,original_gate=gate,origin=str(origin),stage_status=gate['stage_status'],
        planned_queries=4,feature_rows_written=0,paper_ready=False)


def publish(config_path,stage,scene,condition):
    cfg,roster=validate(config_path,stage)
    return _publish(cfg,roster,stage,scene,condition)


def _publish(cfg,roster,stage,scene,condition):
    report=result(cfg,roster,scene,condition)
    dest=Path(stage)/'audit/public_grounding_cohort'/(scene+'_'+condition)
    dest.parent.mkdir(parents=True,exist_ok=True);dest.mkdir()
    (dest/'artifacts').symlink_to(report['origin'],target_is_directory=True)
    shared.sealed._publish_bundle(dest/'terminal',manifest_kind='e6_public_grounding_cohort_unit',
        payloads={'gate.json':(json.dumps(report,indent=2)+'\n').encode()},
        manifest_fields={'source_commit':cfg['source_commit'],'measurement_producer':cfg['measurement_producer'],'paper_ready':False})
    return report


def validate_output(config_path,stage,scene,condition):
    cfg,roster=validate(config_path,stage)
    return _validate_output(cfg,roster,stage,scene,condition)


def _validate_output(cfg,roster,stage,scene,condition):
    dest=Path(stage)/'audit/public_grounding_cohort'/(scene+'_'+condition)
    shared.sealed._validate_bundle(dest/'terminal',root=shared.ROOT,expected_kind='e6_public_grounding_cohort_unit')
    observed=shared.read(dest/'terminal/gate.json');expected=result(cfg,roster,scene,condition)
    if observed!=expected or not (dest/'artifacts').is_symlink() or (dest/'artifacts').resolve()!=Path(expected['origin']).resolve():
        raise ValueError('original grounding attribution or linked artifacts changed')
    return observed


def publish_available(config_path,stage):
    """Validate context once, resume only missing completed units in fixed roster."""
    cfg,roster=validate(config_path,stage);published=[]
    for row in roster['rows']:
        scene,condition=row['scene_id'],row['condition_id']
        _,ref,_=unit_config(cfg,roster,scene,condition)
        origin=Path(ref['path']).parent/'audit/public_grounding'/(scene+'_'+condition)
        dest=Path(stage)/'audit/public_grounding_cohort'/(scene+'_'+condition)
        if (dest/'terminal/seal.json').exists():
            _validate_output(cfg,roster,stage,scene,condition)
        elif (origin/'terminal/seal.json').exists():
            _publish(cfg,roster,stage,scene,condition);published.append([scene,condition])
    return dict(published=published,planned_conditions=18,planned_queries=72,paper_ready=False)


def snapshot(config_path,stage,output):
    cfg,roster=validate(config_path,stage);rows=[]
    for row in roster['rows']:
        scene,condition=row['scene_id'],row['condition_id'];dest=Path(stage)/'audit/public_grounding_cohort'/(scene+'_'+condition)
        if (dest/'terminal/seal.json').exists():
            g=_validate_output(cfg,roster,stage,scene,condition)
            rows.append(dict(scene_id=scene,condition_id=condition,status=g['stage_status'],planned_queries=4,
                             terminal=shared.identity(dest/'terminal/manifest.json')))
        else:rows.append(dict(scene_id=scene,condition_id=condition,status='INCOMPLETE' if dest.exists() else 'NOT_RUN',planned_queries=4,terminal=None))
    counts={s:sum(r['status']==s for r in rows) for s in ('PASS','FAIL','INCOMPLETE','NOT_RUN')}
    payload=dict(schema_version=1,scope=cfg['scope'],source_commit=cfg['source_commit'],freeze_id=cfg['freeze_id'],
        measurement_producer=cfg['measurement_producer'],planned_conditions=18,planned_queries=72,counts=counts,
        complete=counts['INCOMPLETE']+counts['NOT_RUN']==0,feature_rows_written=0,paper_ready=False,rows=rows)
    output=Path(output);shared.sealed.sealed_cpu._inside(output,root=Path(stage),label='cohort snapshot')
    shared.sealed._publish_bundle(output,manifest_kind='e6_public_grounding_cohort_snapshot',
        payloads={'gate.json':(json.dumps(payload,indent=2)+'\n').encode()},manifest_fields={'source_commit':cfg['source_commit'],'paper_ready':False})
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--stage-root',required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--validate',action='store_true');mode.add_argument('--publish',action='store_true');mode.add_argument('--validate-output',action='store_true');mode.add_argument('--snapshot');mode.add_argument('--publish-available',action='store_true')
    p.add_argument('--scene');p.add_argument('--condition');a=p.parse_args()
    if a.validate:validate(a.config,a.stage_root);print('E6_GROUNDING_COHORT_PREFLIGHT=PASS');return
    if a.publish_available:g=publish_available(a.config,a.stage_root)
    elif a.snapshot:g=snapshot(a.config,a.stage_root,a.snapshot)
    else:g=(publish if a.publish else validate_output)(a.config,a.stage_root,a.scene,a.condition)
    print(json.dumps(g,indent=2))

if __name__=='__main__':main()
