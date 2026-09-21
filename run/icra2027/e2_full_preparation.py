"""Full E2 preparation admission and TRAIN-only structural coverage inventory."""
from pathlib import Path
import argparse
import json
import os
import time
import yaml
from agents.edit.inpaint_masks import _identity, _checked, _bound_contract, _preparation
from agents.edit.inpaint_prepare import run_public, PUBLIC_ALGORITHM
from robo.eval import agentic_ablation as e3
from run.icra2027 import e2_full_factorized_protocol as original
from run.icra2027.e2_raw_room import write_new

SCOPE = 'full_e2_training_preparation'
ADMISSION_SHA = 'b1929887a037038c512e6b6577a5a1f15697754937018f1b12ba3c26e8f6d537'
PILOT_SOURCE = '776d37c045fe63647e10b246d0c4ea512ab90a67'


def read(ref):return yaml.safe_load(_checked(ref).read_text())


def preparation_spec(context_path, contract_path, commit):
    c = yaml.safe_load(Path(context_path).read_text())
    directory = e3.REPOSITORY_ROOT/'outputs/icra2027'/c['freeze_id']/'fidelity/removal'/c['scene_id']/'inpaint'
    return dict(directory=str(directory),seal=_identity(directory/'seal.json'),context=_identity(Path(context_path)),
                contract=_identity(Path(contract_path)),producer_commit=commit)


def structural_state(report):
    """Predeclared construction-input gate; no image metric or model outcome."""
    accepted = [r for r in report['objects'] if r['terminal_action']=='accept']
    blocked = []
    for row in accepted:
        reasons=[]
        if row.get('plane_status')!='FIT':reasons.append('NO_PLANE')
        if not row.get('selected_frames'):reasons.append('NO_VIEW')
        elif not row.get('projected_mask_pixels') or row['projected_mask_pixels'][0]<=0:
            reasons.append('EMPTY_PRIMARY_PROJECTED_MASK')
        if reasons:blocked.append(dict(object_slot=row['object_slot'],reasons=reasons))
    if blocked:return 'BLOCKED_UNFILLABLE_ACCEPTED',blocked
    return ('STRUCTURALLY_FILLABLE' if accepted else 'NO_ACCEPTED_OBJECTS'),[]



PRIOR_BATCH_SOURCE = '5d294c87650bb86dc94fd8bd57cedc67f8586ab5'

def prior_batch_preparations(ref):
    """Authenticate completed prior-unit identities; inventory replays each producer validator."""
    pc,pe=_bound_contract(_checked(ref['config']),_checked(ref['contract']),producer_commit=PRIOR_BATCH_SOURCE)
    gate=read(ref['integrity'])
    if (pc['scope']!=SCOPE or gate['scope']!='authenticated_full_e2_preparation_batch'
            or gate['status']!='PASS' or gate['source_commit']!=PRIOR_BATCH_SOURCE
            or gate['config']!=ref['config'] or gate['contract']!=ref['contract']
            or [r['scene_id'] for r in gate['rows']]!=pc['scene_ids']
            or gate['planned_batch_scenes']!=len(pc['scene_ids'])
            or gate['planned_full_scenes']!=50 or gate['planned_full_objects']!=1871
            or gate['planned_full_views']!=400 or gate['official_test_images_read']!=0
            or gate['paper_ready'] is not False):raise ValueError('prior preparation batch identity differs')
    result={}
    for row in gate['rows']:
        scene=row['scene_id'];spec=row['preparation']
        if (spec['producer_commit']!=PRIOR_BATCH_SOURCE or spec['context']!=pc['contexts'][scene]
                or spec['contract']!=ref['contract']):raise ValueError('prior preparation unit source differs')
        for key in ('context','contract','seal'):_checked(spec[key])
        _bound_contract(_checked(spec['context']),_checked(spec['contract']),producer_commit=PRIOR_BATCH_SOURCE)
        result[scene]=spec
    return result

def context(config_path,contract_path):
    cp, ep=Path(config_path).absolute(),Path(contract_path).absolute()
    c, contract=_bound_contract(cp,ep)
    p=original.validate_protocol(_checked(c['full_protocol']))
    prior=prior_batch_preparations(c['prior_batch']) if c.get('prior_batch') else {}
    if c.get('prior_preparations',{})!=prior:raise ValueError('prior preparation roster differs')
    if c.get('prior_batch'):
        pc=read(c['prior_batch']['config'])
        if (pc['full_protocol']!=c['full_protocol'] or pc['pilot_preparation']!=c['pilot_preparation']
                or pc['background_admission']['sha256']!=c['background_admission']['sha256']):
            raise ValueError('prior preparation experiment definition differs')
    admission=read(c['background_admission'])
    if c['background_admission']['sha256'] != ADMISSION_SHA or c['pilot_preparation']['producer_commit'] != PILOT_SOURCE:
        raise ValueError('predeclared admission or original pilot source differs')
    if (c['scope']!=SCOPE or c['schema_version']!=1 or c['paper_ready'] is not False
            or c['scene_ids']!=[sid for sid in p['scene_ids'][1:] if sid in c['contexts']]
            or set(c['contexts'])!=set(c['scene_ids'])
            or sorted(c['scene_ids']+list(prior)+c.get('unavailable_factory_scenes',[]))!=p['scene_ids'][1:]
            or set(c['scene_ids']) & (set(c.get('unavailable_factory_scenes',[])) | set(prior))
            or set(prior) & set(c.get('unavailable_factory_scenes',[]))
            or c['all_planned_scenes']!=50 or c['all_planned_objects']!=1871 or c['all_planned_test_views']!=400
            or c['batch_rule']!='all_currently_available_factories_before_preparation_outcomes'
            or list(c['availability_snapshot'])!=p['scene_ids'][1:]
            or {s for s,r in c['availability_snapshot'].items() if r is not None}!=(set(c['scene_ids'])|set(prior))
            or admission['scope']!='full_e2_training_structural_background_admission'
            or admission['all_planned_scenes']!=50 or admission['all_planned_objects']!=1871
            or admission['all_planned_test_views']!=400 or admission['TEST_quality_used_for_admission'] is not False
            or admission['required_accepted_object_inputs']!=['support_plane_available','at_least_one_selected_training_view','primary_projected_mask_nonempty']
            or admission['mask_fill_pilot_rule']!='first_lexicographic_scene_with_at_least_one_accepted_object_and_all_required_inputs_available'):
        raise ValueError('full preparation roster or structural admission differs')
    pilot=_preparation(c['pilot_preparation'],p['pilot_scene'])
    if (pilot['report']['planned_objects']!=p['population'][p['pilot_scene']]
            or pilot['report']['official_test_images_read']!=0
            or pilot['report']['source_validation']['e3_producer_commit']!=original.CONSTRUCTION_COMMIT):
        raise ValueError('original preparation pilot integrity differs')
    out=e3.REPOSITORY_ROOT/'outputs/icra2027'/c['freeze_id']/'fidelity'
    e3._validate_cli_execution(ep,c['freeze_id'],out,config_paths=[cp])
    original_jobs=original.read(p['resolved_jobs'])
    slots={s['scene_id']:[j['object_slot'] for j in s['jobs']] for s in original_jobs['scenes']}
    for sid,ref in {**{s:r['context'] for s,r in prior.items()},**c['contexts']}.items():
        local=read(ref)
        if (local['scene_id']!=sid or (sid not in prior and local['freeze_id']!=c['freeze_id']) or local['policy_id']!='A4'
                or local['scope']!='automatic_train_only_removal_preparation' or local['algorithm']!=PUBLIC_ALGORITHM
                or local['seed']!=0):raise ValueError('per-scene preparation treatment differs')
        if local['materialization_manifest']!=c['availability_snapshot'][sid]:
            raise ValueError('factory availability snapshot differs')
        manifest=read(local['materialization_manifest'])
        if (manifest['e3_code_commit']!=original.CONSTRUCTION_COMMIT or manifest['e3_freeze_id']!=original.CONSTRUCTION_FREEZE
                or manifest['scene_id']!=sid or manifest['policy_id']!='A4'
                or manifest['roster']['job_count']!=p['population'][sid] or manifest['roster']['object_slots']!=slots[sid]):
            raise ValueError('per-scene full canonical factory identity differs')
    return c,contract,p,out


def run(config_path,contract_path,scene):
    if os.environ.get('SLURM_ARRAY_JOB_ID') or os.environ.get('SLURM_JOB_GPUS'):
        raise ValueError('ordinary CPU preparation only')
    c,contract,p,out=context(config_path,contract_path)
    if scene not in c['scene_ids']:raise ValueError('scene outside remaining49')
    target=out/'preparation_failures'/f'{scene}.json'
    if target.exists() or target.is_symlink():raise FileExistsError('failed preparation attempt already exists')
    cp=_checked(c['contexts'][scene]); local=read(c['contexts'][scene])
    scene_root=out/'removal'/scene
    if any((scene_root/name).exists() or (scene_root/name).is_symlink() for name in
           ('inpaint','inpaint_failed_partial','inpaint_public_failure.json','.inpaint-public.claim.json')):
        raise FileExistsError('preparation output or attempt already exists')
    started=time.monotonic()
    try:return run_public(cp,contract_path)
    except Exception as exc:
        write_new(target,dict(schema_version=1,scope=SCOPE,status='FAILED_PREPARATION',scene_id=scene,
            freeze_id=c['freeze_id'],code_commit=contract['code']['commit'],config=_identity(Path(config_path).absolute()),
            contract=_identity(Path(contract_path).absolute()),context=c['contexts'][scene],
            planned_objects=p['population'][scene],planned_test_views=8,error_type=type(exc).__name__,
            error=str(exc),wall_s=time.monotonic()-started,paper_ready=False))
        raise


def inventory(config_path,contract_path,*,pilot_only=False):
    c,contract,p,out=context(config_path,contract_path);destination=out/('model_pilot_admission' if pilot_only else 'preparation_inventory')
    if destination.exists() or destination.is_symlink():raise FileExistsError('preparation inventory exists')
    rows=[];waiting=[]
    for scene in p['scene_ids']:
        if scene==p['pilot_scene']:
            spec=c['pilot_preparation']
        elif scene in c.get('prior_preparations',{}):
            spec=c['prior_preparations'][scene]
        else:
            if scene not in c['contexts']:
                waiting.append(scene)
                if pilot_only:return dict(status='WAITING',missing_prefix_scene=scene,paper_ready=False)
                continue
            cp=_checked(c['contexts'][scene]);directory=out/'removal'/scene/'inpaint'
            if not (directory/'seal.json').is_file():
                failure=out/'preparation_failures'/f'{scene}.json'
                if not failure.is_file():
                    waiting.append(scene)
                    if pilot_only:return dict(status='WAITING',missing_prefix_scene=scene,paper_ready=False)
                    continue
                if failure.resolve(strict=True)!=failure:raise ValueError('preparation failure uses an alias')
                f=json.loads(failure.read_text())
                if (f['status']!='FAILED_PREPARATION' or f['scene_id']!=scene or f['context']!=c['contexts'][scene]
                        or f['code_commit']!=contract['code']['commit'] or f['freeze_id']!=c['freeze_id']
                        or f['planned_objects']!=p['population'][scene] or read(f['config'])!=c
                        or read(f['contract'])!=contract):raise ValueError('preparation failure binding differs')
                if pilot_only:return dict(status='BLOCKED',failed_prefix_scene=scene,
                    reason='earlier_preparation_failure_does_not_establish_structural_unfillability',
                    failure=_identity(failure),paper_ready=False)
                rows.append(dict(scene_id=scene,status='FAILED_PREPARATION',planned_objects=p['population'][scene],
                    planned_test_views=8,missing_test_views=8,failure=_identity(failure),preparation=None))
                continue
            spec=preparation_spec(cp,Path(contract_path).absolute(),contract['code']['commit'])
        verified=_preparation(spec,scene);report=verified['report']
        if report['planned_objects']!=p['population'][scene]:raise ValueError('preparation object denominator differs')
        state,blocked=structural_state(report)
        rows.append(dict(scene_id=scene,status=state,planned_objects=report['planned_objects'],
            accepted_objects=report['accepted_objects'],planned_training_views=len(verified['jobs']),
            planned_test_views=8,missing_test_views=8 if blocked else None,
            blocked_objects=blocked,preparation=spec,paper_ready=False))
        if pilot_only and state=='STRUCTURALLY_FILLABLE':
            gate=dict(schema_version=1,scope='full_e2_first_structurally_fillable_prefix',status='PASS',
                freeze_id=c['freeze_id'],config=_identity(Path(config_path).absolute()),
                contract=_identity(Path(contract_path).absolute()),full_protocol=c['full_protocol'],background_admission=c['background_admission'],
                planned_full_scenes=50,planned_full_objects=1871,planned_full_test_views=400,
                authenticated_prefix=rows,model_pilot_scene=scene,mask_fill_inference_invoked=False,
                TEST_quality_used_for_admission=False,full_inventory_complete=False,paper_ready=False)
            write_new(destination/'admission.json',gate)
            write_new(destination/'seal.json',dict(schema_version=1,members={'admission.json':_identity(destination/'admission.json')['sha256']}))
            return gate
    if pilot_only:return dict(status='NO_STRUCTURALLY_FILLABLE_SCENE',authenticated_prefix=rows,paper_ready=False)
    if waiting:return dict(status='WAITING',missing_scenes=waiting,paper_ready=False)
    if [r['scene_id'] for r in rows]!=p['scene_ids']:raise ValueError('complete preparation roster differs')
    eligible=[r['scene_id'] for r in rows if r['status']=='STRUCTURALLY_FILLABLE']
    candidate=eligible[0] if eligible else None
    failed_prefix=[r['scene_id'] for r in rows if r['status']=='FAILED_PREPARATION' and (candidate is None or r['scene_id']<candidate)]
    result=dict(schema_version=1,scope=SCOPE,status='PASS',freeze_id=c['freeze_id'],config=_identity(Path(config_path).absolute()),
        contract=_identity(Path(contract_path).absolute()),full_protocol=c['full_protocol'],background_admission=c['background_admission'],
        planned_scenes=50,planned_objects=1871,planned_test_views=400,rows=rows,
        model_pilot_scene=candidate if not failed_prefix else None,failed_model_pilot_prefix=failed_prefix,mask_fill_inference_invoked=False,
        TEST_quality_used_for_admission=False,paper_ready=False)
    write_new(destination/'inventory.json',result)
    write_new(destination/'seal.json',dict(schema_version=1,members={'inventory.json':_identity(destination/'inventory.json')['sha256']}))
    return result


def build_configs(out_dir, *, full_protocol, background_admission, missing_factory_config, pilot_gate, allow_ready_batch=False, prior_batch=None):
    """Freeze an explicit available-factory batch; no producer or metric replay."""
    from robo.eval.freeze import reserve_freeze_id
    out_dir=Path(out_dir).absolute()
    if out_dir.exists():raise FileExistsError('prepared configs already exist')
    pp=Path(full_protocol).absolute();bp=Path(background_admission).absolute()
    p=original.validate_protocol(pp);missing=yaml.safe_load(Path(missing_factory_config).read_text())
    gate=json.loads(Path(pilot_gate).read_text())
    prior=prior_batch_preparations(prior_batch) if prior_batch else {}
    if (gate['status']!='PASS' or gate['source_commit']!=PILOT_SOURCE
            or gate['planned_objects']!=p['population'][p['pilot_scene']]
            or _identity(bp)['sha256']!=ADMISSION_SHA):raise ValueError('preparation pilot or structural protocol differs')
    extra_root=e3.REPOSITORY_ROOT/'outputs/icra2027'/missing['freeze_id']/'fidelity/materialization/factories'
    source_root=Path(p['factory_freeze_root'])/'automatic_candidates'
    factories={s:(extra_root/s/'A4' if s in missing['scene_ids'] else source_root/s/'materialized/A4')
               for s in p['scene_ids'][1:] if s not in prior}
    waiting=[s for s,f in factories.items() if not (f/'materialization_manifest.json').is_file() or not (f/'seal.json').is_file()]
    if waiting and (not allow_ready_batch or len(waiting)==len(factories)):
        return dict(status='WAITING',missing_scenes=waiting,paper_ready=False)
    factories={sid:factory for sid,factory in factories.items() if sid not in waiting}
    # Verify the complete original roster and generation-config link before
    # committing input identities; each real producer authenticates all bytes.
    jobs=original.read(p['resolved_jobs']);job_rows={s['scene_id']:s for s in jobs['scenes']}
    audit_path=Path(p['resolved_jobs']['path']).with_name('inventory_audit.json')
    audit=json.loads(audit_path.read_text());prepared={}
    template=yaml.safe_load(_checked(gate['preparation']['context']).read_text())
    for sid,factory in factories.items():
        manifest=json.loads((factory/'materialization_manifest.json').read_text())
        if (manifest['e3_code_commit']!=original.CONSTRUCTION_COMMIT or manifest['e3_freeze_id']!=original.CONSTRUCTION_FREEZE
                or manifest['scene_id']!=sid or manifest['policy_id']!='A4'
                or manifest['roster']['object_slots']!=[j['object_slot'] for j in job_rows[sid]['jobs']]):
            raise ValueError('factory does not preserve full original construction')
        pool=audit['scene_audits'][sid]['initial_pools']['trellis']
        pool_value=original.read({'path':pool['path'],'bytes':Path(pool['path']).stat().st_size,'sha256':pool['sha256']})
        input_path=Path(pool['path']).parent/'input_manifest.json'
        if _identity(input_path)['sha256']!=pool_value['input_manifest_sha256']:
            raise ValueError('generation input manifest changed')
        inputs=json.loads(input_path.read_text())
        generation=Path('/group/worldcept/PhiRIE/code/SimAny-wt/e3-generation-cohort/configs/experiments/icra2027/trellis_cohort')/(sid+'.yaml')
        if _identity(generation)['sha256']!=inputs['config_sha256']:
            raise ValueError('original generation config differs')
        prepared[sid]={**template,'scene_id':sid,'materialization_manifest':_identity(factory/'materialization_manifest.json'),
                       'generation_config':_identity(generation)}
    freeze_id=reserve_freeze_id(Path(__file__).resolve().parents[2],e3.REPOSITORY_ROOT/'outputs/icra2027')
    out_dir.mkdir(parents=True,exist_ok=False)
    refs={}
    for sid,local in prepared.items():
        local['freeze_id']=freeze_id;path=out_dir/(sid+'.yaml')
        with path.open('x') as f:yaml.safe_dump(local,f,sort_keys=False)
        refs[sid]=_identity(path)
    c=dict(schema_version=1,scope=SCOPE,paper_ready=False,freeze_id=freeze_id,full_protocol=_identity(pp),
           background_admission=_identity(bp),pilot_preparation=gate['preparation'],scene_ids=list(refs),contexts=refs,
           prior_batch=prior_batch,prior_preparations=prior,
           unavailable_factory_scenes=waiting,all_planned_scenes=50,all_planned_objects=1871,all_planned_test_views=400,
           batch_rule='all_currently_available_factories_before_preparation_outcomes',
           availability_snapshot={sid:prepared[sid]['materialization_manifest'] if sid in prepared else (read(prior[sid]['context'])['materialization_manifest'] if sid in prior else None) for sid in p['scene_ids'][1:]})
    with (out_dir/'execution.yaml').open('x') as f:yaml.safe_dump(c,f,sort_keys=False)
    template_root=Path(__file__).resolve().parents[2]/'configs/experiments/icra2027/e2_full_factorized/freeze.yaml'
    freeze=yaml.safe_load(template_root.read_text());freeze['freeze_id']=freeze_id
    inputs=[('preparation_config',out_dir/'execution.yaml','experiment_config'),('full_e2_protocol',pp,'experiment_config'),
            ('background_admission',bp,'experiment_config'),('pilot_preparation_gate',Path(pilot_gate),'source_manifest'),
            ('preparation_python',Path(__import__('sys').executable).resolve(),'python_runtime')]
    if prior_batch:
        inputs += [('prior_batch_'+k,_checked(v),'source_manifest') for k,v in prior_batch.items()]
        inputs += [('prior_'+sid+'_'+k,_checked(spec[k]),'source_manifest') for sid,spec in prior.items() for k in ('seal','context')]
    for sid,ref in refs.items():
        inputs.extend([(f'prepare_{sid}',Path(ref['path']),'experiment_config'),
                       (f'factory_{sid}',Path(prepared[sid]['materialization_manifest']['path']),'source_manifest'),
                       (f'generation_{sid}',Path(prepared[sid]['generation_config']['path']),'experiment_config')])
    freeze['input_roots']=[dict(id=name,path=str(path),kind=kind,required=True) for name,path,kind in inputs]
    freeze['hardware']=dict(gpu_required=False,cpus_per_scene=4,memory_gb_per_scene=32,
                            purpose='Frozen available-input CPU preparation batch; fixed pilot reused, all50 denominators retained')
    with (out_dir/'freeze.yaml').open('x') as f:yaml.safe_dump(freeze,f,sort_keys=False)
    return dict(status='PREPARED',freeze_id=freeze_id,active_scenes=len(refs),unavailable_factory_scenes=waiting,planned_full_objects=1871,paper_ready=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True);ap.add_argument('--contract',required=True)
    ap.add_argument('--phase',choices=['context','run','inventory','pilot-admission'],required=True);ap.add_argument('--scene');a=ap.parse_args()
    if a.phase=='run':run(a.config,a.contract,a.scene)
    elif a.phase in {'inventory','pilot-admission'}:print(json.dumps(inventory(a.config,a.contract,pilot_only=a.phase=='pilot-admission'),sort_keys=True))
    else:c,_,_,_=context(a.config,a.contract);print(json.dumps(dict(status='PASS',remaining_scenes=len(c['scene_ids']))))

if __name__=='__main__':main()
