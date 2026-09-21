"""Recover the frozen 32-object comparison and dispatch bounded ordinary jobs.

This controller calls the existing campaign runner and shared dispatcher. It
never polls indefinitely, submits an array, or launches a policy experiment.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from .core import checked_path, digest, load, receipt, save, source_identity

BASE = Path('/group/worldcept/PhiRIE/code/SimAny')
OLD = BASE / 'outputs/icra2027/todo-followup-20260912'
G = BASE / 'outputs/icra2027/20260908T222513Z-gaussian-system'
SOURCE = Path(__file__).resolve().parents[2]
CPU = str(BASE / '.venv/bin/python')
BACKENDS = ['trellis','trellis2','sam3d','reconviagen']
NAMES = dict(trellis='phi-RIE + TRELLIS', trellis2='phi-RIE + TRELLIS.2',
             sam3d='phi-RIE + SAM 3D Objects', reconviagen='phi-RIE + ReconViaGen (multi-view)')
REGISTRATION = dict(name='existing_signed_source_up_v1',mesh_samples=20000,mesh_sample_seed=42,target_limit=6000,
                    source_up_order=['+z','-z','+x','-x','+y','-y'],yaw_step_deg=10,icp_distance_m=.03)
PROTOCOL = dict(registration=REGISTRATION, triangle_budget=40000, collision='existing s6_physics.coacd_parts: threshold .05, max16 hulls (32 if world extent >1.2m)',
                physics=dict(mass_kg=.3, friction=.5, restitution=.1, source='shared fixed component prior'),
                stability='canonical_factory_report_drop_v1; same PyBullet defaults, 240Hz, 2s zero-velocity settling plus 2s free dynamics, drift<.03m and not sunk',
                placement='observation-estimated rotation and scale baked once; common canonical plane drop without scene translation',
                geometry='existing fidelity_metrics.geometry_metrics; CD and F1@20/40mm; no evaluator ICP',
                appearance='NOT_MEASURED; retain native colors/PBR; Gaussian metrics N/A for TRELLIS.2',
                reconviagen_views='unchanged anchor plus up to11 existing occlusion-aware TRAIN-derived instance projections; at least2 distinct frames; no annotation geometry')
TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def atomic(path, data):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.writing'); q.write_text(json.dumps(data, indent=2) + '\n'); q.replace(p)


def frozen(path, value):
    """Allow recovery to resume only when already written inputs are identical."""
    if Path(path).exists():
        if load(path) != value: raise ValueError('immutable recovery input differs: '+str(path))
    else: save(path, value)


def augment(obj):
    obj = dict(obj); obj['object_key'] = f"{obj['scene_id']}-a{obj['automatic_instance_id']}"
    for key in ['source_manifest_receipt','input_manifest_receipt']:
        checked_path(obj[key])
    if obj.get('availability') == 'AVAILABLE':
        path = checked_path(obj['rgba_receipt']).parent
        obj.update(meta=receipt(path/'meta.json'), observed_surface=receipt(path/'gt_points.ply'),
                   output_hashes=receipt(Path(obj['source_manifest']).parent/'output_hashes.json'))
        from .component_worker import validate_observation
        validate_observation(obj)
    return obj


def validate_generation(gen_ref, task_path, obj, backend, seed):
    gen = load(checked_path(gen_ref)); task = load(task_path)
    if (gen['status'] != 'GENERATION_COMPLETE' or gen['backend'] != backend or gen['seed'] != seed
            or task['params']['backend'] != backend or task['params']['seed'] != seed
            or gen['source']['commit'] != task['params']['source_commit']
            or gen['checkpoint_manifest'] != receipt(task['params']['checkpoint_manifest'])
            or gen['inference'] != task['params'].get('inference', {})
            or task['inputs']['anchor']['sha256'] != obj['rgba_receipt']['sha256']
            or gen['input_images'] != [receipt(checked_path(task['inputs'][name])) for name in task['params']['images']]):
        raise ValueError('generation input/model/seed/source identity differs')
    for ref in gen['artifacts'].values(): checked_path(ref)
    for ref in gen.get('local_dependency_manifests', {}).values(): checked_path(ref)
    return gen


def recover(run):
    run = Path(run); run.mkdir(parents=True, exist_ok=True)
    (run/'tmp').mkdir(exist_ok=True); (run/'inputs').mkdir(exist_ok=True)
    roster_path = OLD/'inputs/component32_roster.json'; roster = load(roster_path)
    if len(roster['jobs']) != 32 or roster['seed'] != 0: raise ValueError('historical roster/seed drift')
    objects = [augment(o) for o in roster['jobs']]
    if len({o['object_key'] for o in objects}) != 32: raise ValueError('duplicate cohort object')
    keys = {o['object_key'] for o in objects}
    outside = [o for o in load(G/'inputs/p05/generator_dev48_roster.json')['jobs']
               if f"{o['scene_id']}-a{o['automatic_instance_id']}" not in keys and o['availability'] == 'AVAILABLE']
    outside.sort(key=lambda o: hashlib.sha256(f"component-engineering-v1/{o['scene_id']}/{o['automatic_instance_id']}".encode()).hexdigest())
    smoke = augment(outside[0])
    matching = BASE/'outputs/icra2027/20260906-357caca-v1/evaluation_matching/evaluation_references.json'
    cohort = dict(kind='bounded_missing_generator_component_v1', dataset='ScanNet++', seed=roster['seed'],
                  scope=roster['scope'], prior_exposure='historically observed DEV; not untouched TEST',
                  selection=roster['selection'], original_roster=receipt(roster_path), objects=objects,
                  planned_objects=32, planned_slots=128, backends=BACKENDS, protocol=PROTOCOL,
                  smoke_object=smoke, smoke_selection='SHA256(component-engineering-v1/scene/automatic ID) among available dev48 objects outside cohort32',
                  evaluation_matching=receipt(matching), allowance_gpu_hours=8., max_active_gpu_jobs=2,
                  max_project_active_jobs=4, max_infrastructure_retries=1,
                  authorization='2026-09-14 missing-generator task; operator explicitly selected true multi-view ReconViaGen',
                  source=source_identity(SOURCE))
    sam = BASE/'outputs/icra2027/six-generators-dev-20260912/inputs'
    model = dict(source=source_identity(BASE/'third_party/sam-3d-objects'),
                 checkpoint_manifest=receipt(sam/'weights-sam3d.json'),
                 pipeline=receipt(BASE/'third_party/sam-3d-objects/checkpoints/hf/pipeline.yaml'))
    frozen(run/'inputs/sam3d-model.json', model)
    slots = load(OLD/'inputs/component96_slots.json'); originals = {(s['scene_id'],s['automatic_instance_id'],s['backend']):s for s in slots}
    dag = {r['unit']:r for r in load(OLD/'jobs/export_dag.json') if r['phase']=='prepare'}
    extra = {r['unit']:r['result'] for r in load(OLD/'inputs/component96-extra-reuse.json')['reused_complete']}
    metrics = {r['unit_id']:r for r in load(OLD/'component_metrics.json')['rows'] if r.get('metric')}
    recovered = []; evaluations = []; legacy = []
    for obj in objects:
        for backend in BACKENDS:
            sid = obj['object_key'] + '-' + backend
            row = dict(slot_id=sid, object_key=obj['object_key'], backend=backend, status='MISSING_GENERATION')
            if obj['availability'] != 'AVAILABLE':
                row.update(status='MISSING_INPUT', reason=obj.get('failure_reason')); recovered.append(row); continue
            if backend == 'sam3d': recovered.append(row); continue
            old = originals[(obj['scene_id'],obj['automatic_instance_id'],backend)]; uid=old['planned_unit_id']; candidate=old.get('candidate')
            if candidate:
                gen_ref = candidate['generation']; task = Path(gen_ref['path']).parents[2]/'tasks'/f'{uid}.json'
            elif uid in dag:
                gen_ref = dag[uid]['candidate']; task = G/'p05-trellis2-full-dev-v1/tasks'/f'{uid}.json'
            else:
                result = Path(extra[uid]['path']) if uid in extra else OLD/'generator-missing3/results'/uid/'result.json'
                actual=load(result); gen_ref=actual['artifacts']['generation_receipt']; task=result.parents[2]/'tasks'/f'{uid}.json'
            gen = validate_generation(gen_ref, task, obj, backend, cohort['seed'])
            old_record=dict(unit_id=uid, generation=gen_ref, generation_task=receipt(task), input_views=len(gen['input_images']),
                            historical_status=old['execution_status'], candidate=candidate)
            if backend == 'reconviagen':
                row.update(status='MISSING_MULTIVIEW_GENERATION', legacy_single_view=old_record)
                legacy.append(old_record); recovered.append(row); continue
            row.update(status='GENERATION_REUSED_EVALUATION_PENDING', **old_record)
            spec = dict(slot_id=sid, object=obj, generation=gen_ref, protocol=PROTOCOL, evaluation_matching=cohort['evaluation_matching'])
            if uid in metrics:
                metric_ref = metrics[uid]['metric']; metric=load(checked_path(metric_ref)); reg=load(checked_path(metric['registration']))
                if metric['status'] != 'MEASURED' or metric['evaluator_alignment'] != 'NONE' or reg['evaluator_GT_read'] is not False:
                    raise ValueError('historical metric is not independently measured')
                spec.update(reuse_registration=metric['registration'], reuse_metric=metric_ref)
                row.update(metric=metric_ref, metric_status='REUSED_MEASURED')
            evaluations.append(spec); recovered.append(row)
    frozen(run/'cohort.json', cohort)
    frozen(run/'recovery.json', dict(slots=recovered, counts=dict(Counter(r['status'] for r in recovered)),
         original_slots=receipt(OLD/'inputs/component96_slots.json'), legacy_single_view_reconviagen=legacy,
         historical_component_metrics=receipt(OLD/'component_metrics.json'), all_original_files_unchanged=True))
    frozen(run/'inputs/evaluation.json', {'slots':evaluations})
    frozen(run/'inputs/views.json', {'objects':[o for o in objects if o['availability']=='AVAILABLE']+[smoke], 'protocol':PROTOCOL})
    common=dict(seed=cohort['seed'], protocol=PROTOCOL, evaluation_matching=cohort['evaluation_matching'], cpu_python=CPU)
    frozen(run/'inputs/sam3d-smoke.json', dict(common, model=model, objects=[dict(smoke,slot_id=smoke['object_key']+'-sam3d-smoke')]))
    frozen(run/'inputs/sam3d-production.json', dict(common, model=model, objects=[dict(o,slot_id=o['object_key']+'-sam3d') for o in objects if o['availability']=='AVAILABLE']))
    if not (run/'registry.json').exists():save(run/'registry.json', dict(max_active_jobs=4, bundles=[], additional_active_job_ids=[], task_allowance_gpu_hours=8., max_active_gpu_jobs=2))
    return dict(run=str(run), planned=128, recovery=dict(Counter(r['status'] for r in recovered)), smoke=smoke['object_key'])


def add_bundle(run, name, mode, config, *, gpu=False, minutes=30, interpreter=CPU, node_profile='default'):
    from .runner import prepare
    run=Path(run); bundle=run/'bundles'/name
    if bundle.exists(): return bundle
    if not 1 <= minutes <= (30 if 'smoke' in name else 120):
        raise ValueError('smoke limit30min / production limit2h applies to every job')
    if node_profile not in ('default','hala'):
        raise ValueError('unknown node profile')
    env=dict(PYTHONPATH=str(SOURCE), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
             TMPDIR=str(run/'tmp'), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
             SIMANY_AUTO='1', SIMANY_MESH_SRC='derived', ATTN_BACKEND='xformers', SPCONV_ALGO='native',
             HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TORCH_HOME=str(BASE/'.cache/icra2027/e2-replacements/torch'),
             XDG_CACHE_HOME=str(run/'tmp/xdg'), TORCH_EXTENSIONS_DIR=str(run/'tmp/torch-extensions'),
             NUMBA_CACHE_DIR=str(run/'tmp/numba'), MPLCONFIGDIR=str(run/'tmp/matplotlib'))
    flags=['--partition=batch','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=64G',
           '--time='+f'{minutes//60:02d}:{minutes%60:02d}:00','--constraint=zone-sof1|zone-gcp-eu1','--exclude=msp3-[0-7],sof1-h200-6','--no-requeue']
    if node_profile=='hala':flags += ['--nodelist=hala']
    if gpu: flags += ['--gres=gpu:a6000:1' if node_profile=='hala' else '--gres=gpu:h200:1']
    if gpu and mode=='generate' and node_profile=='hala':
        # Match the attention family used by the validated H200 SAM run.
        env.update(ATTN_BACKEND='flash_attn',SPARSE_ATTN_BACKEND='flash_attn')
    task=dict(id='gc32-'+name,kind='command',environment='component',split='dev',sensor='posed_rgb',
              inputs={'config':receipt(config)},params=dict(purpose='generator_component',
              argv=[interpreter,'-m','robo.campaign.component_worker',mode,'--config','{config}','--out','{out}/component'],
              output_manifest='{out}/component/outputs.json',timeout_s=minutes*60-30))
    runtime=dict(execution_ready=True,source_root=str(SOURCE),max_active_jobs=4,
                 environments={'component':dict(python=CPU,env=env,sbatch_args=flags)})
    save(run/'inputs'/f'{name}-task.jsonl',[task],jsonl=True); save(run/'inputs'/f'{name}-runtime.json',runtime)
    prepare(run/'inputs'/f'{name}-task.jsonl',run/'inputs'/f'{name}-runtime.json',bundle)
    reg=load(run/'registry.json');reg['bundles'].append(str(bundle));atomic(run/'registry.json',reg)
    return bundle


def batch_path(run, name, filename='batch.json'):
    return Path(run)/'bundles'/name/'results'/('gc32-'+name)/'component'/filename


def renewed_sam_authorization(run):
    """One explicit renewed request; historical retry limits remain recorded."""
    path=Path(run)/'authorizations/sam3d-resubmit-20260915.json'
    if not path.exists():return None
    value=load(path)
    if (value.get('backend')!='sam3d' or value.get('additional_smoke_attempts')!=1
            or value.get('smoke_name')!='sam3d-smoke-retry2'
            or not value.get('user_request')):
        raise ValueError('invalid renewed SAM authorization')
    checked_path(value['historical_blocker']);checked_path(value['smoke_config'])
    if Path(value['smoke_config']['path']) != Path(run)/'inputs/sam3d-smoke-retry2.json':
        raise ValueError('renewed authorization must bind the exact immutable retry config')
    return receipt(path)


def smoke_name(run, backend):
    run=Path(run)
    if backend=='sam3d' and renewed_sam_authorization(run):
        return 'sam3d-smoke-retry2'
    if backend=='reconviagen' and (run/'inputs/reconviagen-smoke-recovery1.json').exists():
        return 'reconviagen-smoke-recovery1'
    if backend=='sam3d' and (run/'inputs/sam3d-smoke-retry1.json').exists():
        return 'sam3d-smoke-retry1'
    return backend+'-smoke'


def prepare_wave(run):
    run=Path(run); cohort=load(run/'cohort.json')
    sam_name=smoke_name(run,'sam3d')
    add_bundle(run,sam_name,'generate',run/'inputs'/f'{sam_name}.json',gpu=True,interpreter=str(BASE/'.envs/sam3d-objects/bin/python'))
    add_bundle(run,'views','views',run/'inputs/views.json',minutes=120)
    add_bundle(run,'reference-evaluation','evaluate-batch',run/'inputs/evaluation.json',minutes=120)
    if (run/'inputs/reference-evaluation-retry1.json').exists():
        add_bundle(run,'reference-evaluation-retry1','evaluate-batch',run/'inputs/reference-evaluation-retry1.json',minutes=120)
    if (run/'inputs/reconviagen-smoke-recovery1.json').exists():
        add_bundle(run,'reconviagen-smoke-recovery1','recover-rvg-smoke',run/'inputs/reconviagen-smoke-recovery1.json',minutes=30)
    tail=run/'inputs/sam3d-tail-resumption.json'
    if tail.exists():
        resume=load(tail)
        predecessor=str(resume['predecessor_job'])
        state=subprocess.check_output(['sacct','-X','-n','-P','-j',predecessor,'--format=JobIDRaw,State'],text=True)
        if not any(line.split('|')[0]==predecessor and line.split('|')[1].split()[0] in TERMINAL for line in state.splitlines()):
            raise ValueError('predecessor must be terminal before resuming unfinished stages')
        add_bundle(run,'sam3d-tail-evaluation','evaluate-batch',checked_path(resume['evaluation_config']),minutes=15,node_profile='hala')
        add_bundle(run,'sam3d-tail-generation','generate',checked_path(resume['generation_config']),gpu=True,minutes=15,
                   interpreter=str(BASE/'.envs/sam3d-objects/bin/python'),node_profile='hala')
    views_path=batch_path(run,'views','views.json')
    if views_path.exists():
        view_rows={r['object_key']:r for r in load(views_path)['rows']}
        reference_task=next((G/'p05-rvg-full-dev-v1/tasks').glob('*.json'))
        params=dict(load(reference_task)['params'], seed=cohort['seed'])
        common=dict(seed=cohort['seed'],protocol=PROTOCOL,evaluation_matching=cohort['evaluation_matching'],cpu_python=CPU,params=params)
        def with_views(obj):
            return dict(obj,slot_id=obj['object_key']+'-reconviagen',views=receipt(views_path.parent/obj['object_key']/'views.json'))
        smoke=cohort['smoke_object']
        if view_rows[smoke['object_key']]['status']=='READY':
            cfg=run/'inputs/reconviagen-smoke.json'
            if not cfg.exists(): save(cfg,dict(common,objects=[with_views(smoke)]))
            add_bundle(run,'reconviagen-smoke','rvg-generate',cfg,gpu=True)
        cfg=run/'inputs/reconviagen-production.json'
        if not cfg.exists(): save(cfg,dict(common,objects=[with_views(o) for o in cohort['objects'] if o['availability']=='AVAILABLE' and view_rows[o['object_key']]['status']=='READY']))
    for backend,mode,python in [('sam3d','generate',str(BASE/'.envs/sam3d-objects/bin/python')),('reconviagen','rvg-generate',CPU)]:
        if (run/'blockers'/f'{backend}.json').exists() and not (backend=='sam3d' and renewed_sam_authorization(run)):continue
        smoke=batch_path(run,smoke_name(run,backend))
        if not smoke.exists() or load(smoke).get('status')!='VALIDATED':continue
        admission=run/'admissions'/f'{backend}.json'
        if not admission.exists():continue
        if load(admission)['smoke'] != receipt(smoke):raise ValueError('smoke review belongs to a different attempt')
        proof=load(smoke)
        for row in proof['rows']:
            if load(checked_path(row['evaluation']))['status']!='EVALUATED':raise ValueError('smoke result failed validation')
        config=load(run/'inputs'/f'{backend}-production.json')
        if backend=='sam3d' and (run/'inputs/sam3d-model-retry1.json').exists():
            config['model']=load(run/'inputs/sam3d-model-retry1.json')
        # Conservatively include inference, export and CPU QA in GPU occupancy.
        seconds=max(60., proof['wall_s'])*1.5
        size=max(1, min(len(config['objects']),int((7200-120)/seconds)))
        for i,start in enumerate(range(0,len(config['objects']),size)):
            objects=config['objects'][start:start+size]
            minutes=min(120,max(10,math.ceil((seconds*len(objects)+120)/60)))
            name=f'{backend}-production-{i:02d}'; cfg=run/'inputs'/f'{name}.json'
            if not cfg.exists():save(cfg,dict(config,objects=objects,smoke_gate=receipt(smoke)))
            add_bundle(run,name,mode,cfg,gpu=True,minutes=minutes,interpreter=python)


def accounting(run):
    run=Path(run);jobs=[]
    for bundle in load(run/'registry.json')['bundles']:
        b=Path(bundle);plan=load(b/'plan.json');flags=plan['runtime']['environments']['component']['sbatch_args']
        gpu=int(any(f.startswith('--gres=gpu') for f in flags))
        limit=next(f.split('=',1)[1] for f in flags if f.startswith('--time=')); h,m,s=map(int,limit.split(':')); reservation=gpu*(h+m/60+s/3600)
        for path in (b/'jobs').glob('*/intent.json'):
            sub=path.parent/'submission.json'
            if not sub.exists():raise RuntimeError('uncertain sbatch intent; reconcile before resubmission: '+str(path))
            d=load(sub)
            if d['returncode']!=0 or not str(d.get('job_id','')).isdigit():raise RuntimeError('failed/ambiguous submission: '+str(sub))
            jobs.append(dict(job_id=str(d['job_id']),bundle=str(b),gpu=gpu,reservation_hours=reservation,submission=receipt(sub)))
    states={}
    if jobs:
        text=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(j['job_id'] for j in jobs),'--format=JobIDRaw,State,ElapsedRaw,ExitCode,NodeList'],text=True)
        for line in text.splitlines():
            jid,state,elapsed,code,node,*_=line.split('|');states[jid]=dict(state=state.split()[0],elapsed_seconds=int(elapsed),exit_code=code,node=node)
    spent=reserved=0.;active=0
    for j in jobs:
        j.update(states.get(j['job_id'],dict(state='UNKNOWN',elapsed_seconds=0)))
        if j['state'] in TERMINAL:spent+=j['gpu']*j['elapsed_seconds']/3600
        else:reserved+=j['reservation_hours'];active+=j['gpu']
    result=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs=jobs,spent_gpu_hours=spent,
                reserved_gpu_hours=reserved,remaining_unreserved_gpu_hours=8-spent-reserved,active_gpu_jobs=active,allowance_gpu_hours=8.)
    atomic(run/'accounting.json',result);return result


def admit(run, backend, reviewed_render_sha256):
    if backend not in ('sam3d','reconviagen'):raise ValueError('select an unresolved backend')
    run=Path(run)
    name=smoke_name(run,backend)
    smoke=batch_path(run,name);data=load(smoke)
    if data.get('status')!='VALIDATED' or len(data['rows'])!=1:raise ValueError('one completed engineering smoke is required')
    row=data['rows'][0];evaluation=load(checked_path(row['evaluation']))
    if evaluation['status']!='EVALUATED' or evaluation.get('collision_import_valid') is not True:
        raise ValueError('smoke did not execute registration, collision import and stability')
    render=evaluation['render'];checked_path(render)
    if not reviewed_render_sha256 or reviewed_render_sha256!=render['sha256']:
        raise ValueError('inspect the actual smoke render and pass its exact SHA256')
    value=dict(smoke=receipt(smoke),evaluation=row['evaluation'],reviewed_render=render,
               scope='engineering validity; reconstruction quality and positive stability are not admission requirements')
    frozen(run/'admissions'/f'{backend}.json',value);return value


def dispatch(run):
    from .finalize import dispatch as shared_dispatch
    run=Path(run)
    with (run/'.budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        prepare_wave(run);budget=accounting(run)
        # Include other active SimAny jobs in the established project cap.
        live=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%i|%j'],text=True)
        other=[line.split('|')[0] for line in live.splitlines() if any(line.split('|',1)[1].startswith(x) for x in ('sar-','fi-','phiview-','p05-','simany-','e3-','e4-'))]
        registry=load(run/'registry.json');registry['additional_active_job_ids']=sorted(set(registry.get('additional_active_job_ids',[])+other));atomic(run/'registry.json',registry)
        reports=[]
        for _ in range(4):
            budget=accounting(run); next_bundle=None
            for folder in registry['bundles']:
                b=Path(folder);plan=load(b/'plan.json');tid=plan['task_ids'][0]
                if (b/'jobs'/tid/'intent.json').exists():continue
                flags=plan['runtime']['environments']['component']['sbatch_args'];gpu=any(f.startswith('--gres=gpu') for f in flags)
                h,m,s=map(int,next(f.split('=',1)[1] for f in flags if f.startswith('--time=')).split(':'))
                if gpu and (budget['active_gpu_jobs']>=2 or h+m/60+s/3600>budget['remaining_unreserved_gpu_hours']):continue
                next_bundle=folder;break
            if next_bundle is None:break
            # Existing dispatcher owns the scheduler lock and sbatch receipts.
            ordered=[next_bundle]+[b for b in registry['bundles'] if b!=next_bundle]
            registry['bundles']=ordered;atomic(run/'registry.json',registry)
            added=shared_dispatch(str(run/'registry.json'),submit=True,max_jobs=1)
            if not added:break
            reports.extend(added)
        return dict(submissions=reports,accounting=accounting(run))


def collect(run):
    run=Path(run);cohort=load(run/'cohort.json');recovery=load(run/'recovery.json');rows=[]
    for original in recovery['slots']:
        row=dict(original);row['generated']=bool(row.get('generation'));evaluations=list((run/'bundles').glob('*/results/*/component/'+row['slot_id']+'/evaluation.json'))
        evaluations+=list((run/'bundles').glob('*/results/*/component/'+row['slot_id']+'/evaluation/evaluation.json'))
        generations=list((run/'bundles').glob('*/results/*/component/'+row['slot_id']+'/generation.json'))
        if generations:
            if len(generations)!=1:raise ValueError('multiple attempts require explicit adjudication')
            g=load(generations[0]);row.update(generation=receipt(generations[0]),status=g['status'],generated=g['status']=='GENERATION_COMPLETE')
        if row.get('generation'):
            g=load(checked_path(row['generation']))
            row.update(generation_seconds=g.get('wall_s'), inference_seconds=g.get('inference_seconds'),
                       peak_gpu_bytes=g.get('peak_gpu_bytes'), conditioning_views=len(g.get('input_images',[])))
        if evaluations:
            if len(evaluations)!=1:raise ValueError('multiple evaluations require explicit adjudication')
            ev=load(evaluations[0]);row.update(evaluation=receipt(evaluations[0]),status=ev['status'],metrics=ev.get('metrics'),
                registered=ev.get('registered',False),exported=ev.get('exported',False),collision_import_valid=ev.get('collision_import_valid'),stability=ev.get('stability'),
                registration_seconds=ev.get('registration_seconds'),evaluation_seconds=ev.get('evaluation_seconds'),
                collision_export_seconds=ev.get('registration_export_seconds'),appearance_metrics=ev.get('appearance_metrics'))
        elif row.get('metric'):
            metric=load(checked_path(row['metric']));row['metrics']=metric['metrics'];row['registered']=True
        rows.append(row)
    blockers={p.stem:dict(load(p),receipt=receipt(p)) for p in (run/'blockers').glob('*.json')}
    authorization=renewed_sam_authorization(run)
    if authorization and 'sam3d' in blockers:
        blockers['sam3d']=dict(status='SUPERSEDED_BY_RENEWED_USER_REQUEST',historical_blocker=blockers['sam3d'],authorization=authorization)
    batch_timing=[]
    for p in (run/'bundles').glob('*/results/*/component/batch.json'):
        batch=load(p)
        if 'wall_s' in batch:
            batch_timing.append(dict(batch=receipt(p),wall_s=batch['wall_s'],initialization_seconds=batch.get('initialization_seconds'),status=batch.get('status')))
    by_backend={b:[r for r in rows if r['backend']==b] for b in BACKENDS}
    supports={b:{r['object_key'] for r in records if r.get('metrics')} for b,records in by_backend.items()}
    common=set.intersection(*supports.values()); summaries=[]
    for b,records in by_backend.items():
        valid=[r for r in records if r.get('metrics')];shared=[r for r in valid if r['object_key'] in common]
        def means(group):
            return {k:sum(r['metrics'][k] for r in group)/len(group) if all(k in r['metrics'] for r in group) else None for k in ['cd_cm','f1_20','f1_40']} if group else None
        summaries.append(dict(backend=b,display_name=NAMES[b],planned=32,generated=sum(r['generated'] for r in records),
            registered=sum(bool(r.get('registered')) for r in records),exported=sum(bool(r.get('exported')) for r in records),
            measured=len(valid),available_support_means=means(valid),common_support=len(shared),common_support_means=means(shared),
            collision_import_valid=sum(r.get('collision_import_valid') is True for r in records),
            stability_tested=sum(bool(r.get('stability')) for r in records),stable=sum(bool(r.get('stability',{}).get('stable')) for r in records),
            engineering_blocker=blockers.get(b),
            status_counts=dict(Counter(r['status'] for r in records))))
    summary=dict(status='PARTIAL',cohort=receipt(run/'cohort.json'),planned_slots=128,backends=summaries,common_support=sorted(common),
                 rows=rows,accounting=accounting(run),paper_modified=False,legacy_rvg_single_view_not_pooled=True,
                 batch_timing=batch_timing,timing_note='Per-object generation records exclude registration/export and may exclude model startup. Batch wall_s includes startup and evaluation. Historical end-to-end cost is unmeasured; stage times are not total asset-conversion latency.')
    dest=run/'results';dest.mkdir(exist_ok=True);atomic(dest/'summary.json',summary)
    with (dest/'per_object.csv').open('w',newline='') as stream:
        fields=['slot_id','backend','object_key','status','cd_cm','f1_20','f1_40','conditioning_views','generation_seconds','inference_seconds','registration_seconds','collision_export_seconds','evaluation_seconds','peak_gpu_bytes'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({k:((r.get('metrics') or {}).get(k) if k in ['cd_cm','f1_20','f1_40'] else r.get(k)) for k in fields})
    tex=['% Bounded observed DEV follow-up; missing results are not zero.','\\begin{tabular}{lrrrr}','Method & Planned & Generated & Measured & F1@20mm \\\\']
    for s in summaries:
        val=s['available_support_means'];score=f"{val['f1_20']:.3f}" if val else '--'
        tex.append(f"{s['display_name']} & 32 & {s['generated']} & {s['measured']} & {score} \\\\")
    tex.append('\\end{tabular}');(dest/'generator_comparison.tex').write_text('\n'.join(tex)+'\n')
    return {k:v for k,v in summary.items() if k!='rows'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['recover','prepare','dispatch','status','collect','admit']);p.add_argument('--run',required=True)
    p.add_argument('--backend',choices=['sam3d','reconviagen']);p.add_argument('--reviewed-render-sha256')
    a=p.parse_args()
    if a.mode=='recover':result=recover(a.run)
    elif a.mode=='prepare':prepare_wave(a.run);result=accounting(a.run)
    elif a.mode=='dispatch':result=dispatch(a.run)
    elif a.mode=='collect':result=collect(a.run)
    elif a.mode=='admit':result=admit(a.run,a.backend,a.reviewed_render_sha256)
    else:result=accounting(a.run)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
