"""Dependency-aware ordinary Slurm jobs. This is not a second robot rollout ledger.

Native experiment tasks invoke the existing paired harness and publish its output
manifest. Scientific tables continue to be produced by robo.eval.paper_pipeline.
"""
from __future__ import annotations
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from .core import canonical, checked_path, digest, load, receipt, rows, save, source_identity, safe_id

KINDS = {'rgb_video_reconstruction', 'generator', 'asset_bridge', 'inpaint', 'text_query', 'chorus', 'semantic_instances', 'state_visual', 'gaussian_color_cache',
         'simfoundry', 'polaris_bundle', 'harmonizer_sequence', 'command'}
TERMINAL = {'COMPLETE', 'METHOD_FAILED', 'BLOCKED_DEPENDENCY', 'ENGINEERING_FAILED'}


def validate_tasks(tasks):
    ids = [safe_id(t['id']) for t in tasks]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError('nonempty unique task IDs required')
    by_id = dict(zip(ids, tasks))
    for t in tasks:
        if t['kind'] not in KINDS or not isinstance(t.get('params'), dict):
            raise ValueError('unknown kind or absent task parameters')
        if t.get('split') not in ('train', 'dev', 'test', 'demo'):
            raise ValueError('declare a split for every task')
        if t.get('sensor') not in ('rgb_video', 'posed_rgb', 'ideal_rgbd', 'observed_rgbd', 'not_applicable'):
            raise ValueError('declare the actual sensor regime')
        if t['sensor'] == 'rgb_video' and t.get('scale_source') not in ('estimated', 'known_robot', 'calibration_marker'):
            raise ValueError('RGB video must disclose metric-scale source')
        if t['sensor'] == 'rgb_video' and t.get('camera_source') != 'estimated':
            raise ValueError('GT/known cameras are posed-RGB, not raw RGB-video')
        if not isinstance(t.get('inputs', {}), dict):
            raise ValueError('inputs must be a named manifest')
        for name, item in t.get('inputs', {}).items():
            safe_id(name)
            if 'task' in item:
                if item['task'] not in by_id or item['task'] == t['id']:
                    raise ValueError('invalid dependency')
                if t['split'] != by_id[item['task']]['split'] and not item.get('frozen_training_model'):
                    raise ValueError('cross-split dependency needs explicit frozen training model')
                safe_id(item['artifact'])
            else:
                checked_path(item)
            if t['kind'] not in ('command','polaris_bundle') and item.get('role') in ('gt_mesh','gt_mask','reference_asset'):
                raise ValueError('hidden evaluation asset cannot enter construction')
            if t['sensor'] == 'rgb_video' and item.get('role') in ('gt_depth', 'gt_pose', 'gt_mesh', 'gt_mask'):
                raise ValueError('GT leaked into RGB-video construction input')
        if t['kind'] == 'command':
            p = t['params']
            if not isinstance(p.get('argv'), list) or not p['argv'] or not p.get('output_manifest'):
                raise ValueError('command needs argv and an output manifest contract')
            if p.get('purpose') == 'native_policy' and not all(p.get(k) for k in
                    ('same_engine_gate', 'rubric_gate', 'fresh_reference', 'official_method_labels')):
                raise ValueError('native policy block lacks source-bound integration gates')
    visited = set(); active = set()
    def visit(tid):
        if tid in active: raise ValueError('cyclic task graph')
        if tid in visited: return
        active.add(tid)
        for item in by_id[tid].get('inputs', {}).values():
            if 'task' in item: visit(item['task'])
        active.remove(tid); visited.add(tid)
    for tid in ids: visit(tid)
    return tasks


def prepare(task_file, runtime_file, out):
    tasks = validate_tasks(rows(task_file)); runtime = load(runtime_file)
    if runtime.get('execution_ready') is not True:
        raise ValueError('runtime not admitted; resolve paths and real DEV gates')
    source = source_identity(runtime['source_root'])
    for name, env in runtime['environments'].items():
        python = Path(env['python']).expanduser().absolute()  # preserve venv symlink!
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError(f'invalid Python executable: {name}')
        env['python'] = str(python)
        if env.get('admission'):
            admission=load(checked_path(env['admission']))
            if admission.get('passed') is not True:
                raise ValueError('environment admission has not passed')
        flags = env.get('sbatch_args', [])
        if any(x.startswith(('--array', '--wrap')) or (x.startswith('-a') and not x.startswith('--')) for x in flags):
            raise ValueError('only ordinary jobs, no arrays or wrap overrides')
    parameter_files = {}
    def seal_parameters(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in ('checkpoint_manifest','approval_file') and v:
                    r=receipt(v);parameter_files[r['path']]=r['sha256']
                else:seal_parameters(v)
        elif isinstance(value,list):
            for v in value:seal_parameters(v)
    for t in tasks:
        seal_parameters(t['params'])
        if t['environment'] not in runtime['environments']:
            raise ValueError(f"missing environment: {t['environment']}")
    root = Path(out).absolute(); root.mkdir(parents=True, exist_ok=False)
    for t in tasks: save(root/'tasks'/f"{t['id']}.json", t)
    save(root/'plan.json', {'schema': 1, 'source': source, 'runtime': runtime,
        'task_ids': [t['id'] for t in tasks], 'tasks_sha256': digest(tasks),
        'task_source': receipt(task_file), 'runtime_source': receipt(runtime_file),
        'parameter_files': parameter_files, 'scientific_results': False})
    return {'tasks': len(tasks), 'bundle': str(root)}


def state(root, tid):
    p = Path(root)/'results'/tid/'result.json'
    return load(p) if p.exists() else None


def resolve_inputs(root, task):
    values = {}
    for name, item in task.get('inputs', {}).items():
        if 'task' in item:
            parent = state(root, item['task'])
            if parent is None: raise RuntimeError('dependency not complete')
            if parent['status'] != 'COMPLETE': raise LookupError('dependency failed')
            values[name] = str(checked_path(parent['artifacts'][item['artifact']]))
        else:
            values[name] = str(checked_path(item))
    return values


def worker(bundle, tid):
    root = Path(bundle).absolute(); plan = load(root/'plan.json')
    task = load(root/'tasks'/f'{safe_id(tid)}.json')
    if source_identity(plan['source']['root']) != plan['source']:
        raise ValueError('source changed since preparation')
    from .core import sha
    for path,h in plan.get('parameter_files',{}).items():
        if sha(path)!=h: raise ValueError('model/approval manifest changed after preparation')
    # A second worker never overwrites or repeats a completed/in-flight attempt.
    work = root/'results'/tid; work.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'task_id': tid, 'task_sha256': digest(task), 'source': plan['source'],
              'artifacts': {}, 'status': 'ENGINEERING_FAILED', 'job_id': os.getenv('SLURM_JOB_ID')}
    try:
        inputs = resolve_inputs(root, task)
        from .adapters import execute
        artifacts = execute(task, inputs, work, plan['runtime'])
        if not artifacts: raise ValueError('no scientific artifacts returned')
        result['artifacts'] = {safe_id(k): receipt(v) for k, v in artifacts.items()}
        result['status'] = 'COMPLETE'
        result['component_execution_only'] = True
    except LookupError:
        result.update(status='BLOCKED_DEPENDENCY', error='upstream task did not complete')
    except Exception as exc:
        # Never serialize provider responses, API headers or secrets.
        result.update(error_type=type(exc).__name__, error=str(exc)[:1000] if task['params'].get('backend') != 'gemini' else 'provider task failed; inspect local sanitized log')
    result['wall_s'] = time.monotonic() - started
    save(work/'result.json', result)
    if result['status'] != 'COMPLETE': raise RuntimeError(f"{tid}: {result['status']}")
    return result


def launch(bundle, *, submit=False, max_jobs=1):
    root = Path(bundle).absolute(); plan = load(root/'plan.json')
    if source_identity(plan['source']['root']) != plan['source']:
        raise ValueError('source changed since preparation')
    tasks = [load(root/'tasks'/f'{tid}.json') for tid in plan['task_ids']]
    if digest(tasks) != plan['tasks_sha256']: raise ValueError('sealed task edited')
    cap = int(plan['runtime'].get('max_active_jobs', 4))
    if max_jobs < 1 or cap < 1: raise ValueError('positive job limits required')
    reports = []
    lockpath = root/'.submit.lock'
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Query scheduler instead of inferring job liveness from files.
        if submit:
            output = subprocess.check_output(['squeue', '--noheader', '--user', os.environ['USER'], '--format=%A'], text=True)
            active = set(output.split())
            owned = {str(load(p).get('job_id')) for p in (root/'jobs').glob('*/submission.json')}
            slots = min(max_jobs, max(0, cap - len(active & owned)))
        else: slots = max_jobs
        for task in tasks:
            if len(reports) >= slots: break
            tid = task['id']; job = root/'jobs'/tid
            if state(root, tid) or (job/'intent.json').exists(): continue
            parents = [state(root, x['task']) for x in task.get('inputs', {}).values() if 'task' in x]
            if any(p is None or p['status'] != 'COMPLETE' for p in parents): continue
            env = plan['runtime']['environments'][task['environment']]
            argv = [env['python'], '-m', 'robo.campaign', 'worker', '--bundle', str(root), '--task', tid]
            flags = env.get('sbatch_args', [])
            script = job/'run.sbatch'
            cmd = ['sbatch', '--parsable', *flags, f'--job-name=sar-{tid[:36]}',
                   f'--output={job}/slurm-%j.log', str(script)]
            reports.append({'task': tid, 'command': cmd, 'worker': argv, 'submit': submit})
            if submit:
                job.mkdir(parents=True, exist_ok=False)
                body = '#!/usr/bin/env bash\nset -euo pipefail\n'
                body += 'cd ' + shlex.quote(plan['source']['root']) + '\n'
                for k, v in env.get('env', {}).items():
                    if not k.replace('_', '').isalnum() or any(s in k.upper() for s in ('TOKEN', 'SECRET', 'KEY')):
                        raise ValueError('environment key unsafe or contains credentials')
                    body += 'export ' + k + '=' + shlex.quote(str(v)) + '\n'
                if env.get('setup_script'): body += 'source ' + shlex.quote(str(checked_path(env['setup_script']))) + '\n'
                body += 'exec ' + shlex.join(argv) + '\n'
                script.write_text(body)
                save(job/'intent.json', reports[-1])
                run = subprocess.run(cmd, capture_output=True, text=True)
                save(job/'submission.json', {'returncode': run.returncode, 'stdout': run.stdout,
                    'stderr': run.stderr, 'job_id': run.stdout.strip().split(';')[0] if run.returncode == 0 else None})
                if run.returncode: raise RuntimeError('submission failed; intent retained, query scheduler before retry')
    return reports


def collect(bundle, out):
    root = Path(bundle); plan = load(root/'plan.json'); report = []
    for tid in plan['task_ids']:
        r = state(root, tid)
        if r:
            for a in r['artifacts'].values(): checked_path(a)
            report.append(r)
        else:
            report.append({'task_id': tid, 'status': 'UNMEASURED', 'artifacts': {}})
    save(Path(out)/'tasks.jsonl', report, jsonl=True)
    counts = {s: sum(r['status'] == s for r in report) for s in sorted({r['status'] for r in report})}
    summary = {'counts': counts, 'planned': len(report), 'all_complete': counts.get('COMPLETE', 0) == len(report),
               'interpretation': 'component execution only; NOT a manipulation success table', 'plan': receipt(root/'plan.json')}
    save(Path(out)/'summary.json', summary)
    return summary


def worker_batch(bundle, task_ids):
    """One allocated process, multiple independent model tasks, cached model weights.

    Never batch jobs already submitted by the ordinary launcher. Native paired
    experiments are excluded because they own their process-level policy state.
    """
    root=Path(bundle);report=[]
    for tid in task_ids:
        task=load(root/'tasks'/f'{safe_id(tid)}.json')
        if task['kind'] not in ('generator','inpaint'):raise ValueError('batch only stateless generator/inpainting tasks')
        if (root/'jobs'/tid/'intent.json').exists():raise ValueError('task already submitted; avoid duplicate execution')
    signatures={(load(root/'tasks'/f'{tid}.json')['environment'],
                 load(root/'tasks'/f'{tid}.json')['params']['backend']) for tid in task_ids}
    if len(signatures)!=1:raise ValueError('batch must share one model environment/backend')
    for tid in task_ids:
        try: report.append(worker(root,tid))
        except Exception as exc:
            r=state(root,tid)
            report.append(r or {'task_id':tid,'status':'ENGINEERING_FAILED','error_type':type(exc).__name__})
    return report
