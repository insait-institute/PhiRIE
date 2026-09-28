"""Build E7 diagnostic records from the complete historical launch roster.

This is a record adapter, not a reconstruction or metric implementation. Legacy
artifacts have no original frozen build/checkpoint manifest and cannot be promoted
by hashing them today. Aggregation remains in ``real_world_metrics``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

import yaml
import numpy as np

from robo.eval.metric_utils import write_csv, write_json
from robo.eval.real_world_metrics import generate


class RecordError(ValueError):
    pass


def fingerprint(path):
    path = Path(path)
    result = {"path": str(path.resolve()), "exists": path.is_file()}
    if path.is_file():
        before = path.stat()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RecordError(f"artifact changed during hashing: {path}")
        result.update(sha256=digest.hexdigest(), bytes=after.st_size)
    return result


def _json(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None


def _finite(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise RecordError('metric must be finite and nonnegative')
    return value


def validate_alignment(record):
    """Independent values require explicit disjoint construction/selection IDs."""
    fields = ('translation_cm', 'rotation_deg')
    if not any(record.get(key) is not None for key in fields):
        return
    evidence = record.get('independent_alignment', {})
    heldout = set(evidence.get('held_out_ids', []))
    used = set(evidence.get('construction_ids', [])) | set(evidence.get('selection_ids', []))
    if not heldout or not used or heldout & used or not evidence.get('reference_sha256'):
        raise RecordError('alignment requires disjoint held-out reference evidence')
    for key in fields:
        _finite(record[key])


def strict_full_build(checks):
    required = ('metric_reconstruction', 'discovery', 'accepted_movable',
                'registered_assets', 'collision_physics', 'background',
                'full_room_collision', 'robot_alignment_or_not_applicable',
                'simulator_load_and_settle', 'immutable_build_manifest')
    return all(checks.get(key) is True for key in required)


def load_roster(config, repo_root):
    if config.get('mode') != 'legacy_audit' or config.get('paper_ready') is not False:
        raise RecordError('legacy import cannot be promoted to a paper-ready experiment')
    path = Path(repo_root) / config['droid']['roster_path']
    if fingerprint(path)['sha256'] != config['droid']['roster_sha256']:
        raise RecordError('original roster hash mismatch')
    roster = yaml.safe_load(path.read_text())
    episodes = [episode for batch in config['droid']['batches'] for episode in roster[batch]]
    jobs = config['droid']['launches']
    if len(episodes) != len(set(episodes)) or [j['episode'] for j in jobs] != episodes:
        raise RecordError('launch roster must preserve every original batch entry in order')
    captures = []
    for job in jobs:
        captures.append(dict(job, source='droid', capture_id=job['episode']))
    captures.extend(dict(c, source='phone') for c in config['phone']['captures'])
    ids = [(c['source'], c['capture_id']) for c in captures]
    workspaces = [c['workspace_id'] for c in captures]
    if len(ids) != len(set(ids)) or len(workspaces) != len(set(workspaces)):
        raise RecordError('duplicate capture or workspace')
    return captures


def build_record(capture, config, freeze_id):
    root = Path(config['artifact_root'])
    scene = capture['scene_id']
    output = root / 'outputs' / capture['workspace_id']
    recon_root = root / 'data/recon_scenes'
    evidence = {}

    def observe(key, path):
        evidence[key] = fingerprint(path)
        return evidence[key]['exists']

    logs = []
    for job_id in capture['job_ids']:
        prefix = 'droid_recon' if capture['source'] == 'droid' else 'video2sim'
        path = root / 'outputs' / f'{prefix}_{job_id}.log'
        if not observe(f'launch_{job_id}', path):
            raise RecordError(f'missing launch identity evidence: {path}')
        body = path.read_text(errors='replace')
        expected_input = (str(Path(config['raw_root']) / capture['episode'])
                          if capture['source'] == 'droid' else str(root / capture['video']))
        header = f'{prefix}: {expected_input} -> scene={scene} out={output}'
        if body.splitlines()[0] != header:
            raise RecordError(f'launch identity mismatch: {path}')
        logs.append({'job_id': job_id, 'done': 'DONE ->' in body,
                     'errors': [line for line in body.splitlines()
                                if 'ABORT:' in line or line.startswith('Traceback')],
                     'sha256': evidence[f'launch_{job_id}']['sha256']})
    inputs = ([Path(config['raw_root']) / capture['episode'] / 'trajectory.h5'] +
              sorted((Path(config['raw_root']) / capture['episode']).glob('metadata_*.json')) +
              sorted((Path(config['raw_root']) / capture['episode'] / 'recordings/MP4').glob('*.mp4'))
              if capture['source'] == 'droid' else [root / capture['video']])
    for index, path in enumerate(inputs):
        observe(f'input_{index}', path)
    report_path = output / 'report.json'
    observe('report', report_path)
    report = _json(report_path)
    accepted = None
    if report is not None:
        accepted = sum(o.get('tier') in {'A', 'B'} and not o.get('rejected')
                       for o in report.get('objects', []))
        if accepted != int(report.get('tier_A', 0)) + int(report.get('tier_B', 0)):
            raise RecordError(f'accepted-object denominator mismatch: {report_path}')
    metric_path = output / 'recon' / ('recon.npz' if capture['source'] == 'droid' else 'recon_metric.npz')
    metric_exists = observe('metric_reconstruction', metric_path)
    camera_schema_valid = False
    if metric_exists:
        with np.load(metric_path, allow_pickle=False) as archive:
            poses, intrinsics = archive['w2c'], archive['K']
            camera_schema_valid = (poses.ndim == 3 and poses.shape[1:] == (4, 4)
                                   and len(poses) > 0 and intrinsics.shape == (3, 3)
                                   and np.isfinite(poses).all() and np.isfinite(intrinsics).all())
    transforms = recon_root / 'data' / scene / 'dslr/nerfstudio/transforms_undistorted.json'
    observe('camera_trajectory', transforms)
    transform_data = _json(transforms)
    splat = observe('gaussian', recon_root / 'splats' / f'{scene}.ply')
    train_report = recon_root / 'splats' / f'{scene}_train_report.json'
    observe('gaussian_training', train_report)
    train = _json(train_report)
    alignment = output / 'recon/align_report.json'
    observe('construction_alignment', alignment)
    aligned = _json(alignment)
    metric_ready = (metric_exists and camera_schema_valid and bool(transform_data and transform_data.get('frames'))
                    and splat and bool(train and train.get('n_gaussians', 0) > 0)
                    and (capture['source'] != 'droid' or aligned is not None))
    observe('scene_xml', output / 'sim_export/scene.xml')
    observe('settle', output / 'sim_export/mujoco_settle.json')
    settle = _json(output / 'sim_export/mujoco_settle.json')
    observe('legacy_alignment', output / 'recon/held_out_eval.json')
    legacy = _json(output / 'recon/held_out_eval.json')
    heldout = legacy.get('held_out_metrics', {}) if legacy else {}
    # Existing evaluator fits offsets on all FK frames before splitting; the
    # original scene transform also used all FK. Never emit it as independent.
    legacy_translation = 100 * _finite(heldout['center_rms_m']) if heldout else None
    legacy_rotation = _finite(heldout['rotation_residual_deg']['median']) if heldout else None
    observe('timings', output / 'timings.txt')
    timing_path = output / 'timings.txt'
    timing_rows = []
    if timing_path.is_file():
        for line in timing_path.read_text().splitlines():
            if line.strip():
                match = re.fullmatch(r'(.+) ([0-9]+)', line)
                if not match:
                    raise RecordError(f'malformed stage timing: {timing_path}')
                timing_rows.append({'stage': match[1], 'seconds': int(match[2])})
    checks = {'metric_reconstruction': bool(metric_ready), 'discovery': report is not None,
              'accepted_movable': accepted is not None and accepted > 0,
              'registered_assets': False, 'collision_physics': False,
              'background': False, 'full_room_collision': False,
              'robot_alignment_or_not_applicable': capture['source'] == 'phone',
              'simulator_load_and_settle': bool(settle and settle.get('n', 0) > 0
                                                and settle.get('stable_3cm') == settle.get('n')),
              'immutable_build_manifest': False}
    failures = [key for key, passed in checks.items() if not passed]
    row = {'freeze_id': freeze_id, 'source': capture['source'],
           'workspace_id': capture['workspace_id'], 'capture_id': capture['capture_id'],
           'paper_ready': False, 'scope': 'legacy_artifact_diagnostic',
           'reconstruction_success': bool(metric_ready),
           'full_build_success': strict_full_build(checks), 'accepted_objects': accepted,
           'input_frames': len(transform_data['frames']) if transform_data else None,
           'translation_cm': None, 'rotation_deg': None, 'alignment_pass': None,
           'runtime_minutes': None,
           'observed_stage_minutes': sum(t['seconds'] for t in timing_rows) / 60 if timing_rows else None,
           'runtime_complete': False, 'build_manifest_path': None,
           'capture_manifest_path': f'captures/{capture["workspace_id"]}.json',
           'failure_stage': failures[0] if failures else None,
           'failure_reason': ';'.join(failures),
           'legacy_alignment_translation_cm': legacy_translation,
           'legacy_alignment_rotation_deg': legacy_rotation,
           'legacy_alignment_independent': False,
           'settle_tested': settle.get('n') if settle else None,
           'settle_stable': settle.get('stable_3cm') if settle else None}
    validate_alignment(row)
    manifest = {'schema_version': 1, 'paper_ready': False, 'capture': capture, 'row': row,
                'checks': checks, 'evidence': evidence, 'launch_attempts': logs,
                'timing_rows': timing_rows,
                'limitations': ['original collective pre-outcome roster not independently timestamped',
                                'original source/config/checkpoint hashes and immutable build manifest absent',
                                'current snapshots do not establish reusable original-generation provenance',
                                'timings may combine resumed attempts and omit aborted terminal stage',
                                'legacy alignment offset selection and construction used all reference frames',
                                'unverified strict criteria are false, not inferred from file existence']}
    return row, manifest


def run(config_path, out_dir, freeze_id, repo_root):
    config_path, repo_root, out_dir = Path(config_path), Path(repo_root), Path(out_dir)
    config = yaml.safe_load(config_path.read_text())
    if out_dir.exists():
        raise RecordError(f'refusing to overwrite existing audit: {out_dir}')
    contract_path = out_dir.parent / 'contract/freeze_manifest.json'
    contract = _json(contract_path)
    commit = subprocess.check_output(['git', '-C', str(repo_root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo_root), 'status', '--porcelain'], text=True).strip()
    if dirty:
        raise RecordError('record generation requires committed clean source')
    if not contract or contract.get('freeze_id') != freeze_id or contract.get('code', {}).get('commit') != commit:
        raise RecordError('matching E0 source/freeze contract required')
    frozen_config = [c for c in contract.get('configs', []) if c['field'] == 'real_world_config']
    if len(frozen_config) != 1 or frozen_config[0]['source_content_sha256'] != fingerprint(config_path)['sha256']:
        raise RecordError('real-world config differs from E0 frozen source')
    captures = load_roster(config, repo_root)
    lock = out_dir.parent / f'.{out_dir.name}.reservation'
    lock.mkdir()  # Exclusive; failures retain the reservation and partial logs.
    stage = out_dir.parent / f'.{out_dir.name}.{uuid.uuid4().hex}.partial'
    stage.mkdir(parents=True)
    try:
        rows = []
        for capture in captures:
            row, manifest = build_record(capture, config, freeze_id)
            rows.append(row)
            write_json(stage / row['capture_manifest_path'], manifest)
        write_json(stage / 'workspaces.json', {'paper_ready': False, 'rows': rows})
        write_csv(stage / 'workspaces.csv', rows)
        result = generate(stage / 'workspaces.json', None, stage)
        result.update(freeze_id=freeze_id, source_commit=commit, paper_ready=False,
                      config=fingerprint(config_path), roster=fingerprint(repo_root / config['droid']['roster_path']),
                      phone_required=config['phone']['minimum_rooms'],
                      phone_available=sum(r['source'] == 'phone' for r in rows),
                      limitations=['diagnostic snapshots only; no prospective full experiment',
                                   'phone capture set below minimum; independent alignment unavailable'])
        write_json(stage / 'real_world_table.json', result)
        if out_dir.exists():
            raise RecordError(f'refusing to overwrite concurrently created audit: {out_dir}')
        os.rename(stage, out_dir)
    except BaseException:
        # Retain a failed staging directory with its evidence; never expose it
        # under the completed output name or silently rerun completed units.
        raise
    return result


def _runtime_identity(python, packages):
    """Pin interpreter bytes and installed distributions for this CPU stage."""
    from agents.recon.droid_extract import prospective_environment
    python = str(Path(python).absolute())
    command = [python, '-c', 'import json,sys;from agents.recon.droid_extract import runtime_manifest;print("E7_RUNTIME_JSON="+json.dumps(runtime_manifest(sys.argv[1:])))', *packages]
    code = Path(__file__).resolve().parents[2]
    output = subprocess.check_output(command, text=True, cwd=code, env=prospective_environment(code))
    payloads = [s[len('E7_RUNTIME_JSON='):] for s in output.splitlines() if s.startswith('E7_RUNTIME_JSON=')]
    if len(payloads) != 1:
        raise RecordError('runtime probe must emit exactly one package closure')
    result = json.loads(payloads[0])
    if result['invocation'] != python:
        raise RecordError('runtime invocation differs')
    return result


def prepare_prospective(config_path, destination, freeze_id, repo_root, h5_python, sfm_python, *, tier='pilot_then_full'):
    """Create metadata plans before any reconstruction or pose-value read."""
    from agents.recon.colmap_poses import write_new_json
    from agents.recon.droid_extract import input_identity, prospective_environment
    from robo.eval.freeze import CANONICAL_FREEZE_ID
    if tier not in {'pilot', 'pilot_then_full'} or not CANONICAL_FREEZE_ID.fullmatch(freeze_id):
        raise RecordError('canonical prospective freeze/tier required')
    repo_root, destination = Path(repo_root).resolve(), Path(destination).absolute()
    original = yaml.safe_load(Path(config_path).read_text())
    captures = [r for r in load_roster(original, repo_root) if r['source'] == 'droid']
    if len(captures) != 10:
        raise RecordError('prospective DROID cohort must retain original10 attempts')
    runtime = {'h5': _runtime_identity(h5_python, ['h5py', 'numpy']),
               'sfm': _runtime_identity(sfm_python, ['pycolmap', 'numpy']),
               'ffmpeg': input_identity('/usr/bin/ffmpeg'), 'ffprobe': input_identity('/usr/bin/ffprobe'),
               'taskset': input_identity('/usr/bin/taskset')}
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for capture in captures:
        path = destination / (capture['workspace_id'] + '.plan.json')
        command = [str(h5_python), '-m', 'agents.recon.droid_extract', '--episode',
                   str(Path(original['raw_root']) / capture['episode']), '--prepare-plan', str(path)]
        subprocess.run(command, cwd=repo_root, env=prospective_environment(repo_root), check=True)
        rows.append({**capture, 'plan': input_identity(path)})
    config = {'schema_version': 1, 'mode': 'prospective_cpu_alignment', 'freeze_id': freeze_id,
              'paper_ready': False, 'tier': tier, 'pose_backend': 'colmap', 'fallback_allowed': False,
              'seed': 0, 'cpu_threads': 8, 'source_commit': 'inherit',
              'original_config': input_identity(config_path), 'runtime': runtime, 'captures': rows,
              'execution_workspace': rows[0]['workspace_id'],
              'gs_training': 'NOT_RUN', 'full_build': 'NOT_RUN', 'phone_additional_capture_blocker': 2}
    write_new_json(destination / 'execution.json', config)
    return config


def validate_prospective(config_path, stage_root, repo_root):
    from agents.recon.droid_extract import input_identity, verify_input, validate_frame_plan
    from robo.manifest.hash import canonical_hash, git_snapshot
    from robo.eval.freeze import CANONICAL_FREEZE_ID
    repo_root, stage_root = Path(repo_root).resolve(), Path(stage_root).absolute()
    config = json.loads(Path(config_path).read_text())
    code = git_snapshot(repo_root)
    if code['dirty'] or code['commit'] == 'nogit':
        raise RecordError('prospective stage requires clean committed source')
    if (config['mode'] != 'prospective_cpu_alignment' or config['paper_ready'] is not False
            or config['pose_backend'] != 'colmap' or config['fallback_allowed'] is not False
            or config['seed'] != 0 or config['cpu_threads'] != 8 or config['source_commit'] != 'inherit'
            or config['gs_training'] != 'NOT_RUN' or config['full_build'] != 'NOT_RUN'
            or config['tier'] not in {'pilot', 'pilot_then_full'}):
        raise RecordError('prospective constructor/tier differs')
    if stage_root.name != config['freeze_id'] or not CANONICAL_FREEZE_ID.fullmatch(stage_root.name):
        raise RecordError('prospective freeze mismatch')
    from robo.eval.agentic_ablation import checked_repo_path
    checked_repo_path(stage_root, 'E7 stage root', must_exist=True)
    contract = json.loads((stage_root / 'contract/freeze_manifest.json').read_text())
    payload = {k: v for k, v in contract.items() if k not in {'created_utc', 'environment', 'contract_sha256'}}
    if (canonical_hash(payload) != contract['contract_sha256'] or contract['freeze_id'] != stage_root.name
            or contract['code']['commit'] != code['commit'] or contract['code']['dirty']):
        raise RecordError('E0 digest/source differs')
    bound = [x for x in contract['configs'] if x['field'] == 'real_world_config']
    if len(bound) != 1 or bound[0]['source_content_sha256'] != input_identity(config_path)['sha256']:
        raise RecordError('prospective config not E0 bound')
    original = yaml.safe_load(verify_input(config['original_config']).read_text())
    captures = [c for c in load_roster(original, repo_root) if c['source'] == 'droid']
    if len(captures) != 10 or [{k: v for k, v in c.items() if k != 'plan'} for c in config['captures']] != captures:
        raise RecordError('complete original ten-episode roster differs')
    expected_workspace = captures[0]['workspace_id']
    if config['execution_workspace'] != expected_workspace:
        raise RecordError('prospective pilot selection changed')
    for row in config['captures']:
        plan = validate_frame_plan(json.loads(verify_input(row['plan']).read_text()))
        if plan['episode'] != str(Path(original['raw_root']) / row['episode']):
            raise RecordError('episode/plan mismatch')
    runtime = config['runtime']
    for key, packages in [('h5', ['h5py', 'numpy']), ('sfm', ['pycolmap', 'numpy'])]:
        if _runtime_identity(runtime[key]['invocation'], packages) != runtime[key]:
            raise RecordError('Python/package runtime changed')
    for key in ('ffmpeg', 'ffprobe', 'taskset'):
        verify_input(runtime[key])
    return config, contract


def validate_prospective_unit(result_path, capture, contract, config):
    from agents.recon.droid_extract import verify_input, input_identity
    from agents.eval.droid_alignment_eval import evaluate_sealed_fit
    result_path = Path(result_path)
    result = json.loads(result_path.read_text())
    if (result['capture'] != capture or result['contract_sha256'] != contract['contract_sha256']
            or result['source_commit'] != contract['code']['commit']
            or result['freeze_id'] != contract['freeze_id']
            or result['status'] not in {'COMPLETE', 'FAILED'}
            or result['reconstruction_success'] is not False or result['full_build_success'] is not False
            or result['gaussian_training_status'] != 'NOT_RUN' or result['full_build_status'] != 'NOT_RUN'):
        raise RecordError('workspace result source/roster/stage mismatch')
    if (result['status'] == 'FAILED') != (result['failure'] is not None):
        raise RecordError('terminal failure/status mismatch')
    attempt = json.loads((result_path.parent / 'attempt.json').read_text())
    frozen_config = [c for c in contract['configs'] if c['field'] == 'real_world_config']
    if (len(frozen_config) != 1 or attempt['config']['sha256'] != frozen_config[0]['source_content_sha256']
            or attempt['source_commit'] != contract['code']['commit']
            or attempt['contract_sha256'] != contract['contract_sha256']
            or attempt['workspace_id'] != capture['workspace_id'] or attempt['runtime'] != config['runtime']):
        raise RecordError('workspace attempt config/runtime/source differs')
    verify_input(attempt['config'])
    for ref in result['artifacts'].values():
        verify_input(ref)
    expected = {str(p.relative_to(result_path.parent)): input_identity(p)
                for p in result_path.parent.rglob('*') if p.is_file() and p != result_path}
    if result['artifacts'] != expected:
        raise RecordError('workspace artifact population/path changed')
    if result['alignment_evaluation'] is not None:
        actual = json.loads((result_path.parent / 'alignment_evaluation.json').read_text())
        replay = evaluate_sealed_fit(result_path.parent / 'alignment', result_path.parent / 'held_out_reference.json')
        if actual != replay or actual != result['alignment_evaluation']:
            raise RecordError('independent alignment metric replay differs')
    elif result['status'] == 'COMPLETE':
        raise RecordError('completed CPU stage lacks independent evaluation')
    return result


def require_cpu_pilot(config, contract, stage_root, workspace_id):
    """Gate remaining preregistered units without rerunning the pilot."""
    pilot = config['captures'][0]
    if workspace_id == pilot['workspace_id']:
        return
    if config['tier'] != 'pilot_then_full':
        raise RecordError('workspace outside frozen pilot tier')
    path = Path(stage_root) / 'real_world/workspaces' / pilot['workspace_id'] / 'result.json'
    result = validate_prospective_unit(path, pilot, contract, config)
    if result['status'] != 'COMPLETE' or not result['alignment_evaluation']['held_out_metrics']:
        raise RecordError('full CPU release requires completed first-IPRL format/coverage pilot')


def run_prospective_workspace(config_path, stage_root, workspace_id, repo_root):
    """Orchestrate canonical CPU producers; retain each attempted failure."""
    import time
    import socket
    from agents.recon.colmap_poses import write_new_json
    from agents.recon.droid_extract import input_identity, verify_input, prospective_environment
    from agents.recon.align_to_traj import validate_train_fit
    config, contract = validate_prospective(config_path, stage_root, repo_root)
    selected = [r for r in config['captures'] if r['workspace_id'] == workspace_id]
    if len(selected) != 1 or (config['tier'] == 'pilot' and workspace_id != config['execution_workspace']):
        raise RecordError('workspace outside frozen execution tier')
    require_cpu_pilot(config, contract, stage_root, workspace_id)
    row = selected[0]; out = Path(stage_root) / 'real_world/workspaces' / workspace_id
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); commands = []; current = 'extract'
    runtime = config['runtime']; h5 = runtime['h5']['invocation']; py = runtime['sfm']['invocation']
    plan = row['plan']['path']; extract = out / 'extraction'; fit = out / 'alignment'
    env = prospective_environment(repo_root)
    affinity = sorted(os.sched_getaffinity(0))[:config['cpu_threads']]
    if len(affinity) < config['cpu_threads']:
        raise RecordError('prospective stage requires8 allocated CPU cores')
    write_new_json(out / 'attempt.json', {'config': input_identity(config_path),
        'source_commit': contract['code']['commit'], 'contract_sha256': contract['contract_sha256'],
        'workspace_id': workspace_id, 'hostname': socket.gethostname(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'cpu_affinity': affinity, 'runtime': runtime, 'started_unix': time.time()})

    def execute(stage, command):
        nonlocal current
        current = stage; before = time.monotonic()
        # Affinity is set only in the child; libraries cannot exceed allocated CPUs.
        prefix = [runtime['taskset']['path'], '-c', ','.join(map(str, affinity))]
        with (out / (stage + '.log')).open('xb') as log:
            result = subprocess.run(prefix + command, cwd=repo_root, env=env, stdout=log, stderr=subprocess.STDOUT)
        commands.append({'stage': stage, 'command': prefix + command, 'returncode': result.returncode,
                         'wall_s': time.monotonic() - before, 'log': input_identity(out / (stage + '.log'))})
        write_new_json(out / (stage + '.execution.json'), commands[-1])
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)

    failure = None; evaluation = None
    try:
        execute('extract', [h5, '-m', 'agents.recon.droid_extract', '--frame-plan', plan, '--prospective-out', str(extract)])
        extraction = json.loads((extract / 'extraction_manifest.json').read_text())
        if extraction['plan'] != row['plan']:
            raise RecordError('extraction plan differs')
        for ref in [extraction['train_trajectory'], *extraction['frames']]:
            verify_input(ref)
        execute('sfm', [py, '-c', 'import pycolmap;pycolmap.set_random_seed(0);from agents.recon.colmap_poses import main;main()',
            '--images-dir', str(extract / 'wrist_frames'), '--out', str(out / 'recon.npz'), '--workdir', str(out / 'colmap'),
            '--num-threads', str(config['cpu_threads'])])
        execute('train_fit', [py, '-m', 'agents.recon.align_to_traj', '--recon', str(out / 'recon.npz'),
            '--traj', str(extract / 'train_trajectory.json'), '--frame-plan', plan, '--out', str(fit)])
        validate_train_fit(fit)
        execute('reference', [h5, '-m', 'agents.recon.droid_extract', '--frame-plan', plan,
            '--reference-fit', str(fit), '--fit-validator-python', py,
            '--prospective-out', str(out / 'held_out_reference.json')])
        execute('evaluate', [py, '-m', 'agents.eval.droid_alignment_eval', '--sealed-fit', str(fit),
            '--traj', str(out / 'held_out_reference.json'), '--out', str(out / 'alignment_evaluation.json')])
        evaluation = json.loads((out / 'alignment_evaluation.json').read_text())
    except (subprocess.CalledProcessError, ValueError, OSError) as exc:
        message = (out / (current + '.log')).read_text(errors='replace') if (out / (current + '.log')).exists() else str(exc)
        classification = 'code_bug'
        if any(x in message for x in ('No module named', 'ImportError', 'cannot open shared object')):
            classification = 'environment_issue'
        elif any(x in message for x in ('MemoryError', 'out of memory', 'Killed')):
            classification = 'resource_failure'
        elif isinstance(exc, subprocess.CalledProcessError) and exc.returncode == -9:
            classification = 'resource_failure'
        elif any(x in message for x in ('mapping produced no reconstruction', 'of frames registered',
                'TRAIN center RMS exceeds', 'insufficient preregistered TRAIN', 'degenerate TRAIN')):
            classification = 'scientific_claim_failure'
        elif isinstance(exc, FileNotFoundError):
            classification = 'missing_data'
        failure = {'stage': current, 'type': type(exc).__name__, 'reason': str(exc),
                   'classification': classification}
    result = {'schema_version': 1, 'scope': 'prospective_cpu_alignment', 'freeze_id': config['freeze_id'],
        'source_commit': contract['code']['commit'], 'contract_sha256': contract['contract_sha256'],
        'capture': row, 'status': 'FAILED' if failure else 'COMPLETE', 'failure': failure,
        'runtime_seconds': time.monotonic() - start, 'runtime_scope': 'new_cpu_sfm_alignment_evaluation_only',
        'commands': commands, 'alignment_evaluation': evaluation,
        'reconstruction_success': False, 'full_build_success': False,
        'gaussian_training_status': 'NOT_RUN', 'full_build_status': 'NOT_RUN',
        'artifacts': {str(p.relative_to(out)): input_identity(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    write_new_json(out / 'result.json', result)
    return result


def summarize_prospective(config_path, stage_root, destination, repo_root):
    from agents.recon.colmap_poses import write_new_json
    from agents.recon.droid_extract import verify_input, input_identity
    config, contract = validate_prospective(config_path, stage_root, repo_root)
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for capture in config['captures']:
        result_path = Path(stage_root) / 'real_world/workspaces' / capture['workspace_id'] / 'result.json'
        result = _json(result_path)
        if result:
            result = validate_prospective_unit(result_path, capture, contract, config)
        metrics = result['alignment_evaluation']['held_out_metrics'] if result and result['alignment_evaluation'] else None
        plan = json.loads(Path(capture['plan']['path']).read_text())
        row = {'source': 'droid', 'workspace_id': capture['workspace_id'], 'freeze_id': config['freeze_id'],
            'capture_id': capture['capture_id'], 'source_commit': contract['code']['commit'],
            'execution_status': result['status'] if result else 'NOT_RUN',
            'reconstruction_success': False, 'full_build_success': False, 'accepted_objects': None,
            'runtime_minutes': None, 'cpu_stage_runtime_seconds': result['runtime_seconds'] if result else None,
            'translation_cm': metrics['center_rms_m'] * 100 if metrics else None,
            'rotation_deg': metrics['rotation_residual_deg']['median'] if metrics else None,
            'alignment_pass': result['alignment_evaluation']['gate']['passed'] if metrics else None,
            'failure_reason': result['failure']['reason'] if result and result['failure'] else None,
            'planned_reference_frames': len(plan['split']['held_out_video_indices']),
            'evaluated_reference_frames': result['alignment_evaluation']['evaluated_reference_frames'] if result and result['alignment_evaluation'] else 0,
            'independent_alignment': {'held_out_ids': plan['split']['held_out_fk_indices'],
                'construction_ids': plan['split']['train_fk_indices'], 'selection_ids': plan['split']['train_fk_indices'],
                'reference_sha256': result['alignment_evaluation']['reference']['sha256'] if metrics else None},
            'result_identity': input_identity(result_path) if result else None}
        validate_alignment(row); rows.append(row)
    write_new_json(destination / 'workspaces.json', {'scope': 'prospective_cpu_alignment', 'paper_ready': False, 'rows': rows})
    write_csv(destination / 'workspaces.csv', rows)
    # Keep the established denominator and unit definitions in the sole aggregator.
    return generate(destination / 'workspaces.json', None, destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--freeze-id', required=True)
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--prospective-prepare', action='store_true')
    parser.add_argument('--prospective-workspace')
    parser.add_argument('--prospective-summary', action='store_true')
    parser.add_argument('--stage-root')
    parser.add_argument('--tier', choices=['pilot', 'pilot_then_full'], default='pilot_then_full')
    parser.add_argument('--h5-python', default=os.environ.get('SIMANY_H5_PY', sys.executable))
    parser.add_argument('--sfm-python', default=os.environ.get('SIMANY_PY', sys.executable))
    args = parser.parse_args(argv)
    if args.prospective_prepare:
        result = prepare_prospective(args.config, args.out, args.freeze_id, args.repo_root,
                                     args.h5_python, args.sfm_python, tier=args.tier)
    elif args.prospective_workspace:
        if not args.stage_root: parser.error('--stage-root required')
        result = run_prospective_workspace(args.config, args.stage_root, args.prospective_workspace, args.repo_root)
    elif args.prospective_summary:
        if not args.stage_root: parser.error('--stage-root required')
        result = summarize_prospective(args.config, args.stage_root, args.out, args.repo_root)
    else:
        result = run(args.config, args.out, args.freeze_id, args.repo_root)
    print(json.dumps(result, indent=2))
    if args.prospective_workspace and result['status'] == 'FAILED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
