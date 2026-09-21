"""Exact-source E0 launcher for fresh train-only Gaussian engineering artifacts."""
from __future__ import annotations
import argparse
import json
import importlib
import importlib.metadata
import os
import re
from pathlib import Path
import socket
import subprocess
import sys
import time
import yaml

CODE = Path(__file__).resolve().parents[2]
from agents.recon.colmap_poses import (file_identity, write_new_json, prepare_training_scene,
    triangulate_training_scene, validate_training_initialization, validate_training_scene)
from robo.manifest.hash import canonical_hash, git_snapshot, hash_checkpoint_path


def training_environment(config):
    env = os.environ.copy()
    env.pop('PYTHONHOME', None)
    env.update(PYTHONPATH=str(CODE) + os.pathsep + config['training_dependency_root'],
        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='42',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
        SIMANY_ROOT=str(CODE), SIMANY_SCENE=config['scene_id'], SIMANY_MESH_SRC='derived',
        TORCH_EXTENSIONS_DIR=config['torch_extensions_root'])
    return env


def runtime_identity():
    """Actual imports plus content anchors and installed-file stat fingerprints.

    Large package trees use the existing metadata fingerprint contract, not
    falsely described full-byte hashing. Entry points/RECORD/executable are
    content hashed; the scoped dependency directory is fully content hashed.
    """
    import site
    if site.ENABLE_USER_SITE or os.environ.get('PYTHONNOUSERSITE') != '1':
        raise ValueError('training runtime must disable user site')
    modules = {}
    for name, distribution in [('torch','torch'),('gsplat','gsplat'),('numpy','numpy'),
            ('scipy','scipy'),('PIL','Pillow'),('plyfile','plyfile'),('pydantic','pydantic'),
            ('pydantic_core','pydantic_core'),('yaml','PyYAML')]:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        dist = importlib.metadata.distribution(distribution)
        records = [p for p in dist.files if str(p).endswith('.dist-info/RECORD')]
        modules[name] = {'path':str(path), 'version':dist.version,
            'entrypoint_content':file_identity(path),
            'installed_tree_stat_fingerprint':hash_checkpoint_path(path.parent if path.name=='__init__.py' else path),
            'record_content':file_identity(dist.locate_file(records[0])) if records else None}
    # Exercise the exact validation/training imports without allocating CUDA.
    from agents.recon import gsplat_train
    return {'python':str(Path(sys.executable).resolve()), 'python_version':sys.version,
        'python_content':file_identity(Path(sys.executable).resolve()), 'modules':modules,
        'user_site_enabled':site.ENABLE_USER_SITE, 'iters':gsplat_train.ITERS}


def verify_runtime(config):
    raw = subprocess.check_output([config['training_python'], '-m',
        'run.icra2027.e3_gaussian_train_only', '--runtime-identity'], cwd=CODE,
        env=training_environment(config), text=True)
    observed = json.loads(raw)
    if observed != config['training_runtime_identity']:
        raise ValueError('training runtime imports/package fingerprints changed')
    dep = Path(config['training_dependency_root'])
    files = {str(p.relative_to(dep)):file_identity(p) for p in sorted(dep.rglob('*')) if p.is_file()}
    if canonical_hash(files) != config['training_dependency_tree_sha256']:
        raise ValueError('scoped Python dependency bytes changed')
    return observed


def cohort_scene_ids(config):
    """Pin the complete E1/E2 population; never enumerate available outputs."""
    populations = []
    for key in ('construction_roster', 'fidelity_roster'):
        spec = config[key]
        path = CODE / spec['path']
        path.relative_to(CODE)
        if path.resolve() != path or file_identity(path)['sha256'] != spec['sha256']:
            raise ValueError('cohort roster path/hash changed')
        population = yaml.safe_load(path.read_text())['population']
        scenes = [str(value) for value in population['scene_ids']]
        if (population['planned_scenes'] != 50 or len(scenes) != 50
                or len(set(scenes)) != 50
                or any(not re.fullmatch('[0-9a-f]{10}', scene) for scene in scenes)
                or population['preserve_failed_scenes'] is not True):
            raise ValueError('complete fifty-scene denominator required')
        populations.append(scenes)
    if populations[0] != populations[1]:
        raise ValueError('construction and fidelity scene rosters differ')
    return populations[0]


def validate_cohort_staging(config, manifest):
    if manifest['boundary']['max_train_frames'] is not None:
        raise ValueError('full population staging must retain all official TRAIN frames')
    if manifest.get('initialization_selection', {}).get('max_frames') != config['initialization_max_frames']:
        raise ValueError('full population initialization budget differs')
    dslr = (Path(config['dataset_root'])/'data'/config['scene_id']/'dslr').resolve()
    names = {'colmap/images.txt', 'nerfstudio/transforms_undistorted.json', 'train_test_lists.json'}
    if set(manifest['source_metadata']) != names or any(
            Path(value['path']) != dslr/name
            for name,value in manifest['source_metadata'].items()):
        raise ValueError('staged source metadata belongs to another planned scene')
    if Path(manifest['boundary']['split']['path']) != dslr/'train_test_lists.json':
        raise ValueError('staged split belongs to another planned scene')
    if any(Path(row['source_rgb']) != dslr/'resized_undistorted_images'/row['name']
           for row in manifest['frames']):
        raise ValueError('staged RGB belongs to another planned scene')


def gaussian_destination(freeze_root, config):
    parent = Path(freeze_root) / 'gaussian_train_only'
    return parent / config['scene_id'] if config.get('scope') == 'fresh_train_only_gaussian_population' else parent


def require_published_full_source(code):
    result = subprocess.run(['git', 'merge-base', '--is-ancestor', code['commit'],
                             'refs/remotes/origin/main'], cwd=CODE, capture_output=True)
    if result.returncode:
        raise ValueError('full execution requires a frozen source commit published on main')


def context(config_path, freeze_root, scene_id=None):
    config_path, freeze_root = Path(config_path).resolve(), Path(freeze_root).resolve()
    config = yaml.safe_load(config_path.read_text())
    population = config.get('scope') == 'fresh_train_only_gaussian_population'
    if config.get('paper_ready') is not False or config.get('scope') not in {
            'fresh_train_only_gaussian_engineering', 'fresh_train_only_gaussian_population'}:
        raise ValueError('Gaussian construction scope changed')
    if population:
        if scene_id not in cohort_scene_ids(config):
            raise ValueError('explicit scene from complete frozen population required')
        if 'scene_id' in config:
            raise ValueError('population config must not narrow the planned scene roster')
        config = dict(config, scene_id=scene_id)
    elif config['scene_id'] != '09c1414f1b' or scene_id is not None:
        raise ValueError('predeclared single-scene pilot changed')
    expected_limit = None if population else 48
    if population and (config.get('initialization_max_frames') != 48 or config.get('initialization_selection') != 'uniform_integer_endpoints_v1'):
        raise ValueError('full initialization protocol differs')
    if (config['max_train_frames'], config['iters'], config['smoke_iters'],
        config['holdout_every'], config['seed'], config['cpu_threads']) != (expected_limit, 15000, 16, 10, 42, 4):
        raise ValueError('predeclared Gaussian recipe or frame boundary changed')
    code = git_snapshot(CODE)
    if code['dirty'] or code['commit'] == 'nogit':
        raise ValueError('clean source required')
    contract = json.loads((freeze_root / 'contract/freeze_manifest.json').read_text())
    payload = {k:v for k,v in contract.items() if k not in {'created_utc','environment','contract_sha256'}}
    if canonical_hash(payload) != contract['contract_sha256']:
        raise ValueError('E0 contract digest differs')
    if contract['code']['commit'] != code['commit'] or contract['code']['dirty'] or contract['freeze_id'] != freeze_root.name:
        raise ValueError('exact-source E0 required')
    bound = [r for r in contract['resource_inventory'] if r['id'] == 'e3_gaussian_config']
    if len(bound) != 1 or bound[0]['sha256'] != file_identity(config_path)['sha256']:
        raise ValueError('Gaussian config not bound by E0')
    return config, code


def run(config_path, freeze_root, phase, scene_id=None):
    config, code = (context(config_path, freeze_root, scene_id) if scene_id is not None
                    else context(config_path, freeze_root))
    if config.get('scope') == 'fresh_train_only_gaussian_population' and phase == 'train-pilot':
        raise ValueError('population 15000-iteration units must use the published train-full phase')
    if phase == 'train-full':
        if config.get('scope') != 'fresh_train_only_gaussian_population':
            raise ValueError('full phase requires complete population config')
        require_published_full_source(code)
    destination = gaussian_destination(freeze_root, config)
    if destination.resolve() != destination.absolute():
        raise ValueError('Gaussian output cannot be a symlink')
    destination.mkdir(parents=True, exist_ok=True)
    receipt = destination / f'{phase}_receipt.json'
    lock = destination / f'{phase}.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    started = time.monotonic()
    report = {'phase': phase, 'code': code, 'paper_ready': False, 'hostname': socket.gethostname(),
              'pid': os.getpid(), 'python': sys.executable, 'job_id': os.environ.get('SLURM_JOB_ID'),
              'scene_id': config.get('scene_id'), 'scope': config.get('scope'),
              'config': file_identity(config_path), 'status': 'RUNNING'}
    try:
        if phase != 'prepare':
            preparation = json.loads((destination/'prepare_receipt.json').read_text())
            if preparation['status'] != 'PASS' or preparation['code'] != code or preparation['config'] != report['config']:
                raise ValueError('same-source/config successful preparation required')
            staged = validate_training_scene(destination/'scene', preparation['input_manifest']['sha256'])
            if config.get('scope') == 'fresh_train_only_gaussian_population':
                if preparation.get('scene_id') != config['scene_id']:
                    raise ValueError('preparation receipt belongs to another planned scene')
                if preparation.get('scope') != config['scope']:
                    raise ValueError('preparation scope differs')
                validate_cohort_staging(config, staged)
            report['prepare_receipt'] = file_identity(destination/'prepare_receipt.json')
        if phase == 'prepare':
            options = {'max_train_frames': config['max_train_frames']}
            if config.get('scope') == 'fresh_train_only_gaussian_population':
                options['initialization_max_frames'] = config['initialization_max_frames']
            result = prepare_training_scene(Path(config['dataset_root'])/'data'/config['scene_id'],
                destination/'scene', **options)
            report['input_manifest'] = file_identity(destination/'scene/training_inputs.json')
            report['frames'] = [r['name'] for r in result['frames']]
        elif phase == 'triangulate':
            report['result'] = triangulate_training_scene(destination/'scene', destination/'initialization',
                seed=config['seed'], num_threads=config['cpu_threads'])
            report['init_manifest'] = file_identity(destination/'initialization/init_manifest.json')
        else:
            init = destination/'initialization'
            triangulation = json.loads((destination/'triangulate_receipt.json').read_text())
            if triangulation['status'] != 'PASS' or triangulation['code'] != code or triangulation['config'] != report['config']:
                raise ValueError('same-source/config successful triangulation required')
            if config.get('scope') == 'fresh_train_only_gaussian_population' and (triangulation.get('scene_id') != config['scene_id'] or triangulation.get('scope') != config['scope']):
                raise ValueError('triangulation receipt belongs to another planned scene')
            if file_identity(init/'init_manifest.json') != triangulation['init_manifest']:
                raise ValueError('triangulation manifest differs from producer receipt')
            report['triangulate_receipt'] = file_identity(destination/'triangulate_receipt.json')
            validate_training_initialization(destination/'scene', init/'init_points.ply', init/'init_manifest.json')
            if phase in {'train-pilot', 'train-full'}:
                smoke = json.loads((destination/'train-smoke_receipt.json').read_text())
                if smoke['status'] != 'PASS' or smoke['code'] != code or smoke.get('config') != report['config']:
                    raise ValueError('full recipe requires exact-source smoke PASS')
                if config.get('scope') == 'fresh_train_only_gaussian_population' and (smoke.get('scene_id') != config['scene_id'] or smoke.get('scope') != config['scope']):
                    raise ValueError('smoke receipt belongs to another planned scene')
                for name, identity in smoke['artifacts'].items():
                    if file_identity(destination/'train-smoke'/name) != identity:
                        raise ValueError('smoke artifacts changed before full recipe')
            report['training_runtime'] = verify_runtime(config)
            node = os.environ.get('SLURMD_NODENAME', '')
            if not (node == 'hala' or node.startswith(('gcp', 'sof1'))):
                raise ValueError('GPU node outside authorized roster')
            gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version',
                                           '--format=csv,noheader'],text=True)
            report['gpu_inventory'] = gpu
            report['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
            if not report['cuda_visible_devices'] or ',' in report['cuda_visible_devices']:
                raise ValueError('exactly one allocated GPU required')
            report['slurm_allocation'] = subprocess.check_output(['scontrol','show','job','-dd',
                os.environ['SLURM_JOB_ID']],text=True)
            out = destination/phase
            out.mkdir()
            command = [config['training_python'], '-m','agents.recon.gsplat_train',
                '--scene-dir',str(destination/'scene'),'--init-ply',str(init/'init_points.ply'),
                '--init-manifest',str(init/'init_manifest.json'),'--out',str(out/'scene.ply'),
                '--iters',str(config['smoke_iters'] if phase == 'train-smoke' else config['iters']),
                '--holdout-every',str(config['holdout_every']),'--seed',str(config['seed'])]
            report['command'] = command
            runtime_env = training_environment(config)
            with (out/'trainer.log').open('x') as log:
                subprocess.run(command,cwd=CODE,env=runtime_env,stdout=log,stderr=subprocess.STDOUT,check=True)
            report['artifacts'] = {p.name:file_identity(p) for p in out.iterdir() if p.is_file()}
            report['training_report'] = json.loads((out/'train_report.json').read_text())
        final_context = (context(config_path, freeze_root, scene_id) if scene_id is not None
                         else context(config_path, freeze_root))
        if final_context[1] != code:
            raise ValueError('source changed during phase')
        for key in ('prepare_receipt', 'triangulate_receipt'):
            if key in report and file_identity(destination/f'{key}.json') != report[key]:
                raise ValueError('upstream receipt changed during phase')
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['wall_s'] = time.monotonic()-started
        write_new_json(receipt, report)
    print(json.dumps({'phase':phase,'status':report['status'],'receipt':str(receipt),'paper_ready':False}))


def main():
    if sys.argv[1:] == ['--runtime-identity']:
        print(json.dumps(runtime_identity(),sort_keys=True))
        return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--freeze-root',required=True)
    parser.add_argument('--scene-id', help='explicit member of a complete population config')
    parser.add_argument('--phase',choices=['prepare','triangulate','train-smoke','train-pilot','train-full'],required=True)
    args=parser.parse_args()
    run(args.config,args.freeze_root,args.phase,args.scene_id)

if __name__ == '__main__': main()
