"""Frozen raw-only room pilot: existing renderer, canonical fidelity metrics."""
from __future__ import annotations

import argparse
import ast
import hashlib
import shutil
import contextlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import yaml

from robo.manifest.hash import canonical_hash, git_snapshot
from run.icra2027.e3_auto_discovery_pilot import identity, sha, write_new, PilotError
from run.icra2027.e3_fresh_generation_contract import checked_identity

CODE = Path(__file__).resolve().parents[2]
METHOD = 'Input scene Gaussian, reconstruction ceiling'
STAGES = ('render', 'metrics')
COHORT_SCOPE = 'fresh_gaussian_raw_room_cohort'
SMOKE_SCENES = ('38d58a7a31', '5748ce6f01')


def cohort_scenes(c):
    from run.icra2027.e3_gaussian_train_only import cohort_scene_ids
    scenes = cohort_scene_ids(c)
    if (c.get('mode') not in {'smoke', 'full'} or c.get('planned_scenes') != 50
            or c.get('smoke_scenes') != list(SMOKE_SCENES)
            or not set(SMOKE_SCENES) <= set(scenes)):
        raise PilotError('fixed full50/two-scene raw-room population differs')
    return list(SMOKE_SCENES) if c['mode'] == 'smoke' else scenes


def receipt_root(c, root):
    path = Path(root)/'raw_room'
    return path/c['scene_id'] if c.get('scope') == COHORT_SCOPE and c.get('scene_id') else path



def build_camera_bank(config_path, root, destination):
    """Seal existing CPU camera plans without recomputing any floating-point pose.

    The returned identity must enter the next config before its new E0 freeze.
    This imports camera inputs only, never completed renders or metric values.
    """
    config_path, root = Path(config_path).resolve(), Path(root).resolve()
    c = yaml.safe_load(config_path.read_text())
    if c.get('scope') != COHORT_SCOPE or c.get('mode') != 'full' or c['freeze_id'] != root.name:
        raise PilotError('camera bank requires the original full cohort config/freeze')
    contract_path = root/'contract/freeze_manifest.json'
    contract = json.loads(contract_path.read_text())
    payload = {k:v for k,v in contract.items() if k not in {'created_utc','environment','contract_sha256'}}
    resources = [r for r in contract['resource_inventory'] if r['id'] == 'e2_raw_room_config']
    if (canonical_hash(payload) != contract['contract_sha256'] or contract['code']['dirty']
            or len(contract['code']['commit']) != 40 or contract['freeze_id'] != root.name
            or len(resources) != 1 or resources[0]['sha256'] != sha(config_path)):
        raise PilotError('original camera contract/config seal differs')
    rows = []
    for scene in cohort_scenes(c):
        receipt_path = root/'raw_room'/scene/'plan_receipt.json'
        receipt = json.loads(receipt_path.read_text())
        plan = json.loads(checked_identity(receipt['plan']).read_text())
        if (receipt['status'] != 'PASS' or receipt['code_commit'] != contract['code']['commit']
                or receipt.get('scene_id') != scene or receipt.get('config_sha256') != sha(config_path)
                or receipt.get('native_train_source_validation') is not True
                or plan['scene_id'] != scene or plan['code_commit'] != contract['code']['commit']
                or plan['freeze_id'] != root.name or plan['contract_sha256'] != contract['contract_sha256']
                or plan.get('scope') != COHORT_SCOPE or plan.get('mode') != 'full'
                or plan.get('active_scenes') != cohort_scenes(c)):
            raise PilotError('original camera plan provenance differs')
        validate_camera_rows(plan['evaluation_images'])
        rows.append(dict(scene_id=scene, plan=identity(checked_identity(receipt['plan'])),
            plan_receipt=identity(receipt_path)))
    bank = dict(schema_version=1, scope='exact_serialized_e2_camera_bank',
        source_config=identity(config_path), source_contract=identity(contract_path),
        source_commit=contract['code']['commit'], source_freeze_id=root.name,
        scenes=rows, paper_ready=False, imported_render_count=0)
    write_new(Path(destination), bank)
    return identity(destination)


def validate_camera_rows(rows):
    if len(rows) != 8 or len({r['frame'] for r in rows}) != 8:
        raise PilotError('camera bank requires eight unique frozen views')
    for row in rows:
        matrix = np.asarray(row['w2c'], dtype=float)
        if (matrix.shape != (4,4) or not np.isfinite(matrix).all()
                or matrix[3].tolist() != [0.,0.,0.,1.]):
            raise PilotError('camera bank contains an invalid homogeneous transform')


def frozen_camera_plan(c):
    """Authenticate exact serialized matrices against the config-bound old plan."""
    bank = json.loads(checked_identity(c['camera_bank']).read_text())
    expected = cohort_scenes(dict(c, mode='full'))
    if (bank.get('schema_version') != 1 or bank.get('scope') != 'exact_serialized_e2_camera_bank'
            or [r['scene_id'] for r in bank['scenes']] != expected):
        raise PilotError('camera bank full population differs')
    source_config = yaml.safe_load(checked_identity(bank['source_config']).read_text())
    contract = json.loads(checked_identity(bank['source_contract']).read_text())
    payload = {k:v for k,v in contract.items() if k not in {'created_utc','environment','contract_sha256'}}
    bound = [r for r in contract['resource_inventory'] if r['id'] == 'e2_raw_room_config']
    if (canonical_hash(payload) != contract['contract_sha256'] or contract['code']['dirty']
            or contract['code']['commit'] != bank['source_commit']
            or contract['freeze_id'] != bank['source_freeze_id']
            or source_config['freeze_id'] != bank['source_freeze_id']
            or source_config.get('mode') != 'full' or cohort_scenes(source_config) != expected
            or len(bound) != 1 or bound[0]['sha256'] != bank['source_config']['sha256']):
        raise PilotError('camera bank source contract differs')
    row = next(r for r in bank['scenes'] if r['scene_id'] == c['scene_id'])
    receipt = json.loads(checked_identity(row['plan_receipt']).read_text())
    plan = json.loads(checked_identity(row['plan']).read_text())
    if (receipt['plan'] != row['plan'] or receipt['status'] != 'PASS'
            or receipt['code_commit'] != bank['source_commit'] or receipt['scene_id'] != c['scene_id']
            or receipt['config_sha256'] != bank['source_config']['sha256']
            or receipt.get('native_train_source_validation') is not True
            or plan['code_commit'] != bank['source_commit'] or plan['freeze_id'] != bank['source_freeze_id']
            or plan['contract_sha256'] != contract['contract_sha256'] or plan['scene_id'] != c['scene_id']):
        raise PilotError('camera bank original plan linkage differs')
    validate_camera_rows(plan['evaluation_images'])
    return plan, row['plan']

def cohort_plan_payload(c, code, contract, *, full_validation=False):
    """E2 reads official TEST only after the TRAIN-only Gaussian is sealed."""
    from agents.eval.fidelity_room_export import select_official_frames
    from agents.recon.colmap_poses import file_identity, pose_headers
    from run.icra2027.e3_auto_discovery_pilot import (
        COHORT_SCOPE as DISCOVERY_SCOPE, bind_cohort_gaussian, gaussian_source_layout,
        validate_gaussian_provenance)
    from run.icra2027.e2_fresh_readiness import check_content_disjoint
    source = bind_cohort_gaussian(dict(c, scope=DISCOVERY_SCOPE,
        source_gaussian_training_provenance='FRESH_OFFICIAL_TRAIN_ONLY'))
    prefix, _, _ = gaussian_source_layout(source)
    inputs_path = Path(source['gaussian_cohort']['freeze_root'])/prefix/'scene/training_inputs.json'
    inputs = json.loads(inputs_path.read_text())
    metadata = {}
    for name, item in inputs['source_metadata'].items():
        if file_identity(item['path']) != {k:item[k] for k in ('sha256','size_bytes')}:
            raise PilotError('Gaussian source camera/split changed')
        metadata[name] = dict(path=item['path'], sha256=item['sha256'])
    frames = inputs['frames']
    if inputs['boundary']['max_train_frames'] is not None:
        raise PilotError('raw cohort requires complete official TRAIN source')
    if full_validation:
        validate_gaussian_provenance(source, inputs['boundary'],
            {r['name']:dict(path=r['source_rgb'],sha256=r['image']['sha256']) for r in frames}, metadata)
    # Match the authenticated training artifact, without any discovery/tool dependency.
    gaussian = identity(source['gaussian'])
    if gaussian['sha256'] != source['gaussian_sha256']:
        raise PilotError('raw Gaussian differs from completed producer receipt')
    split = json.loads(Path(metadata['train_test_lists.json']['path']).read_text())
    poses = pose_headers(metadata['colmap/images.txt']['path'])
    selected = select_official_frames(split, set(poses), count=8, excluded_names=set(split['train']))
    training = [r['name'] for r in frames]
    if set(selected) & set(training): raise PilotError('TEST overlaps full TRAIN roster')
    calibration = inputs['calibration']
    image_root = Path(metadata['train_test_lists.json']['path']).parent/'resized_undistorted_images'
    frozen, original_plan = frozen_camera_plan(c) if c.get('camera_bank') else (None, None)
    if frozen is not None:
        if (frozen['training_inputs'] != identity(inputs_path)
                or frozen['gaussian'] != gaussian or frozen['calibration'] != calibration
                or [r['frame'] for r in frozen['evaluation_images']] != selected):
            raise PilotError('camera bank source inputs/calibration/view selection differs')
    images = []
    for index, name in enumerate(selected):
        camera = frozen['evaluation_images'][index]['w2c'] if frozen is not None else poses[name].tolist()
        row = dict(frame=name, **identity(image_root/name), w2c=camera)
        if frozen is not None and row != frozen['evaluation_images'][index]:
            raise PilotError('camera bank evaluation image bytes differ')
        with Image.open(row['path']) as image:
            if image.size != (calibration['width'],calibration['height']):
                raise PilotError('fixed TEST image dimensions differ from calibration')
        images.append(row)
    check_content_disjoint([r['sha256'] for r in images], [r['image']['sha256'] for r in frames])
    camera_binding = dict(camera_bank=c['camera_bank'], original_camera_plan=original_plan) if frozen is not None else {}
    return dict(schema_version=1, freeze_id=c['freeze_id'], code_commit=code['commit'],
        contract_sha256=contract['contract_sha256'], paper_ready=False, scene_id=c['scene_id'],
        **camera_binding, scope=COHORT_SCOPE, mode=c['mode'], planned_scenes=50, active_scenes=cohort_scenes(c),
        gaussian=gaussian, gaussian_provenance=source['gaussian_provenance'],
        training_inputs=identity(inputs_path), calibration=calibration, evaluation_images=images,
        optimization_input_frames=training, official_train_count=len(split['train']),
        official_test_count=len(split['test']), selected_rgb_training_byte_overlap=[],
        source_gaussian_training_provenance='FRESH_OFFICIAL_TRAIN_ONLY',
        selection_rule='fidelity_room_export.select_official_frames(count=8); no outcome filtering',
        render_recipe={'function':'agents.core.common.render_view','scale':1.0,
            'render_mode':'RGB','background':None,'quantization':'clip(0,1)*255 uint8 PNG'})


def environment(c, stage=None):
    env = dict(os.environ, PYTHONPATH=str(CODE), PYTHONNOUSERSITE='1',
        PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='42', OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', SIMANY_ROOT=str(CODE),
        SIMANY_SCENE=c.get('scene_id', SMOKE_SCENES[0]), SIMANY_MESH_SRC='derived', SIMANY_NO_GT='1',
        SIMANY_AUTO='1', HF_HUB_OFFLINE='1', TORCH_HOME=c['torch_home'],
        E2_LPIPS_WEIGHTS_PATH=c['lpips_backbone']['path'],
        E2_LPIPS_WEIGHTS_SIZE=str(c['lpips_backbone']['bytes']),
        E2_LPIPS_WEIGHTS_SHA256=c['lpips_backbone']['sha256'])
    env.pop('PYTHONHOME', None)
    if stage == 'render':
        env['PYTHONPATH'] += os.pathsep + c['renderer_dependency_root']
    return env


def runtime(stage):
    """Targeted executable/module/RECORD/native hashes, not full dependency closure."""
    import site
    if site.ENABLE_USER_SITE or os.environ.get('PYTHONNOUSERSITE') != '1':
        raise PilotError('runtime requires user site disabled')
    modules = [('torch','torch'), ('numpy','numpy'), ('PIL','Pillow')]
    modules += ([('gsplat','gsplat'), ('plyfile','plyfile'), ('pydantic','pydantic'),
                 ('pydantic_core','pydantic_core')] if stage == 'render' else
                [('scipy','scipy'), ('lpips','lpips'), ('torchvision','torchvision')])
    result = {'python': identity(Path(sys.executable).resolve()), 'version': sys.version,
              'stage': stage, 'user_site_enabled': False, 'modules': {},
              'scope': 'targeted executable, module entrypoint, package RECORD and renderer native bytes'}
    with contextlib.redirect_stdout(sys.stderr):
        for name, package in modules:
            module = importlib.import_module(name)
            dist = importlib.metadata.distribution(package)
            records = [p for p in dist.files if str(p).endswith('.dist-info/RECORD')]
            result['modules'][name] = dict(version=dist.version, entrypoint=identity(module.__file__),
                record=identity(dist.locate_file(records[0])))
        if stage == 'render':
            from gsplat.cuda._backend import _C
            if _C is None: raise PilotError('prebuilt gsplat backend required; no JIT fallback')
            result['native_renderer'] = identity(_C.__file__)
            from pydantic_core import _pydantic_core
            result['native_schema'] = identity(_pydantic_core.__file__)
        else:
            from robo.eval.fidelity_metrics import LPIPSEvaluator
            model = LPIPSEvaluator('cpu')
            if model.error or model.model is None: raise PilotError(f'CPU LPIPS import smoke failed: {model.error}')
            result['lpips_provenance'] = model.provenance
    return result


def probe(c, stage):
    checked_identity(c['lpips_backbone'])
    output = subprocess.check_output([c['python'][stage], '-m', 'run.icra2027.e2_raw_room',
        '--runtime', stage], cwd=CODE, env=environment(c, stage), text=True)
    return json.loads(output)


def context(config_path, root, scene_id=None):
    config_path, root = Path(config_path).resolve(), Path(root).resolve()
    c = yaml.safe_load(config_path.read_text())
    code = git_snapshot(CODE)
    if code['dirty'] or len(code['commit']) != 40: raise PilotError('exact clean source required')
    contract = json.loads((root/'contract/freeze_manifest.json').read_text())
    payload = {k:v for k,v in contract.items() if k not in {'created_utc','environment','contract_sha256'}}
    resources = [r for r in contract['resource_inventory'] if r['id'] == 'e2_raw_room_config']
    if (canonical_hash(payload) != contract['contract_sha256'] or contract['code']['dirty']
            or contract['code']['commit'] != code['commit'] or len(resources) != 1
            or resources[0]['sha256'] != sha(config_path)):
        raise PilotError('exact-source E0/config seal differs')
    if (c['freeze_id'] != root.name or contract['freeze_id'] != root.name
            or c['paper_ready'] is not False or c['scope'] not in {'single_scene_raw_room_smoke', COHORT_SCOPE}
            or c['seed'] != 42 or c['views'] != 8 or c['render_scale'] != 1.0):
        raise PilotError('fixed raw-room pilot scope/camera recipe differs')
    if c['scope'] == COHORT_SCOPE:
        if 'scene_id' in c or (scene_id is not None and scene_id not in cohort_scenes(c)):
            raise PilotError('explicit raw-room unit outside complete frozen population')
        cohort_scenes(c)
        if scene_id is not None: c = dict(c, scene_id=scene_id)
    elif scene_id is not None:
        raise PilotError('legacy raw-room pilot has no scene override')
    out = CODE/'outputs/icra2027'/c['freeze_id']/'raw_room'
    if c['scope'] == COHORT_SCOPE and scene_id is not None: out /= scene_id
    # Preserve canonical evaluator's repository-local, no-symlink restriction.
    from robo.eval.fidelity_metrics import _checked_repository_path
    _checked_repository_path(out, 'raw-room pilot output')
    return c, code, out, contract


def plan_payload(c, code, contract):
    if c.get('scope') == COHORT_SCOPE:
        return cohort_plan_payload(c, code, contract)
    report = json.loads(checked_identity(c['readiness']).read_text())
    if (report['scene_id'] != c['scene_id'] or report['source_dirty']
            or report['source_gaussian_training_provenance'] != 'FRESH_OFFICIAL_TRAIN_ONLY'
            or len(report['evaluation_images']) != 8 or report['metric_records_created'] != 0):
        raise PilotError('frozen readiness scope differs')
    for key in ('train_evaluation_overlap','generator_evaluation_overlap','selected_rgb_training_byte_overlap'):
        if report[key]: raise PilotError('held-out boundary overlaps optimization evidence')
    if report['selected_evaluation_frames'] != [r['frame'] for r in report['evaluation_images']]:
        raise PilotError('camera order differs from fixed readiness')
    checked_identity(c['gaussian'])
    for image in report['evaluation_images']: checked_identity(image)
    return dict(schema_version=1, freeze_id=c['freeze_id'], code_commit=code['commit'],
        contract_sha256=contract['contract_sha256'], paper_ready=False, scene_id=c['scene_id'],
        gaussian=c['gaussian'], readiness=c['readiness'],
        calibration=report['calibration'], evaluation_images=report['evaluation_images'],
        optimization_input_frames=report['optimization_input_frames'],
        source_gaussian_training_provenance='FRESH_OFFICIAL_TRAIN_ONLY',
        render_recipe={'function':'agents.core.common.render_view','scale':1.0,
                       'render_mode':'RGB','background':None,'quantization':'clip(0,1)*255 uint8 PNG'})


def make_plan(config_path, root, scene_id=None):
    c, code, out, contract = context(config_path, root, scene_id)
    if out.exists(): raise FileExistsError('raw-room plan/output already exists')
    if c['scope'] == COHORT_SCOPE:
        if scene_id is None: raise PilotError('raw-room plan requires explicit scene')
        payload = cohort_plan_payload(c, code, contract, full_validation=True)
        probes = {stage:probe(c,stage) for stage in STAGES}
        if probes != c['runtime']: raise PilotError('frozen render/metric runtime differs')
        out.mkdir(parents=True, exist_ok=False)
        write_new(out/'plan.json',payload)
        write_new(receipt_root(c,root)/'plan_receipt.json',dict(status='PASS',
            code_commit=code['commit'], config_sha256=sha(config_path), scene_id=scene_id,
            plan=identity(out/'plan.json'), runtime=probes, paper_ready=False,
            native_train_source_validation=True, model_render_calls=0, metric_records=0))
        return
    from run.icra2027.e2_fresh_readiness import audit
    report = json.loads(checked_identity(c['readiness']).read_text())
    actual = audit(checked_identity(report['config']))
    for key in ('evaluation_images','calibration','optimization_input_frames','generation_input_frames',
                'source_discovery_hashes','gaussian_provenance'):
        if actual[key] != report[key]: raise PilotError(f'readiness inputs changed: {key}')
    # The Gaussian must be the source proved by that exact discovery chain.
    readiness_config = json.loads(Path(report['config']['path']).read_text())
    rvg_config = yaml.safe_load(checked_identity(readiness_config['rvg_config']).read_text())
    discovery_input = json.loads((Path(rvg_config['source_pilot'])/'input_manifest.json').read_text())
    if c['gaussian'] != discovery_input['gaussian']: raise PilotError('different Gaussian than readiness source')
    probes = {stage:probe(c, stage) for stage in STAGES}
    if probes != c['runtime']: raise PilotError('frozen render/metric runtime differs')
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'plan.json', plan_payload(c, code, contract))
    write_new(Path(root)/'raw_room/plan_receipt.json', dict(status='PASS',
        code_commit=code['commit'], plan=identity(out/'plan.json'), runtime=probes,
        paper_ready=False, model_render_calls=0, metric_records=0))


def checked_plan(c, code, out, contract, root):
    receipt = json.loads((receipt_root(c,root)/'plan_receipt.json').read_text())
    if c.get('scope') == COHORT_SCOPE:
        bound=[r for r in contract['resource_inventory'] if r['id']=='e2_raw_room_config']
        if (receipt.get('scene_id') != c['scene_id'] or len(bound)!=1
                or receipt.get('config_sha256') != bound[0]['sha256']
                or receipt.get('native_train_source_validation') is not True):
            raise PilotError('cohort CPU plan scene/config/source-validation differs')
    if receipt['status'] != 'PASS' or receipt['code_commit'] != code['commit']:
        raise PilotError('CPU plan not source-bound')
    plan = json.loads(checked_identity(receipt['plan']).read_text())
    if Path(receipt['plan']['path']) != out/'plan.json' or plan != plan_payload(c, code, contract):
        raise PilotError('sealed plan differs from frozen inputs')
    return plan



def reuse_source_fingerprint(commit):
    """Exact relevant producer/metric source bytes, retaining both full commits."""
    files = ('agents/core/common.py', 'robo/eval/fidelity_metrics.py',
             'run/icra2027/e3_fresh_generation_contract.py')
    result = {}
    for filename in (*files, 'run/icra2027/e2_raw_room.py'):
        source = subprocess.check_output(['git','show',f'{commit}:{filename}'],cwd=CODE)
        if filename.endswith('e2_raw_room.py'):
            decoded = source.decode()
            node = next(n for n in ast.parse(decoded).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'export_raw')
            source = ast.get_source_segment(decoded,node).encode()
            filename += '::export_raw'
        result[filename] = hashlib.sha256(source).hexdigest()
    return result


def build_reuse_bank(config_path, root, camera_bank, destination):
    """Freeze every scene's recovery action; availability never drops a scene."""
    c = yaml.safe_load(Path(config_path).read_text())
    c['camera_bank'] = identity(camera_bank)
    root = Path(root).resolve()
    camera = json.loads(checked_identity(c['camera_bank']).read_text())
    if camera['source_config'] != identity(config_path) or camera['source_freeze_id'] != root.name:
        raise PilotError('reuse camera bank is not from the declared old config/freeze')
    rows = []
    for scene in cohort_scenes(c):
        original, plan_identity = frozen_camera_plan(dict(c,scene_id=scene))
        out = Path(plan_identity['path']).parent
        receipt_path = root/'raw_room'/scene/'execution_receipt.json'
        row = dict(scene_id=scene, action='RENDER')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get('status') != 'PASS' or receipt.get('code_commit') != original['code_commit']
                    or receipt.get('freeze_id') != original['freeze_id'] or receipt.get('scene_id') != scene
                    or [r['stage'] for r in receipt['stages']] != ['render']
                    or any(r['exit_code'] != 0 for r in receipt['stages'])):
                raise PilotError('existing execution is not a completed raw render')
            checked_identity(receipt['artifact'])
            check_render_receipt(out,original,sha(config_path),sha(out/'render_receipt.json'))
            if receipt['artifact'] != identity(out/'bundle/manifest.json'):
                raise PilotError('reuse execution artifact linkage differs')
            canonical_manifest(original,out/'bundle')
            row.update(action='VERIFIED_REUSE', execution_receipt=identity(receipt_path),
                render_receipt=identity(out/'render_receipt.json'), original_plan=plan_identity,
                bundle_manifest=identity(out/'bundle/manifest.json'))
        elif any((out/name).exists() for name in ('execution_claim.json','render_receipt.json','bundle','bundle.partial')):
            raise PilotError('nonterminal original execution cannot be treated as unexecuted')
        rows.append(row)
    bank = dict(schema_version=1, scope='exact_raw_room_render_recovery',
        source_config=identity(config_path), source_commit=camera['source_commit'],
        source_freeze_id=root.name, camera_bank=c['camera_bank'], rows=rows,
        source_fingerprint=reuse_source_fingerprint(camera['source_commit']), paper_ready=False)
    write_new(Path(destination),bank)
    return identity(destination)


def checked_reuse_source(c, code, plan):
    bank = json.loads(checked_identity(c['reuse_bank']).read_text())
    old_config = yaml.safe_load(checked_identity(bank['source_config']).read_text())
    ignored = {'freeze_id','mode','smoke_dependency','scene_id','camera_bank','reuse_bank'}
    if (bank.get('scope') != 'exact_raw_room_render_recovery' or bank.get('schema_version') != 1
            or bank['camera_bank'] != c['camera_bank']
            or [r['scene_id'] for r in bank['rows']] != cohort_scenes(dict(c,mode='full'))
            or {k:v for k,v in c.items() if k not in ignored}
                != {k:v for k,v in old_config.items() if k not in ignored}
            or bank['source_fingerprint'] != reuse_source_fingerprint(bank['source_commit'])
            or bank['source_fingerprint'] != reuse_source_fingerprint(code['commit'])):
        raise PilotError('reuse treatment/runtime/source implementation differs')
    original, original_identity = frozen_camera_plan(c)
    if bank['source_commit'] != original['code_commit'] or bank['source_freeze_id'] != original['freeze_id']:
        raise PilotError('reuse original producer differs')
    omitted = {'freeze_id','code_commit','contract_sha256','camera_bank','original_camera_plan'}
    if {k:v for k,v in plan.items() if k not in omitted} != {k:v for k,v in original.items() if k not in omitted}:
        raise PilotError('reuse camera/Gaussian/image/recipe inputs differ')
    row = next(r for r in bank['rows'] if r['scene_id'] == c['scene_id'])
    if row['action'] != 'VERIFIED_REUSE' or row['original_plan'] != original_identity:
        raise PilotError('scene is not declared for verified reuse')
    receipt = json.loads(checked_identity(row['execution_receipt']).read_text())
    checked_identity(row['render_receipt']); checked_identity(row['bundle_manifest'])
    original_out = Path(original_identity['path']).parent
    if (receipt['status'] != 'PASS' or receipt['code_commit'] != bank['source_commit']
            or receipt['freeze_id'] != bank['source_freeze_id'] or receipt['scene_id'] != c['scene_id']
            or receipt['artifact'] != row['bundle_manifest']
            or Path(row['render_receipt']['path']) != original_out/'render_receipt.json'
            or Path(row['bundle_manifest']['path']) != original_out/'bundle/manifest.json'):
        raise PilotError('reuse original execution linkage differs')
    check_render_receipt(original_out, original, bank['source_config']['sha256'],row['render_receipt']['sha256'])
    canonical_manifest(original,original_out/'bundle')
    return bank, row, original


def import_reused_scene(config_path, root, scene_id):
    c,code,out,contract = context(config_path,root,scene_id)
    if c.get('scope') != COHORT_SCOPE or c.get('mode') != 'full' or not scene_id:
        raise PilotError('reuse requires an explicit full-cohort scene')
    plan = checked_plan(c,code,out,contract,root)
    bank,row,original = checked_reuse_source(c,code,plan)
    write_new(out/'reuse_claim.json',dict(importer_commit=code['commit'],reuse_bank=c['reuse_bank']))
    source = Path(row['bundle_manifest']['path']).parent
    destination = out/'bundle'; staging = out/'bundle.partial'
    staging.mkdir(exist_ok=False)
    manifest = json.loads((source/'manifest.json').read_text())
    for artifact in manifest['artifacts']:
        target = staging/artifact['relative_path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/artifact['relative_path'],target)
        if sha(target) != artifact['sha256']:raise PilotError('copied render bytes differ')
    shutil.copyfile(source/'manifest.json',staging/'original_manifest.json')
    for view in manifest['views']:
        view['render_path'] = str(destination/'input_scene_gaussian'/Path(view['render_path']).name)
        view['gt_path'] = str(destination/'reference'/Path(view['gt_path']).name)
    provenance = dict(operation='verified_byte_copy',importer_commit=code['commit'],
        producer_commit=original['code_commit'],producer_freeze_id=original['freeze_id'],
        producer_config=bank['source_config'],source_fingerprint=bank['source_fingerprint'],
        reuse_bank=c['reuse_bank'],original_artifacts=row,render_or_metric_calls=0)
    manifest['import_provenance'] = provenance
    write_new(staging/'manifest.json',manifest);staging.rename(destination)
    canonical_manifest(original,destination)
    write_new(out/'reuse_receipt.json',dict(status='PASS',**provenance,
        config_sha256=sha(config_path),evaluation_freeze_id=c['freeze_id'],
        evaluation_plan=identity(out/'plan.json'),artifact=identity(destination/'manifest.json')))


def reused_evaluation_manifest(c,code,out,plan,config_path):
    bank,row,original = checked_reuse_source(c,code,plan)
    receipt = json.loads((out/'reuse_receipt.json').read_text())
    if (receipt['status'] != 'PASS' or receipt['importer_commit'] != code['commit']
            or receipt['producer_commit'] != original['code_commit']
            or receipt['producer_freeze_id'] != original['freeze_id']
            or receipt['producer_config'] != bank['source_config']
            or receipt['config_sha256'] != sha(config_path) or receipt['evaluation_freeze_id'] != c['freeze_id']
            or receipt['reuse_bank'] != c['reuse_bank'] or receipt['original_artifacts'] != row
            or receipt['evaluation_plan'] != identity(out/'plan.json')
            or Path(receipt['artifact']['path']) != out/'bundle/manifest.json'
            or sha(out/'bundle/original_manifest.json') != row['bundle_manifest']['sha256']):
        raise PilotError('reuse import receipt attribution differs')
    checked_identity(receipt['artifact'])
    imported = json.loads((out/'bundle/manifest.json').read_text())
    source_manifest = json.loads(checked_identity(row['bundle_manifest']).read_text())
    if imported['artifacts'] != source_manifest['artifacts']:
        raise PilotError('imported artifact declarations differ from frozen original bytes')
    result = canonical_manifest(original,out/'bundle')
    # freeze_id is the evaluation run; producer identities remain explicit below.
    result['freeze_id'] = c['freeze_id']
    result['provenance']['code_commit'] = code['commit']
    record = result['room_methods'][METHOD]['records'][0]
    record['freeze_id'] = c['freeze_id']
    record['render_producer'] = dict(code_commit=original['code_commit'],freeze_id=original['freeze_id'],
        config=bank['source_config'],original_plan=row['original_plan'],render_receipt=row['render_receipt'],
        execution_receipt=row['execution_receipt'],import_receipt=identity(out/'reuse_receipt.json'))
    return result

def gpu_identity():
    import torch
    node = os.environ.get('SLURMD_NODENAME', socket.gethostname())
    if (not os.environ.get('SLURM_JOB_ID') or not (node == 'hala' or node.startswith(('gcp','sof1')))
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise PilotError('one visible GPU on an authorized Slurm node required')
    prop = torch.cuda.get_device_properties(0)
    return dict(node=node, job_id=os.environ['SLURM_JOB_ID'], device_count=1,
                device_name=prop.name, total_bytes=prop.total_memory,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))


def export_raw(plan, destination, common, *, gaussian=None, method_slug="input_scene_gaussian"):
    """Write paired RGB with unchanged camera/SH/background/quantization semantics."""
    if method_slug not in {"input_scene_gaussian", "factorized_auto_discovery"}:
        raise PilotError("undeclared room render method")
    if (gaussian is None) != (method_slug == "input_scene_gaussian"):
        raise PilotError("public composite requires an explicit authenticated Gaussian")
    destination = Path(destination)
    if destination.exists(): raise FileExistsError('raw room bundle already exists')
    staging = destination.with_name(destination.name+'.partial')
    staging.mkdir(parents=True, exist_ok=False)
    for name in ('reference', method_slug): (staging/name).mkdir()
    cal = plan['calibration']; width, height = cal['width'], cal['height']
    source = checked_identity(plan['gaussian'])
    if gaussian is None: gaussian = common.load_gaussians(source)
    artifacts = []; views = []
    for row in plan['evaluation_images']:
        path = checked_identity(row)
        with Image.open(path) as image: target = np.asarray(image.convert('RGB'))
        if target.shape != (height, width, 3): raise PilotError('reference dimensions changed')
        rgb, _, _ = common.render_view(gaussian, np.asarray(row['w2c']), np.asarray(cal['K']), width, height)
        if rgb.shape != target.shape or not np.isfinite(rgb).all(): raise PilotError('invalid rendered RGB')
        filename = Path(row['frame']).stem+'.png'
        pair = {}
        for directory, array in [('reference',target),(method_slug,np.uint8(np.clip(rgb,0,1)*255))]:
            output = staging/directory/filename; Image.fromarray(array).save(output)
            artifact = dict(relative_path=f'{directory}/{filename}', sha256=sha(output), bytes=output.stat().st_size)
            artifacts.append(artifact); pair[directory] = artifact
        views.append(dict(view_id=row['frame'],
            render_path=str(destination/method_slug/filename),
            gt_path=str(destination/'reference'/filename),
            render_sha256=pair[method_slug]['sha256'], gt_sha256=pair['reference']['sha256']))
    if len(views) != 8: raise PilotError('fixed eight-view denominator differs')
    write_new(staging/'manifest.json', dict(plan=plan, artifacts=artifacts, views=views,
                                          paper_ready=False, metric_status='NOT_RUN'))
    staging.rename(destination)
    return views


def canonical_manifest(plan, bundle):
    from robo.eval.fidelity_metrics import ROOM_METHODS, OBJECT_METHODS
    manifest = json.loads((bundle/'manifest.json').read_text())
    if manifest['plan'] != plan: raise PilotError('render plan differs')
    expected = {f'{d}/{Path(r["frame"]).stem}.png' for r in plan['evaluation_images']
                for d in ('reference','input_scene_gaussian')}
    if {a['relative_path'] for a in manifest['artifacts']} != expected or len(manifest['artifacts']) != 16:
        raise PilotError('render artifact denominator differs')
    for a in manifest['artifacts']:
        if sha(bundle/a['relative_path']) != a['sha256']: raise PilotError('render artifact changed')
    by_path = {a['relative_path']:a for a in manifest['artifacts']}
    exact_views = []
    for row in plan['evaluation_images']:
        filename = Path(row['frame']).stem+'.png'
        exact_views.append(dict(view_id=row['frame'],
            render_path=str(bundle/'input_scene_gaussian'/filename),
            gt_path=str(bundle/'reference'/filename),
            render_sha256=by_path[f'input_scene_gaussian/{filename}']['sha256'],
            gt_sha256=by_path[f'reference/{filename}']['sha256']))
    if manifest['views'] != exact_views: raise PilotError('rendered view mapping differs from fixed cameras')
    room = {name: {'source_id':name, 'records':[]} for name in ROOM_METHODS}
    room[METHOD] = dict(source_id='input_scene_gaussian', records=[dict(
        freeze_id=plan['freeze_id'], scene_id=plan['scene_id'], source_build=plan['gaussian']['sha256'],
        render_dir=str(bundle/'input_scene_gaussian'), gt_dir=str(bundle/'reference'),
        evaluation_frames=[Path(r['frame']).stem+'.png' for r in plan['evaluation_images']],
        optimization_input_frames=plan['optimization_input_frames'], n_views=8,
        coverage={'render_only':[], 'gt_only':[]}, views=manifest['views'])])
    return dict(schema_version=2, freeze_id=plan['freeze_id'], paper_ready=False,
        provenance={'code_commit':plan['code_commit'], 'scope':plan.get('scope','single-scene raw-only engineering smoke')},
        room_split={'unit':'held_out_view'}, room_methods=room,
        object_methods={name:{'source_id':name,'records':[]} for name in OBJECT_METHODS},
        coverage={'minimum_room_views_by_method':{'input_scene_gaussian':8},
                  'minimum_room_scenes_by_method':{'input_scene_gaussian':1}})


def check_render_receipt(out, plan, config_sha256, upstream_sha256):
    path = out/'render_receipt.json'
    if not upstream_sha256 or sha(path) != upstream_sha256:
        raise PilotError('driver-bound render receipt differs')
    receipt = json.loads(path.read_text())
    if (receipt['status'] != 'PASS' or receipt['code_commit'] != plan['code_commit']
            or receipt['freeze_id'] != plan['freeze_id'] or receipt['config_sha256'] != config_sha256
            or receipt['plan_sha256'] != sha(out/'plan.json')
            or Path(receipt['artifact']['path']) != out/'bundle/manifest.json'):
        raise PilotError('render receipt source/config/freeze/plan linkage differs')
    checked_identity(receipt['artifact'])


def worker(config_path, root, stage, upstream_sha256=None, scene_id=None):
    c, code, out, contract = context(config_path, root, scene_id)
    plan = checked_plan(c, code, out, contract, root)
    if Path(sys.executable).resolve() != Path(c['python'][stage]).resolve():
        raise PilotError('actual worker interpreter differs from frozen stage')
    if runtime(stage) != c['runtime'][stage]: raise PilotError('actual worker runtime changed')
    device = gpu_identity(); started = time.monotonic()
    if stage == 'render':
        from agents.core import common
        export_raw(plan, out/'bundle', common)
    else:
        from robo.eval.fidelity_metrics import evaluate_manifest
        check_render_receipt(out, plan, sha(config_path), upstream_sha256)
        write_new(out/'fidelity_manifest.json', canonical_manifest(plan, out/'bundle'))
        result = evaluate_manifest(out/'fidelity_manifest.json', out/'table', lpips_device='cuda', bootstrap_seed=42)
        row = next(r for r in result['rows'] if r['method'] == METHOD)
        if (result['paper_ready'] or result['lpips_backend_error']
                or not result['validation']['lpips_provenance_complete']
                or row['n_scenes'] != 1 or row['n_images'] != 8
                or any(row['metric_samples'][m] != 8 for m in ('psnr','ssim','lpips'))):
            raise PilotError('canonical metric completeness failed')
    write_new(out/f'{stage}_receipt.json', dict(status='PASS', code_commit=code['commit'],
        config_sha256=sha(config_path), freeze_id=c['freeze_id'], plan_sha256=sha(out/'plan.json'),
        device=device, wall_s=time.monotonic()-started, paper_ready=False,
        artifact=identity(out/('bundle/manifest.json' if stage == 'render' else 'table/fidelity_table.json'))))


def run(config_path, root, scene_id=None):
    c, code, out, contract = context(config_path, root, scene_id)
    if c.get('scope') == COHORT_SCOPE:
        if scene_id is None: raise PilotError('raw-room execution requires explicit scene')
        if c['mode'] == 'full':
            require_cohort_smoke(c, code)
            if c.get('reuse_bank'):
                bank = json.loads(checked_identity(c['reuse_bank']).read_text())
                rows = [r for r in bank['rows'] if r['scene_id'] == scene_id]
                if len(rows) != 1 or rows[0]['action'] != 'RENDER':
                    raise PilotError('declared reused scene must not invoke the renderer')
    checked_plan(c, code, out, contract, root); device = gpu_identity()
    write_new(out/'execution_claim.json',dict(code_commit=code['commit'], device=device))
    started = time.monotonic(); report = dict(code_commit=code['commit'], paper_ready=False,
        freeze_id=c['freeze_id'], device=device, scope=c['scope'], stages=[])
    try:
        stages = ('render',) if c.get('scope') == COHORT_SCOPE and c['mode']=='full' else STAGES
        for stage in stages:
            command = [c['python'][stage],'-m','run.icra2027.e2_raw_room','--config',str(Path(config_path).resolve()),
                       '--freeze-root',str(Path(root).resolve()),'--phase',stage]
            if scene_id is not None: command += ['--scene-id', scene_id]
            if stage == 'metrics':
                command += ['--upstream-receipt-sha256', sha(out/'render_receipt.json')]
            with (out/f'{stage}.log').open('x') as log:
                result = subprocess.run(command,cwd=CODE,env=environment(c, stage),stdout=log,stderr=subprocess.STDOUT)
            report['stages'].append(dict(stage=stage,command=command,exit_code=result.returncode))
            if result.returncode: raise PilotError(f'{stage} failed; logs and partial evidence preserved')
        report.update(status='PASS', artifact=identity(out/('table/fidelity_table.json' if 'metrics' in stages else 'bundle/manifest.json')))
        if 'metrics' in stages: report['table'] = report['artifact']
        if scene_id is not None: report['scene_id'] = scene_id
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['wall_s']=time.monotonic()-started
        write_new(receipt_root(c,root)/'execution_receipt.json',report)


def combine_cohort_manifests(manifests, *, scenes, freeze_id, code_commit):
    """Combine canonical per-view records; never average per-scene metric tables."""
    from robo.eval.fidelity_metrics import ROOM_METHODS, OBJECT_METHODS
    if len(manifests) != len(scenes) or not scenes or len(scenes)!=len(set(scenes)):
        raise PilotError('raw cohort scene denominator differs')
    room={name:{'source_id':name,'records':[]} for name in ROOM_METHODS}
    records=[]
    for scene, manifest in zip(scenes,manifests):
        rows=manifest['room_methods'][METHOD]['records']
        if (manifest['freeze_id']!=freeze_id or manifest['provenance']['code_commit']!=code_commit
                or len(rows)!=1 or rows[0]['scene_id']!=scene or rows[0]['n_views']!=8):
            raise PilotError('raw cohort per-view source/scene/coverage differs')
        if any(value['records'] for name,value in manifest['room_methods'].items() if name!=METHOD):
            raise PilotError('raw cohort cannot introduce an undeclared method')
        if any(value['records'] for value in manifest['object_methods'].values()):
            raise PilotError('raw cohort cannot introduce object metrics')
        records.extend(rows)
    room[METHOD]={'source_id':'input_scene_gaussian','records':records}
    return {'schema_version':2,'freeze_id':freeze_id,'paper_ready':False,
        'provenance':{'code_commit':code_commit,'scope':COHORT_SCOPE},
        'room_split':{'unit':'held_out_view'},'room_methods':room,
        'object_methods':{name:{'source_id':name,'records':[]} for name in OBJECT_METHODS},
        'coverage':{'minimum_room_views_by_method':{'input_scene_gaussian':8*len(scenes)},
                    'minimum_room_scenes_by_method':{'input_scene_gaussian':len(scenes)}}}


def require_cohort_smoke(c, code):
    from run.icra2027.e3_gaussian_train_only import require_published_full_source
    require_published_full_source(code)
    source=c['smoke_dependency']
    config=checked_identity(source['config'])
    sc=yaml.safe_load(config.read_text())
    root=Path(source['freeze_root'])
    if (sc['mode']!='smoke' or sc['freeze_id']!=root.name or sc['scope']!=COHORT_SCOPE
            or cohort_scenes(sc)!=list(SMOKE_SCENES)):
        raise PilotError('full raw cohort requires the predeclared two-scene smoke')
    # Freeze ID, mode and the dependency itself are the only allowed differences.
    ignored={'mode','freeze_id','smoke_dependency','scene_id'}
    if {k:v for k,v in c.items() if k not in ignored}!={k:v for k,v in sc.items() if k not in ignored}:
        raise PilotError('full raw configuration differs from smoke protocol')
    _, smoke_code, out, contract=context(config,root)
    if smoke_code!=code:raise PilotError('smoke source differs from full source')
    receipt=json.loads((root/'raw_room/aggregate_receipt.json').read_text())
    if (receipt['status']!='PASS' or receipt['code_commit']!=code['commit']
            or receipt['config_sha256']!=sha(config) or receipt['contract_sha256']!=contract['contract_sha256']
            or receipt['scenes']!=list(SMOKE_SCENES) or receipt['n_images']!=16
            or Path(receipt['table']['path'])!=out/'aggregate/table/fidelity_table.json'):
        raise PilotError('complete source-bound sixteen-view smoke is required')
    table=json.loads(checked_identity(receipt['table']).read_text())
    metric_row=next((row for row in table['rows'] if row['method']==METHOD),None)
    if (table['paper_ready'] or table['lpips_backend_error']
            or not table['validation']['lpips_provenance_complete'] or metric_row is None
            or metric_row['n_scenes']!=2 or metric_row['n_images']!=16
            or metric_row['metric_samples']!={'psnr':16,'ssim':16,'lpips':16}):
        raise PilotError('smoke table does not contain sixteen complete canonical metric samples')
    coverage=json.loads(checked_identity(receipt['coverage']).read_text())
    if (coverage['planned_scenes']!=2 or coverage['planned_views']!=16
            or coverage['completed_scenes']!=2 or [r['scene_id'] for r in coverage['rows']]!=list(SMOKE_SCENES)
            or any(r['status']!='PASS' for r in coverage['rows'])):
        raise PilotError('smoke coverage differs from the two-scene population')
    return receipt


def aggregate(config_path, root):
    c,code,out,contract=context(config_path,root)
    if c.get('scope')!=COHORT_SCOPE:raise PilotError('aggregate requires complete declared cohort')
    scenes=cohort_scenes(c)
    if c['mode']=='full':require_cohort_smoke(c,code)
    destination=out/'aggregate';destination.mkdir(parents=True,exist_ok=False)
    manifests=[];units=[]
    for scene in scenes:
        row={'scene_id':scene,'status':'NOT_RUN'}
        try:
            unit_c,unit_code,unit_out,unit_contract=context(config_path,root,scene)
            plan=checked_plan(unit_c,unit_code,unit_out,unit_contract,root)
            if (unit_out/'reuse_receipt.json').exists():
                manifests.append(reused_evaluation_manifest(unit_c,code,unit_out,plan,config_path))
                row.update(status='PASS',source_action='VERIFIED_REUSE',reuse_receipt=identity(unit_out/'reuse_receipt.json'))
                units.append(row)
                continue
            receipt=json.loads((receipt_root(unit_c,root)/'execution_receipt.json').read_text())
            if (receipt['status']!='PASS' or receipt['scene_id']!=scene
                    or receipt['code_commit']!=code['commit'] or receipt['freeze_id']!=c['freeze_id']):
                raise PilotError('raw scene execution not complete or source-bound')
            checked_identity(receipt['artifact'])
            check_render_receipt(unit_out,plan,sha(config_path),sha(unit_out/'render_receipt.json'))
            manifests.append(canonical_manifest(plan,unit_out/'bundle'))
            row.update(status='PASS',execution_receipt=identity(receipt_root(unit_c,root)/'execution_receipt.json'))
        except Exception as exc:
            row.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
        units.append(row)
    write_new(destination/'coverage.json',{'planned_scenes':len(scenes),'planned_views':8*len(scenes),
        'completed_scenes':sum(r['status']=='PASS' for r in units),'rows':units,'paper_ready':False})
    if any(row['status']!='PASS' for row in units):
        raise PilotError('incomplete raw cohort; all missing/failed scenes retained in coverage.json')
    manifest=combine_cohort_manifests(manifests,scenes=scenes,freeze_id=c['freeze_id'],code_commit=code['commit'])
    write_new(destination/'fidelity_manifest.json',manifest)
    if Path(sys.executable).resolve()!=Path(c['python']['metrics']).resolve() or runtime('metrics')!=c['runtime']['metrics']:
        raise PilotError('aggregate metric runtime differs')
    device=gpu_identity()
    from robo.eval.fidelity_metrics import evaluate_manifest
    result=evaluate_manifest(destination/'fidelity_manifest.json',destination/'table',lpips_device='cuda',bootstrap_seed=42)
    row=next(r for r in result['rows'] if r['method']==METHOD)
    if (result['paper_ready'] or result['lpips_backend_error'] or not result['validation']['lpips_provenance_complete']
            or row['n_scenes']!=len(scenes) or row['n_images']!=8*len(scenes)
            or any(row['metric_samples'][m]!=8*len(scenes) for m in ('psnr','ssim','lpips'))):
        raise PilotError('canonical raw cohort metric completeness failed')
    write_new(Path(root)/'raw_room/aggregate_receipt.json',{'status':'PASS','code_commit':code['commit'],
        'config_sha256':sha(config_path),'contract_sha256':contract['contract_sha256'],
        'scenes':scenes,'n_images':8*len(scenes),'table':identity(destination/'table/fidelity_table.json'),
        'coverage':identity(destination/'coverage.json'),'device':device,'paper_ready':False})


def aggregate_driver(config_path, root):
    c,code,out,contract=context(config_path,root)
    if c.get('scope')!=COHORT_SCOPE:raise PilotError('aggregate requires declared cohort')
    command=[c['python']['metrics'],'-m','run.icra2027.e2_raw_room',
        '--config',str(Path(config_path).resolve()),'--freeze-root',str(Path(root).resolve()),
        '--phase','aggregate-worker']
    with (out/'aggregate_driver.log').open('x') as log:
        subprocess.run(command,cwd=CODE,env=environment(c,'metrics'),stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config');p.add_argument('--freeze-root')
    p.add_argument('--phase',choices=['reuse-bank','reuse','camera-bank','plan','run','aggregate','aggregate-worker',*STAGES]);p.add_argument('--runtime',choices=STAGES)
    p.add_argument('--upstream-receipt-sha256');p.add_argument('--scene-id');p.add_argument('--camera-bank-out');p.add_argument('--reuse-bank-out');p.add_argument('--camera-bank')
    a=p.parse_args()
    if a.runtime: print(json.dumps(runtime(a.runtime)));return
    if not all((a.config,a.freeze_root,a.phase)): p.error('config/freeze-root/phase required')
    if a.phase=='reuse-bank':
        if not a.reuse_bank_out or not a.camera_bank:p.error('reuse-bank-out and camera-bank required')
        print(json.dumps(build_reuse_bank(a.config,a.freeze_root,a.camera_bank,a.reuse_bank_out)))
    elif a.phase=='reuse':import_reused_scene(a.config,a.freeze_root,a.scene_id)
    elif a.phase=='camera-bank':
        if not a.camera_bank_out: p.error('camera-bank-out required')
        print(json.dumps(build_camera_bank(a.config,a.freeze_root,a.camera_bank_out)))
    elif a.phase=='plan':make_plan(a.config,a.freeze_root,a.scene_id)
    elif a.phase=='run':run(a.config,a.freeze_root,a.scene_id)
    elif a.phase=='aggregate':aggregate_driver(a.config,a.freeze_root)
    elif a.phase=='aggregate-worker':aggregate(a.config,a.freeze_root)
    else:worker(a.config,a.freeze_root,a.phase,a.upstream_receipt_sha256,a.scene_id)


if __name__=='__main__':main()
