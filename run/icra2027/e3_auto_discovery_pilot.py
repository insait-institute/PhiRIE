#!/usr/bin/env python3
"""Bounded GT-isolated discovery/crop engineering pilot; no asset generation.

The first declared scene and first 48 official training frames are frozen before
any model output. Existing discovery/preparation producers own their algorithms.
Inherited Gaussian inputs remain UNKNOWN; fresh inputs require an authenticated
training/initialization receipt chain and the identical official TRAIN boundary.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import yaml

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
COHORT_SCOPE = 'gt_isolated_full_cohort_discovery_crops'


class PilotError(ValueError):
    pass


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    result = dict(path=str(path), bytes=before.st_size, mtime_ns=before.st_mtime_ns,
                  sha256=sha(path))
    if (path.stat().st_size, path.stat().st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise PilotError(f'input changed during hashing: {path}')
    return result


def validate_config(config, roster):
    cohort = config.get('scope') == COHORT_SCOPE
    if config.get('paper_ready') is not False or config.get('scope') not in {COHORT_SCOPE, 'gt_isolated_engineering_discovery_crops'}:
        raise PilotError('engineering scope cannot be promoted')
    if cohort:
        from run.icra2027.e3_gaussian_train_only import cohort_scene_ids
        scenes = cohort_scene_ids(config)
        if (config['scene_id'] not in scenes or config['planned_scenes'] != 50
                or config['full_vocabulary'] != 1 or config.get('crop_comparison_reference') is not None
                or config['scene_roster'] != config['construction_roster']['path']
                or config['source_gaussian_training_provenance'] != 'FRESH_OFFICIAL_TRAIN_ONLY'):
            raise PilotError('complete fresh full-vocabulary scene population required')
        parsed = ast.parse((CODE/'agents/discover/auto_segment.py').read_text())
        prompt_node = next(node.value for node in parsed.body if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == 'PROMPTS' for target in node.targets))
        prompts = ast.literal_eval(prompt_node.left) + ast.literal_eval(prompt_node.right.body)
        if config['discovery_prompts'] != prompts:
            raise PilotError('full discovery prompt roster differs from frozen producer')
    elif config['scene_id'] != str(roster['population']['scene_ids'][0]):
        raise PilotError('pilot must use first predeclared scene')
    if config['max_train_frames'] != (None if cohort else 48) or config['discovery_stride'] != 12 or config['render_stride'] != 3:
        raise PilotError('fixed pre-generation frame boundary or stride changed')
    if config.get('generate_assets') is not False:
        raise PilotError('asset generation is outside this pilot')
    for name, digest in config.get('unchanged_producer_files', {}).items():
        if name not in {'agents/core/common.py', *(
                f'agents/discover/{module}.py' for module in (
                'derive_mesh_from_splat','auto_segment','factory_prepare','factory_refine_masks','training_views'))} or sha(CODE/name) != digest:
            raise PilotError('frozen discovery producer code changed')


def destination_for(config, freeze_root):
    root = Path(freeze_root) / 'auto_discovery_pilot'
    if config.get('scope') == COHORT_SCOPE:
        root /= config['scene_id']
    if root.absolute() != root.resolve():
        raise PilotError('discovery output cannot contain symlinks')
    return root


def gaussian_source_layout(config):
    cohort = config.get('scope') == COHORT_SCOPE
    prefix = f"gaussian_train_only/{config['scene_id']}" if cohort else 'gaussian_train_only'
    phase = 'train-full' if cohort else 'train-pilot'
    required = {'contract/freeze_manifest.json', *(
        f'{prefix}/{name}' for name in ('prepare_receipt.json', 'triangulate_receipt.json',
        'scene/training_inputs.json', 'initialization/init_manifest.json',
        f'{phase}_receipt.json', f'{phase}/train_report.json'))}
    return prefix, phase, required


def bind_cohort_gaussian(config):
    """Bind completed units from one predeclared exact-source Gaussian freeze."""
    from agents.recon.colmap_poses import file_identity
    source = config['gaussian_cohort']
    root = Path(source['freeze_root'])
    if root.absolute() != root.resolve() or file_identity(root/'contract/freeze_manifest.json') != source['contract']:
        raise PilotError('frozen Gaussian cohort contract changed')
    prefix, phase, required = gaussian_source_layout(config)
    anchors = {name:file_identity(root/name) for name in sorted(required)}
    from robo.manifest.hash import canonical_hash
    contract = json.loads((root/'contract/freeze_manifest.json').read_text())
    payload = {k:v for k,v in contract.items() if k not in {'created_utc','environment','contract_sha256'}}
    resources = [r for r in contract['resource_inventory'] if r['id']=='e3_gaussian_config']
    if (canonical_hash(payload) != contract['contract_sha256'] or contract['freeze_id'] != root.name
            or contract['code']['commit'] != source['producer_commit'] or contract['code']['dirty']
            or len(resources)!=1 or resources[0]['sha256'] != source['config_sha256']):
        raise PilotError('Gaussian cohort source/config E0 closure differs')
    receipts = {name:json.loads((root/prefix/f'{name}_receipt.json').read_text())
                for name in ('prepare','triangulate',phase)}
    for item in receipts.values():
        if (item.get('status') != 'PASS' or item.get('scene_id') != config['scene_id']
                or item.get('scope') != 'fresh_train_only_gaussian_population'
                or item['code']['commit'] != source['producer_commit'] or item['code']['dirty']
                or item['config']['sha256'] != source['config_sha256']):
            raise PilotError('completed same-scene/source/config Gaussian full unit unavailable')
    receipt = receipts[phase]
    for name,item in receipts.items():
        links = {'prepare':'prepare_receipt','triangulate':'triangulate_receipt'} if name==phase else (
            {'prepare':'prepare_receipt'} if name=='triangulate' else {})
        if any(item[key] != file_identity(root/prefix/f'{parent}_receipt.json') for parent,key in links.items()):
            raise PilotError('Gaussian cohort phase receipt linkage differs')
    if receipts['prepare']['input_manifest'] != anchors[f'{prefix}/scene/training_inputs.json'] or receipts['triangulate']['init_manifest'] != anchors[f'{prefix}/initialization/init_manifest.json']:
        raise PilotError('Gaussian cohort input/initialization linkage differs')
    report = json.loads((root/prefix/phase/'train_report.json').read_text())
    if (receipt['training_report'] != report or report['iters'] != 15000 or report['seed'] != 42
            or report['provenance']['official_test_images_read'] != 0
            or report['independent_heldout_evaluation'] is not False):
        raise PilotError('Gaussian cohort training report differs')
    return dict(config, gaussian=str(root/prefix/phase/'scene.ply'),
        gaussian_sha256=receipt['artifacts']['scene.ply']['sha256'],
        gaussian_provenance={'freeze_root':str(root), 'producer_commit':source['producer_commit'],
                             'anchors':anchors})


def cohort_resources(config, freeze_root, commit):
    """Hash large shared weights once; subsequent scene plans check their stats."""
    from robo.eval.fidelity_replacements import _tree_inventory
    path = Path(freeze_root)/'contract/discovery_resources.json'
    expected = {'code_commit':commit, 'sam3_checkpoint':config['sam3_checkpoint'],
                'sam3_source':config['sam3_source']}
    if path.exists():
        record = json.loads(path.read_text())
        if record['binding'] != expected:
            raise PilotError('shared discovery resource binding changed')
        item = record['checkpoint']
        if (Path(item['path']) != Path(config['sam3_checkpoint']['path']).resolve()
                or item['sha256'] != config['sam3_checkpoint']['sha256']
                or item['bytes'] != config['sam3_checkpoint']['bytes']
                or record['source']['tree_sha256'] != config['sam3_source']['tree_sha256']):
            raise PilotError('cached discovery resource content differs from frozen config')
        stat = Path(item['path']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item['bytes'], item['mtime_ns']):
            raise PilotError('shared SAM3 checkpoint changed')
        source = _tree_inventory(Path(config['sam3_source']['path']), label='SAM3 source')
        if source != record['source']:
            raise PilotError('shared SAM3 source changed')
        return record['checkpoint'], source
    checkpoint = identity(config['sam3_checkpoint']['path'])
    if (checkpoint['sha256'],checkpoint['bytes']) != (config['sam3_checkpoint']['sha256'],config['sam3_checkpoint']['bytes']):
        raise PilotError('SAM3 checkpoint differs from pinned content')
    source = _tree_inventory(Path(config['sam3_source']['path']), label='SAM3 source')
    if source['tree_sha256'] != config['sam3_source']['tree_sha256']:
        raise PilotError('SAM3 source tree differs')
    write_new(path, {'binding':expected, 'checkpoint':checkpoint, 'source':source})
    return checkpoint, source


def validate_cohort_runtime(config):
    from run.icra2027.e3_trellis_generation_pilot import runtime_identity, targeted_runtime_identity
    for stage in ('render','sam3'):
        python = config[f'{stage}_python']
        actual = {'packages_sha256':runtime_identity(python)[1],
                  'targeted_bytes':targeted_runtime_identity(python)}
        if actual != config['runtime'][stage]:
            raise PilotError(f'{stage} discovery runtime changed')
    from agents.recon.colmap_poses import file_identity
    from robo.manifest.hash import canonical_hash
    dependency = Path(config['renderer_dependency_root'])
    files = {str(p.relative_to(dependency)):file_identity(p) for p in sorted(dependency.rglob('*')) if p.is_file()}
    if canonical_hash(files) != config['renderer_dependency_tree_sha256']:
        raise PilotError('scoped renderer dependency bytes changed')


def validate_input_stats(manifest):
    for item in [*manifest['input_images'].values(), *manifest['metadata'].values(),
                 manifest['gaussian'], manifest['sam3_checkpoint']]:
        current = Path(item['path']).stat()
        if (current.st_size, current.st_mtime_ns) != (item['bytes'], item['mtime_ns']):
            raise PilotError(f'input changed after plan: {item["path"]}')


def checked_manifest(config, commit, config_path, destination):
    path = destination/'input_manifest.json'
    manifest = json.loads(path.read_text())
    if manifest['code_commit'] != commit or manifest['config_sha256'] != sha(config_path):
        raise PilotError('input plan source/config differs')
    if config.get('scope') == COHORT_SCOPE:
        receipt = json.loads((destination/'plan_receipt.json').read_text())
        if receipt != {'scene_id':config['scene_id'], 'code_commit':commit,
                       'config_sha256':sha(config_path), 'input_manifest_sha256':sha(path)}:
            raise PilotError('cohort input plan seal differs')
        if (manifest['scene_id'] != config['scene_id'] or manifest['scope'] != COHORT_SCOPE
                or manifest['freeze_id'] != config['freeze_id']):
            raise PilotError('cohort input plan scene/scope differs')
        fresh = bind_cohort_gaussian(config)
        if (manifest['gaussian']['path'] != fresh['gaussian']
                or manifest['gaussian']['sha256'] != fresh['gaussian_sha256']
                or manifest['gaussian_provenance']['anchors'] != fresh['gaussian_provenance']['anchors']):
            raise PilotError('cohort input plan Gaussian source differs')
        proof = manifest['gaussian_provenance']
        if (proof['producer_commit'] != config['gaussian_cohort']['producer_commit']
                or proof['freeze_id'] != Path(config['gaussian_cohort']['freeze_root']).name
                or proof['status'] != 'FRESH_OFFICIAL_TRAIN_ONLY'
                or manifest['source_gaussian_training_provenance'] != proof['status']):
            raise PilotError('cohort input plan fresh source attribution differs')
        source_inputs = json.loads((Path(fresh['gaussian_provenance']['freeze_root'])/
            gaussian_source_layout(config)[0]/'scene/training_inputs.json').read_text())
        frames = source_inputs['frames']
        if (manifest['boundary']['training_frames'] != [r['name'] for r in frames]
                or proof['training_frames'] != [r['name'] for r in frames]
                or proof['training_iterations'] != 15000
                or manifest['boundary']['max_train_frames'] is not None
                or set(manifest['input_images']) != {r['name'] for r in frames}):
            raise PilotError('cohort discovery must retain complete source TRAIN roster')
        if any(manifest['input_images'][r['name']]['sha256'] != r['image']['sha256']
               or manifest['input_images'][r['name']]['path'] != r['source_rgb'] for r in frames):
            raise PilotError('cohort discovery RGB source differs')
        if any(manifest['metadata'][name]['sha256'] != item['sha256']
               or manifest['metadata'][name]['path'] != item['path']
               for name,item in source_inputs['source_metadata'].items()):
            raise PilotError('cohort discovery metadata source differs')
    return manifest


def validate_gaussian_provenance(config, boundary, input_images, metadata):
    """Authenticate a fresh Gaussian without promoting legacy input provenance."""
    status = config.get('source_gaussian_training_provenance', 'UNKNOWN')
    if status == 'UNKNOWN':
        if config.get('gaussian_provenance') is not None:
            raise PilotError('UNKNOWN Gaussian cannot carry a fresh provenance claim')
        return {'status': 'UNKNOWN', 'independent_heldout_evaluation': False}
    if status != 'FRESH_OFFICIAL_TRAIN_ONLY':
        raise PilotError('unknown Gaussian provenance status')
    from agents.recon.colmap_poses import file_identity, validate_training_initialization
    from robo.manifest.hash import canonical_hash
    proof = config['gaussian_provenance']
    root = Path(proof['freeze_root']).resolve(strict=True)
    prefix, phase, required = gaussian_source_layout(config)
    if set(proof['anchors']) != required:
        raise PilotError('Gaussian source closure is incomplete')
    records = {}
    for name, expected in proof['anchors'].items():
        path = root/name
        if path.is_symlink() or file_identity(path) != expected:
            raise PilotError(f'Gaussian provenance anchor changed: {name}')
        records[name] = json.loads(path.read_text())
    contract = records['contract/freeze_manifest.json']
    payload = {k:v for k,v in contract.items() if k not in {'created_utc','environment','contract_sha256'}}
    if canonical_hash(payload) != contract['contract_sha256'] or contract['freeze_id'] != root.name:
        raise PilotError('Gaussian E0 seal differs')
    code = contract['code']
    if code['commit'] != proof['producer_commit'] or code['dirty']:
        raise PilotError('Gaussian producer source differs')
    d = root/prefix
    inputs, initialization = validate_training_initialization(d/'scene',
        d/'initialization/init_points.ply', d/'initialization/init_manifest.json')
    preparation = records[f'{prefix}/prepare_receipt.json']
    triangulation = records[f'{prefix}/triangulate_receipt.json']
    training = records[f'{prefix}/{phase}_receipt.json']
    source_configs = [r for r in contract['resource_inventory'] if r['id'] == 'e3_gaussian_config']
    if len(source_configs) != 1:
        raise PilotError('Gaussian E0 training config is absent')
    for receipt in (preparation, triangulation, training):
        if receipt['status'] != 'PASS' or receipt['code']['commit'] != code['commit'] or receipt['code']['dirty'] or receipt['config']['sha256'] != source_configs[0]['sha256']:
            raise PilotError('Gaussian producer phase lacks same-source/config PASS')
        if config.get('scope') == COHORT_SCOPE and (receipt.get('scene_id') != config['scene_id']
                or receipt.get('scope') != 'fresh_train_only_gaussian_population'):
            raise PilotError('Gaussian cohort phase scene/scope differs')
    if config.get('scope') == COHORT_SCOPE:
        source = config['gaussian_cohort']
        if source_configs[0]['sha256'] != source['config_sha256']:
            raise PilotError('Gaussian cohort config differs')
        if inputs['boundary']['max_train_frames'] is not None or inputs.get('initialization_selection', {}).get('max_frames') != 48:
            raise PilotError('full discovery requires complete TRAIN Gaussian staging and uniform48 initialization')
        expected_dslr = (Path(config['dataset_root'])/'data'/config['scene_id']/'dslr').resolve()
        if any(Path(value['path']) != expected_dslr/name for name,value in inputs['source_metadata'].items()):
            raise PilotError('Gaussian cohort metadata belongs to another scene')
    links = [(preparation['input_manifest'], d/'scene/training_inputs.json'),
             (triangulation['prepare_receipt'], d/'prepare_receipt.json'),
             (triangulation['init_manifest'], d/'initialization/init_manifest.json'),
             (training['prepare_receipt'], d/'prepare_receipt.json'),
             (training['triangulate_receipt'], d/'triangulate_receipt.json')]
    if any(file_identity(path) != expected for expected,path in links):
        raise PilotError('Gaussian phase receipt lineage differs')
    report = records[f'{prefix}/{phase}/train_report.json']
    if training['training_report'] != report or report['iters'] != 15000 or report['seed'] != 42 or report['independent_heldout_evaluation'] is not False:
        raise PilotError('Gaussian frozen training recipe/diagnostic scope differs')
    if report['provenance']['initialization'] != initialization or report['provenance']['official_test_images_read'] != 0:
        raise PilotError('Gaussian initialization or official TRAIN lineage differs')
    gaussian = Path(config['gaussian']).resolve(strict=True)
    if gaussian != d/phase/'scene.ply' or file_identity(gaussian) != training['artifacts']['scene.ply'] or sha(gaussian) != config['gaussian_sha256']:
        raise PilotError('Gaussian bytes differ from authenticated training artifact')
    frames = [r['name'] for r in inputs['frames']]
    if frames != boundary['training_frames'] or (config.get('scope') != COHORT_SCOPE and len(frames) != 48):
        raise PilotError('discovery and Gaussian training frame rosters differ')
    if any(r['image']['sha256'] != input_images[r['name']]['sha256'] for r in inputs['frames']):
        raise PilotError('discovery RGB differs from Gaussian training RGB')
    if any(value['sha256'] != metadata[name]['sha256'] for name,value in inputs['source_metadata'].items()):
        raise PilotError('discovery camera/split metadata differs from Gaussian source')
    return {'status': status, 'producer_commit': code['commit'], 'freeze_id':root.name,
            'training_frames':frames, 'init_points':initialization['n_points'],
            'gaussians':report['n_gaussians'], 'training_iterations':report['iters'],
            'independent_heldout_evaluation':False, 'anchors':proof['anchors']}


def load_context(config_path, freeze_root, scene_id=None):
    config_path, freeze_root = Path(config_path).resolve(), Path(freeze_root).resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get('scope') == COHORT_SCOPE:
        if 'scene_id' in config or scene_id is None:
            raise PilotError('cohort requires an explicit unit without narrowing its frozen config')
        config = dict(config, scene_id=scene_id)
    elif scene_id is not None:
        raise PilotError('legacy pilot does not accept a scene override')
    roster = yaml.safe_load((CODE / config['scene_roster']).read_text())
    validate_config(config, roster)
    contract = json.loads((freeze_root / 'contract/freeze_manifest.json').read_text())
    from robo.manifest.hash import canonical_hash
    payload = {k:v for k,v in contract.items() if k not in {'created_utc','environment','contract_sha256'}}
    if canonical_hash(payload) != contract['contract_sha256']:
        raise PilotError('E0 canonical seal differs')
    commit = subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain'], text=True).strip()
    if dirty or contract['code']['commit'] != commit or contract['code']['dirty']:
        raise PilotError('clean exact-source E0 contract required')
    if contract['freeze_id'] != freeze_root.name or config['freeze_id'] != freeze_root.name:
        raise PilotError('freeze ID differs')
    resources = contract['resource_inventory']
    bound = [r for r in resources if r['id'] == 'e3_auto_pilot_config']
    if len(bound) != 1 or bound[0]['sha256'] != sha(config_path):
        raise PilotError('pilot config differs from E0 frozen bytes')
    return config, commit


def plan(config_path, freeze_root, scene_id=None):
    from agents.core.common import load_colmap_w2c
    from agents.discover.training_views import select_training_views
    from robo.eval.fidelity_replacements import _tree_inventory

    config, commit = load_context(config_path, freeze_root, scene_id)
    destination = destination_for(config, freeze_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # Exclusive: failed plans keep their output reservation.
    if config.get('scope') == COHORT_SCOPE:
        config = bind_cohort_gaussian(config)
    source_scene = Path(config['dataset_root']) / 'data' / config['scene_id']
    views = sorted(load_colmap_w2c(source_scene / 'dslr/colmap/images.txt').items())
    _, boundary = select_training_views(views, source_scene / 'dslr/train_test_lists.json',
                                        1, config['max_train_frames'])
    frames = boundary['training_frames']
    images = source_scene / 'dslr/resized_undistorted_images'
    inputs = {name: identity(images / name) for name in frames}
    metadata = {name: identity(source_scene / 'dslr' / name) for name in (
        'colmap/images.txt', 'nerfstudio/transforms_undistorted.json', 'train_test_lists.json')}
    gaussian = identity(config['gaussian'])
    if gaussian['sha256'] != config['gaussian_sha256']:
        raise PilotError('inherited Gaussian differs from declared source bytes')
    gaussian_provenance = validate_gaussian_provenance(config, boundary, inputs, metadata)
    checkpoint = (cohort_resources(config, freeze_root, commit)[0]
                  if config.get('scope') == COHORT_SCOPE else identity(config['sam3_checkpoint']['path']))
    if checkpoint['sha256'] != config['sam3_checkpoint']['sha256'] or checkpoint['bytes'] != config['sam3_checkpoint']['bytes']:
        raise PilotError('SAM3 checkpoint differs from pinned content')
    source = _tree_inventory(Path(config['sam3_source']['path']), label='SAM3 source')
    if source['tree_sha256'] != config['sam3_source']['tree_sha256']:
        raise PilotError('SAM3 source tree differs from pinned content')
    if config.get('scope') == COHORT_SCOPE:
        validate_cohort_runtime(config)
    # The stage-visible scene physically has no scans directory or unselected
    # image. Incorrect legacy fallbacks therefore fail instead of reading GT.
    staged = destination / 'inputs/data' / config['scene_id'] / 'dslr'
    for name in metadata:
        target = staged / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_scene / 'dslr' / name, target)
    staged_images = staged / 'resized_undistorted_images'
    staged_images.mkdir()
    for name in frames:
        (staged_images / name).symlink_to(images / name)
    splats = destination / 'inputs/splats'
    splats.mkdir()
    (splats / f'{config["scene_id"]}.ply').symlink_to(config['gaussian'])
    write_new(destination / 'inputs/frame_allowlist.json', {'frames': frames})
    record = dict(schema_version=1, freeze_id=Path(freeze_root).name, code_commit=commit,
                  config_sha256=sha(config_path), paper_ready=False, scope=config['scope'],
                  scene_id=config['scene_id'],
                  source_gaussian_training_provenance=gaussian_provenance['status'],
                  gaussian_provenance=gaussian_provenance, boundary=boundary,
                  input_images=inputs, metadata=metadata, gaussian=gaussian,
                  sam3_checkpoint=checkpoint, sam3_source=source,
                  planned_stages=['render', 'fuse', 'discover', 'prepare', 'refine'],
                  instance_population='all automatic discovery instances, including preparation-filtered instances',
                  artifact_semantics={'gt_object_id': 'legacy key for automatic instance ID 1000+index',
                                      'gt_points.ply': 'derived construction-surface samples, not evaluation GT',
                                      'projected_mask_fallback': 'derived-surface projection; no GT annotation'})
    if config.get('scope') == COHORT_SCOPE:
        record.update(full_vocabulary=config['full_vocabulary'], discovery_prompts=config['discovery_prompts'],
                      runtime=config['runtime'], planned_scenes=50)
    write_new(destination / 'input_manifest.json', record)
    if config.get('scope') == COHORT_SCOPE:
        write_new(destination/'plan_receipt.json', {'scene_id':config['scene_id'],
            'code_commit':commit, 'config_sha256':sha(config_path),
            'input_manifest_sha256':sha(destination/'input_manifest.json')})
    print(json.dumps({'plan': str(destination / 'input_manifest.json'),
                      'scene': config['scene_id'], 'training_frames': len(frames),
                      'render_frames': len(frames[::3]), 'discovery_frames': len(frames[::12]),
                      'checkpoint_sha256': checkpoint['sha256'], 'paper_ready': False}))


def enforce_read_boundary(event, arguments, *, forbidden_scene, image_root, allowed_images):
    if event != 'open' or not isinstance(arguments[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(arguments[0])).resolve()
    # This guards Python reads; the staged scene independently omits scan GT,
    # including for C++ extensions that do not emit Python audit events.
    if path == forbidden_scene / 'scans' or forbidden_scene / 'scans' in path.parents:
        raise PilotError(f'forbidden evaluation geometry/annotation access: {path}')
    if image_root in path.parents and path.name not in allowed_images:
        raise PilotError(f'image outside frozen training boundary: {path}')


def validate_staged_inputs(destination, config, manifest):
    """Recheck actual worker-visible bytes and symlink identities before each stage."""
    scene = Path(destination) / 'inputs/data' / config['scene_id']
    if (scene / 'scans').exists():
        raise PilotError('stage-visible scene must not contain evaluation geometry')
    images = scene / 'dslr/resized_undistorted_images'
    frames = manifest['boundary']['training_frames']
    if sorted(p.name for p in images.iterdir()) != frames or sorted(manifest['input_images']) != frames:
        raise PilotError('staged image roster differs from frozen boundary')
    for name, item in manifest['input_images'].items():
        if (images / name).resolve(strict=True) != Path(item['path']):
            raise PilotError(f'staged image target differs: {name}')
    for name, item in manifest['metadata'].items():
        if sha(scene / 'dslr' / name) != item['sha256']:
            raise PilotError(f'staged metadata differs: {name}')
    allowlist = json.loads((Path(destination) / 'inputs/frame_allowlist.json').read_text())
    if allowlist != {'frames': frames}:
        raise PilotError('staged frame allowlist differs')
    gaussian = Path(destination) / 'inputs/splats' / f'{config["scene_id"]}.ply'
    if gaussian.resolve(strict=True) != Path(manifest['gaussian']['path']):
        raise PilotError('staged Gaussian target differs')
    if sha(gaussian) != manifest['gaussian']['sha256']:
        raise PilotError('staged Gaussian bytes differ')


def validate_gpu_memory(gpu):
    # A nominal 48 GB A6000 reports 46,068 MiB usable with the current driver.
    if gpu['total_bytes'] < 44 * 1024**3 or gpu['free_bytes'] < 30 * 1024**3:
        raise PilotError('single assigned GPU lacks the declared free memory budget')


def worker(config_path, freeze_root, stage, scene_id=None):
    """Execute one existing producer with a read boundary and peak-memory log."""
    import runpy
    config, _commit = load_context(config_path, freeze_root, scene_id)
    destination = destination_for(config, freeze_root)
    manifest = checked_manifest(config, _commit, config_path, destination)
    scene = destination / 'inputs/data' / config['scene_id']
    output = destination / 'construction'
    expected_env = {'SIMANY_ROOT': str(CODE), 'SIMANY_OUT': str(output),
                    'SIMANY_AUTO': '1', 'SIMANY_FULL': str(config.get('full_vocabulary', 0)), 'SIMANY_MESH_SRC': 'derived',
                    'SIMANY_SCANNETPP_ROOT': str(destination / 'inputs'),
                    'SIMANY_SAM3_CKPT': config['sam3_checkpoint']['path']}
    if config.get('scope') == COHORT_SCOPE:
        from run.icra2027.e3_gaussian_train_only import require_published_full_source
        require_published_full_source({'commit':_commit})
        interpreter = config['render_python'] if stage == 'render' else config['sam3_python']
        if Path(sys.executable).resolve() != Path(interpreter).resolve():
            raise PilotError('worker interpreter differs from frozen stage runtime')
        node = os.environ.get('SLURMD_NODENAME', '')
        if not os.environ.get('SLURM_JOB_ID') or not (node == 'hala' or node.startswith(('gcp','sof1'))):
            raise PilotError('full discovery worker requires an authorized Slurm node')
        expected_env.update(SIMANY_SCENE=config['scene_id'], SIMANY_NO_GT='1',
            SIMANY_SPLATS_ROOT=str(destination/'inputs/splats'), PYTHONNOUSERSITE='1')
    if any(os.environ.get(k) != v for k, v in expected_env.items()):
        raise PilotError('worker environment differs from GT-isolated protocol')
    validate_staged_inputs(destination, config, manifest)
    if config.get('scope') == COHORT_SCOPE:
        validate_input_stats(manifest)
        if not (Path(freeze_root)/'contract/discovery_resources.json').is_file():
            raise PilotError('planned shared discovery resource receipt is missing')
        checkpoint,source = cohort_resources(config, freeze_root, _commit)
        if manifest['sam3_checkpoint'] != checkpoint or manifest['sam3_source'] != source:
            raise PilotError('worker model/source differs from frozen plan')
        validate_cohort_runtime(config)
    split = str(scene / 'dslr/train_test_lists.json')
    limit_args = [] if config['max_train_frames'] is None else ['--max-train-frames', str(config['max_train_frames'])]
    commands = {
        'render': ('agents.discover.derive_mesh_from_splat', ['render', '--train-split', split, *limit_args]),
        'fuse': ('agents.discover.derive_mesh_from_splat', ['fuse', '--train-split', split]),
        'discover': ('agents.discover.auto_segment', ['--scene-dir', str(scene), '--out-dir', str(output), '--mesh-path', str(output / 'derived_mesh.ply'), '--train-split', split, *limit_args]),
        'prepare': ('agents.discover.factory_prepare', ['--frame-allowlist', str(destination / 'inputs/frame_allowlist.json'), '--read-frames-out', str(output / 'prepare_read_frames.json')]),
        'refine': ('agents.discover.factory_refine_masks', ['--images-dir', str(scene / 'dslr/resized_undistorted_images'), '--out-dir', str(output)]),
    }
    source_scene = Path(config['dataset_root']) / 'data' / config['scene_id']
    sys.addaudithook(lambda event, args: enforce_read_boundary(
        event, args, forbidden_scene=source_scene,
        image_root=(source_scene / 'dslr/resized_undistorted_images').resolve(),
        allowed_images=set(manifest['boundary']['training_frames'])))
    module, arguments = commands[stage]
    sys.argv = [module, *arguments]
    start = time.monotonic()
    runpy.run_module(module, run_name='__main__')
    import torch
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else None
    import resource
    print(json.dumps({'stage': stage, 'wall_s': time.monotonic() - start,
                      'peak_cuda_allocated_bytes': peak,
                      'peak_host_memory_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}), flush=True)


def summarize_instances(labels, objects):
    by_id = {int(o['gt_object_id']): o for o in objects}
    if len(by_id) != len(objects) or not set(by_id) <= {1000 + i for i in range(len(labels))}:
        raise PilotError('prepared objects duplicate or outside discovery denominator')
    rows = []
    for index, label in enumerate(labels):
        obj = by_id.get(1000 + index)
        rows.append({'automatic_instance_id': 1000 + index, 'label': str(label),
                     'prepared': obj is not None, 'output_index': obj['index'] if obj else None,
                     'failure_reason': None if obj else 'filtered_by_existing_preparation_gates',
                     'mask_source': obj.get('mask_source', 'derived_projection') if obj else None})
    return rows


STAGED_PHASES = {
    'run-render': ('render',),
    'run-fuse': ('fuse',),
    'run-objects': ('discover', 'prepare', 'refine'),
}


def staged_predecessors(phase):
    phases = list(STAGED_PHASES)
    if phase not in phases:
        raise PilotError('unknown staged execution phase')
    return phases[:phases.index(phase)]


def validate_stage_seal(destination, phase, commit, config_path):
    """Authenticate exact-source completed stages before releasing downstream work."""
    path = destination / f'{phase}_seal.json'
    seal = json.loads(path.read_text())
    if (seal['phase'] != phase or seal['code_commit'] != commit
            or seal['config_sha256'] != sha(config_path)
            or seal['input_manifest_sha256'] != sha(destination/'input_manifest.json')):
        raise PilotError('stage seal source/config/input differs')
    if set(seal['predecessors']) != set(staged_predecessors(phase)):
        raise PilotError('stage seal dependency set differs')
    for previous, digest in seal['predecessors'].items():
        if sha(destination/f'{previous}_seal.json') != digest:
            raise PilotError('stage predecessor seal changed')
    if [row['stage'] for row in seal['stages']] != list(STAGED_PHASES[phase]):
        raise PilotError('stage seal action set differs')
    if any(row.get('exit_code') != 0 for row in seal['stages']):
        raise PilotError('stage seal contains a failed action')
    if phase == 'run-render':
        actual = {str(p.relative_to(destination/'construction')) for p in (destination/'construction/mesh_derive').glob('*') if p.is_file()}
        if actual != set(seal['artifacts']):
            raise PilotError('render artifact roster changed')
    for relative, expected in seal['artifacts'].items():
        path = destination/'construction'/relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or path.is_symlink():
            raise PilotError('stage artifact path escapes construction')
        if identity(path) != expected:
            raise PilotError('stage artifact changed')
    required = ('mesh_derive/training_views.json' if phase == 'run-render' else 'derived_mesh.ply')
    if phase in {'run-render','run-fuse'} and required not in seal['artifacts']:
        raise PilotError('stage seal lacks its required artifact')
    if phase == 'run-render' and not any(name.startswith('mesh_derive/view_') and name.endswith('.npz') for name in seal['artifacts']):
        raise PilotError('stage seal lacks rendered views')
    return seal


def seal_stage(destination, phase, commit, config_path, completed):
    output = destination/'construction'
    paths = sorted((output/'mesh_derive').glob('*')) if phase == 'run-render' else [output/'derived_mesh.ply']
    if phase == 'run-objects':
        paths = sorted(output.rglob('*'))
    record = {'phase':phase, 'code_commit':commit, 'config_sha256':sha(config_path),
        'input_manifest_sha256':sha(destination/'input_manifest.json'),
        'predecessors':{name:sha(destination/f'{name}_seal.json') for name in staged_predecessors(phase)},
        'stages':completed, 'artifacts':{str(path.relative_to(output)):identity(path) for path in paths if path.is_file()},
        'job_id':os.environ['SLURM_JOB_ID'], 'paper_ready':False}
    write_new(destination/f'{phase}_seal.json', record)
    validate_stage_seal(destination, phase, commit, config_path)


def execute(config_path, freeze_root, scene_id=None, phase=None):
    from robo.eval.metric_utils import write_csv
    config, commit = load_context(config_path, freeze_root, scene_id)
    destination = destination_for(config, freeze_root)
    manifest = checked_manifest(config, commit, config_path, destination)
    if config.get('scope') == COHORT_SCOPE:
        from run.icra2027.e3_gaussian_train_only import require_published_full_source
        require_published_full_source({'commit':commit})
        config = bind_cohort_gaussian(config)
        validate_cohort_runtime(config)
    if manifest['code_commit'] != commit or manifest['config_sha256'] != sha(config_path):
        raise PilotError('input plan source/config differs')
    # Expensive content fingerprints were computed once before GPU launch.
    # Abort on source size/mtime drift rather than silently reusing the plan.
    validate_input_stats(manifest)
    from robo.eval.fidelity_replacements import _tree_inventory
    current_source = _tree_inventory(Path(config['sam3_source']['path']), label='SAM3 source')
    if current_source['tree_sha256'] != manifest['sam3_source']['tree_sha256']:
        raise PilotError('SAM3 source changed after input plan')
    validate_staged_inputs(destination, config, manifest)
    proof = validate_gaussian_provenance(config, manifest['boundary'], manifest['input_images'], manifest['metadata'])
    if proof != manifest.get('gaussian_provenance', {'status':'UNKNOWN','independent_heldout_evaluation':False}):
        raise PilotError('planned Gaussian provenance changed')
    node = os.environ.get('SLURMD_NODENAME', socket.gethostname())
    if not os.environ.get('SLURM_JOB_ID') or not (node == 'hala' or node.startswith(('gcp', 'sof1'))):
        raise PilotError('run requires Slurm on an authorized Hala/gcp-/sof1- node')
    selected = list(STAGED_PHASES[phase]) if phase is not None else manifest['planned_stages']
    completed = []
    if phase is not None:
        if config.get('scope') != COHORT_SCOPE:
            raise PilotError('separate stages require a complete frozen cohort')
        for previous in staged_predecessors(phase):
            completed.extend(validate_stage_seal(destination, previous, commit, config_path)['stages'])
        with (destination/f'{phase}.lock').open('x') as stream:
            stream.write(os.environ['SLURM_JOB_ID'])
    if phase != 'run-fuse':
        probe = subprocess.check_output([config['render_python'], '-c',
            'import torch,json; assert torch.cuda.device_count()==1; '
            'free,total=torch.cuda.mem_get_info(); '
            'print(json.dumps(dict(name=torch.cuda.get_device_name(),free_bytes=free,total_bytes=total,capability=torch.cuda.get_device_capability())))'], text=True)
        gpu = json.loads(probe.strip().splitlines()[-1])
        validate_gpu_memory(gpu)
    else:
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '', '-1', 'NoDevFiles'):
            raise PilotError('CPU fusion must not reserve or expose a GPU')
        gpu = {'gpu_reserved':False}
    runtime_name = f'{phase}_runtime.json' if phase else 'gpu_runtime.json'
    write_new(destination/runtime_name, dict(node=node, job_id=os.environ['SLURM_JOB_ID'], **gpu))
    output = destination/'construction'
    if phase in (None, 'run-render'):
        output.mkdir()
    elif not output.is_dir() or output.is_symlink():
        raise PilotError('staged construction directory is missing or redirected')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               PYTHONPATH=str(CODE) + ':' + config['sam3_source']['path'],
               SIMANY_ROOT=str(CODE), SIMANY_OUT=str(output), SIMANY_AUTO='1', SIMANY_FULL=str(config.get('full_vocabulary', 0)),
               SIMANY_SCENE=config['scene_id'], SIMANY_MESH_SRC='derived', SIMANY_NO_GT='1',
               SIMANY_SCANNETPP_ROOT=str(destination / 'inputs'),
               SIMANY_SPLATS_ROOT=str(destination / 'inputs/splats'),
               SIMANY_SAM3_CKPT=config['sam3_checkpoint']['path'],
               HF_HUB_OFFLINE='1', HF_HOME=config['hf_home'])
    phase_completed = []
    for stage in selected:
        if stage == 'refine' and not json.loads((output / 'objects/objects.json').read_text()):
            row = {'stage': stage, 'status': 'NOT_RUN', 'reason': 'empty_prepared_population', 'exit_code':0}
            completed.append(row)
            phase_completed.append(row)
            continue
        interpreter = config['render_python'] if stage == 'render' else config['sam3_python']
        log = destination / f'{stage}.log'
        command = [interpreter, str(Path(__file__).resolve()), '--config', str(Path(config_path).resolve()),
                   '--freeze-root', str(Path(freeze_root).resolve()), '--phase', 'worker', '--stage', stage]
        if scene_id is not None:
            command += ['--scene-id', scene_id]
        stage_env = dict(env)
        if stage == 'render' and config.get('renderer_dependency_root'):
            stage_env['PYTHONPATH'] += os.pathsep + config['renderer_dependency_root']
        start = time.monotonic()
        with log.open('x') as stream:
            code = subprocess.call(command, env=stage_env, cwd=CODE, stdout=stream, stderr=subprocess.STDOUT)
        row = dict(stage=stage, exit_code=code, wall_s=time.monotonic() - start, command=command, log=str(log))
        completed.append(row)
        phase_completed.append(row)
        write_new(destination / f'{stage}_execution.json', row)
        if code:
            write_new(destination / 'failure.json', {'failed_stage': stage, 'paper_ready': False, 'stages': completed})
            raise PilotError(f'{stage} failed; preserve {log} and use a new freeze after fixing source/config')
    if phase is not None:
        if load_context(config_path, freeze_root, scene_id)[1] != commit:
            raise PilotError('source changed during staged discovery')
        seal_stage(destination, phase, commit, config_path, phase_completed)
        if phase != 'run-objects':
            return {'phase':phase, 'status':'PASS', 'paper_ready':False}
    import numpy as np
    with np.load(output / 'auto_instances.npz', allow_pickle=False) as instances:
        labels = instances['labels'].tolist()
    objects = json.loads((output / 'objects/objects.json').read_text())
    rows = summarize_instances(labels, objects)
    if load_context(config_path, freeze_root, scene_id)[1] != commit:
        raise PilotError('source changed during discovery')
    result = dict(scope=config['scope'], freeze_id=Path(freeze_root).name, code_commit=commit,
                  scene_id=config['scene_id'], paper_ready=False,
                  source_gaussian_training_provenance=proof['status'], gaussian_provenance=proof,
                  discovered_instances=len(rows), prepared_instances=len(objects),
                  prepared_coverage=len(objects) / len(rows) if rows else None,
                  degenerate_empty_discovery=not rows, stages=completed, rows=rows)
    write_new(destination / 'pilot_summary.json', result)
    write_csv(destination / 'instance_records.csv', rows)
    write_new(destination / 'output_hashes.json', {str(p.relative_to(output)): identity(p)
              for p in sorted(output.rglob('*')) if p.is_file()})
    all_jobs = {'schema_version':1, 'kind':'complete_automatic_discovery_jobs',
        'scene_id':config['scene_id'], 'freeze_id':Path(freeze_root).name,
        'code_commit':commit, 'paper_ready':False, 'planned_jobs':len(rows),
        'rows':rows, 'input_manifest_sha256':sha(destination/'input_manifest.json'),
        'summary_sha256':sha(destination/'pilot_summary.json'),
        'output_hashes_sha256':sha(destination/'output_hashes.json'),
        'source_gaussian_training_provenance':proof['status']}
    write_new(destination/'all_jobs_manifest.json', all_jobs)
    write_new(destination/'postrun_audit.json', {
        'source_commit':commit, 'job_id':os.environ['SLURM_JOB_ID'], 'paper_ready':False,
        'summary_sha256':all_jobs['summary_sha256'],
        'output_hashes_sha256':all_jobs['output_hashes_sha256'],
        'input_manifest_sha256':all_jobs['input_manifest_sha256'],
        'all_jobs_manifest_sha256':sha(destination/'all_jobs_manifest.json'),
        'source_gaussian_training_provenance':proof['status'],
        'scope':'producer completion seal; scheduler terminal accounting remains separate'})
    print(json.dumps(result, indent=2))


def compare_crops(config_path, freeze_root):
    """Compare bytes across complete populations; never infer cross-run object IDs."""
    from run.icra2027.e3_trellis_generation_pilot import source_jobs
    from agents.recon.colmap_poses import file_identity
    config, commit = load_context(config_path, freeze_root)
    destination = Path(freeze_root)/'auto_discovery_pilot'
    reference = config['crop_comparison_reference']
    previous = Path(reference['directory'])
    required = {'pilot_summary.json','output_hashes.json','input_manifest.json','postrun_audit.json'}
    if set(reference['anchors']) != required:
        raise PilotError('crop-comparison reference closure differs')
    for name, expected in reference['anchors'].items():
        if file_identity(previous/name) != expected:
            raise PilotError('crop-comparison reference changed')
    old = source_jobs(previous)
    current = source_jobs(destination)
    summary = json.loads((destination/'pilot_summary.json').read_text())
    if summary['code_commit'] != commit or summary['freeze_id'] != Path(freeze_root).name:
        raise PilotError('new crop population source differs')
    rows = []
    for job in current:
        matches = [r['job_id'] for r in old if r['prepared'] and job['prepared']
                   and r['input']['sha256'] == job['input']['sha256']]
        rows.append({'job_id':job['job_id'], 'prepared':job['prepared'],
            'input_sha256':job['input']['sha256'] if job['prepared'] else None,
            'exact_matching_prior_jobs':matches,
            'exact_rgba_match':bool(matches) if job['prepared'] else None})
    result = {'scope':'input-byte comparison only; no cross-run object identity assumption',
        'code_commit':commit, 'freeze_id':Path(freeze_root).name, 'paper_ready':False,
        'new_planned_jobs':len(current), 'prior_planned_jobs':len(old),
        'exact_match_prepared_inputs':sum(bool(r['exact_rgba_match']) for r in rows),
        'artifact_reuse_performed':False, 'generation_performed':False,
        'reuse_requirement':'exact RGBA bytes are necessary; model/seed/runtime identity must also match before raw artifact reuse; registration is recomputed',
        'reference':reference, 'new_all_jobs_sha256':sha(destination/'all_jobs_manifest.json'),
        'rows':rows}
    write_new(destination/'crop_hash_comparison.json',result)
    print(json.dumps(result,indent=2))


def cohort_status(config_path, freeze_root, output_path):
    """Snapshot all planned scenes, retaining unavailable/failed/empty units."""
    from run.icra2027.e3_gaussian_train_only import cohort_scene_ids
    from run.icra2027.e3_trellis_generation_pilot import source_jobs
    raw = yaml.safe_load(Path(config_path).read_text())
    scenes = cohort_scene_ids(raw)
    config, commit = load_context(config_path, freeze_root, scenes[0])
    if config['scope'] != COHORT_SCOPE:
        raise PilotError('cohort status requires complete population scope')
    rows = []
    for scene in scenes:
        unit = destination_for(dict(config, scene_id=scene), freeze_root)
        row = {'scene_id':scene, 'state':'NOT_RUN', 'planned_jobs':None, 'prepared_jobs':None,
               'output_path':str(unit), 'failure':None}
        if (unit/'postrun_audit.json').exists():
            summary = json.loads((unit/'pilot_summary.json').read_text())
            if (summary['scene_id'], summary['code_commit'], summary['freeze_id']) != (scene,commit,config['freeze_id']):
                raise PilotError('cohort completion source/scene differs')
            jobs = source_jobs(unit)
            row.update(state='COMPLETE', planned_jobs=len(jobs), prepared_jobs=sum(j['prepared'] for j in jobs))
        elif (unit/'failure.json').exists():
            row.update(state='BLOCKED',failure=json.loads((unit/'failure.json').read_text()))
        elif (unit/'unit_failure.json').exists():
            row.update(state='BLOCKED',failure=json.loads((unit/'unit_failure.json').read_text()))
        elif (unit/'input_manifest.json').exists():
            row['state'] = 'NOT_STARTED' if not (unit/'construction').exists() else 'FULL_RUNNING'
            row['input_plan_ready'] = True
        rows.append(row)
    result = {'scope':COHORT_SCOPE, 'paper_ready':False, 'code_commit':commit,
        'freeze_id':config['freeze_id'], 'config_sha256':sha(config_path), 'planned_scenes':50,
        'completed_scenes':sum(r['state']=='COMPLETE' for r in rows), 'rows':rows,
        'unknown_job_denominators_are_not_zero':True}
    write_new(output_path,result)
    return result


def run_unit_phase(config_path, freeze_root, phase, scene_id):
    """Keep a typed failed-unit receipt without deleting or restarting outputs."""
    config, commit = load_context(config_path, freeze_root, scene_id)
    try:
        if phase in STAGED_PHASES:
            return execute(config_path,freeze_root,scene_id,phase=phase)
        return {'plan':plan, 'run':execute}[phase](config_path,freeze_root,scene_id)
    except Exception as exc:
        if config.get('scope') == COHORT_SCOPE and not isinstance(exc, FileExistsError):
            path = destination_for(config,freeze_root)/'unit_failure.json'
            if not path.exists():
                write_new(path, {'scene_id':scene_id, 'code_commit':commit,
                    'config_sha256':sha(config_path), 'phase':phase, 'paper_ready':False,
                    'classification':'missing_data_or_pending_dependency' if isinstance(exc,FileNotFoundError) else 'validation_or_stage_failure',
                    'error':f'{type(exc).__name__}: {exc}'})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--freeze-root', required=True)
    parser.add_argument('--phase', choices=['plan', 'run', 'worker', 'compare-crops', 'cohort-status', *STAGED_PHASES], required=True)
    parser.add_argument('--stage', choices=['render', 'fuse', 'discover', 'prepare', 'refine'])
    parser.add_argument('--scene-id', help='explicit unit of a complete frozen cohort')
    parser.add_argument('--status-out', help='new non-overwriting complete-cohort status snapshot')
    args = parser.parse_args()
    if args.phase in {'plan','run', *STAGED_PHASES}:
        run_unit_phase(args.config, args.freeze_root, args.phase, args.scene_id)
    elif args.phase == 'cohort-status':
        if not args.status_out: parser.error('cohort-status requires --status-out')
        cohort_status(args.config,args.freeze_root,args.status_out)
    elif args.phase == 'compare-crops':
        if args.scene_id is not None:
            parser.error('cohort runs do not adopt legacy comparison crops')
        compare_crops(args.config, args.freeze_root)
    else:
        worker(args.config, args.freeze_root, args.stage, args.scene_id)


if __name__ == '__main__':
    main()
