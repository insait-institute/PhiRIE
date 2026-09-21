"""CPU-only E2 input readiness for a sealed fresh automatic engineering pilot.

This audits source receipts and prospective official-test cameras. It produces
no metric records, renders, masks, registration, or new evaluation surfaces.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import yaml
from PIL import Image

from agents.eval.fidelity_room_export import select_official_frames
from agents.recon.colmap_poses import pose_headers
from robo.eval.fidelity_metrics import ROOM_METHODS, OBJECT_METHODS
from run.icra2027.e3_auto_discovery_pilot import identity, sha, write_new, PilotError
from run.icra2027.e3_fresh_generation_contract import checked_identity, FRESH
from run.icra2027.e3_rvg_generation_pilot import source_boundary, validate_view_receipt


def disjoint_evaluation(split, registered, training, generator_frames, *, count=8):
    if len(training) != len(set(training)) or not set(training) <= set(split['train']):
        raise PilotError('training roster is not a unique official TRAIN subset')
    if not set(generator_frames) <= set(training):
        raise PilotError('generation view is outside complete training boundary')
    # Exclude the whole official TRAIN split, not merely its selected 48 frames.
    selected = select_official_frames(split, set(registered), count=count,
                                     excluded_names=set(split['train']) | set(generator_frames))
    if set(selected) & (set(training) | set(generator_frames)):
        raise PilotError('evaluation overlaps optimization inputs')
    return selected


def check_content_disjoint(evaluation_hashes, training_hashes):
    if set(evaluation_hashes) & set(training_hashes):
        raise PilotError('held-out RGB bytes duplicate an optimization input')


def check_pool(spec, jobs, source_hashes):
    files = {name: checked_identity(record) for name, record in spec.items()}
    pool, inputs, audit = (json.loads(files[n].read_text()) for n in
                          ('proposal_pool.json', 'input_manifest.json', 'postrun_audit.json'))
    for value in (pool, inputs, audit):
        if value['source_gaussian_training_provenance'] != FRESH:
            raise PilotError('mixed Gaussian provenance in generation pool')
        if value['source_discovery_hashes'] != source_hashes:
            raise PilotError('mixed discovery source in generation pool')
    if (pool['input_manifest_sha256'] != sha(files['input_manifest.json'])
            or audit['input_manifest_sha256'] != sha(files['input_manifest.json'])
            or audit['proposal_pool_sha256'] != sha(files['proposal_pool.json'])
            or pool['code_commit'] != inputs['code_commit']
            or pool['code_commit'] != audit.get('producer_source_commit', audit.get('source_commit'))):
        raise PilotError('generation completion receipt link differs')
    for rows in (pool['rows'], inputs['jobs']):
        if [(r['job_id'], r['prepared']) for r in rows] != [(r['job_id'], r['prepared']) for r in jobs]:
            raise PilotError('generation pool dropped or changed a discovery job')
    if pool['planned_jobs'] != len(jobs):
        raise PilotError('generation denominator differs')
    # Completion was independently audited by the producer; retain its sealed
    # receipt. This readiness audit does not substitute a new terminal validator.
    return pool, inputs


def audit(config_path):
    config_path = Path(config_path)
    c = json.loads(config_path.read_text())
    recipe_path = checked_identity(c['rvg_config'])
    recipe = yaml.safe_load(recipe_path.read_text())
    # Existing pose helpers import common's path constants; keep that import in
    # the automatic/derived context even though this audit opens no mesh.
    os.environ.update(SIMANY_SCENE=recipe['scene_id'], SIMANY_AUTO='1',
                      SIMANY_MESH_SRC='derived', SIMANY_NO_GT='1')
    jobs, boundary, binding = source_boundary(recipe)
    training = boundary['boundary']['training_frames']
    if len(training) != 48 or binding['gaussian_provenance']['training_frames'] != training:
        raise PilotError('fresh Gaussian and discovery 48-frame boundary differs')
    pools = {}
    for name, spec in c['pools'].items():
        pools[name], _ = check_pool(spec, jobs, binding['source_discovery_hashes'])
    if set(pools) != {'trellis', 'reconviagen'}:
        raise PilotError('both frozen initial tools are required')
    rvg_out = Path(c['pools']['reconviagen']['proposal_pool.json']['path']).parent
    rvg_inputs = json.loads((rvg_out/'input_manifest.json').read_text())
    views = validate_view_receipt(recipe, rvg_out, rvg_inputs, boundary)
    if (pools['reconviagen']['view_manifest_sha256'] != sha(rvg_out/'view_manifest.json')
            or pools['reconviagen']['views_receipt_sha256'] != sha(rvg_out/'views_receipt.json')):
        raise PilotError('RVG pool does not bind its multiview receipts')
    # Validate the sealed per-object consumed PNG identities as recorded by the
    # existing producer, without rerunning its collector or a model.
    from run.icra2027.e3_rvg_generation_pilot import verify_consumed_views
    for row in views['rows']:
        if row['views']:
            observed = verify_consumed_views(rvg_out, row['object_index'], row['views'])
            pool_row = next(r for r in pools['reconviagen']['rows']
                            if r['automatic_instance_id'] == row['automatic_instance_id'])
            if observed != pool_row['runtime']['consumed_views']:
                raise PilotError('consumed RVG view receipt differs')
    generated = sorted({j['frame'] for j in jobs if j['prepared']} |
                       {v['frame'] for row in views['rows'] for v in row['views']})
    metadata = {k: checked_identity(v) for k, v in boundary['metadata'].items()}
    split = json.loads(metadata['train_test_lists.json'].read_text())
    poses = pose_headers(metadata['colmap/images.txt'])
    selected = disjoint_evaluation(split, poses, training, generated)
    image_root = metadata['train_test_lists.json'].parent/'resized_undistorted_images'
    # Missing cameras/images never cause replacement by later, easier frames.
    available = []
    for name in split['test']:
        if Path(name).name != name or name in {'.', '..'}:
            raise PilotError('unsafe official-test filename')
        path = image_root/name
        available.append({'frame': name, 'registered': name in poses,
                          'exists': path.is_file(), 'selected': name in selected})
    selected_images = [dict(frame=name, **identity(image_root/name),
                            w2c=poses[name].tolist()) for name in selected]
    calibration_path = Path(recipe['source_discovery_config']['path'])
    discovery_config = yaml.safe_load(calibration_path.read_text())
    training_manifest = (Path(discovery_config['gaussian_provenance']['freeze_root']) /
                         'gaussian_train_only/scene/training_inputs.json')
    calibration = json.loads(training_manifest.read_text())['calibration']
    for row in selected_images:
        with Image.open(row['path']) as image:
            if image.size != (calibration['width'], calibration['height']):
                raise PilotError('held-out RGB dimensions differ from source calibration')
            row['width'], row['height'] = image.size
    train_hashes = [v['sha256'] for v in boundary['input_images'].values()]
    check_content_disjoint([v['sha256'] for v in selected_images], train_hashes)
    source = Path(recipe['source_pilot'])
    outputs = json.loads((source/'output_hashes.json').read_text())
    surfaces = {name: value for name, value in outputs.items()
                if name == 'derived_mesh.ply' or name.endswith('/gt_points.ply')}
    objects = []
    for job in jobs:
        objects.append(dict(job_id=job['job_id'], prepared=job['prepared'],
            initial_tool_status={k: next(r['status'] for r in p['rows']
                if r['job_id'] == job['job_id']) for k, p in pools.items()},
            appearance_ready=False, geometry_ready=False,
            missing_inputs=['independent_heldout_photo_masks', 'source_bound_registered_predictions',
                            'independent_evaluation_surface_and_metric_scale_proof'],
            preparation_failure=job.get('failure_reason') if not job['prepared'] else None))
    checkout = Path(__file__).resolve().parents[2]
    return dict(schema_version=1, scope='single_scene_fresh_E2_input_readiness',
        paper_ready=False, metric_records_created=0, gpu_launched=False,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=checkout, text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','status','--porcelain'], cwd=checkout, text=True)),
        config=identity(config_path), scene_id=boundary['scene_id'],
        public_camera_metadata=boundary['metadata'],
        calibration=calibration,
        source_gaussian_training_provenance=FRESH,
        gaussian_provenance=binding['gaussian_provenance'],
        source_discovery_hashes=binding['source_discovery_hashes'],
        generation_anchors=c['pools'], planned_objects=len(jobs),
        prepared_objects=sum(j['prepared'] for j in jobs), objects=objects,
        official_train_count=len(split['train']), official_test_count=len(split['test']),
        optimization_input_frames=training, generation_input_frames=generated,
        selected_evaluation_frames=selected, evaluation_images=selected_images,
        official_test_availability=available, train_evaluation_overlap=[],
        generator_evaluation_overlap=[], selected_rgb_training_byte_overlap=[],
        selection_rule='existing fidelity_room_export.select_official_frames(count=8); no outcome filtering',
        room_input_readiness='OFFICIAL_TEST_RGB_AND_POSES_AVAILABLE_RENDER_BUNDLE_MISSING',
        room_methods={name: {'metric_ready': False, 'missing': (
            ['frozen_raw_gaussian_render_bundle', 'exact_source_E0_and_pilot_smoke'] if i == 0 else
            ['fresh_method_scene_build', 'paired_heldout_render_bundle'])}
            for i, name in enumerate(ROOM_METHODS)},
        object_methods={name: {'metric_ready': False} for name in OBJECT_METHODS},
        prepared_target_surface_source_frames=training,
        prepared_target_surface_evaluation_frame_overlap=[],
        actual_registration_receipt_available=False,
        registration_status='raw initial pools have no authenticated registration result; future targets must be TRAIN-only',
        excluded_evaluation_surfaces=surfaces,
        surface_exclusion_reason='derived from the same TRAIN Gaussian used for discovery, masks and registration targets; different samples or hashes do not establish independence',
        next_room_step='minimal raw-Gaussian render builder using existing renderer/camera selector, then canonical fidelity_metrics; separate source/E0/config and smoke required',
        limitations=['One scene readiness is not the fixed 50-scene E2 population.',
            'No GT-controlled producer, scans, semantic labels or oracle outputs were read.',
            'Official camera calibration may use dataset-wide SfM; image optimization boundary is separately frozen.',
            'Object held-out visibility and masks are unverified; no geometry independence is inferred from file names or unequal hashes.',
            'Internal v30 diagnostics are not independent held-out metrics.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError('refusing to overwrite readiness audit')
    report = audit(args.config)
    write_new(args.out, report)
    print(json.dumps({'output': identity(args.out), 'planned_objects': report['planned_objects'],
                      'room_input_readiness': report['room_input_readiness']}))


if __name__ == '__main__':
    main()
