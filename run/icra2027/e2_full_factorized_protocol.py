"""Authenticate the full E2 roster; delegate construction to existing producers."""
from pathlib import Path
import argparse
import json
import yaml

from agents.edit.inpaint_masks import _checked, _identity, _bound_contract
from run.icra2027 import e2_raw_room as raw

SCOPE = 'full_public_factorized_official_test_protocol'
CONSTRUCTION_COMMIT = '0a8b4caa6005dd9201b58ee028f11c26ef06c34a'
CONSTRUCTION_FREEZE = '20260905-859f51d-v1'
CONSTRUCTION_GATE_SHA = '2a5c0285bf734d20dc803c21d2bd10920ebe118e0404fd1efde0730f64d21252'
SOURCE_ANCHORS = {'resolved_jobs': '266b37a3319b63b3fe69133ad2c7e93c0cda2497a0ef83d38b0b9ca3ac1006cb',
    'camera_bank': 'd3c9d7b2d77e2535de3ef5c9770a1362eb2818032f0b28cd09768792b3e532c4',
    'factory_execution': 'e252c76307ad1d342922d4746cd1641507b61fa61f982bd925470d377627c58b'}


def read(ref):
    path = _checked({**ref, 'bytes': ref.get('bytes', ref.get('size_bytes'))})
    return yaml.safe_load(path.read_text())


def validate_protocol(path):
    """Read sealed construction and camera metadata only; never RGB or metrics."""
    p = yaml.safe_load(Path(path).read_text())
    if (p['schema_version'] != 1 or p['scope'] != SCOPE or p['policy_id'] != 'A4'
            or p['planned_scenes'] != 50 or p['planned_objects'] != 1871
            or p['planned_views'] != 400 or p['views_per_scene'] != 8
            or p['paper_ready'] is not False or p['thresholds_changed'] is not False
            or p['pilot_rule'] != 'first_lexicographic_scene_before_background_outcomes'
            or p['missing_background_policy'] != 'retain_all_eight_missing_views_no_raw_fallback'
            or p['quality_support'] != 'all_common_available_views_after_complete_coverage_accounting'
            or p['construction_gate']['sha256'] != CONSTRUCTION_GATE_SHA):
        raise ValueError('full E2 predeclared scope differs')
    if any(p[k]['sha256'] != digest for k, digest in SOURCE_ANCHORS.items()):
        raise ValueError('full E2 source anchor differs')
    gate = read(p['construction_gate']); jobs = read(p['resolved_jobs'])
    if (gate['status'] != 'PASS' or gate['scope'] != 'complete_construction_integrity_only'
            or gate['source_commit'] != CONSTRUCTION_COMMIT or gate['freeze_id'] != CONSTRUCTION_FREEZE
            or gate['planned_scenes'] != 50 or gate['planned_jobs'] != 1871
            or gate['evaluation_geometry_read'] is not False
            or jobs['freeze_id'] != CONSTRUCTION_FREEZE
            or jobs['counts'] != {'scenes': 50, 'jobs': 1871, 'policy_object_rows': 9355}):
        raise ValueError('full E2 construction gate or denominator differs')
    roster = [r['scene_id'] for r in jobs['scenes']]
    population = {r['scene_id']: len(r['jobs']) for r in jobs['scenes']}
    if (len(set(roster)) != 50 or roster != sorted(roster) or p['scene_ids'] != roster
            or p['population'] != population or p['pilot_scene'] != roster[0]
            or [r['scene_id'] for r in gate['scenes']] != roster):
        raise ValueError('full E2 roster/count/pilot differs')
    bank = read(p['camera_bank'])
    if (bank['scope'] != 'exact_serialized_e2_camera_bank'
            or [r['scene_id'] for r in bank['scenes']] != roster
            or set(p['evaluation_frames']) != set(roster)):
        raise ValueError('full E2 original camera roster differs')
    for row in bank['scenes']:
        plan = read(row['plan']); sid = row['scene_id']
        raw.validate_camera_rows(plan['evaluation_images'])
        frames = [r['frame'] for r in plan['evaluation_images']]
        if (plan['scene_id'] != sid or p['evaluation_frames'][sid] != frames
                or set(frames) & set(plan['optimization_input_frames'])
                or plan['source_gaussian_training_provenance'] != 'FRESH_OFFICIAL_TRAIN_ONLY'):
            raise ValueError('full E2 TRAIN/TEST camera plan differs')
    # Factory producer is a distinct immutable source; no qualifier gate or
    # task eligibility may select the E2 population.
    source = read(p['factory_execution'])
    if (Path(p['factory_freeze_root']).name != source['freeze_id']
            or source['source']['control_audit'] != p['construction_gate']
            or source['source']['e3_root'] != str(Path(p['resolved_jobs']['path']).parents[1])
            or set(source['source']['scenes']) != set(roster)):
        raise ValueError('full E2 factory source differs')
    return p


def validate_prepare(config_path, contract_path, protocol_path):
    c, contract = _bound_contract(Path(config_path).absolute(), Path(contract_path).absolute())
    p = validate_protocol(protocol_path)
    entries = [r for r in contract['resource_inventory']
               if r.get('resolved_path') == str(Path(protocol_path).absolute())]
    if len(entries) != 1 or entries[0]['sha256'] != _identity(Path(protocol_path).absolute())['sha256']:
        raise ValueError('full E2 protocol is not E0-bound')
    sid = c['scene_id']
    factory_root = Path(p['factory_freeze_root'])
    if (sid != p['pilot_scene'] or c['policy_id'] != 'A4'
            or _checked(c['materialization_manifest']) != factory_root/'automatic_candidates'/sid/'materialized/A4/materialization_manifest.json'):
        raise ValueError('only predeclared full E2 preparation pilot is admitted')
    manifest = read(c['materialization_manifest'])
    if (manifest['e3_code_commit'] != CONSTRUCTION_COMMIT
            or manifest['e3_freeze_id'] != CONSTRUCTION_FREEZE
            or manifest['roster']['job_count'] != p['population'][sid]
            or manifest['roster']['object_slots'] != [r['object_slot'] for scene in read(p['resolved_jobs'])['scenes']
                                                   if scene['scene_id'] == sid for r in scene['jobs']]):
        raise ValueError('full E2 factory population differs')
    return c, contract, p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True); ap.add_argument('--config')
    ap.add_argument('--contract'); ap.add_argument('--phase', choices=['protocol', 'prepare'], required=True)
    args = ap.parse_args()
    if args.phase == 'protocol':
        result = validate_protocol(args.protocol)
        print(json.dumps(dict(status='PASS', planned_scenes=result['planned_scenes'], planned_objects=result['planned_objects'], pilot_scene=result['pilot_scene'])))
    else:
        validate_prepare(args.config, args.contract, args.protocol)
        from agents.edit.inpaint_prepare import run_public
        run_public(args.config, args.contract)

if __name__ == '__main__':
    main()
