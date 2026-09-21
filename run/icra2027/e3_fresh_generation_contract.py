"""Fresh discovery and pure TRELLIS reuse bindings; never invokes a model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from run.icra2027.e3_auto_discovery_pilot import (
    CODE, PilotError, identity, sha, validate_gaussian_provenance,
)
from robo.manifest.hash import canonical_hash

FRESH = 'FRESH_OFFICIAL_TRAIN_ONLY'
DISCOVERY_FILES = frozenset({
    'input_manifest.json', 'pilot_summary.json', 'output_hashes.json',
    'postrun_audit.json', 'all_jobs_manifest.json',
})
RAW_FILES = ('trellis_mesh.ply', 'trellis_gs.ply', 'mesh_sim.ply', 'mesh_sim.obj')
ALGORITHM_FILES = frozenset({'models/s4_trellis.py', 'agents/core/common.py'})
PRIOR_FILES = frozenset({
    'input_manifest.json', 'proposal_pool.json', 'postrun_audit.json',
    'python_runtime.json', 'runtime.json',
})


def checked_identity(record):
    path = Path(record['path'])
    if path.is_symlink():
        raise PilotError('source anchor may not be a symlink')
    actual = identity(path)
    if any(actual[k] != record[k] for k in ('path', 'bytes', 'sha256')):
        raise PilotError(f'source anchor changed: {path}')
    return path


def checked_contract(record, commit, config_sha256, resource_id):
    path = checked_identity(record)
    contract = json.loads(path.read_text())
    payload = {k:v for k,v in contract.items()
               if k not in {'created_utc', 'environment', 'contract_sha256'}}
    resources = [r for r in contract['resource_inventory'] if r['id'] == resource_id]
    if (canonical_hash(payload) != contract['contract_sha256']
            or contract['code']['commit'] != commit or contract['code']['dirty']
            or len(resources) != 1 or resources[0]['sha256'] != config_sha256):
        raise PilotError('source E0/config/code closure differs')
    return contract


def discovery_binding(config, jobs_loader):
    """Config hashes are trust anchors; staging cannot reseal edited source data."""
    source = Path(config['source_pilot'])
    summary = json.loads((source/'pilot_summary.json').read_text())
    manifest = json.loads((source/'input_manifest.json').read_text())
    status = config.get('source_gaussian_training_provenance', 'UNKNOWN')
    if (summary.get('source_gaussian_training_provenance', 'UNKNOWN') != status
            or manifest.get('source_gaussian_training_provenance', 'UNKNOWN') != status):
        raise PilotError('source Gaussian provenance status differs')
    if status not in {'UNKNOWN', FRESH}:
        raise PilotError('unsupported Gaussian provenance status')
    if (sha(source/'pilot_summary.json') != config['source_summary_sha256']
            or sha(source/'output_hashes.json') != config['source_output_hashes_sha256']):
        raise PilotError('frozen source identity differs')
    result = {'source_gaussian_training_provenance':status}
    if status == FRESH:
        hashes = config.get('source_discovery_hashes', {})
        if set(hashes) != DISCOVERY_FILES:
            raise PilotError('fresh source requires exact five discovery hashes')
        for name, digest in hashes.items():
            if (source/name).is_symlink() or sha(source/name) != digest:
                raise PilotError(f'frozen discovery hash differs: {name}')
        config_path = checked_identity(config['source_discovery_config'])
        source_config = yaml.safe_load(config_path.read_text())
        commit = config['source_discovery_commit']
        contract = checked_contract(config['source_discovery_contract'], commit,
                                    sha(config_path), 'e3_auto_pilot_config')
        if source_config.get('scope') == 'gt_isolated_full_cohort_discovery_crops':
            from run.icra2027.e3_auto_discovery_pilot import bind_cohort_gaussian
            from run.icra2027.e3_gaussian_train_only import cohort_scene_ids
            scene = summary['scene_id']
            if 'scene_id' in source_config or scene not in cohort_scene_ids(source_config):
                raise PilotError('generation source scene outside frozen discovery cohort')
            expected_source = Path(config['source_discovery_contract']['path']).parent.parent / 'auto_discovery_pilot' / scene
            if source.resolve() != expected_source.resolve() or manifest.get('scene_id') != scene:
                raise PilotError('generation discovery scene path differs')
            source_config = bind_cohort_gaussian(dict(source_config, scene_id=scene))
        if (manifest['config_sha256'] != sha(config_path)
                or manifest['code_commit'] != commit or summary['code_commit'] != commit
                or summary['freeze_id'] != contract['freeze_id']
                or source_config['freeze_id'] != contract['freeze_id']
                or summary['scene_id'] != source_config['scene_id']):
            raise PilotError('fresh discovery source/config/frame identity differs')
        proof = validate_gaussian_provenance(source_config, manifest['boundary'],
                                             manifest['input_images'], manifest['metadata'])
        if (proof['status'] != FRESH or manifest.get('gaussian_provenance') != proof
                or summary.get('gaussian_provenance') != proof):
            raise PilotError('fresh Gaussian receipt proof differs')
        result.update(source_discovery_hashes=dict(hashes), gaussian_provenance=proof)
    elif config.get('source_discovery_hashes') or config.get('raw_reuse'):
        raise PilotError('fresh discovery/reuse fields cannot label UNKNOWN source')
    return jobs_loader(source), result


def validate_crop_comparison(config, jobs):
    source = Path(config['source_pilot'])
    path = source/'crop_hash_comparison.json'
    if sha(path) != config['crop_comparison_sha256']:
        raise PilotError('crop comparison hash differs')
    comparison = json.loads(path.read_text())
    if (comparison['paper_ready'] is not False
            or comparison['generation_performed'] is not False
            or comparison['artifact_reuse_performed'] is not False
            or comparison['new_all_jobs_sha256'] != config['source_discovery_hashes']['all_jobs_manifest.json']
            or comparison['code_commit'] != config['source_discovery_commit']
            or comparison['new_planned_jobs'] != len(jobs)
            or [r['job_id'] for r in comparison['rows']] != [j['job_id'] for j in jobs]):
        raise PilotError('crop comparison population/source differs')
    for row, job in zip(comparison['rows'], jobs):
        if (row['prepared'] != job['prepared'] or row['input_sha256'] !=
                (job['input']['sha256'] if job['prepared'] else None)):
            raise PilotError('crop comparison input identity differs')
    return comparison


def historical_algorithm_identity(prior_commit):
    return {name:hashlib.sha256(subprocess.check_output(
        ['git','show',f'{prior_commit}:{name}'],cwd=CODE)).hexdigest()
        for name in sorted(ALGORITHM_FILES)}


def algorithm_identity(config, prior_commit):
    expected = config['raw_reuse']['algorithm_files']
    if set(expected) != ALGORITHM_FILES:
        raise PilotError('raw algorithm closure differs')
    if historical_algorithm_identity(prior_commit) != expected:
        raise PilotError('frozen historical raw source algorithm differs')
    matches=all((CODE/name).is_file() and sha(CODE/name)==digest
                for name,digest in expected.items())
    return {'files':dict(expected),'current_matches':matches}


def runtime_reuse_eligibility(config, old_config, old_inputs):
    """Version equality is not proof of original runtime bytes."""
    old = old_config.get('runtime_byte_identity')
    new = config.get('runtime_byte_identity')
    if old is None or new is None or old_inputs.get('runtime_byte_identity') != old:
        return False, 'original_independently_anchored_runtime_bytes_unavailable'
    checked_identity(old); checked_identity(new)
    # No historical producer in this study emitted a validated closed runtime
    # inventory. A caller-supplied "complete" label cannot create that evidence.
    # Adding a supported issuer/schema requires a separate reviewed producer.
    return False, 'unsupported_original_runtime_byte_manifest_producer_or_schema'


def prior_trellis_source(config, jobs_loader):
    """Authenticate original generation, preserving its original source/runtime."""
    spec = config['raw_reuse']
    root = Path(spec['directory'])
    if set(spec['anchors']) != PRIOR_FILES:
        raise PilotError('prior TRELLIS source closure differs')
    records = {}
    for name, record in spec['anchors'].items():
        if Path(record['path']) != root/name:
            raise PilotError('prior source anchor path differs')
        records[name] = json.loads(checked_identity(record).read_text())
    cfg_path = checked_identity(spec['config'])
    old_config = yaml.safe_load(cfg_path.read_text())
    commit = spec['producer_commit']
    contract = checked_contract(spec['contract'], commit, sha(cfg_path), 'e3_trellis_config')
    audit, pool, inputs = (records[n] for n in
        ('postrun_audit.json', 'proposal_pool.json', 'input_manifest.json'))
    if (audit['state'] != 'COMPLETED' or audit['exit_code'] != '0:0'
            or audit['producer_source_commit'] != commit or pool['code_commit'] != commit
            or inputs['code_commit'] != commit or pool['exit_code'] != 0
            or pool['freeze_id'] != contract['freeze_id']
            or old_config['freeze_id'] != contract['freeze_id']
            or pool['paper_ready'] is not False or inputs['paper_ready'] is not False):
        raise PilotError('prior generation lacks authenticated terminal success')
    for expected, actual in [
        (audit['input_manifest_sha256'], sha(root/'input_manifest.json')),
        (pool['input_manifest_sha256'], sha(root/'input_manifest.json')),
        (audit['proposal_pool_sha256'], sha(root/'proposal_pool.json')),
        (audit['config_sha256'], sha(cfg_path)), (inputs['config_sha256'], sha(cfg_path)),
    ]:
        if expected != actual:
            raise PilotError('prior generation receipt link differs')
    if inputs['seed'] != old_config['seed'] or config['seed'] != 42:
        raise PilotError('frozen seed recipe differs')
    runtime = records['python_runtime.json']
    runtime_hash = hashlib.sha256(json.dumps({k:v for k,v in runtime.items() if k != 'runtime_sha256'},
                                           sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if runtime_hash != runtime['runtime_sha256'] or runtime_hash != old_config['runtime_sha256']:
        raise PilotError('prior runtime receipt differs')
    # Rebuild the original denominator/crop identities, not merely pool rows.
    old_jobs = jobs_loader(Path(old_config['source_pilot']))
    if (sha(Path(old_config['source_pilot'])/'pilot_summary.json') != old_config['source_summary_sha256']
            or sha(Path(old_config['source_pilot'])/'output_hashes.json') != old_config['source_output_hashes_sha256']):
        raise PilotError('prior discovery inputs differ')
    expected_jobs = []
    for job in old_jobs:
        job = dict(job)
        job['proposal_id'] = f"{contract['freeze_id']}:{job['job_id']}:trellis:initial:seed42"
        expected_jobs.append(job)
    if (inputs['jobs'] != expected_jobs or pool['planned_jobs'] != len(old_jobs)
            or [r['job_id'] for r in pool['rows']] != [j['job_id'] for j in old_jobs]):
        raise PilotError('prior raw input denominator differs')
    algorithm = algorithm_identity(config, commit)
    eligible, reason = runtime_reuse_eligibility(config, old_config, inputs)
    reasons=[] if eligible else [reason]
    for key in ('models','seed','runtime_sha256','python'):
        if old_config[key]!=config[key]:reasons.append(f'current_{key}_differs_from_original')
    if not algorithm['current_matches']:reasons.append('current_raw_algorithm_differs_from_original')
    return dict(spec=spec, pool=pool, jobs=expected_jobs, algorithm_files=algorithm['files'],
                runtime=records['runtime.json'], reuse_eligible=not reasons,
                reuse_ineligible_reason=reasons[0] if reasons else None,
                reuse_ineligible_reasons=reasons)


def reuse_rows(config, manifest, prior, comparison):
    """Map raw bytes across IDs. Missing inputs remain typed denominator rows."""
    prior_jobs = {j['job_id']:j for j in prior['jobs']}
    prior_rows = {r['job_id']:r for r in prior['pool']['rows']}
    comparisons = {r['job_id']:r for r in comparison['rows']}
    rows, receipts = [], []
    for job in manifest['jobs']:
        row = {k:job[k] for k in ('job_id', 'automatic_instance_id', 'proposal_id', 'prepared')}
        row.update(tool='trellis', seed=42, paper_ready=False,
                   shared_initial_policy_rows=['A1', 'A2', 'A3', 'A4'])
        matches = sorted(j['job_id'] for j in prior['jobs'] if j['prepared'] and job['prepared']
                         and j['input']['sha256'] == job['input']['sha256'])
        if sorted(comparisons[job['job_id']]['exact_matching_prior_jobs']) != matches:
            raise PilotError('crop comparison differs from authenticated original inputs')
        available = [key for key in matches if prior_rows[key]['status'] == 'available'
                     and prior.get('reuse_eligible') is True]
        if not available:
            row.update(status='unavailable', reason='initial_tool_not_run' if job['prepared']
                       else 'preparation_unavailable', artifacts={})
            rows.append(row)
            continue
        old_id = available[0]  # Stable ID order, never quality or geometry selection.
        old_job, old_row = prior_jobs[old_id], prior_rows[old_id]
        checked_identity(job['input']); checked_identity(old_job['input'])
        if (old_row['input_sha256'] != job['input']['sha256']
                or old_row['proposal_id'] != old_job['proposal_id'] or old_row['seed'] != 42
                or old_row['tool'] != 'trellis' or old_row['runtime']['seed'] != 42
                or old_row['runtime']['status'] != 'generated'
                or old_row['runtime']['model_source'] != config['models']['trellis_snapshot']['path']
                or set(old_row['artifacts']) != set(RAW_FILES)):
            raise PilotError('prior raw artifact input/recipe differs')
        for name, record in old_row['artifacts'].items():
            expected_path = Path(prior['spec']['directory'])/'construction/objects'/f"obj_{old_job['output_index']:02d}"/name
            if Path(record['path']) != expected_path or record['bytes'] <= 0:
                raise PilotError('prior raw artifact path/size differs')
            checked_identity(record)
        receipt = dict(schema_version=1, kind='validated_pure_trellis_raw_reuse',
            status='PASS', paper_ready=False, generation_performed=False,
            source_discovery_hashes=manifest['source_discovery_hashes'],
            source_gaussian_training_provenance=FRESH, new_job_id=job['job_id'],
            new_proposal_id=job['proposal_id'], input_sha256=job['input']['sha256'],
            original_job_id=old_id, original_proposal_id=old_row['proposal_id'],
            original_generator_commit=prior['spec']['producer_commit'],
            original_gaussian_training_provenance=prior['pool']['source_gaussian_training_provenance'],
            original_source_anchors=prior['spec'], original_gpu_runtime=prior['runtime'],
            original_object_runtime=old_row['runtime'], algorithm_files=prior['algorithm_files'],
            models=config['models'], runtime_sha256=config['runtime_sha256'], seed=42,
            runtime_byte_identity=config.get('runtime_byte_identity'),
            crop_comparison_sha256=config['crop_comparison_sha256'], artifacts=old_row['artifacts'],
            scope='Raw image-to-3D outputs only. Current scene registration and observations must be constructed anew; no RVG provenance claim.')
        row.update(status='available', reason=None, input_sha256=job['input']['sha256'],
                   frame=job['frame'], artifacts=old_row['artifacts'], runtime=old_row['runtime'],
                   generation_mode='validated_raw_reuse', original_generator_commit=prior['spec']['producer_commit'])
        receipts.append(receipt); rows.append(row)
    return rows, receipts
