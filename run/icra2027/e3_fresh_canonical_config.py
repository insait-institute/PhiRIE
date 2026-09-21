"""Prepare canonical E3 configs only after both real initial producers finish."""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import yaml

from robo.eval import agentic_ablation as e3
from robo.manifest.hash import canonical_hash
from run.icra2027.e3_fresh_generation_contract import DISCOVERY_FILES, FRESH
from run.icra2027.e3_trellis_generation_pilot import source_jobs, runtime_identity

CODE = Path(__file__).resolve().parents[2]
POLICIES = 'configs/experiments/icra2027/agentic_automatic_policies.yaml'
RUNTIME = 'configs/experiments/icra2027/e3_gaussian_train_only.yaml'


def completion_identity(audit):
    if 'state' in audit:
        return audit['state'],audit['exit_code'],audit['producer_source_commit']
    # RVG's completed audit embeds parsable scheduler output instead of
    # duplicating its fields. Require the exact requested top-level job row.
    lines=audit.get('sacct','').splitlines()
    if not lines or lines[0].split('|')[:3]!=['JobID','State','ExitCode']:
        raise ValueError('initial producer lacks scheduler completion evidence')
    rows=[line.split('|') for line in lines[1:] if line.split('|')[0]==audit['job_id']]
    if len(rows)!=1 or len(rows[0])<3:
        raise ValueError('initial producer scheduler job identity differs')
    return rows[0][1],rows[0][2],audit['source_commit']


def authenticate_pool(directory, source_hashes, jobs, *, freeze_root=None, terminal_audit=None):
    directory = e3.checked_repo_path(directory, 'completed initial pool', kind='dir')
    pool_path = directory/'proposal_pool.json'
    pool = json.loads(pool_path.read_text())
    audit_path = (e3.checked_repo_path(terminal_audit, 'terminal audit', kind='file')
                  if terminal_audit is not None else directory/'postrun_audit.json')
    audit = json.loads(audit_path.read_text())
    manifest = json.loads((directory/'input_manifest.json').read_text())
    # Nested cohort pools require an explicit parent freeze. Never scan upwards
    # for a convenient contract or accept a contract outside that producer root.
    root = e3.checked_repo_path(freeze_root if freeze_root is not None else directory.parent,
                                'initial producer freeze', kind='dir')
    relative = directory.relative_to(root)
    if len(relative.parts) not in (1, 2) or relative.parts[0] not in {'trellis_initial', 'rvg_initial'}:
        raise ValueError('initial pool must be a direct or per-scene child of its producer freeze')
    if len(relative.parts) == 2:
        scene = e3._require_scene_id(relative.parts[1])
        if any(not job['job_id'].startswith(scene + ':auto:') for job in jobs):
            raise ValueError('nested initial pool scene differs from discovery jobs')
    contract_path = e3.checked_repo_path(root/'contract/freeze_manifest.json',
                                         'initial producer E0 contract', kind='file')
    contract = json.loads(contract_path.read_text())
    terminal_spec = None
    if terminal_audit is not None:
        from agents.orchestrator.automatic_inventory import terminal_pool_rows
        terminal_spec = dict(path=str(audit_path), sha256=e3.sha256_file(audit_path), freeze_root=str(root))
        terminal_pool_rows(directory, pool, manifest, terminal_spec)
        commit = audit['producer_source_commit']
        audit = dict(proposal_pool_sha256=audit['proposal_pool_json_sha256'],
                     input_manifest_sha256=audit['input_manifest_json_sha256'])
        completion_ok = True  # Terminal state and each retained artifact were replayed above.
    else:
        state,exit_code,commit=completion_identity(audit)
        completion_ok = state == 'COMPLETED' and exit_code == '0:0' and pool['exit_code'] == 0
    if (not completion_ok
            or audit['proposal_pool_sha256'] != e3.sha256_file(pool_path)
            or audit['input_manifest_sha256'] != e3.sha256_file(directory/'input_manifest.json')
            or pool['input_manifest_sha256'] != audit['input_manifest_sha256']
            or pool['paper_ready'] is not False
            or pool['code_commit'] != commit
            or pool['code_commit'] != manifest['code_commit']
            or contract['code']['commit'] != pool['code_commit']
            or contract['code']['dirty'] is not False
            or contract['freeze_id'] != pool['freeze_id']
            or root.name != pool['freeze_id']):
        raise ValueError('initial producer completion/source binding differs')
    digest = canonical_hash({k:v for k,v in contract.items()
                             if k not in {'created_utc','environment','contract_sha256'}})
    if digest != contract['contract_sha256']:
        raise ValueError('initial producer E0 digest differs')
    if not any(r.get('sha256') == manifest['config_sha256']
               and r.get('kind') == 'experiment_config'
               for r in contract['resource_inventory']):
        raise ValueError('initial producer config lacks E0 binding')
    for payload in (pool, manifest):
        if (payload.get('source_gaussian_training_provenance') != FRESH
                or payload.get('source_discovery_hashes') != source_hashes):
            raise ValueError('initial pool has different discovery provenance')
    if (pool['planned_jobs'] != len(jobs)
            or [r['job_id'] for r in pool['rows']] != [j['job_id'] for j in jobs]):
        raise ValueError('initial pool lost complete planned population')
    if terminal_spec is None and any(r['prepared'] and r.get('reason') == 'initial_tool_not_run' for r in pool['rows']):
        raise ValueError('initial producer has unattempted prepared jobs')
    records = directory/'proposal_records.jsonl'
    if (pool['proposal_records_sha256'] != e3.sha256_file(records)
            or [json.loads(line) for line in records.read_text().splitlines()] != pool['rows']):
        raise ValueError('initial pool records differ')
    result = {'path':str(pool_path), 'sha256':e3.sha256_file(pool_path),
            'producer_provenance': {
                'freeze_id': pool['freeze_id'], 'source_commit': pool['code_commit'],
                'config_sha256': manifest['config_sha256'],
                'contract_path': str(contract_path),
                'contract_file_sha256': e3.sha256_file(contract_path),
                'contract_sha256': contract['contract_sha256'],
                'input_manifest_sha256': pool['input_manifest_sha256'],
                'proposal_records_sha256': pool['proposal_records_sha256'],
                'completion_audit_kind': 'terminal' if terminal_spec else 'postrun',
                'completion_audit_path': str(audit_path),
                'completion_audit_sha256': e3.sha256_file(audit_path)}}
    if terminal_spec is not None:
        result['terminal_audit'] = terminal_spec
    else:
        result['producer_provenance']['postrun_audit_sha256'] = e3.sha256_file(audit_path)
    return result


def _missing_inputs(discovery, trellis, rvg, *, trellis_terminal_audit=None, rvg_terminal_audit=None):
    directories = {'discovery':Path(discovery), 'trellis':Path(trellis), 'reconviagen':Path(rvg)}
    audits = {'trellis':trellis_terminal_audit, 'reconviagen':rvg_terminal_audit}
    missing = []
    for tool, directory in directories.items():
        members = DISCOVERY_FILES if tool == 'discovery' else ('proposal_pool.json','input_manifest.json','proposal_records.jsonl')
        missing.extend(str(directory/name) for name in members if not (directory/name).is_file())
        if tool != 'discovery':
            audit = Path(audits[tool]) if audits[tool] is not None else directory/'postrun_audit.json'
            if not audit.is_file(): missing.append(str(audit))
    return sorted(missing)


def _source_payload(discovery, trellis, rvg, *, trellis_freeze_root=None, rvg_freeze_root=None,
                    trellis_terminal_audit=None, rvg_terminal_audit=None):
    directories = {'trellis': Path(trellis), 'reconviagen': Path(rvg)}
    roots = {'trellis': trellis_freeze_root, 'reconviagen': rvg_freeze_root}
    audits = {'trellis': trellis_terminal_audit, 'reconviagen': rvg_terminal_audit}
    source = e3.checked_repo_path(discovery, 'fresh discovery', kind='dir')
    hashes = {name:e3.sha256_file(source/name) for name in sorted(DISCOVERY_FILES)}
    jobs = source_jobs(source)
    manifest = json.loads((source/'input_manifest.json').read_text())
    if manifest['source_gaussian_training_provenance'] != FRESH:
        raise ValueError('canonical fresh pilot requires authenticated TRAIN-only discovery')
    scene = json.loads((source/'all_jobs_manifest.json').read_text())['scene_id']
    pools = {tool:authenticate_pool(directories[tool], hashes, jobs, freeze_root=roots[tool], terminal_audit=audits[tool])
             for tool in ('trellis','reconviagen')}
    def normalize(identity):
        return {key:identity[source_key] for key,source_key in
                [('path','path'),('size_bytes','bytes'),('sha256','sha256')]}
    source_payload = dict(scene_id=scene, planned_jobs=len(jobs),
        discovery_directory=str(source), discovery_hashes=hashes, initial_pools=pools,
        scene_sources={'gaussian':normalize(manifest['gaussian']),
            'intrinsics':normalize(manifest['metadata']['nerfstudio/transforms_undistorted.json']),
            'poses':normalize(manifest['metadata']['colmap/images.txt'])})
    return source_payload, jobs


def prepare(discovery, trellis, rvg, *, freeze_id=None, config_directory=None,
            trellis_freeze_root=None, rvg_freeze_root=None,
            trellis_terminal_audit=None, rvg_terminal_audit=None):
    audits = dict(trellis_terminal_audit=trellis_terminal_audit, rvg_terminal_audit=rvg_terminal_audit)
    missing = _missing_inputs(discovery, trellis, rvg, **audits)
    if missing:
        return dict(status='WAITING_REAL_INITIAL_POOLS', paper_ready=False,
                    missing_inputs=sorted(missing), config_written=False)
    source_payload, jobs = _source_payload(discovery, trellis, rvg,
        trellis_freeze_root=trellis_freeze_root, rvg_freeze_root=rvg_freeze_root, **audits)
    scene = source_payload['scene_id']
    pools = source_payload['initial_pools']
    hashes = source_payload['discovery_hashes']
    payload = dict(schema_version=1, study_scope='automatic_training_only_engineering',
        paper_ready=False, output_dir='outputs/icra2027/{freeze_id}/agentic',
        automatic_sources=source_payload)
    # Validation delegates to the existing normalizer; this object is never
    # published as an inventory or represented as an executing source contract.
    from agents.orchestrator.automatic_inventory import build_payloads
    _,_,audit,_ = build_payloads(payload, {'contract_sha256':'0'*64,
        'freeze_id':'config-validation','code':{'commit':'0'*40}}, 'config-validation')
    if not audit['initial_pool_complete']:
        raise ValueError('both initial producer populations must be terminal')
    result = dict(status='READY_FOR_RESERVED_FREEZE', paper_ready=False,
        config_written=False, planned_jobs=len(jobs), planned_policy_object_rows=5*len(jobs),
        prepared_inputs=sum(j['prepared'] for j in jobs), initial_pools=pools,
        source_discovery_hashes=hashes)
    if config_directory is None:
        return result
    return _write_configs(payload, [(source_payload, jobs)], result, freeze_id, config_directory)


def _write_configs(payload, bundles, result, freeze_id, config_directory):
    evidence = e3.REPOSITORY_ROOT/'outputs/icra2027'
    if (not freeze_id or not (evidence/'.freeze_ids'/freeze_id).is_dir()
            or (evidence/freeze_id).exists()):
        raise ValueError('unused canonical reservation required')
    destination = Path(config_directory).absolute()
    destination.relative_to(CODE)
    if destination.resolve() != destination:
        raise ValueError('source config directory may not contain symlinks')
    paths = {key:destination/name for key,name in
        [('jobs','agentic_fresh_jobs.yaml'),('execution','agentic_fresh_execution.yaml'),
         ('freeze','agentic_fresh_freeze.yaml')]}
    if any(p.exists() or p.is_symlink() for p in paths.values()):
        raise FileExistsError('fresh configs already exist')
    control_python = '/group/worldcept/PhiRIE/code/SimAny/.venv/bin/python'
    runtime = yaml.safe_load((CODE/RUNTIME).read_text())
    cohort = isinstance(payload['automatic_sources'], list)
    execution = dict(schema_version=2 if cohort else 1,scope='fresh_canonical_engineering',paper_ready=False,
        freeze_id=freeze_id,planned_jobs=result['planned_jobs'],
        planned_policy_object_rows=result['planned_policy_object_rows'],
        jobs_config=str(paths['jobs'].relative_to(CODE)),policies_config=POLICIES,
        policies_sha256=e3.sha256_file(CODE/POLICIES),
        observation_runtime_config=RUNTIME,observation_runtime_config_sha256=e3.sha256_file(CODE/RUNTIME),
        observation_python=runtime['training_python'],control_python=control_python,
        control_runtime_sha256=runtime_identity(control_python)[1])
    scene_slots = {source['scene_id']: [f"obj_{j['automatic_instance_id']}" for j in jobs]
                   for source, jobs in bundles}
    if cohort:
        execution.update(planned_scenes=len(bundles), scene_ids=list(scene_slots),
                         scene_object_slots=scene_slots)
    else:
        scene = bundles[0][0]['scene_id']
        execution.update(scene_id=scene, object_slots=scene_slots[scene])
    freeze = yaml.safe_load((CODE/'configs/experiments/icra2027/agentic_automatic_freeze.yaml').read_text())
    freeze.update(freeze_id=freeze_id,hardware={'gpu_required':False,
        'purpose':'CPU E0 for fresh canonical inventory; observe/control have separate resource receipts'})
    freeze['input_roots'] = [dict(id=key,path=path,kind='experiment_config',required=True)
        for key,path in [('agentic_fresh_jobs',execution['jobs_config']),
                         ('agentic_automatic_policies',POLICIES),
                         ('agentic_fresh_execution',str(paths['execution'].relative_to(CODE))),
                         ('observation_runtime_config',RUNTIME)]]
    if cohort:
        freeze['input_roots'].append(dict(id='canonical_scene_roster',
            path=payload['population']['scene_roster_config'],kind='experiment_config',required=True))
    destination.mkdir(parents=True,exist_ok=True)
    for key,value in [('jobs',payload),('execution',execution),('freeze',freeze)]:
        with paths[key].open('x') as f:yaml.safe_dump(value,f,sort_keys=False)
    return dict(result,status='CONFIG_WRITTEN_E0_REQUIRED',config_written=True,
                files={k:str(p) for k,p in paths.items()},freeze_id=freeze_id)


def terminal_audit_for_scene(roots, scene):
    """Allow declared audit batches without choosing between conflicting reports."""
    if roots is None: return None
    roots = [roots] if isinstance(roots, (str, Path)) else list(roots)
    roots = [e3.checked_repo_path(root, 'terminal audit batch', kind='dir') for root in roots]
    matches = [root/(scene+'.json') for root in roots if (root/(scene+'.json')).is_file()]
    matches = [e3.checked_repo_path(path, 'terminal audit report', kind='file') for path in matches]
    if not matches: return None
    hashes = {e3.sha256_file(path) for path in matches}
    if len(hashes) != 1:
        raise ValueError('conflicting terminal audits for scene '+scene)
    return str(matches[0])


# This recovery treatment was declared before inspecting canonical outcomes.
RVG_OVERRIDE_SCENES = frozenset({'40aec5fffa', '3f15a9266d'})


def load_pool_overrides(path, scenes):
    if path is None:
        return {}, None
    path = e3._checked_code_path(path, 'predeclared pool override manifest')
    value = yaml.safe_load(path.read_text())
    if set(value) != {'schema_version', 'overrides'} or value['schema_version'] != 1:
        raise ValueError('pool override manifest schema differs')
    overrides = {}
    for record in value['overrides']:
        if set(record) != {'scene_id','tool','path','freeze_root','source_commit','original_pool'}:
            raise ValueError('pool override fields differ')
        scene = e3._require_scene_id(record['scene_id'])
        if scene not in RVG_OVERRIDE_SCENES or scene not in scenes or scene in overrides or record['tool'] != 'reconviagen':
            raise ValueError('pool override scene/tool outside predeclared recovery or duplicated')
        root = e3.checked_repo_path(record['freeze_root'], 'replacement freeze root', must_exist=False)
        target = e3.checked_repo_path(record['path'], 'replacement pool', must_exist=False)
        if target != root/'rvg_initial'/scene/'proposal_pool.json':
            raise ValueError('pool override path does not match scene/tool/freeze')
        if len(record['source_commit']) != 40 or any(c not in '0123456789abcdef' for c in record['source_commit']):
            raise ValueError('pool override source commit must be exact')
        original = record['original_pool']
        if set(original) != {'path','sha256','freeze_root','terminal_audit'}:
            raise ValueError('original failed pool provenance fields differ')
        old_root = e3.checked_repo_path(original['freeze_root'], 'original freeze root', must_exist=False)
        old_path = e3.checked_repo_path(original['path'], 'original failed pool', must_exist=False)
        if old_path != old_root/'rvg_initial'/scene/'proposal_pool.json' or old_path == target:
            raise ValueError('original and replacement pool paths differ from declared recovery')
        e3._require_sha256(original['sha256'], 'original failed pool hash')
        if set(original['terminal_audit']) != {'path','sha256'}:
            raise ValueError('original terminal audit anchor differs')
        e3._require_sha256(original['terminal_audit']['sha256'], 'original terminal audit hash')
        overrides[scene] = record
    if set(overrides) != RVG_OVERRIDE_SCENES:
        raise ValueError('pool override manifest must declare both frozen RVG recovery scenes')
    return overrides, dict(path=str(path), sha256=e3.sha256_file(path))


def override_runtime_history(record, source, jobs):
    """Retain the failed original process cost without reusing it as a proposal."""
    scene = source['scene_id']; selected = source['initial_pools']['reconviagen']
    if (selected['path'] != record['path']
            or selected['producer_provenance']['source_commit'] != record['source_commit']
            or selected['producer_provenance']['freeze_id'] != Path(record['freeze_root']).name):
        raise ValueError('selected pool override source/freeze identity differs')
    original = record['original_pool']; terminal = original['terminal_audit']
    for anchor in (original, terminal):
        path = e3.checked_repo_path(anchor['path'], 'original recovery provenance', kind='file')
        if e3.sha256_file(path) != anchor['sha256']:
            raise ValueError('original failed recovery provenance changed')
    authenticated = authenticate_pool(Path(original['path']).parent, source['discovery_hashes'], jobs,
        freeze_root=original['freeze_root'], terminal_audit=terminal['path'])
    if authenticated['sha256'] != original['sha256']:
        raise ValueError('original failed pool receipt differs')
    # Reuse the aggregate's small-file closure validator for direct ancestry.
    from robo.eval.agentic_runtime_accounting import _pool
    current = _pool(selected, scene, 'reconviagen')
    previous = _pool(original, scene, 'reconviagen')
    if (current['required_parent_pool_paths'] != [original['path']]
            or previous['required_parent_pool_paths'] or previous['exit_code'] == 0
            or current['job_ids'] != previous['job_ids']
            or current['discovery_hashes'] != previous['discovery_hashes']):
        raise ValueError('recovery runtime ancestry or failed original inputs differ')
    return dict(scene_id=scene, tool='reconviagen', pools=[
        {key:anchor[key] for key in ('path','sha256')} for anchor in (original,selected)])


def cohort_readiness(roster_config, discovery_root, trellis_root, rvg_root, *,
                     trellis_freeze_root, rvg_freeze_root, freeze_id=None, config_directory=None,
                     trellis_terminal_audit_root=None, rvg_terminal_audit_root=None,
                     pool_override_manifest=None):
    """Full-roster gate; config writing requires both complete producer pools.

    Roots are the population directories (auto_discovery_pilot, trellis_initial,
    rvg_initial); explicit producer roots authenticate their respective E0s.
    """
    roster_path = e3._checked_code_path(roster_config, 'canonical scene roster')
    roster = yaml.safe_load(roster_path.read_text())
    scenes = roster.get('population', {}).get('scene_ids')
    if not isinstance(scenes, list) or len(scenes) != 50 or len(set(scenes)) != 50:
        raise ValueError('canonical cohort requires the complete frozen 50-scene roster')
    scenes = [e3._require_scene_id(scene) for scene in scenes]
    roots = [Path(discovery_root), Path(trellis_root), Path(rvg_root)]
    overrides, override_anchor = load_pool_overrides(pool_override_manifest, scenes)
    def scene_directories(scene):
        return (roots[0]/scene, roots[1]/scene,
                Path(overrides[scene]['path']).parent if scene in overrides else roots[2]/scene)
    def audit_paths(scene):
        paths = {}
        for name, bases in [('trellis_terminal_audit',trellis_terminal_audit_root),
                            ('rvg_terminal_audit',rvg_terminal_audit_root)]:
            if scene in overrides and name == 'rvg_terminal_audit':
                continue  # An original failed audit cannot certify a replacement pool.
            selected = terminal_audit_for_scene(bases, scene)
            if selected is not None: paths[name] = selected
        return paths
    missing_by_scene = {scene: _missing_inputs(*scene_directories(scene), **audit_paths(scene)) for scene in scenes}
    for scene, record in overrides.items():
        original = record['original_pool']
        required = [Path(original['path']).parent/name for name in ('proposal_pool.json','input_manifest.json','proposal_records.jsonl')]
        required.append(Path(original['terminal_audit']['path']))
        missing_by_scene[scene].extend(str(path) for path in required if not path.is_file())
    missing = {scene: paths for scene, paths in missing_by_scene.items() if paths}
    if missing:
        return dict(status='WAITING_REAL_INITIAL_POOLS', paper_ready=False,
            config_written=False, planned_scenes=len(scenes),
            scenes_with_required_files=len(scenes)-len(missing), missing_inputs=missing)
    population = dict(scene_roster_config=str(roster_path),
        scene_roster_config_file_sha256=e3.sha256_file(roster_path),
        scene_roster_sha256=hashlib.sha256(('\n'.join(scenes)+'\n').encode()).hexdigest(),
        planned_scenes=len(scenes))
    sources, originals, bundles, runtime_histories = [], [], [], []
    for scene in scenes:
        source, jobs = _source_payload(*scene_directories(scene),
            trellis_freeze_root=trellis_freeze_root,
            rvg_freeze_root=overrides[scene]['freeze_root'] if scene in overrides else rvg_freeze_root,
            **audit_paths(scene))
        if source['scene_id'] != scene:
            raise ValueError('cohort source scene differs from frozen roster')
        if scene in overrides:
            runtime_histories.append(override_runtime_history(overrides[scene], source, jobs))
        sources.append(source); originals.extend(jobs); bundles.append((source, jobs))
    population.update(planned_jobs_per_policy=len(originals),
                      planned_policy_object_rows=5*len(originals))
    payload = dict(schema_version=1, study_scope='automatic_training_only_engineering',
        paper_ready=False, output_dir='outputs/icra2027/{freeze_id}/agentic',
        population=population, automatic_sources=sources)
    if overrides:
        if len(originals) != 1871:
            raise ValueError('mixed-pool recovery cohort must retain all 1871 planned jobs')
        payload['runtime_accounting'] = dict(schema_version=1, process_histories=runtime_histories)
        payload['pool_override_manifest'] = override_anchor
    from agents.orchestrator.automatic_inventory import build_payloads
    _, proposals, audit, _ = build_payloads(payload, {'contract_sha256':'0'*64,
        'freeze_id':'config-validation', 'code':{'commit':'0'*40}}, 'config-validation')
    if not audit['initial_pool_complete']:
        raise ValueError('both full-cohort producer populations must be terminal')
    result = dict(status='READY_FOR_RESERVED_FREEZE', paper_ready=False,
        config_written=False, planned_scenes=len(scenes), planned_jobs=len(originals),
        prepared_inputs=sum(j['prepared'] for j in originals),
        planned_policy_object_rows=5*len(originals), proposal_counts=proposals['counts'],
        population=population, initial_pools={s['scene_id']:s['initial_pools'] for s in sources})
    if overrides:
        result.update(pool_override_manifest=override_anchor, runtime_accounting=payload['runtime_accounting'])
    if config_directory is None:
        return result
    return _write_configs(payload, bundles, result, freeze_id, config_directory)


def publish_readiness(readiness_path, readiness_sha256, *, freeze_id, config_directory,
                      discovery_root, roster_config):
    """Publish configs from a completed closure, rechecking its small anchors.

    This is config preparation only. The canonical inventory producer must
    still authenticate every actual artifact after E0; this function never
    publishes an inventory or substitutes for that execution gate.
    """
    readiness_path = Path(readiness_path).resolve(strict=True)
    if e3.sha256_file(readiness_path) != readiness_sha256:
        raise ValueError('readiness receipt hash differs')
    result = json.loads(readiness_path.read_text())
    if (result.get('status') != 'READY_FOR_RESERVED_FREEZE' or result.get('config_written') is not False
            or result.get('paper_ready') is not False
            or (result.get('planned_scenes'), result.get('planned_jobs'), result.get('planned_policy_object_rows')) != (50,1871,9355)):
        raise ValueError('complete full-cohort readiness required')
    roster = e3._checked_code_path(roster_config, 'canonical scene roster')
    scenes = [e3._require_scene_id(s) for s in yaml.safe_load(roster.read_text())['population']['scene_ids']]
    population = result['population']
    if (len(scenes) != 50 or len(set(scenes)) != 50
            or list(result['initial_pools']) != scenes or population['scene_roster_config'] != str(roster)
            or population['scene_roster_config_file_sha256'] != e3.sha256_file(roster)
            or population['scene_roster_sha256'] != hashlib.sha256(('\n'.join(scenes)+'\n').encode()).hexdigest()
            or (population['planned_scenes'], population['planned_jobs_per_policy'], population['planned_policy_object_rows']) != (50,1871,9355)):
        raise ValueError('readiness full roster anchor differs')
    override = result['pool_override_manifest']
    if e3.sha256_file(override['path']) != override['sha256']:
        raise ValueError('readiness override manifest changed')
    records, declared_override = load_pool_overrides(override['path'], scenes)
    if declared_override != override:
        raise ValueError('readiness override identity differs')
    histories = result['runtime_accounting']
    expected_histories = []
    for scene, record in records.items():
        selected = result['initial_pools'][scene]['reconviagen']
        original = record['original_pool']
        for anchor in (original, original['terminal_audit']):
            path = e3.checked_repo_path(anchor['path'], 'original recovery metadata', kind='file')
            if e3.sha256_file(path) != anchor['sha256']:
                raise ValueError('readiness original recovery provenance changed')
        if selected['path'] != record['path'] or selected['producer_provenance']['source_commit'] != record['source_commit']:
            raise ValueError('readiness replacement source differs')
        expected_histories.append(dict(scene_id=scene, tool='reconviagen', pools=[
            {key:anchor[key] for key in ('path','sha256')} for anchor in (original,selected)]))
    if (histories.get('schema_version') != 1 or sorted(histories['process_histories'], key=lambda h:h['scene_id'])
            != sorted(expected_histories, key=lambda h:h['scene_id'])):
        raise ValueError('readiness original/recovery runtime histories differ')
    bundles = []
    for scene in scenes:
        directory = e3.checked_repo_path(Path(discovery_root)/scene,'discovery metadata',kind='dir')
        hashes = {name:e3.sha256_file(directory/name) for name in sorted(DISCOVERY_FILES)}
        pools = result['initial_pools'][scene]
        for pool in pools.values():
            path = e3.checked_repo_path(pool['path'],'readiness pool',kind='file')
            provenance = pool['producer_provenance']
            anchors = [(path,pool['sha256']), (path.parent/'input_manifest.json',provenance['input_manifest_sha256']),
                (path.parent/'proposal_records.jsonl',provenance['proposal_records_sha256']),
                (Path(provenance['contract_path']),provenance['contract_file_sha256']),
                (Path(provenance['completion_audit_path']),provenance['completion_audit_sha256'])]
            if any(e3.sha256_file(p) != digest for p,digest in anchors):
                raise ValueError('readiness pool/config/audit anchor changed')
            if json.loads((path.parent/'input_manifest.json').read_text())['source_discovery_hashes'] != hashes:
                raise ValueError('readiness discovery anchors changed')
        manifest = json.loads((directory/'input_manifest.json').read_text())
        original = json.loads((directory/'all_jobs_manifest.json').read_text())
        def normalize(record):
            return {key:record[source] for key,source in [('path','path'),('size_bytes','bytes'),('sha256','sha256')]}
        source = dict(scene_id=scene,planned_jobs=original['planned_jobs'],discovery_directory=str(directory),
            discovery_hashes=hashes,initial_pools=pools,scene_sources={
                'gaussian':normalize(manifest['gaussian']),
                'intrinsics':normalize(manifest['metadata']['nerfstudio/transforms_undistorted.json']),
                'poses':normalize(manifest['metadata']['colmap/images.txt'])})
        if original['scene_id'] != scene or original['planned_jobs'] != len(original['rows']):
            raise ValueError('readiness metadata population differs')
        bundles.append((source,original['rows']))
    if sum(len(jobs) for _,jobs in bundles) != 1871:
        raise ValueError('readiness object population differs')
    payload = dict(schema_version=1,study_scope='automatic_training_only_engineering',paper_ready=False,
        output_dir='outputs/icra2027/{freeze_id}/agentic',population=population,
        automatic_sources=[source for source,_ in bundles],runtime_accounting=result['runtime_accounting'],
        pool_override_manifest=override)
    published = _write_configs(payload,bundles,result,freeze_id,config_directory)
    return dict(published,readiness_receipt=dict(path=str(readiness_path),sha256=readiness_sha256),
                canonical_inventory_validation_required=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('discovery','trellis','rvg'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--freeze-id');parser.add_argument('--config-directory')
    parser.add_argument('--trellis-freeze-root');parser.add_argument('--rvg-freeze-root')
    parser.add_argument('--pool-override-manifest', help='predeclared two-scene RVG recovery pool selection')
    parser.add_argument('--cohort-roster', help='complete 50-scene readiness and optional config writing')
    parser.add_argument('--publish-readiness', help='complete prior readiness JSON; metadata-only config publication')
    parser.add_argument('--readiness-sha256')
    parser.add_argument('--trellis-terminal-audit');parser.add_argument('--rvg-terminal-audit')
    parser.add_argument('--trellis-terminal-audit-root',action='append');parser.add_argument('--rvg-terminal-audit-root',action='append')
    args=parser.parse_args()
    roots=dict(trellis_freeze_root=args.trellis_freeze_root,rvg_freeze_root=args.rvg_freeze_root)
    if args.publish_readiness:
        if not all((args.readiness_sha256,args.cohort_roster,args.freeze_id,args.config_directory)):
            parser.error('readiness publication needs hash, roster, reservation and config destination')
        result=publish_readiness(args.publish_readiness,args.readiness_sha256,freeze_id=args.freeze_id,
            config_directory=args.config_directory,discovery_root=args.discovery,roster_config=args.cohort_roster)
    elif args.cohort_roster:
        if args.trellis_terminal_audit or args.rvg_terminal_audit:
            parser.error('cohort mode requires audit directories rather than one-scene audits')
        if not all(roots.values()) or bool(args.freeze_id) != bool(args.config_directory):
            parser.error('cohort requires both producer roots; config writes require a freeze reservation')
        result=cohort_readiness(args.cohort_roster,args.discovery,args.trellis,args.rvg,
            freeze_id=args.freeze_id,config_directory=args.config_directory,**roots,
            trellis_terminal_audit_root=args.trellis_terminal_audit_root,
            rvg_terminal_audit_root=args.rvg_terminal_audit_root,
            pool_override_manifest=args.pool_override_manifest)
    else:
        if args.pool_override_manifest:
            parser.error('pool overrides require complete cohort mode')
        if args.trellis_terminal_audit_root or args.rvg_terminal_audit_root:
            parser.error('single-scene mode requires explicit audit files')
        result=prepare(args.discovery,args.trellis,args.rvg,
            freeze_id=args.freeze_id,config_directory=args.config_directory,**roots,
            trellis_terminal_audit=args.trellis_terminal_audit,rvg_terminal_audit=args.rvg_terminal_audit)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
