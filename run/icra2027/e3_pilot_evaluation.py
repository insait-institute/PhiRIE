"""Prepare/run the existing independent evaluator after the full-cohort pilot seals.

No metric, matching rule, proposal selection, or partial full-cohort aggregate
is defined here. Configuration publication never reads GT before construction
validation; run delegates to the existing sealed matching/evaluation producers.
"""
from __future__ import annotations
import argparse
import copy
import json
import shutil
from collections import Counter
from pathlib import Path
import yaml
from agents.eval import automatic_matching_manifest as matching
from robo.eval import agentic_ablation as e3

CODE = Path(__file__).resolve().parents[2]
DISCOVERY_FIELDS = ('source_pilot','source_summary_sha256','source_output_hashes_sha256',
    'source_gaussian_training_provenance','source_discovery_hashes','source_discovery_config',
    'source_discovery_commit','source_discovery_contract')


def identity(path):
    return matching._absolute_identity(Path(path).absolute())


def pin_gt_after_construction(config):
    matching.validate_construction(config)  # Includes E0/full-roster/pilot binding; no GT access.
    result = copy.deepcopy(config)
    for scene in result['scenes']:
        scene['gt_inputs'] = {name:identity(Path('/data/ScanNetpp/data')/scene['scene_id']/'scans'/name)
                              for name in matching.GT_FILES}
    return result


def prepare(execution_config, freeze_id, destination, mode="pilot"):
    if mode not in {"pilot", "full"}:
        raise ValueError("evaluation preparation mode must be pilot or full")
    execution_path = Path(execution_config).resolve(strict=True)
    execution = yaml.safe_load(execution_path.read_text())
    construction = e3.checked_repo_path(e3.REPOSITORY_ROOT/'outputs/icra2027'/execution['freeze_id']/'agentic',
                                       'sealed pilot construction',kind='dir')
    # Resolve against the actual clean construction source, not an evaluation copy.
    contract = json.loads((construction.parent/'contract/freeze_manifest.json').read_text())
    candidates = [r for r in contract['resource_inventory']
                  if r.get('id')=='agentic_fresh_jobs' and r.get('hash_method')=='content_sha256']
    if len(candidates)!=1:
        raise ValueError('construction jobs resource is not uniquely bound')
    jobs_path = Path(candidates[0]['resolved_path'])
    if e3.sha256_file(jobs_path)!=candidates[0]['sha256']:
        raise ValueError('construction jobs changed')
    jobs = yaml.safe_load(jobs_path.read_text())
    roster = yaml.safe_load(Path(jobs['population']['scene_roster_config']).read_text())
    scenes = ([execution['pilot']['scene_id']] if mode == 'pilot'
              else [str(s) for s in roster['population']['scene_ids']])
    specs = []
    for scene in scenes:
        directory, shard, _ = e3._load_control_scene(construction, scene)
        sources = [s for s in jobs['automatic_sources'] if s['scene_id'] == scene]
        if len(sources) != 1 or shard['job_count'] != sources[0]['planned_jobs']:
            raise ValueError('construction population differs')
        source = sources[0]
        if mode == 'pilot' and source['planned_jobs'] != execution['pilot']['planned_jobs']:
            raise ValueError('pilot construction population differs')
        provenance = source['initial_pools']['trellis']['producer_provenance']
        producer_e0 = json.loads(Path(provenance['contract_path']).read_text())
        records = [r for r in producer_e0['resource_inventory']
                   if r.get('sha256') == provenance['config_sha256'] and r.get('kind') == 'experiment_config']
        if len(records) != 1:
            raise ValueError('generation config is not uniquely bound')
        generation_path = Path(records[0]['resolved_path'])
        if e3.sha256_file(generation_path) != provenance['config_sha256']:
            raise ValueError('generation config changed')
        generation = yaml.safe_load(generation_path.read_text())
        specs.append(dict(scene_id=scene, planned_jobs=source['planned_jobs'],
            construction_root=str(construction), control_seal=identity(directory/'seal.json'),
            discovery={key:generation[key] for key in DISCOVERY_FIELDS}))
    count = sum(spec['planned_jobs'] for spec in specs)
    config = dict(schema_version=1,freeze_id=freeze_id,mode=mode,paper_ready=False,
        dataset_root='/data/ScanNetpp',planned_scenes=len(specs),planned_jobs=count,protocol=matching.PROTOCOL,
        pilot_execution_config=identity(execution_path),
        population_roster=identity(jobs['population']['scene_roster_config']),scenes=specs)
    if mode == 'full':
        config['agentic_uncertainty'] = copy.deepcopy(e3.AGENTIC_UNCERTAINTY_PROTOCOL)
        # The existing pilot binding also authenticates the complete 50-scene
        # execution/E0/inventory closure. Do this before the first GT access.
        pilot_spec = [spec for spec in specs if spec['scene_id'] == execution['pilot']['scene_id']]
        probe = dict(config, mode='pilot', planned_scenes=1,
                     planned_jobs=execution['pilot']['planned_jobs'], scenes=pilot_spec)
        matching._pilot_from_execution(probe, scenes)
    config = pin_gt_after_construction(config)
    reservation = e3.REPOSITORY_ROOT/'outputs/icra2027/.freeze_ids'/freeze_id
    if not reservation.is_dir() or (reservation.parent.parent/freeze_id).exists():
        raise ValueError('new unused evaluation freeze reservation required')
    destination = Path(destination).absolute()
    relative = destination.relative_to(CODE)
    if destination.resolve()!=destination:
        raise ValueError('evaluation config destination may not contain symlinks')
    policies = [r for r in contract['resource_inventory'] if r.get('id')=='agentic_automatic_policies']
    if len(policies)!=1 or e3.sha256_file(policies[0]['resolved_path'])!=policies[0]['sha256']:
        raise ValueError('construction policy resource changed')
    freeze = yaml.safe_load((CODE/'configs/experiments/icra2027/e3_matching_pilot_38d/freeze.yaml').read_text())
    freeze.update(freeze_id=freeze_id,hardware={'gpu_required':False,
        'purpose':('Independent evaluation-only schema/integrity pilot; no outcome-based release'
                   if mode == 'pilot' else 'Complete frozen cohort independent evaluation; unchanged metrics')})
    names = {'e3_matching_config':'matching.yaml','e3_matching_jobs':'construction_jobs.yaml',
             'e3_matching_policies':'construction_policies.yaml'}
    freeze['input_roots'] = [dict(id=k,path=str(relative/v),kind='experiment_config',required=True)
                            for k,v in names.items()]
    freeze['input_roots'].extend([dict(id='e3_pilot_execution',path=str(execution_path),kind='experiment_config',required=True),
        dict(id='e3_matching_roster',path=config['population_roster']['path'],kind='experiment_config',required=True)])
    payloads={'matching.yaml':yaml.safe_dump(config,sort_keys=False).encode(),
        'construction_jobs.yaml':jobs_path.read_bytes(),
        'construction_policies.yaml':Path(policies[0]['resolved_path']).read_bytes(),
        'freeze.yaml':yaml.safe_dump(freeze,sort_keys=False).encode()}
    destination.mkdir(parents=True,exist_ok=False)
    for name,payload in payloads.items():
        with (destination/name).open('xb') as stream:stream.write(payload)
    return dict(status='CONFIG_WRITTEN_E0_REQUIRED',freeze_id=freeze_id,mode=mode,scene_ids=scenes,
                planned_jobs=count,planned_policy_rows=5*count,files={n:identity(destination/n) for n in payloads},paper_ready=False)


def run(config_path, contract_path):
    config_path=Path(config_path).resolve(strict=True)
    config=yaml.safe_load(config_path.read_text())
    if config.get('mode', 'pilot') != 'pilot' or len(config['scenes']) != 1:
        raise ValueError('run is pilot-only; full evaluation uses canonical export and per-scene CLI')
    scene=config['scenes'][0]['scene_id']
    construction=Path(config['scenes'][0]['construction_root'])
    out=e3.REPOSITORY_ROOT/'outputs/icra2027'/config['freeze_id']
    e3._validate_cli_execution(contract_path,config['freeze_id'],out/'evaluation_matching',config_paths=[config_path])
    matching.validate_construction(config)
    reference=out/'evaluation_matching/evaluation_references.json'
    if not reference.exists():matching.export(config_path,contract_path,reference.parent)
    digest=e3.sha256_file(reference)
    directory,shard,seal=e3._load_control_scene(construction,scene)
    inventory,_=e3._load_inventory(construction,controller_safe=True)
    job_ids=[j['job_id'] for s in inventory['scenes'] if s['scene_id']==scene for j in s['jobs']]
    _, reference_manifest=matching.load_references(reference,digest,construction,scene,seal['members']['controller_shard.json'],job_ids)
    if reference_manifest['config_sha256']!=e3.sha256_file(config_path):
        raise ValueError('resumed matching output belongs to a different config')
    if not (out/'agentic/evaluation'/scene).exists():
        code=e3.main(['--evaluate','--jobs',str(config_path.parent/'construction_jobs.yaml'),
            '--policies',str(config_path.parent/'construction_policies.yaml'),'--contract-manifest',str(contract_path),
            '--freeze-id',construction.parent.name,'--scene-id',scene,'--out',str(construction),
            '--evaluation-manifest',str(reference),'--evaluation-manifest-sha256',digest])
        if code:raise RuntimeError(f'canonical evaluation exited {code}')
    result=e3._load_eval_scene(out/'agentic',scene,construction_root=construction)
    expected_identity = {'external_evaluation_manifest_sha256':digest,
        'evaluation_freeze_id':config['freeze_id'], 'construction_freeze_id':construction.parent.name,
        'freeze_id':construction.parent.name, 'evaluation_code_commit':reference_manifest['code_commit']}
    if any(result.get(key)!=value for key,value in expected_identity.items()):
        raise ValueError('resumed evaluation source/freeze/reference identity differs')
    if (result['job_count']!=config['planned_jobs'] or len(result['rows'])!=5*config['planned_jobs']
            or result['geometry_reference_jobs']+result['geometry_unmatched_jobs']!=config['planned_jobs']
            or set(result['proposal_metrics'])!={p['proposal_id'] for p in shard['proposals']}):
        raise ValueError('pilot evaluation lost planned jobs or proposals')
    return dict(status='PASS',scope='independent_pilot_integrity_only',freeze_id=config['freeze_id'],
        scene_id=scene,planned_jobs=config['planned_jobs'],policy_rows=len(result['rows']),
        matched_jobs=result['geometry_reference_jobs'],unmatched_jobs=result['geometry_unmatched_jobs'],
        evaluation_shard=identity(out/'agentic/evaluation'/scene/'eval_shard.json'),paper_ready=False,
        full_cohort_aggregation='NOT_RUN; requires all 50 evaluated scenes')


def _check_evaluation_rows(shard, control, ledger, references, expected_identity):
    """Check the complete policy product without defining another metric."""
    if any(shard.get(k) != v for k, v in expected_identity.items()):
        raise ValueError('full evaluation source/freeze/reference identity differs')
    expected = Counter((j, p) for j in references for p in e3.POLICY_IDS)
    if (Counter((r['job_id'], r['policy_id']) for r in ledger) != expected
            or Counter((r['job_id'], r['policy_id']) for r in shard['rows']) != expected
            or shard['job_count'] != len(references)
            or shard['geometry_reference_jobs'] != sum(r['status'] == 'matched' for r in references.values())
            or shard['geometry_unmatched_jobs'] != sum(r['status'] == 'unmatched' for r in references.values())):
        raise ValueError('full evaluation planned policy/job product differs')
    proposals = {p['proposal_id']: p for p in control['proposals']}
    if len(proposals) != len(control['proposals']) or set(shard['proposal_metrics']) != set(proposals):
        raise ValueError('full evaluation proposal population differs')
    for pid, metric in shard['proposal_metrics'].items():
        proposal = proposals[pid]
        reference = references[proposal['job_id']]
        if reference['status'] == 'unmatched':
            if (metric.get('geometry_evaluation_status') != 'unmatched'
                    or any(metric.get(k) is not None for k in ('f1_20', 'cd_cm', 'collapse'))):
                raise ValueError('unmatched geometry must remain null')
        elif (metric.get('evaluation_surface_sha256') != reference['evaluation_surface']['sha256']
                or metric.get('registration_surface_sha256') != proposal['input_hashes']['visible_observation']
                or metric['evaluation_surface_sha256'] == metric['registration_surface_sha256']):
            raise ValueError('independent evaluation surface binding differs')
        if any(metric.get(k) != proposal['evidence'][k] for k in ('settle_stable', 'settle_drift_m')):
            raise ValueError('construction physical probe differs')
    original = {(r['job_id'], r['policy_id']): r for r in ledger}
    for row in shard['rows']:
        source = original[(row['job_id'], row['policy_id'])]
        selected = source.get('selected_proposal_id')
        physical = ({k: proposals[selected]['evidence'][k] for k in ('settle_stable','settle_drift_m')}
                    if selected in proposals else None)
        expected_row = dict(source, accepted=source['terminal_action'] == 'accept',
                            metrics=shard['proposal_metrics'].get(selected), physical_metrics=physical)
        if row != expected_row:
            raise ValueError('evaluation row changed a sealed controller decision')


def _check_retry_artifacts(control):
    proposals = {p['proposal_id']: p for p in control['proposals']}
    retries = set()
    for proposal in proposals.values():
        if proposal['tool'] != 'registration_retry':
            continue
        parents = proposal['parent_proposal_ids']
        if len(parents) != 1 or parents[0] not in proposals or parents[0] == proposal['proposal_id']:
            raise ValueError('retry parent/new proposal identity differs')
        parent = proposals[parents[0]]
        if parent['job_id'] != proposal['job_id'] or parent['tool'] == 'registration_retry':
            raise ValueError('retry must belong to its initial job proposal')
        for role in ('retry', 'transform', 'registration'):
            path = e3.checked_repo_path(proposal['artifact_paths'][role], 'retry artifact', kind='file')
            if (e3.sha256_file(path) != proposal['artifact_hashes'][role]
                    or path.stat().st_size != proposal['artifact_sizes'][role]):
                raise ValueError('retry artifact changed')
        if proposal['artifact_paths']['transform'] == parent['artifact_paths']['transform']:
            raise ValueError('retry reused the original transform artifact path')
        action = json.loads(e3.checked_repo_path(proposal['artifact_paths']['retry'], 'retry action', kind='file').read_text())
        registration = json.loads(e3.checked_repo_path(proposal['artifact_paths']['registration'], 'retry registration', kind='file').read_text())
        if (action['action'] != 'registration_signed_source_up_restart'
                or action['proposal_id'] != proposal['proposal_id'] or action['parent_proposal_id'] != parents[0]
                or action.get('source_up_hypothesis') != proposal.get('source_up_hypothesis')
                or action.get('source_up_hypothesis') != registration.get('source_up_hypothesis')
                or action.get('source_up_hypothesis') not in {'+x','-x','+y','-y','-z'}
                or parent.get('source_up_hypothesis') != '+z'):
            raise ValueError('retry did not record the declared different action')
        import numpy as np
        transform = np.load(proposal['artifact_paths']['transform'] if Path(proposal['artifact_paths']['transform']).is_absolute()
                            else e3.REPOSITORY_ROOT/proposal['artifact_paths']['transform'], allow_pickle=False)
        if transform.shape != (4,4) or not np.isfinite(transform).all() or not np.array_equal(transform, registration['T']):
            raise ValueError('retry transform differs from registration record')
        retries.add(proposal['job_id'])
    return retries


def _check_reference_bindings(rows, root, scene):
    for row in rows:
        if row['status'] == 'unmatched':
            if row.get('matched_gt_id') is not None or row.get('evaluation_surface') is not None:
                raise ValueError('unmatched reference has invented GT')
        elif row['status'] == 'matched':
            gid = row.get('matched_gt_id')
            surface = row.get('evaluation_surface')
            if (type(gid) is not int or gid < 0 or not isinstance(surface,dict)
                    or Path(surface['path']) != root/'surfaces'/scene/f'gt_{gid}.npy'):
                raise ValueError('matched surface scene/path differs')
            matching._checked_anchor(surface)
        else:
            raise ValueError('unknown reference status')


def _check_selected_tree(root, seal):
    e3._reject_symlink_components(root)
    if not root.is_dir():
        raise ValueError('aggregate selected-asset root missing')
    children = list(root.rglob('*'))
    if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in children):
        raise ValueError('aggregate selected tree contains a symlink or special file')
    files = {p.relative_to(root).as_posix():p for p in children if p.is_file()}
    directories = sorted(p.relative_to(root).as_posix() for p in children if p.is_dir())
    if set(files) != set(seal['selected_asset_members']) or directories != seal['selected_asset_directories']:
        raise ValueError('aggregate selected-asset population differs')
    for name,path in files.items():
        value = seal['selected_asset_members'][name]
        if path.stat().st_size != value['size_bytes'] or e3.sha256_file(path) != value['sha256']:
            raise ValueError('aggregate selected asset changed')
    return list(files.values())


def audit_full(config_path, contract_path):
    """Authenticate all shards and replay the existing aggregate, never geometry generation.

    The replay has a separate directory under the same evaluation freeze and
    uses the canonical producer unchanged. It cannot alter controller choices,
    matching thresholds, geometric metrics, or the measured source artifacts.
    """
    config_path = Path(config_path).resolve(strict=True)
    config = yaml.safe_load(config_path.read_text())
    if (config.get('mode') != 'full' or config['planned_scenes'] != 50 or config['planned_jobs'] != 1871):
        raise ValueError('full completion audit requires the declared 50/1871 cohort')
    out = e3.REPOSITORY_ROOT/'outputs/icra2027'/config['freeze_id']
    publication = out/'agentic'
    destination = out/'independent_evaluation_completion_audit.json'
    if destination.exists():
        raise FileExistsError('refusing to overwrite completed full audit')
    if config.get('agentic_uncertainty') != e3.AGENTIC_UNCERTAINTY_PROTOCOL:
        raise ValueError('full uncertainty protocol is missing or changed')
    jobs_path, policies_path = (config_path.parent/n for n in ('construction_jobs.yaml','construction_policies.yaml'))
    e3._validate_cli_execution(contract_path, config['freeze_id'], publication,
                              config_paths=[config_path,jobs_path,policies_path])
    matching.validate_construction(config)  # Every construction seal checked before GT.
    construction = Path(config['scenes'][0]['construction_root'])
    if any(Path(s['construction_root']) != construction for s in config['scenes']):
        raise ValueError('full evaluation mixes construction roots')
    inventory, _ = e3._load_inventory(construction, controller_safe=True)
    roster = [s['scene_id'] for s in config['scenes']]
    eval_children = list((publication/'evaluation').iterdir())
    if ({p.name for p in eval_children} != set(roster)
            or any(p.is_symlink() or not p.is_dir() for p in eval_children)):
        raise ValueError('evaluation directory differs from complete scene roster')
    if ([s['scene_id'] for s in inventory['scenes']] != roster
            or inventory['counts'] != {'scenes':50,'jobs':1871,'policy_object_rows':9355}):
        raise ValueError('full evaluation inventory roster differs')
    # Authenticate the complete matching seal once (including every surface).
    reference = out/'evaluation_matching/evaluation_references.json'
    first = roster[0]
    _, _, first_seal = e3._load_control_scene(construction, first)
    _, manifest = matching.load_references(reference, e3.sha256_file(reference), construction, first,
        first_seal['members']['controller_shard.json'], [j['job_id'] for j in inventory['scenes'][0]['jobs']])
    if (manifest['config_sha256'] != e3.sha256_file(config_path)
            or manifest['planned_scenes'] != 50 or manifest['planned_jobs'] != 1871
            or [s['scene_id'] for s in manifest['scenes']] != roster
            or Counter((r['scene_id'],r['job_id']) for r in manifest['rows']) !=
               Counter((s['scene_id'],j['job_id']) for s in inventory['scenes'] for j in s['jobs'])):
        raise ValueError('full matching config/roster identity differs')
    # Recheck the original independently declared GT inputs after construction.
    for spec in config['scenes']:
        for anchor in spec['gt_inputs'].values():
            matching._checked_anchor(anchor)
    evidence = {}
    def remember(path):
        anchor = identity(path)
        evidence[anchor['path']] = {k:anchor[k] for k in ('sha256','size_bytes')}
    for path in (config_path,jobs_path,policies_path,Path(contract_path),reference,reference.parent/'seal.json'):
        remember(path)
    all_ledger, retry_jobs = [], set()
    matched = unmatched = 0
    for scene, scene_inventory, binding in zip(roster, inventory['scenes'], manifest['scenes']):
        directory, control, seal = e3._load_control_scene(construction, scene)
        ledger = [json.loads(line) for line in (directory/'job_ledger.jsonl').read_text().splitlines() if line.strip()]
        e3.validate_ledger(ledger)
        references = [r for r in manifest['rows'] if r['scene_id'] == scene]
        _check_reference_bindings(references,reference.parent,scene)
        if (len(references) != len(scene_inventory['jobs'])
                or {r['job_id'] for r in references} != {j['job_id'] for j in scene_inventory['jobs']}
                or Path(binding['construction_root']) != construction
                or binding['controller_shard_sha256'] != seal['members']['controller_shard.json']):
            raise ValueError('full matching scene/control binding differs')
        shard = e3._load_eval_scene(publication, scene, construction_root=construction)
        _check_evaluation_rows(shard, control, ledger, {r['job_id']:r for r in references}, {
            'external_evaluation_manifest_sha256':e3.sha256_file(reference),
            'evaluation_freeze_id':config['freeze_id'], 'construction_freeze_id':construction.parent.name,
            'freeze_id':construction.parent.name, 'evaluation_code_commit':manifest['code_commit']})
        produced = _check_retry_artifacts(control)
        declared = {r['job_id'] for r in ledger if r['policy_id']=='A4' and r.get('retry_produced') is True}
        if produced != declared:
            raise ValueError('retry artifact count differs from terminal ledger')
        retry_jobs.update(produced); all_ledger.extend(ledger)
        matched += shard['geometry_reference_jobs']; unmatched += shard['geometry_unmatched_jobs']
        for path in (directory/'seal.json', directory/'controller_shard.json', directory/'job_ledger.jsonl',
                     publication/'evaluation'/scene/'seal.json', publication/'evaluation'/scene/'eval_shard.json'):
            remember(path)
    if len(all_ledger) != 9355 or matched+unmatched != 1871:
        raise ValueError('incomplete full aggregate denominator')
    payload = json.loads((publication/'agentic_ablation.json').read_text())
    if (payload['counts'] != {'scenes':50,'jobs_per_policy':1871,'policy_object_rows':9355,'genuine_retry_jobs':len(retry_jobs)}
            or payload.get('headline_eligible') is not False or payload.get('paper_ready') is not False):
        raise ValueError('aggregate denominator or nonheadline gate differs')
    # Refuse reuse of a partial audit replay; preserve it for diagnosis.
    replay = out/'integrity_replay_agentic'
    replay.mkdir(exist_ok=False)
    shutil.copytree(publication/'evaluation', replay/'evaluation', symlinks=False)
    jobs, policies = (yaml.safe_load(p.read_text()) for p in (jobs_path,policies_path))
    replayed = e3.run_aggregate(jobs, policies, construction.parent.name, construction,
                              evaluation_root=replay, uncertainty_protocol=config['agentic_uncertainty'])
    if ({k:v for k,v in replayed.items() if k!='created_utc'} != {k:v for k,v in payload.items() if k!='created_utc'}):
        raise ValueError('canonical full aggregate replay differs')
    original_seal = json.loads((publication/'aggregate_seal.json').read_text())
    replay_seal = json.loads((replay/'aggregate_seal.json').read_text())
    uncertainty_members = {'agentic_paired_uncertainty.json', 'agentic_paired_uncertainty.csv'}
    if not uncertainty_members.issubset(original_seal['members']):
        raise ValueError('full aggregate lacks predeclared uncertainty outputs')
    if (set(original_seal['members']) != set(replay_seal['members'])
            or original_seal['selected_asset_members'] != replay_seal['selected_asset_members']
            or original_seal['selected_asset_directories'] != replay_seal['selected_asset_directories']):
        raise ValueError('full aggregate artifact closure differs')
    for path in _check_selected_tree(publication/'selected_assets',original_seal):
        remember(path)
    for name, digest in original_seal['members'].items():
        path = e3.checked_repo_path(publication/name,'full aggregate member',kind='file')
        if e3.sha256_file(path) != digest or (name!='agentic_ablation.json' and digest!=replay_seal['members'][name]):
            raise ValueError('full aggregate sealed member/replay differs')
        remember(path)
    remember(publication/'aggregate_seal.json')
    result = dict(schema_version=1,status='PASS',scope='complete_cohort_independent_evaluation_integrity',
        freeze_id=config['freeze_id'],construction_freeze_id=construction.parent.name,
        source_commit=manifest['code_commit'],planned_scenes=50,planned_jobs=1871,terminal_policy_rows=9355,
        matched_jobs=matched,unmatched_jobs=unmatched,unique_retry_actions=len(retry_jobs),
        aggregate_rows_replayed=replayed['rows'],evidence_hashes=evidence,runtime_scope=payload['runtime_scope'],
        runtime_accounting_status={r['policy_id']:r['runtime_accounting_status'] for r in replayed['rows']},
        paper_ready=False,headline_eligible=False,claim_gate='NOT_RUN',
        checks={k:'PASS' for k in ('construction_train_only_discovery_binding','complete_frozen_roster',
            'control_frozen_before_evaluation','independent_GT_input_hashes','evaluation_metric_shards_authenticated',
            'unmatched_geometry_null','aggregate_replay_exact','runtime_accounting_replay_exact',
            'retry_new_proposal_transform_and_different_action','paired_scene_bootstrap_replay_exact',
            'nonheadline_gates')},
        limitations='Authenticates canonical metric shards and replays aggregation; no independent second geometry estimator; physical stability is construction evidence, not held-out manipulation.')
    e3._atomic_write_json(destination,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='phase',required=True)
    p=sub.add_parser('prepare');p.add_argument('--execution-config',required=True);p.add_argument('--freeze-id',required=True);p.add_argument('--destination',required=True);p.add_argument('--mode',choices=('pilot','full'),default='pilot')
    p=sub.add_parser('run');p.add_argument('--config',required=True);p.add_argument('--contract',required=True)
    p=sub.add_parser('audit-full');p.add_argument('--config',required=True);p.add_argument('--contract',required=True)
    args=parser.parse_args()
    result=(prepare(args.execution_config,args.freeze_id,args.destination,args.mode) if args.phase=='prepare'
            else audit_full(args.config,args.contract) if args.phase=='audit-full' else run(args.config,args.contract))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
