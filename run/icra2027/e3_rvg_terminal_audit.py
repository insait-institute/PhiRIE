"""Audit failed RVG processes and completed pools with invalid artifacts.

No metrics, model calls, acceptance decisions, or edits to original producers.
Original successful artifacts are independently checked and publicly replayed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

import yaml

from agents.orchestrator.automatic_inventory import terminal_pool_rows, terminal_unavailable_reason
from robo.eval import agentic_ablation as e3
from run.icra2027 import e3_rvg_generation_pilot as rvg
from run.icra2027.e3_auto_discovery_pilot import identity, sha, write_new, PilotError
from run.icra2027.e3_fresh_generation_contract import checked_contract

CLOSURE=('input_manifest.json','proposal_pool.json','proposal_records.jsonl',
         'execution_claim.json','generation.log','runtime.json','view_manifest.json','views_receipt.json')


def failure_rows(pool):
    rows=[]
    for original in pool['rows']:
        if original['status'] not in {'unavailable','generation_failed'} or original.get('runtime') or original['artifacts']:
            raise PilotError('RVG pre-generation audit cannot classify generated or attempted artifacts')
        rows.append(dict(job_id=original['job_id'],proposal_id=original['proposal_id'],
            producer_status=original['status'],normalized_status='unavailable',
            reason='preparation_unavailable' if not original['prepared'] else
                'no_per_object_completion_record_process_exit_'+str(pool['exit_code']),
            per_object_attempt_proven=False,artifact_validation_errors=[],verified_artifacts=[]))
    return rows


def terminal_rows(out, pool, manifest):
    """Independently audit raw artifacts; terminal_pool_rows replays each result.

    Completion records and consumed image bytes authenticate attempts. Missing
    records remain unknown, including the suffix after a process failure.
    """
    import numpy as np
    from plyfile import PlyData

    directory = Path(out)
    views = json.loads((directory/'view_manifest.json').read_text())
    expected_views = {row['object_index']: row['views'] for row in views['rows']}
    rows, expected_records = [], set()
    for original in pool['rows']:
        if original['status'] not in {'available', 'generation_failed', 'unavailable'}:
            raise PilotError('unknown RVG producer status')
        runtime = original.get('runtime')
        record_identity = None
        classification = 'preparation_unavailable' if not original['prepared'] else 'unknown_unattempted_suffix'
        if runtime is not None:
            idx = original['output_index']
            record = e3.checked_repo_path(directory/'producer_records'/f'object_{idx:02d}.json',
                                          'RVG per-object runtime', kind='file')
            expected_records.add(record.name)
            if json.loads(record.read_text()) != runtime:
                raise PilotError('RVG per-object runtime differs from original pool')
            record_identity = identity(record)
            if runtime.get('status') not in {'generated', 'generation_failed', 'unavailable'}:
                raise PilotError('unknown RVG per-object runtime status')
            if runtime['status'] in {'generated', 'generation_failed'}:
                wall = runtime.get('wall_s')
                if isinstance(wall, bool) or not isinstance(wall, (float, int)) or not math.isfinite(wall) or wall < 0:
                    raise PilotError('RVG per-object runtime must be finite and nonnegative')
                classification = 'completed_generation' if runtime['status'] == 'generated' else 'proven_failed_attempt'
            else:
                classification = 'view_eligibility_unavailable'
            if runtime['status'] == 'generated':
                expected = expected_views[idx]
                if (runtime.get('view_manifest_sha256') != sha(directory/'view_manifest.json')
                        or runtime.get('source_discovery_hashes') != manifest['source_discovery_hashes']
                        or runtime.get('consumed_views') != rvg.verify_consumed_views(directory, idx, expected)):
                    raise PilotError('RVG generated runtime consumed-image/source binding differs')
        errors, verified = [], []
        for name, artifact_identity in original['artifacts'].items():
            artifact = e3.checked_repo_path(artifact_identity['path'], 'terminal artifact', must_exist=False)
            artifact.relative_to(directory)
            if not artifact.is_file():
                errors.append({'artifact':name,'reason':'missing'}); continue
            if artifact.stat().st_size != artifact_identity['bytes'] or e3.sha256_file(artifact) != artifact_identity['sha256']:
                errors.append({'artifact':name,'reason':'hash_or_size_mismatch'}); continue
            verified.append({'artifact':name, 'path':str(artifact), 'sha256':artifact_identity['sha256'], 'bytes':artifact_identity['bytes']})
            if name.endswith('.ply'):
                try:
                    vertices = PlyData.read(str(artifact))['vertex'].data
                    if not len(vertices): errors.append({'artifact':name,'reason':'empty_vertices'})
                    for field in vertices.dtype.names:
                        values = vertices[field]
                        if np.issubdtype(values.dtype,np.number) and not np.isfinite(values).all():
                            errors.append({'artifact':name,'reason':'nonfinite_numeric_field','field':field,
                                'nan_count':int(np.isnan(values).sum()),'positive_inf_count':int(np.isposinf(values).sum()),
                                'negative_inf_count':int(np.isneginf(values).sum())})
                except Exception as error:
                    errors.append({'artifact':name,'reason':'ply_parse_failure','exception':repr(error)})
        if original['status'] == 'available':
            runtime = original.get('runtime')
            if not runtime or runtime.get('status') != 'generated' or runtime.get('seed') != 42:
                errors.append({'reason':'missing_valid_generated_runtime'})
            status = 'artifact_invalid' if errors else 'available_verified'
            reason = 'artifact_validation_failed' if errors else None
        else:
            status = 'unavailable'
            reason = terminal_unavailable_reason(original, pool['exit_code'])
        rows.append(dict(job_id=original['job_id'], proposal_id=original['proposal_id'],
            producer_status=original['status'], normalized_status=status, reason=reason,
            per_object_attempt_proven=bool(runtime), artifact_validation_errors=errors,
            verified_artifacts=verified, attempt_classification=classification,
            generation_attempt_proven=classification in {'completed_generation', 'proven_failed_attempt'},
            producer_reason=original.get('reason'), producer_runtime_record=record_identity))
    record_directory = directory/'producer_records'
    actual_records = {p.name for p in record_directory.iterdir()} if record_directory.is_dir() else set()
    if actual_records != expected_records:
        raise PilotError('RVG per-object completion record population differs')
    return rows


def failure_classification(pool, rows, log):
    if pool['exit_code'] == 0:
        if not any(row['normalized_status'] == 'artifact_invalid' for row in rows):
            raise PilotError('completed RVG pool without invalid artifacts requires the normal postrun audit')
        return 'artifact_validation_failure'
    if 'RVG view mask differs from source-derived recomputation' in log:
        return 'environment_view_mask_replay_incompatibility'
    if 'input.numel() == 0' in log or 'input.numel()==0' in log:
        return 'tool_generation_empty_sparse_coordinates'
    return 'producer_process_failure_unclassified'


def audit(config_path,freeze_root,destination):
    config_path=Path(config_path).absolute();root=Path(freeze_root).absolute();destination=Path(destination).absolute()
    destination.relative_to(root/'terminal_audit')
    if destination.exists():raise FileExistsError('RVG terminal audit already exists')
    config=yaml.safe_load(config_path.read_text());out=rvg.output_directory(config,root)
    manifest,boundary=rvg.read_manifest(config,out)
    rvg.validate_view_receipt(config,out,manifest,boundary)
    source=Path(subprocess.check_output(['git','-C',str(config_path.parent),'rev-parse','--show-toplevel'],text=True).strip())
    git=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
    if dirty or git!=manifest['code_commit'] or sha(config_path)!=manifest['config_sha256']:
        raise PilotError('RVG terminal original source/config must remain clean and exact')
    contract=checked_contract(identity(root/'contract/freeze_manifest.json'),git,sha(config_path),config['contract_resource_id'])
    if contract['freeze_id']!=root.name:raise PilotError('RVG terminal E0 freeze differs')
    pool=json.loads((out/'proposal_pool.json').read_text())
    rows=terminal_rows(out,pool,manifest)
    if pool['rows']!=[json.loads(line) for line in (out/'proposal_records.jsonl').read_text().splitlines()]:
        raise PilotError('RVG original pool and rows differ')
    if ([r['job_id'] for r in pool['rows']] != [r['job_id'] for r in manifest['jobs']]
            or len(rows)!=pool['planned_jobs']):
        raise PilotError('RVG terminal completion population differs')
    jid=str(json.loads((out/'runtime.json').read_text())['job_id'])
    scheduler=subprocess.check_output(['sacct','-n','-P','-X','-j',jid,'--format=JobID,State,ExitCode,NodeList,ElapsedRaw'],text=True)
    log=(out/'generation.log').read_text()
    classification=failure_classification(pool,rows,log)
    report=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),
        scope='terminal RVG per-job artifact and runtime audit; no acceptance or metrics',
        producer_source_commit=git,source_checkout_clean=True,independent_source_binding_pass=True,
        producer_freeze_id=root.name,scene_id=rvg.scene_id(config),job_id=jid,scheduler=scheduler,
        producer_process_exit_code=pool['exit_code'],process_failure_classification=classification,
        config_sha256=sha(config_path),producer_config=identity(config_path),
        source_discovery_hashes=manifest['source_discovery_hashes'],planned_jobs=len(rows),
        counts=dict(Counter(r['normalized_status'] for r in rows)),
        attempt_counts=dict(Counter(r['attempt_classification'] for r in rows)),rows=rows,paper_ready=False,
        limitations=['Absent completion records do not prove an object was never attempted; the suffix remains unknown.',
                     'per_object_attempt_proven is the legacy record-presence field; generation_attempt_proven excludes view-only unavailable records.',
                     'Typed failure and view-unavailable records retain distinct reasons; raw producer details are preserved separately.'],
        audit_script_sha256=sha(Path(__file__)),
        audit_source_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[2]),'rev-parse','HEAD'],text=True).strip())
    report.update({name.replace('.','_')+'_sha256':sha(out/name) for name in CLOSURE})
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Validate the identical public adapter against a temporary audit in this
    # audit directory before publishing its final no-overwrite name.
    temporary=destination.with_name('.'+destination.name+'.validation')
    write_new(temporary,report)
    try:
        terminal_pool_rows(out,pool,manifest,dict(path=str(temporary),sha256=sha(temporary),freeze_root=str(root)))
        write_new(destination,report)
    finally:temporary.unlink()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer-config',required=True);parser.add_argument('--producer-freeze',required=True)
    parser.add_argument('--out',required=True);args=parser.parse_args()
    report=audit(args.producer_config,args.producer_freeze,args.out)
    print(json.dumps({k:report[k] for k in ('scene_id','job_id','planned_jobs','counts','process_failure_classification')}))


if __name__=='__main__':main()
