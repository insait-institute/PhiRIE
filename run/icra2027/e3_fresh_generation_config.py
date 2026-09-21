"""CPU-only config preparation. Missing discovery receipts never become hashes."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

import yaml

from run.icra2027.e3_auto_discovery_pilot import PilotError, identity, sha
from run.icra2027.e3_fresh_generation_contract import (
    DISCOVERY_FILES, FRESH, PRIOR_FILES, discovery_binding, historical_algorithm_identity,
    prior_trellis_source, validate_crop_comparison,
)
from run.icra2027.e3_trellis_generation_pilot import source_jobs


def prepare(source_root, source_config, source_commit, recipe_config, prior_root,
            *, freeze_id=None, output_config=None):
    source_root=Path(source_root).resolve()
    source=source_root/'auto_discovery_pilot'
    required=sorted(DISCOVERY_FILES | {'crop_hash_comparison.json'})
    missing=[name for name in required if not (source/name).is_file()]
    report=dict(schema_version=1,kind='fresh_generation_config_preparation',
                source_freeze=str(source_root),source_commit=source_commit,
                paper_ready=False,model_calls_performed=False,config_written=False,
                planned_jobs=None,prepared_inputs=None,missing_inputs=missing)
    if missing:
        return dict(report,status='WAITING_DISCOVERY')
    if len(source_commit)!=40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise PilotError('explicit expected discovery source commit required')
    source_config=Path(source_config).resolve()
    recipe_config=Path(recipe_config).resolve()
    c=copy.deepcopy(yaml.safe_load(recipe_config.read_text()))
    # No ID is emitted or written until the caller supplies a reserved freeze.
    c.pop('freeze_id',None)
    c.update(source_pilot=str(source),source_gaussian_training_provenance=FRESH,
             source_summary_sha256=sha(source/'pilot_summary.json'),
             source_output_hashes_sha256=sha(source/'output_hashes.json'),
             source_discovery_hashes={name:sha(source/name) for name in sorted(DISCOVERY_FILES)},
             source_discovery_config=identity(source_config),source_discovery_commit=source_commit,
             source_discovery_contract=identity(source_root/'contract/freeze_manifest.json'),
             crop_comparison_sha256=sha(source/'crop_hash_comparison.json'))
    prior_root=Path(prior_root).resolve()
    audit=json.loads((prior_root/'postrun_audit.json').read_text())
    c['raw_reuse']=dict(directory=str(prior_root),producer_commit=audit['producer_source_commit'],
        config=identity(recipe_config),contract=identity(prior_root.parent/'contract/freeze_manifest.json'),
        anchors={name:identity(prior_root/name) for name in sorted(PRIOR_FILES)},
        algorithm_files=historical_algorithm_identity(audit['producer_source_commit']))
    jobs,binding=discovery_binding(c,source_jobs)
    validate_crop_comparison(c,jobs)
    prior=prior_trellis_source(c,source_jobs)
    report.update(status='READY_FOR_RESERVED_FREEZE',planned_jobs=len(jobs),
                  prepared_inputs=sum(j['prepared'] for j in jobs),
                  source_discovery_hashes=binding['source_discovery_hashes'],
                  raw_reuse_eligible=prior['reuse_eligible'],
                  raw_reuse_ineligible_reason=prior['reuse_ineligible_reason'],
                  raw_reuse_ineligible_reasons=prior.get('reuse_ineligible_reasons',[]),
                  runtime_and_model_cpu_plan_validation_pending=True)
    if output_config is None:
        return report
    if not freeze_id or not re.fullmatch(r'\d{8}-[a-f0-9]{7}-v[1-9]\d*',freeze_id) or freeze_id==source_root.name:
        raise PilotError('new caller-reserved freeze ID required before config write')
    # This checks allocator directory presence only, not reservation ownership.
    # Exact-source E0 remains the execution gate; this helper never allocates IDs.
    reservation=source_root.parent/'.freeze_ids'/freeze_id
    if not reservation.is_dir() or (source_root.parent/freeze_id).exists():
        raise PilotError('canonical new freeze reservation is absent')
    c['freeze_id']=freeze_id
    output=Path(output_config)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        yaml.safe_dump(c,stream,sort_keys=False)
    report.update(status='CONFIG_WRITTEN_E0_REQUIRED',config_written=True,
                  config=identity(output),freeze_id=freeze_id,
                  execution_ready=False)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-freeze-root',required=True)
    parser.add_argument('--source-config',required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--recipe-config',required=True)
    parser.add_argument('--prior-generation-root',required=True)
    parser.add_argument('--freeze-id')
    parser.add_argument('--output-config')
    args=parser.parse_args()
    result=prepare(args.source_freeze_root,args.source_config,args.source_commit,
                   args.recipe_config,args.prior_generation_root,
                   freeze_id=args.freeze_id,output_config=args.output_config)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
