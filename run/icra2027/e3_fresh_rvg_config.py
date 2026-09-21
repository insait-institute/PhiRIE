"""Prepare fresh RVG configs from sealed discovery; never reuse old RVG views."""
import argparse
import copy
import json
from pathlib import Path
import re

import yaml

from run.icra2027.e3_auto_discovery_pilot import identity,sha,PilotError
from run.icra2027.e3_fresh_generation_contract import DISCOVERY_FILES,FRESH
from run.icra2027.e3_rvg_generation_pilot import source_boundary


def prepare(source_root,source_config,source_commit,recipe_config,*,freeze_id=None,output_config=None,scene_id=None):
    root=Path(source_root).resolve();source=root/'auto_discovery_pilot'
    if scene_id is not None:
        if not re.fullmatch(r'[a-f0-9]{10}', str(scene_id)):
            raise PilotError('invalid requested RVG scene ID')
        source = source / scene_id
    missing=sorted(name for name in DISCOVERY_FILES if not (source/name).is_file())
    report=dict(paper_ready=False,model_calls_performed=False,config_written=False,
                planned_jobs=None,prepared_inputs=None,missing_inputs=missing)
    if missing:return dict(report,status='WAITING_DISCOVERY')
    if not re.fullmatch('[a-f0-9]{40}',source_commit):raise PilotError('expected discovery commit required')
    source_config=Path(source_config).resolve()
    c=copy.deepcopy(yaml.safe_load(Path(recipe_config).read_text()))
    c.pop('freeze_id',None)
    source_spec=yaml.safe_load(source_config.read_text())
    if scene_id is not None:
        from run.icra2027.e3_auto_discovery_pilot import COHORT_SCOPE
        from run.icra2027.e3_gaussian_train_only import cohort_scene_ids
        if source_spec.get('scope') != COHORT_SCOPE or scene_id not in cohort_scene_ids(source_spec):
            raise PilotError('RVG scene outside frozen discovery cohort')
        source_spec=dict(source_spec,scene_id=scene_id)
        c.update(output_scene_id=scene_id,contract_resource_id='e3_rvg_config_'+scene_id)
    else:
        c.pop('output_scene_id',None)
        c['contract_resource_id']='e3_rvg_config'
    c.update(source_pilot=str(source),scene_id=source_spec['scene_id'],
        source_gaussian_training_provenance=FRESH,
        source_summary_sha256=sha(source/'pilot_summary.json'),source_output_hashes_sha256=sha(source/'output_hashes.json'),
        source_input_manifest_sha256=sha(source/'input_manifest.json'),
        source_discovery_hashes={name:sha(source/name) for name in sorted(DISCOVERY_FILES)},
        source_discovery_config=identity(source_config),source_discovery_commit=source_commit,
        source_discovery_contract=identity(root/'contract/freeze_manifest.json'))
    jobs,boundary,binding=source_boundary(c)
    report.update(status='READY_FOR_RESERVED_FREEZE',planned_jobs=len(jobs),prepared_inputs=sum(j['prepared'] for j in jobs),
                  source_discovery_hashes=binding['source_discovery_hashes'],
                  actual_multiview_collection_pending=True,old_rvg_reuse_performed=False)
    if output_config is None:return report
    if (not freeze_id or not re.fullmatch(r'\d{8}-[a-f0-9]{7}-v[1-9]\d*',freeze_id)
            or not (root.parent/'.freeze_ids'/freeze_id).is_dir() or (root.parent/freeze_id).exists()):
        raise PilotError('unused canonical reservation directory required; E0 remains the execution gate')
    c['freeze_id']=freeze_id
    output=Path(output_config);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:yaml.safe_dump(c,f,sort_keys=False)
    return dict(report,status='CONFIG_WRITTEN_E0_REQUIRED',config_written=True,config=identity(output),execution_ready=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-freeze-root','source-config','source-commit','recipe-config'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--freeze-id');parser.add_argument('--output-config');parser.add_argument('--scene-id');args=parser.parse_args()
    print(json.dumps(prepare(args.source_freeze_root,args.source_config,args.source_commit,args.recipe_config,
                            freeze_id=args.freeze_id,output_config=args.output_config,scene_id=args.scene_id),indent=2,allow_nan=False))


if __name__=='__main__':main()
