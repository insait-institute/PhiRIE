"""Resolve the predeclared scope subset and prepare TRAIN-only role construction.

This plans scope units beside the canonical ledger. It neither fabricates rollout
rows nor admits missing L2 capabilities or cross-process reference reuse.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import yaml
from robo.manifest.hash import canonical_hash
from robo.roundtrip.matrix import read_rows,save_new,sha


def scope_units(primary,subset):
    if subset['kind']!='native_scope_subset':raise ValueError('wrong subset kind')
    if subset['replacement_scopes']!=['L1_target_destination','L2_task_workspace']:
        raise ValueError('scope ladder differs from frozen protocol')
    methods=subset['controller_methods'];layouts=subset['layout_ids'];out=[];controls={}
    selected=[r for r in primary if r['layout_id'] in layouts and r['controller_method'] in methods]
    expected=subset['canonical_instances_planned']*subset['resets_per_instance']*len(methods)
    if len(selected)!=expected:raise ValueError('incomplete scope denominator')
    keys=set()
    for row in selected:
        key=(row['canonical_instance_id'],row['reset_id'],row['controller_method'])
        if key in keys:raise ValueError('duplicate scope reset/method')
        keys.add(key)
        for scope in subset['replacement_scopes']:
            item={k:row[k] for k in ['cohort_id','canonical_instance_id','instance_slot_id','layout_id','task_id','generation_seed','reset_id','policy_rng_seed','controller_method','policy_id','sensor_regime','renderer']}
            item.update(scope=scope,execution_protocol='scope_same_engine_v1',primary_unit_id=row.get('unit_id',row.get('plan_unit_id')),
                        state='NOT_STARTED',executed=None,success=None)
            item['scope_unit_id']='scope-'+canonical_hash({k:v for k,v in item.items() if k not in ['state','executed','success']})[:24]
            out.append(item)
        for method in ['REF_NATIVE',row['controller_method']]:
            controls[(key[0],key[1],method)]={'canonical_instance_id':key[0],'reset_id':key[1],'controller_method':method,
                'purpose':'fresh same-process control; original primary episode cannot be reused across engines'}
    if len({r['canonical_instance_id'] for r in out})!=subset['canonical_instances_planned']:raise ValueError('wrong instance count')
    if len(out)!=subset['extra_units_planned']:raise ValueError('scope count differs from predeclaration')
    return out,list(controls.values())


def public_destination(task):
    if task=='PickPlaceCounterToSink':return {'object_role':'receptacle','component_kind':'sink_basin','target_prompt':'sink basin'}
    if task=='PickPlaceSinkToCounter':return {'object_role':'receptacle','target_prompt':None}
    if task=='PickPlaceCounterToCabinet':return {'object_role':'support','component_kind':'cabinet_bottom_shelf','target_prompt':'bottom cabinet shelf'}
    raise ValueError('unadmitted destination task')


def prepare(*,primary,bindings,admission,subset,base_config,scope_dev_receipt,code,out):
    from robo.roundtrip.build import validate_config,task_role_prompt
    out=Path(out);out.mkdir(parents=True,exist_ok=False);code=Path(code).resolve()
    original=json.loads(Path(admission).read_text());s=Path(subset).resolve()
    matched=[r for r in original['frozen_inputs'] if Path(r['path']).name=='scope_subset.yaml']
    if len(matched)!=1 or sha(s)!=matched[0]['sha256']:raise ValueError('subset not bound by original pre-outcome admission')
    subset_config=yaml.safe_load(s.read_text());rows,controls=scope_units(read_rows(primary),subset_config)
    binding_rows=read_rows(bindings);by_id={b['canonical_instance_id']:b for b in binding_rows}
    if len(by_id)!=len(binding_rows):raise ValueError('duplicate canonical bindings')
    selected={r['canonical_instance_id']:r for r in rows};base=json.loads(Path(base_config).read_text())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=code,text=True).strip():raise ValueError('dirty tracked source')
    construction=[]
    for cid,row in sorted(selected.items(),key=lambda p:(p[1]['layout_id'],p[1]['task_id'],p[1]['generation_seed'])):
        b=by_id[cid];capture=Path(b['capture_public']);config=copy.deepcopy(base)
        config.pop('component_kind',None)
        config.update(schema_version=2,tier='TEST',scope='role_asset',declared_scope='L1_target_destination',source_commit=commit,
            canonical_instance_id=cid,cohort_id=b['cohort_id'],capture_manifest_sha256=b['capture_manifest_sha256'],
            dev_admission_sha256=sha(scope_dev_receipt),**public_destination(row['task_id']))
        # Language grounding only. The actual constructor validates sealed RGB-D
        # and masks in its existing isolated boundary; no hidden asset is read here.
        public_phrase=task_role_prompt((capture/'task_instruction.txt').read_text(),config['object_role'],config['target_prompt'],config.get('component_kind'))
        validate_config(config);config_path=out/'configs'/f'{row["instance_slot_id"]}.json';save_new(config_path,config)
        artifact=out/'destination_builds'/row['instance_slot_id'];receipt=out/'dispatch'/f'{row["instance_slot_id"]}_isolation.json'
        argv=[sys.executable,'run/sim_recon_sim/build_isolated.py','--code',str(code),
              '--config',str(config_path),'--capture',str(capture),'--out',str(artifact),'--receipt',str(receipt),
              '--forbidden',str(Path(b['bundle_dir']).parent),'--forbidden',os.environ.get('SIMANY_NATIVE_THIRD_PARTY',str(Path(code)/'third_party')),
              '--bubblewrap',os.environ.get('SIMANY_BWRAP',shutil.which('bwrap') or 'bwrap')]
        script=out/'dispatch'/f'build_{row["instance_slot_id"]}.sh';script.parent.mkdir(exist_ok=True)
        script.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(code))+'\nexport TMPDIR=${TMPDIR:-/tmp}\nexec '+shlex.join(argv)+'\n')
        construction.append({'canonical_instance_id':cid,'instance_slot_id':row['instance_slot_id'],'task_id':row['task_id'],
            'layout_id':row['layout_id'],'generation_seed':row['generation_seed'],'capture_public':str(capture),
            'capture_manifest_sha256':b['capture_manifest_sha256'],'public_prompt':public_phrase,'config':str(config_path),
            'config_sha256':sha(config_path),'command_script':str(script),'script_sha256':sha(script),'output':str(artifact),
            'state':'NOT_STARTED','terminal_failure_rule':'missing-mask/build/import failure remains failed scope unit; no L0 fallback'})
    save_new(out/'planned_scope_units.jsonl',rows,jsonl=True);save_new(out/'additional_control_units.jsonl',controls,jsonl=True)
    save_new(out/'construction_roster.jsonl',construction,jsonl=True)
    save_new(out/'scope_admission_plan.json',{'kind':'scope_preparation_not_execution_admission','source_commit':commit,
        'predeclaration_source_commit':original['source_commit'],'predeclaration_file_sha256':sha(s),'predeclaration_admission':str(Path(admission).resolve()),
        'predeclaration_admission_sha256':sha(admission),'primary_roster_sha256':sha(primary),'bindings_sha256':sha(bindings),
        'scope_dev_receipt':str(Path(scope_dev_receipt).resolve()),'scope_dev_receipt_sha256':sha(scope_dev_receipt),
        'planned_instances':len(selected),'planned_scope_units':len(rows),'planned_additional_same_engine_controls':len(controls),
        'capture_rule':'existing sealed TRAIN only; no new TEST trajectories','execution_ready':False,
        'remaining_gates':['Role-wise B3/B4 destination artifacts and fixed N3 repair protocol','Cabinet generated-role DEV integration',
            'B4 scope runner admission','L2 workspace support/obstacle implementation','Fresh same-engine scope controls and no cross-process REF reuse'],
        'l2_status':'NOT_RUN; never substituted by L0/L1','scientific_claim_gate':'NOT_RUN'})
    return construction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['primary','bindings','admission','subset','base-config','scope-dev-receipt','code','out']:p.add_argument('--'+name,required=True)
    a=p.parse_args();prepare(**vars(a))
if __name__=='__main__':main()
