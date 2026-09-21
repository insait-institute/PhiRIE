"""Read-only E1 reference inventory. Does not produce construction-table rows.

Existing construction_metrics owns all metric aggregation. This metadata audit
cannot admit artifacts for reuse or certify a new physical measurement.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import yaml
from robo.eval.construction_metrics import PAPER_SCENE_IDS, REQUIRED_REGIMES

ROOT=Path('/group/worldcept/code/SimAny')
E3=ROOT/'outputs/icra2027/20260906-357caca-v1'
JOBS=Path('/group/worldcept/code/SimAny-wt/e3-full-canonical/configs/experiments/icra2027/e3_full_canonical/agentic_fresh_jobs.yaml')
E4=ROOT/'outputs/icra2027/20260906-0e7579a-v1'
E4_CONFIG=Path('/group/worldcept/code/SimAny-wt/e4-full-execution/configs/experiments/icra2027/e4_full_qualification/execution.json')
AUDIT_SHA='68d6f8cc9d3b0a7bbd05c7243f9de0e1f1b3821f12b8cb428c20a6ba30b24ecb'
LEGACY=ROOT/'outputs/icra2027/icra2027-contract-v1-e1-c22afef-v1/construction/scene_records.csv'


def identity(path,expected=None):
    path=Path(path);data=path.read_bytes();sha=hashlib.sha256(data).hexdigest()
    if expected is not None and sha!=expected:raise ValueError(f'source hash differs: {path}')
    return {'path':str(path),'sha256':sha,'size_bytes':len(data)}


def read_json(path,expected=None):
    path=Path(path);data=path.read_bytes();sha=hashlib.sha256(data).hexdigest()
    if expected is not None and sha!=expected:raise ValueError(f'source hash differs: {path}')
    return json.loads(data),{'path':str(path),'sha256':sha,'size_bytes':len(data)}


def sealed_metadata(directory,name):
    """Authenticate named metadata through its seal, without replaying geometry."""
    directory=Path(directory)
    if not (directory/'seal.json').is_file():return None
    seal,sref=read_json(directory/'seal.json')
    ref=seal['members'][name]
    expected=ref['sha256'] if isinstance(ref,dict) else ref
    value,vref=read_json(directory/name,expected)
    identity(directory/'seal.json',sref['sha256'])
    return {'seal':sref,'file':vref,'payload':value}


def scene_plan(scene,discovery,controller,e4):
    result=[]
    for regime in REQUIRED_REGIMES:
        automatic=regime==REQUIRED_REGIMES[3]
        result.append({'scene_id':scene,'regime':regime,'execution_state':'NOT_RUN',
            'table_i_record_admissible':False,
            'input_family_status':'MATCH_CURRENT_FULL_E3' if automatic else 'FRESH_PROTOCOL_REQUIRED',
            'discovery_source':discovery if automatic else None,
            'controller_evidence':controller if automatic else None,
            'e4_export_snapshot':e4 if automatic else None,
            'table_i_measurements':{'input_instances':None,'accepted_instances':None,'f1_20':None,
                'f1_weight':None,'stable_instances':None,'tested_instances':None,'runtime_minutes':None},
            'missing_criteria':(['prospective_common_constructor_and_acceptance_definition',
                'E1_drop_protocol_and_collision_identity','complete_runtime_scope',
                'all_50_scene_simulator_export_accounting'] if automatic else [
                'source_bound_fresh_input_preparation','same_declared_constructor_as_other_regimes',
                'independent_nonconstruction_geometry_reference','E1_drop_protocol','complete_runtime_scope']),
            'reuse_admitted':False})
    return result


def audit():
    payload,aref=read_json(E3/'independent_evaluation_completion_audit.json',AUDIT_SHA)
    if payload['status']!='PASS' or payload['planned_scenes']!=50 or payload['planned_jobs']!=1871:
        raise ValueError('full E3 population changed')
    seal_path=E3/'agentic/aggregate_seal.json'
    seal,sref=read_json(seal_path,payload['evidence_hashes'][str(seal_path)]['sha256'])
    ledger_path=E3/'agentic/job_ledger.jsonl';lref=identity(ledger_path,seal['members']['job_ledger.jsonl'])
    ledger=[json.loads(line) for line in ledger_path.read_text().splitlines() if line]
    if len(ledger)!=9355 or len({(r['job_id'],r['policy_id']) for r in ledger})!=9355:
        raise ValueError('full E3 ledger roster changed')
    identity(JOBS,'a2e7e49c7e5625157676e3bcd3c2369618f9b0e337d9768eb54e914f56b7f803')
    jobs=yaml.safe_load(JOBS.read_text());sources={s['scene_id']:s for s in jobs['automatic_sources']}
    if set(sources)!=set(PAPER_SCENE_IDS):raise ValueError('source scene roster differs from E1')
    e4_config,e4ref=read_json(E4_CONFIG,'e252c76307ad1d342922d4746cd1641507b61fa61f982bd925470d377627c58b')
    rows=[];snapshots=[]
    for scene in PAPER_SCENE_IDS:
        source=sources[scene];directory=Path(source['discovery_directory'])
        inp,iref=read_json(directory/'input_manifest.json',source['discovery_hashes']['input_manifest.json'])
        summary,pref=read_json(directory/'pilot_summary.json',source['discovery_hashes']['pilot_summary.json'])
        _,postref=read_json(directory/'postrun_audit.json',source['discovery_hashes']['postrun_audit.json'])
        if (inp['source_gaussian_training_provenance']!='FRESH_OFFICIAL_TRAIN_ONLY'
                or inp['planned_stages']!=['render','fuse','discover','prepare','refine']
                or inp['boundary']['heldout_overlap'] or any(s['exit_code']!=0 for s in summary['stages'])):
            raise ValueError(f'discovery input boundary changed: {scene}')
        a4=[r for r in ledger if r['scene_id']==scene and r['policy_id']=='A4']
        if len(a4)!=source['planned_jobs'] or len(a4)!=summary['discovered_instances']:
            raise ValueError(f'discovery/controller denominator changed: {scene}')
        discovery={'input_manifest':iref,'summary':pref,'original_postrun_audit':postref,
            'discovered_instances':summary['discovered_instances'],'prepared_instances':summary['prepared_instances'],
            'vocabulary':'FULL','instance_population':inp['instance_population'],
            'mesh_semantics':'TSDF from TRAIN-only Gaussian rendered depth; not scan mesh'}
        counts=Counter(r['terminal_action'] for r in a4)
        controller={'scope':'A4_controller_terminal_decisions_not_simulator_builds',
            'planned_jobs':len(a4),'terminal_counts':dict(counts),'evidence_reference':lref,
            'stability_scope':'isolated_single_convex_hull_corrected_AABB_drop_with_frozen_contact_dynamics',
            'runtime_scope':payload['runtime_scope'],'table_i_values_promoted':False}
        factory=E4/'automatic_candidates'/scene/'materialized/A4'
        material=sealed_metadata(factory,'materialization_manifest.json')
        population=sealed_metadata(E4/'automatic_candidates'/scene/'population','gate.json')
        snapshot={'scene_id':scene,'materialization':None,'population':None,
                  'export_files':{},'state':'NOT_AVAILABLE_OR_IN_FLIGHT','reuse_admitted':False,
                  'measurement_scope':'MuJoCo full-room export settle; not E1 drop or manipulation'}
        if material:
            m=material.pop('payload')
            if m['scene_id']!=scene or m['policy_id']!='A4':raise ValueError('E4 materialization identity differs')
            snapshot['materialization']=material
        if population:
            p=population.pop('payload')
            if p['scene_id']!=scene or p['code']['commit']!='553db0575241a6858921b6d648d24c29392161fd':
                raise ValueError('E4 population source differs')
            snapshot['population']=population
            for name,ref in p['exports']['A4'].get('artifacts',{}).items():
                snapshot['export_files'][name]=identity(ROOT/ref['path'],ref['sha256'])
            snapshot['state']=('SEALED_EXPORT_METADATA_ONLY' if {'scene_xml','mujoco_settle','room_collision_report','isaac_manifest'}<=set(snapshot['export_files']) else 'SEALED_POPULATION_NO_COMPLETE_EXPORT')
        snapshots.append(snapshot)
        rows.extend(scene_plan(scene,discovery,controller,snapshot))
    if len(rows)!=250 or len({(r['scene_id'],r['regime']) for r in rows})!=250:
        raise ValueError('five-input denominator changed')
    return {'schema_version':1,'scope':'E1_metadata_reference_audit_and_missing_rerun_protocol',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'audit_code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'audit_code_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        'planned_scenes':50,'planned_scene_regime_cells':250,'paper_ready':False,
        'table_i_claim_gate':'FAIL','reuse_admitted':False,
        'reference_validation_scope':'metadata content hashes and saved seals; no geometry/physics replay or new measurement',
        'sources':{'full_e3_audit':aref,'full_e3_seal':sref,'full_e3_ledger':lref,
            'construction_jobs':identity(JOBS),'e4_config':e4ref,
            'legacy_250_invalid_records':identity(LEGACY,'b9168f94943613e250feeb21c829b516d75dd17cc1483e35d0f9bbbf1e6315de')},
        'summary':{'current_controller_planned_jobs':sum(r['controller_evidence']['planned_jobs'] for r in rows if r['controller_evidence']),
            'current_A4_controller_accepts':sum(r['controller_evidence']['terminal_counts'].get('accept',0) for r in rows if r['controller_evidence']),
            'sealed_A4_export_metadata_scenes':sum(s['state']=='SEALED_EXPORT_METADATA_ONLY' for s in snapshots),
            'table_i_admissible_records':0},'rows':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);args=parser.parse_args()
    out=Path(args.out).resolve()
    if out.exists():raise FileExistsError(f'refusing to overwrite audit: {out}')
    value=audit();out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'out':str(out),'sha256':identity(out)['sha256'],**value['summary']}))

if __name__=='__main__':main()
