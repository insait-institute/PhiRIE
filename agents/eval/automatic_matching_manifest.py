"""Evaluation-only matching export for already sealed automatic construction.

Called through agents.eval.eval_vs_gt; metric and matching implementations stay
with existing producers. No generated proposal or policy decision selects GT.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import yaml

from robo.eval import agentic_ablation as e3
from robo.manifest.hash import canonical_hash
from run.icra2027.e3_fresh_generation_contract import discovery_binding, FRESH
from run.icra2027.e3_trellis_generation_pilot import source_jobs

PROTOCOL = dict(matcher='existing_mutual_nearest_neighbor_min_coverage',
    match_threshold=0.25, overlap_tolerance_m=0.02, overlap_points=4000,
    gt_surface_samples=20000, gt_surface_seed=42, coordinate_frame='scannetpp_world',
    unit='metre', same_mesh=False, policy_independent=True,
    duplicate_rule='retain_and_diagnose', unmatched_rule='null_geometry_keep_planned_job')
GT_FILES = ('mesh_aligned_0.05.ply', 'segments.json', 'segments_anno.json')


def _checked_anchor(record):
    anchor=e3._source_identity(record,'evaluation input')
    if e3.sha256_file(anchor['path']) != anchor['sha256']:
        raise ValueError('evaluation input content differs')
    return anchor


def _absolute_identity(path):
    path=Path(path).absolute()
    return dict(path=str(path),size_bytes=path.stat().st_size,sha256=e3.sha256_file(path))


def _pilot_from_execution(config, declared_roster):
    """Authenticate a predeclared pilot from the construction E0, before GT.

    Legacy pilot configs keep their fixed 38d/15 population. The opt-in anchor
    names a full-cohort execution config frozen before any controller outcome.
    It changes neither matching nor evaluation-surface selection.
    """
    from run.icra2027.e3_fresh_canonical_phase import scene_slots

    anchor = _checked_anchor(config['pilot_execution_config'])
    execution = yaml.safe_load(Path(anchor['path']).read_text())
    slots = scene_slots(execution)
    pilot = execution.get('pilot')
    expected = min((len(values), scene) for scene, values in slots.items() if values)
    if (execution.get('schema_version') != 2 or execution.get('scope') != 'fresh_canonical_engineering'
            or execution.get('paper_ready') is not False or list(slots) != declared_roster
            or len(slots) != 50 or sum(map(len, slots.values())) != 1871
            or execution.get('planned_policy_object_rows') != 9355
            or pilot != dict(selection_rule='smallest_positive_population_then_scene_id',
                             scene_id=expected[1], planned_jobs=expected[0])
            or type(pilot.get('planned_jobs')) is not int
            or [s['scene_id'] for s in config['scenes']] != [expected[1]]
            or type(config['planned_jobs']) is not int
            or config['planned_jobs'] != expected[0]
            or config['scenes'][0]['planned_jobs'] != expected[0]):
        raise ValueError('predeclared execution pilot scene/count/roster differs')
    root = e3.checked_repo_path(config['scenes'][0]['construction_root'], 'pilot construction', kind='dir')
    contract = json.loads((root.parent/'contract/freeze_manifest.json').read_text())
    digest = canonical_hash({k:v for k,v in contract.items()
                            if k not in {'created_utc','environment','contract_sha256'}})
    jobs, _ = e3._load_inventory(root, controller_safe=True)
    resource_matches = [r for r in contract['resource_inventory']
        if r.get('resolved_path') == anchor['path'] and r.get('hash_method') == 'content_sha256'
        and r.get('sha256') == anchor['sha256']]
    if (len(resource_matches) != 1 or contract.get('contract_sha256') != digest
            or contract['code'].get('dirty') is not False
            or contract['freeze_id'] != root.parent.name or execution['freeze_id'] != contract['freeze_id']
            or jobs['freeze_id'] != contract['freeze_id']
            or jobs['source_contract']['contract_sha256'] != digest
            or jobs['source_contract']['code_commit'] != contract['code']['commit']
            or jobs['counts'] != dict(scenes=50,jobs=1871,policy_object_rows=9355)
            or [s['scene_id'] for s in jobs['scenes']] != list(slots)
            or {s['scene_id']: [j['object_slot'] for j in s['jobs']] for s in jobs['scenes']} != slots):
        raise ValueError('pilot execution source/E0/inventory binding differs')
    return [expected[1]], expected[0]


def validate_construction(config):
    """Finish all construction checks before the first GT read/hash."""
    if config.get('protocol') != PROTOCOL or config.get('paper_ready') is not False:
        raise ValueError('frozen evaluation matching protocol differs')
    specs=config['scenes']; ids=[s['scene_id'] for s in specs]
    if len(ids)!=len(set(ids)) or len(ids)!=config['planned_scenes']:
        raise ValueError('matching scene denominator differs')
    if config.get('mode')=='full' and (len(ids)!=50 or config['planned_jobs']!=1871):
        raise ValueError('full matching requires all 50 scenes and 1871 jobs')
    if config.get('mode') not in {'smoke','pilot','full'}:
        raise ValueError('unknown matching mode')
    if config['mode'] != 'smoke':
        if config['dataset_root'] != '/data/ScanNetpp':
            raise ValueError('real evaluation requires original ScanNet++ dataset root')
        roster=_checked_anchor(config['population_roster'])
        declared=[str(x) for x in yaml.safe_load(Path(roster['path']).read_text())['population']['scene_ids']]
        expected=declared if config['mode']=='full' else ['38d58a7a31']
        expected_jobs=15
        if config['mode']=='pilot' and 'pilot_execution_config' in config:
            expected, expected_jobs = _pilot_from_execution(config, declared)
        if ids!=expected or (config['mode']=='pilot' and config['planned_jobs']!=expected_jobs):
            raise ValueError('predeclared evaluation roster differs')
    prepared=[];count=0
    for spec in specs:
        scene=e3._require_scene_id(spec['scene_id'])
        root=e3.checked_repo_path(spec['construction_root'],'sealed construction',kind='dir')
        directory,shard,seal=e3._load_control_scene(root,scene)
        anchor=_checked_anchor(spec['control_seal'])
        if Path(anchor['path'])!=directory/'seal.json':raise ValueError('controller seal path differs')
        if spec['discovery'].get('source_gaussian_training_provenance') != FRESH:
            raise ValueError('evaluation requires sealed official TRAIN-only construction')
        jobs,binding=discovery_binding(spec['discovery'],source_jobs)
        source=Path(spec['discovery']['source_pilot'])
        if json.loads((source/'pilot_summary.json').read_text())['scene_id']!=scene:
            raise ValueError('discovery scene binding differs')
        hashes=json.loads((source/'output_hashes.json').read_text())
        geometry={}
        for name in ('derived_mesh.ply','auto_instances.npz'):
            path=source/'construction'/name
            if path.is_symlink() or e3.sha256_file(path)!=hashes[name]['sha256']:
                raise ValueError('frozen discovery geometry differs')
            geometry[name]=_absolute_identity(path)
        expected=[f'{scene}/obj_{j["automatic_instance_id"]:02d}' for j in jobs]
        ledger=[json.loads(l) for l in (directory/'job_ledger.jsonl').read_text().splitlines() if l.strip()]
        e3.validate_ledger(ledger)
        if Counter((r['job_id'],r['policy_id']) for r in ledger)!=Counter((j,p) for j in expected for p in e3.POLICY_IDS):
            raise ValueError('complete controller policy/job denominator differs')
        if len(expected)!=spec['planned_jobs']:raise ValueError('matching job denominator differs')
        count+=len(jobs)
        prepared.append(dict(spec=spec,jobs=jobs,geometry=geometry,construction=dict(
            control_seal=anchor,controller_shard_sha256=seal['members']['controller_shard.json'],
            construction_root=str(root),discovery_binding=binding)))
    if count!=config['planned_jobs']:raise ValueError('complete planned matching population differs')
    return prepared


def match_scene(jobs, predicted_vertices, archive, gt_vertices, gt_faces, gts):
    """Match one frozen discovered identity once, never a policy's proposal."""
    from agents.eval import eval_vs_gt as matching
    if (matching.MATCH_THR,matching.OVERLAP_TOL_M,matching.N_OVERLAP_PTS,
            matching.N_GT_SAMPLES)!=(0.25,0.02,4000,20000):
        raise ValueError('existing matching implementation constants drifted')
    gts=[dict(g) for g in gts if g['label'].strip().lower() not in matching.C.STRUCTURAL_EXCLUDE]
    for g in gts:
        points=gt_vertices[g['vert_idx']];g['lo'],g['hi']=points.min(axis=0),points.max(axis=0)
    rows=[];surfaces={}
    for job in jobs:
        oid=job['automatic_instance_id'];idx=np.asarray(archive[f'vert_idx_{oid-1000}'])
        if oid<1000 or idx.ndim!=1 or not np.issubdtype(idx.dtype,np.integer) or not len(idx) or idx.min()<0 or idx.max()>=len(predicted_vertices):
            raise ValueError('discovered instance indices differ')
        match,score=matching.best_match(idx,predicted_vertices[idx],gts,gt_vertices,False)
        gid=int(match['object_id']) if match is not None else None
        if match is not None and gid not in surfaces:
            surface=matching.submesh_points(gt_vertices,gt_faces,match['vert_idx'],20000,seed=42)
            if not len(surface) or not np.isfinite(surface).all():raise ValueError('invalid independent GT surface')
            surfaces[gid]=surface
        rows.append(dict(automatic_instance_id=oid,prepared=job['prepared'],matched_gt_id=gid,
            matching_score=float(score),status='matched' if match is not None else 'unmatched'))
    counts=Counter(r['matched_gt_id'] for r in rows if r['status']=='matched')
    return rows,surfaces,{str(k):v for k,v in counts.items() if v>1}


def export(config_path, contract_path, destination):
    config_path=Path(config_path);config=yaml.safe_load(config_path.read_text())
    contract=e3._validate_cli_execution(contract_path,config['freeze_id'],Path(destination),config_paths=[config_path])
    expected=e3.REPOSITORY_ROOT/'outputs/icra2027'/config['freeze_id']/'evaluation_matching'
    destination=e3.checked_repo_path(destination,'matching output',must_exist=False)
    if destination!=expected:raise ValueError('fixed evaluation-only output path required')
    if destination.exists():raise FileExistsError('refusing matching overwrite')
    prepared=validate_construction(config)
    # No GT access occurs above this line; all A0--A4 construction is now sealed.
    from agents.eval import eval_vs_gt as matching
    from plyfile import PlyData
    rows=[];scene_records=[];members={}
    with e3._atomic_directory(destination) as staging:
        for item in prepared:
            spec=item['spec'];scene=spec['scene_id'];gt_inputs={}
            if set(spec['gt_inputs'])!=set(GT_FILES):raise ValueError('GT input closure differs')
            for name in GT_FILES:
                expected=Path(config['dataset_root'])/'data'/scene/'scans'/name
                record=spec['gt_inputs'][name]
                if Path(record['path'])!=expected:raise ValueError('GT reference must be original scan input')
                gt_inputs[name]=_checked_anchor(record)
            if gt_inputs['mesh_aligned_0.05.ply']['sha256']==item['geometry']['derived_mesh.ply']['sha256']:
                raise ValueError('construction mesh cannot serve as independent GT')
            gv,gf,gts=matching.load_scene_gt(scene,dataset_root=config['dataset_root'])
            vertices=PlyData.read(item['geometry']['derived_mesh.ply']['path'])['vertex']
            pv=np.stack([vertices[k] for k in 'xyz'],axis=1)
            with np.load(item['geometry']['auto_instances.npz']['path'],allow_pickle=False) as archive:
                matches,surfaces,duplicates=match_scene(item['jobs'],pv,archive,gv,gf,gts)
            refs={}
            for gid,points in surfaces.items():
                name=f'surfaces/{scene}/gt_{gid}.npy';path=staging/name;path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as f:np.save(f,points,allow_pickle=False)
                members[name]=e3.sha256_file(path)
                refs[gid]=dict(path=str(destination/name),size_bytes=path.stat().st_size,sha256=members[name])
            for row in matches:
                slot=f'obj_{row["automatic_instance_id"]:02d}'
                rows.append(dict(row,job_id=f'{scene}/{slot}',scene_id=scene,object_slot=slot,
                    evaluation_surface=refs.get(row['matched_gt_id'])))
            for anchor in [*gt_inputs.values(), *item['geometry'].values()]:
                _checked_anchor(anchor)
            scene_records.append(dict(scene_id=scene,**item['construction'],gt_inputs=gt_inputs,
                discovery_geometry=item['geometry'],planned_jobs=len(matches),matched_jobs=len([r for r in matches if r['status']=='matched']),
                unique_matched_gt=len(surfaces),duplicate_gt_matches=duplicates))
        result=dict(schema_version=1,kind='evaluation_only_automatic_matching',paper_ready=False,
            freeze_id=config['freeze_id'],code_commit=contract['code']['commit'],config_sha256=e3.sha256_file(config_path),
            contract_sha256=contract['contract_sha256'],execution_contract=_absolute_identity(contract_path),
            matching_config=_absolute_identity(config_path),protocol=PROTOCOL,planned_scenes=len(prepared),
            planned_jobs=len(rows),matched_jobs=sum(r['status']=='matched' for r in rows),
            unmatched_jobs=sum(r['status']=='unmatched' for r in rows),scenes=scene_records,rows=rows)
        e3._write_json_inside(staging/'evaluation_references.json',result)
        members['evaluation_references.json']=e3.sha256_file(staging/'evaluation_references.json')
        e3._write_json_inside(staging/'seal.json',dict(schema_version=1,members=members))
    return result


def load_references(path, digest, construction_root, scene, control_hash, job_ids):
    path=e3.checked_repo_path(path,'evaluation-only references',kind='file')
    if path.name!='evaluation_references.json' or path.parent.name!='evaluation_matching':
        raise ValueError('invalid evaluation-only manifest location')
    if e3.sha256_file(path)!=e3._require_sha256(digest,'evaluation manifest hash'):
        raise ValueError('external evaluation manifest changed')
    seal=json.loads((path.parent/'seal.json').read_text())
    members=seal.get('members',{})
    if any(p.is_symlink() for p in path.parent.rglob('*')):
        raise ValueError('evaluation seal contains a symlink')
    actual={p.relative_to(path.parent).as_posix() for p in path.parent.rglob('*') if p.is_file() and p!=path.parent/'seal.json'}
    if not members or set(members)!=actual:raise ValueError('evaluation seal closure differs')
    for name,sha in members.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('invalid evaluation seal member')
        member=e3.checked_repo_path(path.parent/name,'sealed evaluation member',kind='file')
        if e3.sha256_file(member)!=e3._require_sha256(sha,'evaluation member hash'):
            raise ValueError('evaluation member changed')
    payload=json.loads(path.read_text())
    if payload.get('schema_version')!=1 or payload.get('paper_ready') is not False or payload.get('kind')!='evaluation_only_automatic_matching' or payload.get('protocol')!=PROTOCOL:
        raise ValueError('external evaluation reference protocol differs')
    contract_anchor=_checked_anchor(payload['execution_contract'])
    config_anchor=_checked_anchor(payload['matching_config'])
    contract=json.loads(Path(contract_anchor['path']).read_text())
    contract_digest=canonical_hash({k:v for k,v in contract.items() if k not in {'created_utc','environment','contract_sha256'}})
    if (contract_digest!=payload['contract_sha256'] or contract_digest!=contract.get('contract_sha256')
            or contract.get('code',{}).get('commit')!=payload['code_commit']
            or contract.get('code',{}).get('dirty') is not False
            or contract.get('freeze_id')!=payload['freeze_id']
            or config_anchor['sha256']!=payload['config_sha256']):
        raise ValueError('evaluation E0 provenance differs')
    matches=[r for r in contract.get('resource_inventory',[]) if r.get('resolved_path')==config_anchor['path'] and r.get('hash_method')=='content_sha256']
    if not matches or any(r.get('sha256')!=config_anchor['sha256'] for r in matches):
        raise ValueError('evaluation config absent from E0 closure')
    expected=e3.REPOSITORY_ROOT/'outputs/icra2027'/payload['freeze_id']/'evaluation_matching'
    if path.parent!=expected:raise ValueError('evaluation freeze output differs')
    e3._validate_cli_execution(contract_anchor['path'],payload['freeze_id'],path.parent,config_paths=[config_anchor['path']])
    specs=[s for s in payload['scenes'] if s['scene_id']==scene]
    if len(specs)!=1 or Path(specs[0]['construction_root'])!=construction_root or specs[0]['controller_shard_sha256']!=control_hash:
        raise ValueError('external reference construction binding differs')
    references=[r for r in payload['rows'] if r['scene_id']==scene]
    if len(references)!=len(job_ids) or set(r['job_id'] for r in references)!=set(job_ids):
        raise ValueError('external reference job denominator differs')
    for r in references:
        surface=r.get('evaluation_surface')
        if r['status']=='unmatched':
            if surface is not None or r['matched_gt_id'] is not None:raise ValueError('unmatched job has invented GT')
        elif r['status']=='matched':
            target=path.parent/f'surfaces/{scene}/gt_{r["matched_gt_id"]}.npy'
            if surface is None or Path(surface['path'])!=target:raise ValueError('matched surface path differs')
            _checked_anchor(surface)
        else:raise ValueError('unknown matching status')
    return {r['job_id']:r for r in references},payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matching-config',required=True);parser.add_argument('--contract-manifest',required=True)
    parser.add_argument('--out',required=True);args=parser.parse_args()
    result=export(args.matching_config,args.contract_manifest,args.out)
    print(json.dumps({k:result[k] for k in ('freeze_id','planned_scenes','planned_jobs','matched_jobs','unmatched_jobs')}))


if __name__=='__main__':main()
