from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from agents.eval import automatic_matching_manifest as matching
from agents.eval import eval_vs_gt
from robo.eval import agentic_ablation as e3
from robo.manifest.hash import canonical_hash

ROOT=Path(__file__).resolve().parents[1]


def test_existing_matcher_positive_unmatched_and_duplicate_identity():
    import trimesh
    mesh=trimesh.creation.icosphere(subdivisions=1,radius=.1)
    vertices=np.asarray(mesh.vertices)
    predicted=np.concatenate([vertices,vertices+10])
    archive={'vert_idx_0':np.arange(len(vertices)), 'vert_idx_1':np.arange(len(vertices)),
             'vert_idx_2':np.arange(len(vertices),len(predicted))}
    jobs=[dict(automatic_instance_id=1000+i,prepared=i!=2) for i in range(3)]
    gt=[dict(object_id=7,label='chair',vert_idx=np.arange(len(vertices)))]
    rows,surfaces,duplicates=matching.match_scene(jobs,predicted,archive,vertices,mesh.faces,gt)
    assert [r['matched_gt_id'] for r in rows]==[7,7,None]
    assert rows[2]['prepared'] is False
    assert duplicates=={'7':2}
    assert surfaces[7].shape==(20000,3)
    again=matching.match_scene(jobs,predicted,archive,vertices,mesh.faces,gt)[1]
    np.testing.assert_array_equal(surfaces[7],again[7])


def test_existing_threshold_drift_rejected(monkeypatch):
    monkeypatch.setattr(eval_vs_gt,'MATCH_THR',.24)
    with pytest.raises(ValueError,match='constants drifted'):
        matching.match_scene([],None,None,None,None,[])


def test_controller_seal_required_before_gt_access(tmp_path,monkeypatch):
    root=tmp_path/'construction';root.mkdir()
    monkeypatch.setattr(e3,'REPOSITORY_ROOT',tmp_path)
    config=dict(protocol=matching.PROTOCOL,paper_ready=False,mode='smoke',planned_scenes=1,
                planned_jobs=1,scenes=[dict(scene_id='38d58a7a31',construction_root=str(root))])
    def unsealed(*args,**kwargs):raise ValueError('controller seal missing')
    def gt_forbidden(*args,**kwargs):pytest.fail('GT accessed before sealed controller')
    monkeypatch.setattr(e3,'_load_control_scene',unsealed)
    monkeypatch.setattr(eval_vs_gt,'load_scene_gt',gt_forbidden)
    monkeypatch.setattr(matching,'_checked_anchor',gt_forbidden)
    with pytest.raises(ValueError,match='controller seal missing'):
        matching.validate_construction(config)


@pytest.mark.parametrize('path',['evaluation_references.json','surfaces/scene/gt_7.npy'])
def test_controller_rejects_evaluation_matching_paths(tmp_path,monkeypatch,path):
    monkeypatch.setattr(e3,'REPOSITORY_ROOT',tmp_path)
    file=tmp_path/'evaluation_matching'/path;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'x')
    with pytest.raises(ValueError):e3._checked_controller_file(file,'test leak')


def _fixture_manifest(tmp_path,monkeypatch):
    monkeypatch.setattr(e3,'REPOSITORY_ROOT',tmp_path)
    monkeypatch.setattr(e3,'_validate_cli_execution',lambda *args,**kwargs: {})
    directory=tmp_path/'outputs/icra2027/freeze/evaluation_matching';directory.mkdir(parents=True)
    config=tmp_path/'matching.yaml';config.write_text('paper_ready: false\n')
    contract=dict(freeze_id='freeze',code=dict(commit='a'*40,dirty=False),resource_inventory=[
        dict(resolved_path=str(config),hash_method='content_sha256',sha256=e3.sha256_file(config))])
    contract['contract_sha256']=canonical_hash(contract)
    contract_path=tmp_path/'contract.json';contract_path.write_text(json.dumps(contract))
    surface=directory/'surfaces/38d58a7a31/gt_7.npy';surface.parent.mkdir(parents=True)
    np.save(surface,np.ones((10,3)))
    payload=dict(schema_version=1,kind='evaluation_only_automatic_matching',paper_ready=False,
        freeze_id='freeze',protocol=matching.PROTOCOL,code_commit='a'*40,
        execution_contract=matching._absolute_identity(contract_path),matching_config=matching._absolute_identity(config),
        contract_sha256=contract['contract_sha256'],config_sha256=e3.sha256_file(config),
        scenes=[dict(scene_id='38d58a7a31',construction_root=str(tmp_path/'construction'),controller_shard_sha256='b'*64)],
        rows=[dict(job_id='38d58a7a31/obj_1000',scene_id='38d58a7a31',matched_gt_id=7,status='matched',evaluation_surface=matching._absolute_identity(surface)),
              dict(job_id='38d58a7a31/obj_1001',scene_id='38d58a7a31',matched_gt_id=None,status='unmatched',evaluation_surface=None)])
    path=directory/'evaluation_references.json';path.write_text(json.dumps(payload))
    seal=dict(schema_version=1,members={p.relative_to(directory).as_posix():e3.sha256_file(p) for p in (path,surface)})
    (directory/'seal.json').write_text(json.dumps(seal))
    args=(path,e3.sha256_file(path),tmp_path/'construction','38d58a7a31','b'*64,[r['job_id'] for r in payload['rows']])
    return args,surface


def test_external_manifest_loads_matched_and_explicit_null(tmp_path,monkeypatch):
    args,surface=_fixture_manifest(tmp_path,monkeypatch)
    rows,payload=matching.load_references(*args)
    assert len(rows)==2
    assert rows['38d58a7a31/obj_1001']['evaluation_surface'] is None
    surface.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='member changed'):matching.load_references(*args)


def test_external_manifest_rejects_dropped_job_and_unsealed_extra(tmp_path,monkeypatch):
    args,_=_fixture_manifest(tmp_path,monkeypatch)
    dropped=(*args[:-1],args[-1][:1])
    with pytest.raises(ValueError,match='denominator'):matching.load_references(*dropped)
    (args[0].parent/'extra.npy').write_bytes(b'extra')
    with pytest.raises(ValueError,match='closure'):matching.load_references(*args)


def test_geometry_denominator_does_not_condition_physical_stability(monkeypatch):
    monkeypatch.setattr(e3,'_policy_runtime_seconds',lambda *args:1.)
    rows=[]
    for policy in e3.POLICY_IDS:
        for i in range(3):
            rows.append(dict(policy_id=policy,job_id=f'scene/obj_{i}',scene_id='scene',accepted=i<2,
                terminal_action='accept' if i<2 else 'abstain',retry_invoked=False,
                metrics=dict(f1_20=.8,cd_cm=1.,collapse=False) if i==0 else None,
                physical_metrics=dict(settle_stable=i==0) if i<2 else None))
    result=e3._aggregate_rows(rows,{'scene':{}},3,1)
    for row in result:
        assert row['planned_jobs']==3 and row['accepted_jobs']==2
        assert row['build_coverage']==pytest.approx(2/3)
        assert row['geometry_evaluated_jobs']==1 and row['f1_20']==.8
        assert row['physical_tested_jobs']==2 and row['physical_stable_jobs']==1
        assert row['stable_fraction']==.5


def test_export_uses_original_gt_and_never_overwrites(tmp_path,monkeypatch):
    import trimesh
    monkeypatch.setattr(e3,'REPOSITORY_ROOT',tmp_path)
    scene='38d58a7a31';dataset=tmp_path/'scan_dataset';scan=dataset/'data'/scene/'scans';scan.mkdir(parents=True)
    mesh=trimesh.creation.icosphere(subdivisions=1,radius=.1)
    mesh.export(scan/'mesh_aligned_0.05.ply')
    (scan/'segments.json').write_text(json.dumps({'segIndices':[1]*len(mesh.vertices)}))
    (scan/'segments_anno.json').write_text(json.dumps({'segGroups':[{'objectId':7,'label':'chair','segments':[1]}]}))
    discovery=tmp_path/'discovery';discovery.mkdir()
    # A separate frozen observed mesh with an extra unmatched discovered body.
    observed=trimesh.util.concatenate([mesh,mesh.copy().apply_translation([10,0,0])])
    observed.export(discovery/'derived_mesh.ply')
    np.savez(discovery/'auto_instances.npz',vert_idx_0=np.arange(len(mesh.vertices)),vert_idx_1=np.arange(len(mesh.vertices),len(observed.vertices)))
    spec=dict(scene_id=scene,gt_inputs={name:matching._absolute_identity(scan/name) for name in matching.GT_FILES})
    prepared=[dict(spec=spec,jobs=[dict(automatic_instance_id=1000,prepared=True),dict(automatic_instance_id=1001,prepared=False)],
        geometry={name:matching._absolute_identity(discovery/name) for name in ('derived_mesh.ply','auto_instances.npz')},
        construction=dict(construction_root=str(tmp_path/'construction'),controller_shard_sha256='b'*64))]
    monkeypatch.setattr(matching,'validate_construction',lambda config:prepared)
    config_path=tmp_path/'matching.yaml';config_path.write_text(yaml.safe_dump(dict(freeze_id='freeze',dataset_root=str(dataset))))
    contract=dict(freeze_id='freeze',code=dict(commit='a'*40,dirty=False),resource_inventory=[
        dict(resolved_path=str(config_path),hash_method='content_sha256',sha256=e3.sha256_file(config_path))])
    contract['contract_sha256']=canonical_hash(contract)
    contract_path=tmp_path/'contract.json';contract_path.write_text(json.dumps(contract))
    monkeypatch.setattr(e3,'_validate_cli_execution',lambda *args,**kwargs:contract)
    destination=tmp_path/'outputs/icra2027/freeze/evaluation_matching'
    result=matching.export(config_path,contract_path,destination)
    assert (result['planned_jobs'],result['matched_jobs'],result['unmatched_jobs'])==(2,1,1)
    manifest=destination/'evaluation_references.json'
    rows,_=matching.load_references(manifest,e3.sha256_file(manifest),tmp_path/'construction',scene,'b'*64,
                                    [f'{scene}/obj_1000',f'{scene}/obj_1001'])
    assert rows[f'{scene}/obj_1001']['evaluation_surface'] is None
    with pytest.raises(FileExistsError):matching.export(config_path,contract_path,destination)


def test_export_cannot_relabel_construction_surface_as_gt(tmp_path,monkeypatch):
    monkeypatch.setattr(e3,'REPOSITORY_ROOT',tmp_path)
    scene='38d58a7a31';dataset=tmp_path/'scan_dataset';scan=dataset/'data'/scene/'scans';scan.mkdir(parents=True)
    for name in matching.GT_FILES:(scan/name).write_bytes(b'identical')
    spec=dict(scene_id=scene,gt_inputs={name:matching._absolute_identity(scan/name) for name in matching.GT_FILES})
    prepared=[dict(spec=spec,geometry={'derived_mesh.ply':matching._absolute_identity(scan/matching.GT_FILES[0])})]
    monkeypatch.setattr(matching,'validate_construction',lambda config:prepared)
    monkeypatch.setattr(e3,'_validate_cli_execution',lambda *args,**kwargs:{})
    config_path=tmp_path/'matching.yaml';config_path.write_text(yaml.safe_dump(dict(freeze_id='freeze',dataset_root=str(dataset))))
    with pytest.raises(ValueError,match='construction mesh cannot'):
        matching.export(config_path,tmp_path/'unused',tmp_path/'outputs/icra2027/freeze/evaluation_matching')
    assert not (tmp_path/'outputs/icra2027/freeze/evaluation_matching').exists()


def test_coverage_sweep_retains_accepted_unmatched_physics():
    policies=yaml.safe_load((ROOT/'configs/experiments/icra2027/agentic_policies.yaml').read_text())
    proposal=dict(proposal_id='p',job_id='scene/obj_1000',tool='trellis',
                  evidence=e3._smoke_evidence(residual=.001))
    controls={'scene':{'proposals':[proposal]}}
    evaluations={'scene':dict(proposal_metrics={'p':dict(f1_20=None,cd_cm=None,collapse=None)},
                             rows=[dict(job_id='scene/obj_1000')])}
    rows=e3._coverage_sweep(policies,controls,evaluations,1)
    assert rows and all(r['accepted_jobs']==1 and r['geometry_evaluated_jobs']==0 for r in rows)
    assert all(r['build_coverage']==1. and r['stable_fraction']==1. for r in rows)
    assert all(r['physical_tested_jobs']==1 and r['f1_20'] is None for r in rows)


def test_external_evaluate_writes_new_freeze_and_preserves_unmatched(tmp_path,monkeypatch):
    monkeypatch.setattr(e3,'REPOSITORY_ROOT',tmp_path)
    old=tmp_path/'outputs/icra2027/old/agentic';control=old/'control/38d58a7a31';control.mkdir(parents=True)
    proposal=dict(proposal_id='p',job_id='38d58a7a31/obj_1000',object_id='obj_1000',
                  evidence=e3._smoke_evidence(residual=.01))
    shard=dict(freeze_id='old',proposals=[proposal])
    ledger=[dict(job_id=proposal['job_id'],policy_id=p,terminal_action='accept',selected_proposal_id='p') for p in e3.POLICY_IDS]
    (control/'job_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger))
    monkeypatch.setattr(e3,'_load_control_scene',lambda *args:(control,shard,{'members':{'controller_shard.json':'b'*64}}))
    monkeypatch.setattr(e3,'_load_inventory',lambda *args,**kwargs:({},{}))
    monkeypatch.setattr(e3,'_scene_inventory',lambda *args:{'jobs':[{'job_id':proposal['job_id']}]})
    monkeypatch.setattr(e3,'validate_ledger',lambda rows:None)
    monkeypatch.setattr(matching,'load_references',lambda *args:({proposal['job_id']:{'status':'unmatched'}},
                                                               {'freeze_id':'new','code_commit':'a'*40}))
    result=e3.run_evaluate('old',old,'38d58a7a31',evaluation_manifest='manifest',evaluation_manifest_sha256='a'*64)
    assert result['geometry_reference_jobs']==0 and result['geometry_unmatched_jobs']==1
    assert len(result['rows'])==5 and all(r['physical_metrics']['settle_stable'] for r in result['rows'])
    assert not (old/'evaluation').exists()
    assert (tmp_path/'outputs/icra2027/new/agentic/evaluation/38d58a7a31/eval_shard.json').is_file()


def test_external_aggregate_rejects_changed_policy_before_evaluation(tmp_path,monkeypatch):
    policies=yaml.safe_load((ROOT/'configs/experiments/icra2027/agentic_policies.yaml').read_text())
    sealed_hash=e3._canonical_digest(policies)
    changed=copy.deepcopy(policies)
    changed['gates']['max_registration_residual_m'] += .001
    old=tmp_path/'old';new=tmp_path/'new';new.mkdir()
    jobs={'automatic_sources':dict(scene_id='38d58a7a31',planned_jobs=15)}
    inventory=dict(counts=dict(scenes=1,jobs=15,policy_object_rows=75),scenes=[{'scene_id':'38d58a7a31'}])
    monkeypatch.setattr(e3,'_load_inventory',lambda *args,**kwargs:(inventory,{}))
    monkeypatch.setattr(e3,'_load_control_scene',lambda *args:(old,{'policy_config_sha256':sealed_hash},{}))
    monkeypatch.setattr(e3,'_load_eval_scene',lambda *args,**kwargs:pytest.fail('evaluation opened after policy drift'))
    with pytest.raises(ValueError,match='policy differs from sealed controller'):
        e3.run_aggregate(jobs,changed,'old',old,evaluation_root=new)
    assert not list(new.iterdir())
