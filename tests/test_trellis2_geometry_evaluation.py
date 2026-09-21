import copy
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from robo.manifest.hash import canonical_hash
from run.icra2027 import e3_trellis2_mesh_probe as g
from run.icra2027.e3_auto_discovery_pilot import identity


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return identity(path)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / 'old-freeze'
    out = root / 'trellis2_initial/38d58a7a31'
    probe = out / 'mesh_probe'
    obs = tmp_path / 'observations.json'
    obs_spec = put(obs, {'records': [dict(job_id=f'38d58a7a31/obj_{i}', observation_sha256='a'*64) for i in range(1000,1015)]})
    config = dict(freeze_id=root.name, output_scene_id='38d58a7a31', observation_source={'manifest':obs_spec})
    config_path = tmp_path / 'original.yaml'
    config_spec = put(config_path, config)
    e0 = dict(code={'commit':'b'*40,'dirty':False},freeze_id=root.name)
    e0['contract_sha256'] = canonical_hash(e0)
    e0_spec = put(root/'contract/freeze_manifest.json',e0)
    manifest = {'source_discovery_hashes': {'input_manifest.json':'c'*64}, 'gaussian_provenance': {'TRAIN':True}}
    manifest_spec = put(out/'input_manifest.json', manifest)
    rows, pools = [], []
    for i in range(1000,1015):
        directory = probe / f'obj_{i}'
        mesh = tmp_path / f'mesh_{i}.ply'
        trimesh.creation.box().export(mesh)
        spec = identity(mesh)
        row = dict(job_id=f'38d58a7a31:auto:{i}',automatic_instance_id=i,proposal_id=f't2-{i}',
                   code_commit='b'*40,evaluation_ground_truth_read=False,status='PROBED',source_status='available',
                   input_hashes={'raw_mesh':spec['sha256'],'visible_observation':'a'*64},
                   runtime={'alignment':{'T':np.eye(4).tolist()}},settle_stable=False)
        put(directory/'record.json',row)
        put(directory/'runtime/registration.json',row['runtime']['alignment'])
        rows.append(row)
        pools.append(dict(job_id=row['job_id'],proposal_id=row['proposal_id'],artifacts={'trellis2_mesh.ply':spec}))
    pool_spec = put(out/'proposal_pool.json',{'rows':pools})
    post_spec = put(out/'postrun_audit.json', {'status':'PASS'})
    input_spec = put(probe/'inputs.json',dict(config=config_spec,new_e0=e0_spec,observation_source=config['observation_source'],
                                            input_manifest=manifest_spec,proposal_pool=pool_spec,postrun_audit=post_spec))
    summary = dict(rows=rows,planned_jobs=15,probed_jobs=15,code_commit='b'*40,freeze_id=root.name,
                   inputs_sha256=input_spec['sha256'],evaluation_ground_truth_read=False)
    summary_spec = put(probe/'summary.json',summary)
    seal_spec = put(probe/'seal.json',{'members':{p.relative_to(probe).as_posix():identity(p) for p in probe.rglob('*') if p.is_file()}})
    audit = dict(status='PASS',source_commit='b'*40,generation_audit={'status':'PASS'},
                 summary_sha256=summary_spec['sha256'],seal_sha256=seal_spec['sha256'])
    audit_spec = put(root/'trellis2_pilot_completion_audit.json',audit)
    source = dict(config=config_spec,e0=e0_spec,probe_seal=seal_spec,completion_audit=audit_spec,code_root=str(tmp_path),code_commit='b'*40)
    monkeypatch.setattr(g,'_geometry_source_process',lambda *args:dict(config=config,generation_audit={'status':'PASS'},input_manifest=manifest))
    monkeypatch.setattr(g,'_validate_runtime',lambda *args:None)  # Existing runtime validator has its own fixture tests.
    return source,probe


def test_full_probe_authentication_preserves15(bundle):
    source,_=bundle
    result=g._validate_probe_for_geometry(source)
    assert len(result['summary']['rows']) == 15


@pytest.mark.parametrize('kind',['mesh_registration','extra_member','missing_member','source_commit','e0','summary','observation'])
def test_probe_tamper_rejected(bundle,kind):
    source,probe=bundle
    if kind=='mesh_registration':
        put(probe/'obj_1001/runtime/registration.json',{'T':np.zeros((4,4)).tolist()})
    elif kind=='extra_member':put(probe/'extra.json',{})
    elif kind=='missing_member':(probe/'obj_1002/record.json').unlink()
    elif kind=='source_commit':source['code_commit']='f'*40
    elif kind=='e0':put(Path(source['e0']['path']),{})
    elif kind=='summary':put(probe/'summary.json',{})
    else:put(Path(source['config']['path']),{})
    with pytest.raises((g.PilotError,KeyError)):g._validate_probe_for_geometry(source)


@pytest.fixture
def geometry(bundle,tmp_path):
    source,_=bundle
    construction=g._validate_probe_for_geometry(source)
    points=trimesh.sample.sample_surface(trimesh.creation.box(),20000,seed=42)[0]
    target=tmp_path/'target.npy';np.save(target,points)
    refs=[]
    for i in range(1000,1015):
        matched=i in (1001,1006,1008)
        refs.append(dict(job_id=f'38d58a7a31/obj_{i}',status='matched' if matched else 'unmatched',
                         matched_gt_id=i if matched else None,evaluation_surface=identity(target) if matched else None))
    return construction,dict(rows=refs),target


def test_canonical_geometry_keeps_unmatched_and_no_appearance(geometry):
    construction,refs,_=geometry
    rows=g._geometry_rows(construction,refs)
    assert len(rows)==15
    assert sum(r['geometry_status']=='matched' for r in rows)==3
    assert all(r['cd_cm'] is None and r['f1_20'] is None and r['collapse'] is None for r in rows if r['geometry_status']=='unmatched')
    assert all(r['psnr'] is None and r['ssim'] is None and r['lpips'] is None for r in rows)
    assert rows[1]['f1_20'] > .9
    assert rows==g._geometry_rows(construction,refs)


def test_registration_gt_separation(geometry):
    construction,refs,target=geometry
    construction['summary']['rows'][1]['input_hashes']['visible_observation']=identity(target)['sha256']
    with pytest.raises(g.PilotError,match='coincide'):g._geometry_rows(construction,refs)


def test_reference_bytes_tamper(geometry):
    construction,refs,target=geometry
    np.save(target,np.zeros((20000,3)))
    with pytest.raises(g.PilotError):g._geometry_rows(construction,refs)


def test_source_is_rejected_before_any_gt_access(tmp_path,monkeypatch):
    config=put(tmp_path/'config.json',dict(schema_version=1,scope=g.GEOMETRY_SCOPE,planned_jobs=15,paper_ready=False,
                                         freeze_id='new',construction_source={},reference_source={}))
    from robo.eval import agentic_ablation
    monkeypatch.setattr(agentic_ablation,'_validate_cli_execution',lambda *a,**k:{'code':{'commit':'b'*40}})
    def bad(*args):raise g.PilotError('construction rejected')
    monkeypatch.setattr(g,'_validate_probe_for_geometry',bad)
    monkeypatch.setattr(g,'_validate_geometry_references',lambda *a:pytest.fail('GT opened before construction validated'))
    with pytest.raises(g.PilotError,match='construction rejected'):
        g.evaluate_geometry(config['path'],tmp_path/'new/contract/freeze_manifest.json',tmp_path/'new/trellis2_geometry')


def test_original_source_identity_failclosed(tmp_path):
    import subprocess
    subprocess.run(['git','init',str(tmp_path)],check=True,capture_output=True)
    with pytest.raises(subprocess.CalledProcessError):
        g._geometry_source_process(dict(code_root=str(tmp_path),code_commit='b'*40),'raise Exception("must not run")',[])


@pytest.fixture
def reference_case(geometry,tmp_path,monkeypatch):
    construction,refs,_=geometry
    construction['input_manifest']['source_gaussian_training_provenance']=g.FRESH
    payload=dict(refs,code_commit='d'*40,planned_jobs=15,planned_scenes=1,matched_jobs=3,
                 scenes=[dict(discovery_binding=copy.deepcopy(construction['input_manifest']))])
    path=tmp_path/'eval/evaluation_matching/evaluation_references.json'
    spec=put(path,payload)
    checks={key:'PASS' for key in ('construction_train_only_discovery_binding','control_frozen_before_evaluation',
         'independent_GT_input_hashes','matching_and_surface_replay_exact','matching_manifest_seal_E0_source_config_binding',
         'unmatched_geometry_null','nonheadline_gates')}
    audit=put(path.parent.parent/'independent_evaluation_completion_audit.json',
              dict(paper_ready=False,headline_eligible=False,claim_gate='NOT_RUN',checks=checks,
                   evidence_hashes={str(path):{'sha256':spec['sha256']}}))
    source=dict(manifest=spec,completion_audit=audit,code_commit='d'*40)
    monkeypatch.setattr(g,'_geometry_source_process',lambda *args:payload)
    return source,construction,payload


def test_reference_same_discovery_and_fixed15(reference_case):
    source,construction,payload=reference_case
    assert g._validate_geometry_references(source,construction)==payload


@pytest.mark.parametrize('kind',['discovery','empty_binding','population','matched','missing_rows','duplicate','code','claim','missing_check'])
def test_reference_closure_negatives(reference_case,kind):
    source,construction,payload=reference_case
    if kind=='discovery':payload['scenes'][0]['discovery_binding']['source_discovery_hashes']={}
    elif kind=='empty_binding':payload['scenes'][0]['discovery_binding']={}
    elif kind=='population':payload['planned_jobs']=14
    elif kind=='matched':payload['matched_jobs']=4
    elif kind=='missing_rows':payload['rows'].pop()
    elif kind=='duplicate':payload['rows'][-1]=payload['rows'][0]
    elif kind=='code':payload['code_commit']='e'*40
    else:
        path=Path(source['completion_audit']['path']);audit=json.loads(path.read_text())
        if kind=='claim':audit['paper_ready']=True
        else:audit['checks'].pop('control_frozen_before_evaluation')
        source['completion_audit']=put(path,audit)
    with pytest.raises(g.PilotError):g._validate_geometry_references(source,construction)


def test_publishes_canonical15_geometry_and_csv_no_overwrite(geometry,tmp_path,monkeypatch):
    from robo.eval import agentic_ablation
    construction,refs,_=geometry
    source=tmp_path/'new'
    config=put(tmp_path/'config.json',dict(schema_version=1,scope=g.GEOMETRY_SCOPE,planned_jobs=15,paper_ready=False,
                                         freeze_id='new',construction_source={},reference_source={}))
    monkeypatch.setattr(agentic_ablation,'_validate_cli_execution',lambda *a,**k:{'code':{'commit':'b'*40}})
    monkeypatch.setattr(g,'_validate_probe_for_geometry',lambda *a:construction)
    monkeypatch.setattr(g,'_validate_geometry_references',lambda *a:refs)
    result=g.evaluate_geometry(config['path'],source/'contract/freeze_manifest.json',source/'trellis2_geometry')
    assert result['planned_jobs']==15 and result['geometry_evaluated_jobs']==3 and result['unmatched_jobs']==12
    assert result['native_gaussian'] is False and result['paper_ready'] is False
    assert len((source/'trellis2_geometry/geometry.csv').read_text().splitlines())==16
    with pytest.raises(FileExistsError):
        g.evaluate_geometry(config['path'],source/'contract/freeze_manifest.json',source/'trellis2_geometry')


def test_shared_evidence_publication_and_seal(geometry,tmp_path,monkeypatch):
    from robo.eval import agentic_ablation as e3, fidelity_metrics as fidelity
    construction,refs,_=geometry
    shared=tmp_path/'shared';shared.mkdir()
    source=shared/'new'
    config=put(tmp_path/'config.json',dict(schema_version=1,scope=g.GEOMETRY_SCOPE,planned_jobs=15,paper_ready=False,
                                         freeze_id='new',construction_source={},reference_source={}))
    # Production e3 validates this same-Git shared root at import. Prove output
    # uses its authenticated root even though legacy fidelity remains local.
    monkeypatch.setattr(e3,'REPOSITORY_ROOT',shared)
    monkeypatch.setattr(fidelity,'REPOSITORY_ROOT',tmp_path/'code')
    monkeypatch.setattr(e3,'_validate_cli_execution',lambda *a,**k:{'code':{'commit':'b'*40}})
    monkeypatch.setattr(g,'_validate_probe_for_geometry',lambda *a:construction)
    monkeypatch.setattr(g,'_validate_geometry_references',lambda *a:refs)
    g.evaluate_geometry(config['path'],source/'contract/freeze_manifest.json',source/'trellis2_geometry')
    seal=e3._verify_sealed_directory(source/'trellis2_geometry')
    assert set(seal['members'])=={'geometry.json','geometry.csv'}
    (source/'trellis2_geometry/geometry.csv').write_text('tampered')
    with pytest.raises(ValueError):e3._verify_sealed_directory(source/'trellis2_geometry')


def test_shared_publisher_rejects_escape_and_symlink(tmp_path,monkeypatch):
    from robo.eval import agentic_ablation as e3
    root=tmp_path/'shared';root.mkdir()
    monkeypatch.setattr(e3,'REPOSITORY_ROOT',root)
    with pytest.raises(ValueError):
        with e3._atomic_directory(tmp_path/'outside'):pass
    (root/'link').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError):
        with e3._atomic_directory(root/'link/out'):pass
