import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from robo.eval import agentic_ablation as e3
from robo.eval import e3_factory_materializer as materializer
from tests.test_agentic_missing_observations import inventory
from tests.test_e3_factory_materializer import _write_json, _reseal_inventory


@pytest.fixture
def automatic_factory_input(tmp_path,monkeypatch):
    from agents.core import common
    from plyfile import PlyData,PlyElement
    def forbidden(*a,**kw):raise AssertionError('GT/implicit instance loader must not run')
    monkeypatch.setattr(common,'load_gt_instances',forbidden)
    monkeypatch.setattr(common,'load_instances',forbidden)
    monkeypatch.setattr(e3,'REPOSITORY_ROOT',tmp_path)
    monkeypatch.setattr(materializer,'REPOSITORY_ROOT',tmp_path)
    monkeypatch.setattr(materializer,'_require_clean_code_snapshot',lambda:{'commit':'d'*40,'dirty':False,'status':[], 'code_root':str(materializer.CODE_ROOT)})
    source=tmp_path/'discovery';construction=source/'construction';construction.mkdir(parents=True)
    verts=np.array([(0.,0.,0.),(.1,0.,0.),(0.,.1,0.),(0.,0.,.1)],dtype=[('x','f4'),('y','f4'),('z','f4')])
    faces=np.array([([0,1,2],),([0,1,3],),([0,2,3],),([1,2,3],)],dtype=[('vertex_indices','i4',(3,))])
    PlyData([PlyElement.describe(verts,'vertex'),PlyElement.describe(faces,'face')],text=True).write(construction/'derived_mesh.ply')
    np.savez(construction/'auto_instances.npz',labels=np.array(['mouse']),scores=np.array([.8]),vert_idx_0=np.arange(4))
    _write_json(construction/'objects/objects.json',[])
    ids={name:{'path':str(construction/name),'bytes':(construction/name).stat().st_size,'sha256':e3.sha256_file(construction/name)} for name in ['derived_mesh.ply','auto_instances.npz','objects/objects.json']}
    _write_json(source/'output_hashes.json',ids)
    jobs,proposals=inventory();scene=jobs['scenes'][0]
    origin={**scene['source_scene_gaussian']};origin['bytes']=origin.pop('size_bytes')
    _write_json(source/'input_manifest.json',{'gaussian':origin,'metadata':{'nerfstudio/transforms_undistorted.json':origin,'colmap/images.txt':origin}})
    _write_json(source/'pilot_summary.json',{'stages':[{'exit_code':0}], 'rows':[{'automatic_instance_id':1000,'prepared':False}]})
    _write_json(source/'postrun_audit.json',{'summary_sha256':e3.sha256_file(source/'pilot_summary.json'),'output_hashes_sha256':e3.sha256_file(source/'output_hashes.json')})
    hashes={name:e3.sha256_file(source/name) for name in ['pilot_summary.json','input_manifest.json','output_hashes.json','postrun_audit.json']}
    audit={'paper_ready':False,'source_discovery':hashes,'source_gaussian_training_provenance':'UNKNOWN',
           'jobs':[{'job_id':'09c1414f1b/obj_1000','source_job_id':'09c1414f1b:auto:1000','automatic_instance_id':1000,'prepared_output_index':None}]}
    out=tmp_path/'agentic';inv=out/'input_inventory';inv.mkdir(parents=True)
    for name,value in [('resolved_jobs.json',jobs),('resolved_proposals.json',proposals),('inventory_audit.json',audit)]:_write_json(inv/name,value)
    (inv/'artifact_hashes.sha256').write_text('')
    _reseal_inventory(out)
    e3.run_observe('freeze',out,'09c1414f1b')
    policies=yaml.safe_load((materializer.CODE_ROOT/'configs/experiments/icra2027/agentic_policies.yaml').read_text())
    e3.run_control(policies,{'code':{'commit':'d'*40}},'freeze',out,'09c1414f1b')
    descriptor=tmp_path/'automatic_scene.json'
    _write_json(descriptor,{'schema_version':1,'scene_id':'09c1414f1b','discovery_directory':str(source),'discovery_hashes':hashes})
    return tmp_path,out,descriptor,jobs,audit


@pytest.mark.parametrize('policy,action',[('A0','reject'),('A4','abstain')])
def test_automatic_materializer_keeps_missing_jobs_without_gt_or_aggregate(automatic_factory_input,policy,action):
    root,out,descriptor,jobs,audit=automatic_factory_input
    destination=root/policy
    manifest=materializer.materialize_factory_variant(e3_root=out,scene_id='09c1414f1b',policy_id=policy,out=destination,automatic_scene_contract=descriptor)
    assert manifest['roster']['job_count']==1 and manifest['roster']['accepted_count']==0
    assert manifest['selected_records'][0]['terminal_action']==action
    assert manifest['selected_records'][0]['object_slot']=='obj_1000'
    assert not list(destination.rglob('*.urdf'))
    assert not (out/'aggregate_seal.json').exists()
    objects=json.loads((destination/'objects/objects.json').read_text())
    assert objects[0]['automatic_instance_id']==1000 and 'gt_object_id' not in objects[0]
    aligned=json.loads((destination/'objects/obj_1000/aligned.json').read_text())
    assert aligned['tier'] is None and aligned['construction_eligible'] is False
    report=materializer.validate_materialized_factory(destination,expected_scene_id='09c1414f1b',expected_policy_id=policy)
    assert report['paper_ready'] is False
    with pytest.raises(FileExistsError):materializer.materialize_factory_variant(e3_root=out,scene_id='09c1414f1b',policy_id=policy,out=destination,automatic_scene_contract=descriptor)


@pytest.mark.parametrize('mutation',['scene','mapping','mesh','segmentation','missing_mesh','source_hash'])
def test_automatic_source_rejects_identity_or_geometry_drift(automatic_factory_input,mutation):
    root,out,descriptor,jobs,audit=automatic_factory_input
    d=json.loads(descriptor.read_text());construction=root/'discovery/construction'
    if mutation=='scene':d['scene_id']='0123456789';_write_json(descriptor,d)
    if mutation=='mapping':audit['jobs'][0]['automatic_instance_id']=999
    if mutation=='mesh':(construction/'derived_mesh.ply').write_bytes(b'changed')
    if mutation=='segmentation':np.savez(construction/'auto_instances.npz',labels=np.array(['mouse']),scores=np.array([.8]),vert_idx_0=np.array([999]))
    if mutation=='missing_mesh':(construction/'derived_mesh.ply').unlink()
    if mutation=='source_hash':d['discovery_hashes']['input_manifest.json']='0'*64;_write_json(descriptor,d)
    with pytest.raises((ValueError,FileNotFoundError)):
        materializer._automatic_source_context(jobs,audit,'09c1414f1b',descriptor)


def test_static_export_rejects_unset_auto_or_gt_mesh_before_loading(automatic_factory_input,monkeypatch):
    from robo.sim.export_mjcf import validate_automatic_export_context
    root,out,descriptor,*_=automatic_factory_input;destination=root/'A0'
    materializer.materialize_factory_variant(e3_root=out,scene_id='09c1414f1b',policy_id='A0',out=destination,automatic_scene_contract=descriptor)
    for auto,mesh in [(None,destination/'derived_mesh.ply'),('1',root/'scans/mesh_aligned_0.05.ply')]:
        with pytest.raises(ValueError,match='AUTO=1'):
            validate_automatic_export_context(destination,auto_mode=auto,pipeline_mesh=mesh)
    assert validate_automatic_export_context(destination,auto_mode='1',pipeline_mesh=destination/'derived_mesh.ply')['paper_ready'] is False


def test_background_automatic_namespace_guard_runs_before_mesh_read(monkeypatch):
    from agents.core import common
    from robo.sim.s7_sim import build_background
    monkeypatch.setattr(common,'env',lambda *a:None)
    with pytest.raises(ValueError,match='automatic namespace'):
        build_background([{'index':1000,'automatic_instance_id':1000,'instance_namespace':'automatic'}],[{}],{})


def test_actual_automatic_background_retains_failed_jobs_and_never_loads_gt(automatic_factory_input,monkeypatch):
    from robo.sim import export_mjcf,s7_sim
    from agents.core import common
    root,out,descriptor,*_=automatic_factory_input
    factories=[]
    for policy in ('A0','A4'):
        path=root/policy
        materializer.materialize_factory_variant(e3_root=out,scene_id='09c1414f1b',policy_id=policy,out=path,automatic_scene_contract=descriptor)
        factories.append(path)
    monkeypatch.setattr(common,'SCENE_ID','09c1414f1b')
    monkeypatch.setattr(common,'OUT',factories[0])
    monkeypatch.setattr(common,'PIPELINE_MESH_PLY',factories[0]/'derived_mesh.ply')
    monkeypatch.setattr(common,'env',lambda name,*a:'1' if name=='AUTO' else None)
    objects,hulls,sources=export_mjcf.load_common_carve_inputs(factories)
    assert len(objects)==1 and hulls==[] and sources['policy_ids']==['A0','A4']
    path,report=s7_sim.build_background(objects,[{}],{},carve_hulls=hulls,
        paired_policy_ids=sources['policy_ids'],return_report=True)
    assert report['kept_face_count']==4 and report['source_face_count']==4
    assert report['discovered_slots']==['obj_1000'] and report['carved_slots']==[]
    assert report['mode']=='paired_policy_union' and path.is_file()


def test_mesh_cache_switches_source_and_same_path_changed_bytes(tmp_path,monkeypatch):
    from agents.assets import s5_align
    from agents.core import common
    from plyfile import PlyData,PlyElement
    def mesh(path,offset):
        verts=np.array([(offset,0.,0.),(offset+.1,0.,0.),(offset,.1,0.)],dtype=[('x','f4'),('y','f4'),('z','f4')])
        faces=np.array([([0,1,2],)],dtype=[('vertex_indices','i4',(3,))])
        PlyData([PlyElement.describe(verts,'vertex'),PlyElement.describe(faces,'face')],text=True).write(path)
    first=tmp_path/'synthetic_gt.ply';second=tmp_path/'derived_mesh.ply'
    mesh(first,0);mesh(second,10)
    monkeypatch.setattr(s5_align,'_MESH_CACHE',{})
    monkeypatch.setattr(common,'PIPELINE_MESH_PLY',first)
    assert s5_align.scene_mesh_arrays()[0][0,0]==0
    monkeypatch.setattr(common,'PIPELINE_MESH_PLY',second)
    assert s5_align.scene_mesh_arrays()[0][0,0]==10
    mesh(second,20)
    assert s5_align.scene_mesh_arrays()[0][0,0]==20
    previous_identity = s5_align._MESH_CACHE['source_identity']
    monkeypatch.setattr(common, 'SCENE_ID', 'another_scene')
    assert s5_align.scene_mesh_arrays()[0][0,0]==20
    assert s5_align._MESH_CACHE['source_identity'] != previous_identity


def test_automatic_eligibility_is_explicit_and_not_a_copied_quality_tier():
    from robo.tasks.pi05_tasks import _is_graspable
    from robo.eval.e4_candidate_screen import _eligibility_reasons
    row={'label':'mouse','tier':None,'dims':np.array([.1,.1,.1]),'mass':.3,'drift':0.,'construction_eligible':True}
    assert _is_graspable(row) and _eligibility_reasons(row)==[]
    row['construction_eligible']=False
    assert not _is_graspable(row) and 'tier_not_A_or_B' in _eligibility_reasons(row)


def test_accepted_automatic_asset_uses_sealed_proposal_not_gt_factory_path(tmp_path, monkeypatch):
    from tests.test_e3_factory_materializer import _accepted_record, _proposal, SCENE, FREEZE
    monkeypatch.setattr(e3, 'REPOSITORY_ROOT', tmp_path)
    monkeypatch.setattr(materializer, 'REPOSITORY_ROOT', tmp_path)
    out = tmp_path/'agentic'
    record = _accepted_record(tmp_path, out, tmp_path/'automatic_generation', 'A4', 1000)
    # Generation files use prepared indices, while controller/export IDs stay 1000.
    asset = record['selected_asset']
    for role in ('raw_mesh', 'raw_gaussian'):
        old = tmp_path/asset['artifact_paths'][role]
        new = tmp_path/'automatic_generation/prepared_00'/old.name
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
        asset['artifact_paths'][role] = str(new.relative_to(tmp_path))
    slot = 'obj_1000'
    job = {'job_id': f'{SCENE}/{slot}', 'object_slot': slot}
    proposal = _proposal(job, 'trellis', selected=record)
    selected_path = out/'control'/SCENE/'selected_assets/A4'/f'{slot}.json'
    _write_json(selected_path, record)
    source = {'automatic': True, 'selected_scene_root':str(selected_path.parent.parent),
              'jobs':[job], 'object_slots':[slot],
              'object_rows':[{'index':1000,'automatic_instance_id':1000,'label':'mouse'}]}
    args = dict(e3_root=out,scene_id=SCENE,policy_id='A4',freeze_id=FREEZE,
                proposals={'proposals':[proposal]},source=source,
                aggregate={'selected_records':{f'A4/{SCENE}/{slot}.json':materializer._identity(selected_path)}})
    prepared = materializer._prepare_selected_records(**args)
    assert prepared[0]['action']=='accept'
    assert prepared[0]['object_row']['index']==1000
    assert 'prepared_00' in prepared[0]['artifacts']['raw_mesh']['path_display']
    # Same bytes at a different path are not an authorized source-job binding.
    proposal['artifact_paths']['raw_mesh'] = 'other_scene/trellis_mesh.ply'
    with pytest.raises(ValueError,match='identity differs from resolved proposal'):
        materializer._prepare_selected_records(**args)


def test_actual_union_carve_removes_only_replaced_automatic_instance(tmp_path, monkeypatch):
    from agents.core import common
    from plyfile import PlyData, PlyElement
    from robo.sim.s7_sim import build_background
    vertices = np.array([[0,0,0],[.1,0,0],[0,.1,0],[0,0,.1]], dtype=float)
    all_vertices = np.vstack([vertices, vertices+[1,0,0]])
    faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
    all_faces = np.vstack([faces, faces+4])
    ply_vertices = np.array([tuple(v) for v in all_vertices], dtype=[('x','f4'),('y','f4'),('z','f4')])
    ply_faces = np.array([(face,) for face in all_faces], dtype=[('vertex_indices','i4',(3,))])
    mesh = tmp_path/'derived_mesh.ply'
    PlyData([PlyElement.describe(ply_vertices,'vertex'),PlyElement.describe(ply_faces,'face')],text=True).write(mesh)
    monkeypatch.setattr(common,'OUT',tmp_path)
    monkeypatch.setattr(common,'PIPELINE_MESH_PLY',mesh)
    monkeypatch.setattr(common,'env',lambda name,*a:'1' if name=='AUTO' else None)
    objects = [{'index':1000+i,'automatic_instance_id':1000+i,'instance_namespace':'automatic',
                'aabb':[all_vertices[i*4:(i+1)*4].min(0).tolist(),all_vertices[i*4:(i+1)*4].max(0).tolist()]}
               for i in range(2)]
    hulls = [{'name':'A4/obj_1000/part_00.obj','policy_id':'A4','slot':'obj_1000',
              'vertices':vertices,'support_clip_z_m':.005}]
    _, report = build_background(objects,[{},{}],
        {1000:{'vert_idx':np.arange(4)},1001:{'vert_idx':np.arange(4,8)}},
        carve_hulls=hulls,paired_policy_ids=['A0','A4'],return_report=True)
    assert report['source_face_count']==8 and report['kept_face_count']==4
    assert report['discovered_slots']==['obj_1000','obj_1001']
    assert report['carved_slots']==['obj_1000']
    assert report['policy_ids']==['A0','A4']


@pytest.fixture
def fresh_factory_input(automatic_factory_input):
    from run.icra2027.e3_auto_discovery_pilot import summarize_instances
    root,out,descriptor,jobs,audit=automatic_factory_input
    source=root/'discovery'
    manifest=json.loads((source/'input_manifest.json').read_text())
    manifest.update(scene_id='09c1414f1b',source_gaussian_training_provenance='FRESH_OFFICIAL_TRAIN_ONLY',
                    planned_stages=['render','fuse','discover','prepare','refine'],boundary={'training_frames':['train.jpg']})
    _write_json(source/'input_manifest.json',manifest)
    rows=summarize_instances(['mouse'],[])
    summary=dict(scene_id='09c1414f1b',paper_ready=False,freeze_id='fresh-discovery',code_commit='a'*40,
                 stages=[{'stage':s,'exit_code':0} for s in manifest['planned_stages'][:-1]]+
                        [{'stage':'refine','status':'NOT_RUN','reason':'empty_prepared_population'}],
                 rows=rows,discovered_instances=1,prepared_instances=0,
                 source_gaussian_training_provenance='FRESH_OFFICIAL_TRAIN_ONLY')
    _write_json(source/'pilot_summary.json',summary)
    all_jobs=dict(kind='complete_automatic_discovery_jobs',schema_version=1,paper_ready=False,
                  rows=rows,planned_jobs=1,scene_id='09c1414f1b',freeze_id='fresh-discovery',code_commit='a'*40,
                  source_gaussian_training_provenance='FRESH_OFFICIAL_TRAIN_ONLY',
                  input_manifest_sha256=e3.sha256_file(source/'input_manifest.json'),
                  summary_sha256=e3.sha256_file(source/'pilot_summary.json'),
                  output_hashes_sha256=e3.sha256_file(source/'output_hashes.json'))
    _write_json(source/'all_jobs_manifest.json',all_jobs)
    _write_json(source/'postrun_audit.json',dict(summary_sha256=all_jobs['summary_sha256'],
        output_hashes_sha256=all_jobs['output_hashes_sha256'],all_jobs_manifest_sha256=e3.sha256_file(source/'all_jobs_manifest.json')))
    names=['pilot_summary.json','input_manifest.json','output_hashes.json','postrun_audit.json','all_jobs_manifest.json']
    hashes={name:e3.sha256_file(source/name) for name in names}
    audit.update(source_discovery=hashes,source_gaussian_training_provenance='FRESH_OFFICIAL_TRAIN_ONLY')
    _write_json(out/'input_inventory/inventory_audit.json',audit);_reseal_inventory(out)
    d=json.loads(descriptor.read_text());d['discovery_hashes']=hashes;_write_json(descriptor,d)
    return root,out,descriptor,jobs,audit


def test_fresh_materialization_reuses_complete_source_validator(fresh_factory_input):
    root,out,descriptor,*_=fresh_factory_input
    result=materializer.materialize_factory_variant(e3_root=out,scene_id='09c1414f1b',policy_id='A4',
        out=root/'fresh-A4',automatic_scene_contract=descriptor)
    assert result['roster']['abstained_count']==1
    assert result['source_scene']['source_gaussian_training_provenance']=='FRESH_OFFICIAL_TRAIN_ONLY'
    assert result['paper_ready'] is False
    materializer.validate_materialized_factory(root/'fresh-A4')


@pytest.mark.parametrize('mutation',['missing_all_jobs','swapped_all_jobs','missing_completion_anchor','source_proof'])
def test_fresh_source_rejects_missing_or_swapped_population_proof(fresh_factory_input,mutation):
    root,out,descriptor,jobs,audit=fresh_factory_input;source=root/'discovery'
    d=json.loads(descriptor.read_text())
    if mutation=='missing_all_jobs':d['discovery_hashes'].pop('all_jobs_manifest.json')
    elif mutation=='swapped_all_jobs':
        value=json.loads((source/'all_jobs_manifest.json').read_text());value['scene_id']='0123456789'
        _write_json(source/'all_jobs_manifest.json',value)
        d['discovery_hashes']['all_jobs_manifest.json']=e3.sha256_file(source/'all_jobs_manifest.json')
    elif mutation=='missing_completion_anchor':
        value=json.loads((source/'postrun_audit.json').read_text());value.pop('all_jobs_manifest_sha256')
        _write_json(source/'postrun_audit.json',value)
        d['discovery_hashes']['postrun_audit.json']=e3.sha256_file(source/'postrun_audit.json')
    else:audit['source_gaussian_training_provenance']='UNKNOWN'
    audit['source_discovery']=d['discovery_hashes'];_write_json(descriptor,d)
    with pytest.raises((ValueError,RuntimeError)):
        materializer._automatic_source_context(jobs,audit,'09c1414f1b',descriptor)


@pytest.mark.parametrize('mutation',['selected_outer_only','selected_and_ledger_outer_only','observation_source','selected_ledger_disagreement'])
def test_automatic_rejects_resealed_disconnected_control_members(automatic_factory_input,mutation):
    root,out,descriptor,*_=automatic_factory_input
    control=out/'control/09c1414f1b';seal=json.loads((control/'seal.json').read_text())
    shard=json.loads((control/'controller_shard.json').read_text())
    relative='selected_assets/A4/obj_1000.json';selected_path=control/relative
    if mutation in ('selected_outer_only','selected_and_ledger_outer_only'):
        selected=json.loads(selected_path.read_text());selected['reason_codes']=['tampered_reason']
        _write_json(selected_path,selected)
        seal['selected_asset_hashes'][relative]=e3.sha256_file(selected_path)
        seal['selected_asset_sizes'][relative]=selected_path.stat().st_size
    if mutation in ('selected_and_ledger_outer_only','selected_ledger_disagreement'):
        path=control/'job_ledger.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]['reason_codes']=['tampered_reason']
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        seal['members']['job_ledger.jsonl']=e3.sha256_file(path)
        if mutation=='selected_ledger_disagreement':shard['ledger_sha256']=e3.sha256_file(path)
    if mutation=='observation_source':
        directory=out/'observations/09c1414f1b';path=directory/'manifest.json'
        observed=json.loads(path.read_text());observed['code_commit']='b'*40;_write_json(path,observed)
        observed_seal=json.loads((directory/'seal.json').read_text())
        for key in ('members','controller_members'):observed_seal[key]['manifest.json']=e3.sha256_file(path)
        _write_json(directory/'seal.json',observed_seal)
        shard['observation_manifest_sha256']=e3.sha256_file(path)
    _write_json(control/'controller_shard.json',shard)
    seal['members']['controller_shard.json']=e3.sha256_file(control/'controller_shard.json')
    _write_json(control/'seal.json',seal)
    with pytest.raises(ValueError,match='(shard differs|observation source identity|selected decision differs)'):
        materializer.materialize_factory_variant(e3_root=out,scene_id='09c1414f1b',policy_id='A4',
            out=root/'bad',automatic_scene_contract=descriptor)
    assert not (root/'bad').exists()


def test_direct_static_producer_rejects_wrong_active_mesh_before_instance_read(automatic_factory_input,monkeypatch):
    from agents.core import common
    from robo.sim import export_mjcf
    root,out,descriptor,*_=automatic_factory_input
    factories=[]
    for policy in ('A0','A4'):
        path=root/policy
        materializer.materialize_factory_variant(e3_root=out,scene_id='09c1414f1b',policy_id=policy,
            out=path,automatic_scene_contract=descriptor)
        factories.append(path)
    monkeypatch.setattr(common,'SCENE_ID','09c1414f1b')
    monkeypatch.setattr(common,'OUT',factories[0])
    monkeypatch.setattr(common,'PIPELINE_MESH_PLY',root/'scans/mesh_aligned_0.05.ply')
    monkeypatch.setattr(common,'env',lambda name,*a:'1' if name=='AUTO' else None)
    # Fixture's implicit/GT loader raises AssertionError if reached.
    with pytest.raises(ValueError,match='exact derived mesh'):
        export_mjcf._room_static_report(factories,None,root/'collision')
