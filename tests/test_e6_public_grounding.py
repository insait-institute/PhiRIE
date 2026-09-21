import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from robo.certification import public_grounding as g
from run.icra2027 import e6_public_grounding as driver
from agents.discover import derive_mesh_from_splat as derive


def query(i='q01'):
    return dict(scene_id='scene',task_id=i,task_family='place_in_region',query_sha256='a'*64,source_state_anchor='rgb0',
                roles={'manipulated_object':dict(description='red bottle',bbox_xyxy_normalized=[.2,.2,.8,.8]),
                       'target':dict(description='left target patch',bbox_xyxy_normalized=[.2,.2,.8,.8],
                                     polygon_xy_normalized=[[.2,.2],[.8,.2],[.8,.8],[.2,.8]]),
                       'robot_visual_reference':dict(description='robot')})


def poses(points):
    out={}
    for i,p in enumerate(points):
        M=np.eye(4);M[:3,3]=-np.array(p);out[str(i)]=M
    return out


def test_present_association_preserves_ambiguity_and_object_identity():
    a=dict(id='observed_a',label='bottle',projection=dict(bbox_normalized=[.2,.2,.8,.8]))
    result=g.associate_role(query()['roles']['manipulated_object'],[a])
    assert result['status']=='resolved' and result['selected_object_id']=='observed_a'
    b=copy.deepcopy(a);b['id']='observed_b'
    result=g.associate_role(query()['roles']['manipulated_object'],[a,b])
    assert result['status']=='ambiguous' and result['selected_object_id'] is None
    b['label']='chair'
    assert g.associate_role(query()['roles']['manipulated_object'],[b])['status']=='unresolved'


def test_missing_anchor_keeps_72_unresolved_without_surface_callback():
    rows=[]
    for scene in range(6):
        for condition in ('clean','mild','severe'):
            for q in range(4):
                item=query(f'q{q}');item['scene_id']=str(scene)
                row=g.ground_query(item,source_present=False,projected_instances=[],surface=lambda a:pytest.fail('missing view cannot read depth'))
                row['condition']=condition;rows.append(row)
    assert len(rows)==72
    assert len({(r['scene_id'],r['condition'],r['task_id']) for r in rows})==72
    assert all(r['robot_frame'] is None and r['physics_verified'] is False for r in rows)
    assert all(r['roles']['manipulated_object']['reason']=='query_source_view_missing' for r in rows)


def test_distinct_region_queries_never_collapse_or_create_physical_volume():
    depth=np.ones((40,40));alpha=np.ones((40,40));K=np.diag([20.,20.,1.])
    patch=lambda ann:g.region_surface(ann,depth,alpha,K,np.eye(4))
    a=g.ground_query(query('q1'),source_present=True,projected_instances=[],surface=patch)
    b=g.ground_query(query('q2'),source_present=True,projected_instances=[],surface=patch)
    assert a['roles']['target']['region_id']!=b['roles']['target']['region_id']
    assert a['roles']['target']['metric_surface']['physical_volume_known'] is False
    assert g.region_surface(query()['roles']['target'],depth,np.zeros_like(alpha),K,np.eye(4)) is None


def test_projected_instances_require_predicted_depth_visibility():
    x,y=np.meshgrid(np.arange(2,9),np.arange(2,9));v=np.stack([x.ravel(),y.ravel(),np.ones(x.size)],axis=1)
    K=np.eye(3);depth=np.ones((12,12));alpha=np.ones((12,12))
    assert g.projected_instance(v,np.arange(len(v)),K,np.eye(4),depth,alpha)['visible_vertices']==len(v)
    assert g.projected_instance(v,np.arange(len(v)),K,np.eye(4),depth*2,alpha) is None
    with pytest.raises(ValueError,match='identity'):g.projected_instance(v,np.array([999]),K,np.eye(4),depth,alpha)


def test_se3_alignment_preserves_scale_error_and_rank_failure():
    ref=poses([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    angle=.3;T=np.eye(4);T[:3,:3]=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]];T[:3,3]=[.2,.3,.4]
    moving={k:v@T for k,v in ref.items()}
    result=g.rigid_camera_alignment(ref,moving)
    assert result['status']=='aligned_public_frame'
    assert np.allclose(result['transform'],T) and result['scale_applied']==1.
    scaled=poses([[0,0,0],[2,0,0],[0,2,0],[2,2,0]])
    result=g.rigid_camera_alignment(ref,scaled)
    assert result['status']=='unresolved' and result['reason']=='public_camera_rigid_alignment_residual'
    assert result['baseline_scale_ratio']==.5 and result['scale_applied']==1. and result['transform'] is None
    line=poses([[0,0,0],[1,0,0],[2,0,0]])
    assert g.rigid_camera_alignment(line,line)['reason']=='degenerate_public_camera_baseline'
    assert g.rigid_camera_alignment({},ref)['reason']=='insufficient_common_public_cameras'


def test_only_explicit_predicted_inputs_and_existing_modules(tmp_path):
    runtime=dict(render_python='gs',sam3_python='sam');rgb=dict(python='cpu')
    commands=driver.commands(runtime,rgb,tmp_path/'source',tmp_path/'out')
    assert [x[0] for x in commands]==['render','fuse','discover']
    assert commands[0][3][commands[0][3].index('--frame-stride')+1]=='1'
    assert '--splat-ply' in commands[0][3] and '--scene-dir' in commands[1][3]
    assert commands[2][3][-2:]==['--mesh-path',str(tmp_path/'out/derived_mesh.ply')]
    assert not any('s3_lift' in str(x) for x in commands)


def test_read_boundary_forbids_legacy_and_vault_without_opening_them(tmp_path):
    for path in ['/data/ScanNetpp/a/scans/mesh.ply',str(driver.shared.ROOT/'data/recon_scenes/data/a/depth.npz'),str(tmp_path/'vault/labels.json')]:
        with pytest.raises(PermissionError):driver.enforce_public_read('open',(path,'r'))
    driver.enforce_public_read('open',(str(tmp_path/'approved/recon_metric.npz'),'r'))
    driver.enforce_public_read('open',(4,'r'))


def test_sparse_render_all_selected_views_and_explicit_cameras(monkeypatch,tmp_path):
    monkeypatch.setattr(derive.C,'OUT',tmp_path)
    names=['a.jpg','b.jpg','c.jpg'];poses_={n:np.eye(4) for n in names}
    monkeypatch.setattr(derive.C,'load_intrinsics',lambda path:(np.eye(3),4,4,{}))
    monkeypatch.setattr(derive.C,'load_colmap_w2c',lambda path:poses_)
    monkeypatch.setattr(derive.C,'load_gaussians',lambda path:('explicit',path))
    monkeypatch.setattr(derive.C,'load_gt_instances',lambda:pytest.fail('GT loader forbidden'))
    monkeypatch.setattr(derive.C,'render_view',lambda *a,**kw:(np.ones((2,2,3)),np.ones((2,2)),np.ones((2,2))))
    args=SimpleNamespace(frame_stride=1,scene_dir=str(tmp_path/'public_scene'),splat_ply='public_scene.ply')
    derive.seg_render(args)
    files=sorted((tmp_path/'mesh_derive').glob('view_*.npz'))
    assert len(files)==3
    assert [str(np.load(p)['fname']) for p in files]==names
    args.frame_stride=0
    with pytest.raises(ValueError,match='positive'):derive.seg_render(args)


def test_virtual_robot_declaration_never_imputes_mass_or_physics(monkeypatch):
    from robo.tasks import pi05_tasks as planner
    objects=[dict(id='public_bottle',label='bottle',aabb=[[0.,0.,.7],[.05,.05,.9]])]
    result=g.declare_virtual_robot(objects)
    assert result['real_calibration_claimed'] is False and result['physical_stability_known'] is False
    assert result['status']=='declared_virtual_frame'
    monkeypatch.setattr(planner,'_place_robot',lambda *a,**k:dict(base_pos=[.02,.02,.7],base_yaw=0.))
    result=g.declare_virtual_robot(objects)
    assert result['status']=='unresolved' and result['base_pos'] is None
    assert result['reason']=='existing_planner_returned_unclear_fallback'
    assert g.declare_virtual_robot([])['base_pos'] is None


def test_reconstruction_failure_is_not_mislabeled_missing_input_view():
    result=g.ground_query(query(),source_present=False,projected_instances=[],unavailable_reason='upstream_reconstruction_rejected')
    assert result['roles']['target']['reason']=='upstream_reconstruction_rejected'


def admission(monkeypatch,tmp_path):
    from run.icra2027 import e3_auto_discovery_pilot as d
    s=driver.shared;root=tmp_path;stage=root/'outputs/icra2027/20260906-abcdef0-v1';stage.mkdir(parents=True)
    checkpoint=root/'sam3.pt';checkpoint.write_bytes(b'weights')
    runtime={k:{} for k in driver.RUNTIME_KEYS}
    runtime.update(sam3_source=dict(path=str(root),tree_sha256='tree'),sam3_checkpoint=dict(path=str(checkpoint),
        sha256=s.file_identity(checkpoint)['sha256'],bytes=checkpoint.stat().st_size,mtime_ns=checkpoint.stat().st_mtime_ns))
    s.write_new_json(root/'runtime.json',runtime);s.write_new_json(root/'rgb.json',{})
    rgb=s.identity(root/'rgb.json')
    s.write_new_json(root/'construction.json',dict(source_commit=driver.CONSTRUCTION_SHA,scope='e6_public_rgb_full18',runtime=rgb))
    cfg=dict(schema_version=1,scope='e6_public_grounding_v1',freeze_id=stage.name,source_commit='new_source',
        construction=s.identity(root/'construction.json'),discovery_runtime=s.identity(root/'runtime.json'),rgb_runtime=rgb,protocol=g.PROTOCOL)
    path=stage/'execution.json';s.write_new_json(path,cfg)
    refs={k:cfg[k] for k in ('construction','discovery_runtime','rgb_runtime')};refs['config']=s.identity(path)
    resources=[dict(id='e6_grounding_'+k,resolved_path=v['path'],sha256=v['sha256']) for k,v in refs.items()]
    resources.append(dict(id='sam3_checkpoint',resolved_path=str(checkpoint),sha256=runtime['sam3_checkpoint']['sha256']))
    contract=dict(freeze_id=stage.name,code=dict(commit='new_source',dirty=False),resource_inventory=resources)
    contract['contract_sha256']=s.canonical_hash(contract)
    (stage/'contract').mkdir();s.write_new_json(stage/'contract/freeze_manifest.json',contract)
    monkeypatch.setattr(s,'ROOT',root)
    monkeypatch.setattr(s,'git_snapshot',lambda p:dict(commit=driver.CONSTRUCTION_SHA if str(p)==driver.CONSTRUCTION_CODE else 'new_source',dirty=False))
    monkeypatch.setattr(d,'validate_cohort_runtime',lambda r:None)
    from robo.eval import fidelity_replacements
    monkeypatch.setattr(fidelity_replacements,'_tree_inventory',lambda *a,**k:dict(tree_sha256='tree'))
    return path,stage,cfg,checkpoint


def test_grounding_admission_exact_e0_and_source(monkeypatch,tmp_path):
    path,stage,cfg,checkpoint=admission(monkeypatch,tmp_path)
    assert driver.validate(path,stage)[0]==cfg


@pytest.mark.parametrize('change',['weights','config','source','protocol','reference','unknown'])
def test_grounding_admission_rejects_drift(monkeypatch,tmp_path,change):
    path,stage,cfg,checkpoint=admission(monkeypatch,tmp_path)
    if change=='weights':checkpoint.write_bytes(b'changed model')
    elif change=='config':cfg['freeze_id']='changed'
    elif change=='source':cfg['source_commit']='changed'
    elif change=='protocol':cfg['protocol']=dict(g.PROTOCOL,discovery_stride=12)
    elif change=='reference':Path(cfg['construction']['path']).write_text('{}')
    else:cfg['hidden_reference_path']='unbound'
    path.write_text(driver.json.dumps(cfg))
    with pytest.raises(ValueError):driver.validate(path,stage)


def test_failed_reconstruction_preserves_four_queries_without_producer(monkeypatch,tmp_path):
    s=driver.shared;root=tmp_path;stage=root/'stage';stage.mkdir()
    code=root/'code';querydir=code/'configs/experiments/icra2027/public_task_queries';querydir.mkdir(parents=True)
    queries=[query('q'+str(i)) for i in range(4)]
    s.write_new_json(querydir/'queries.json',dict(queries=queries))
    roster=dict(rows=[dict(scene_id='scene',condition_id='mild',query_hashes={q['task_id']:q['query_sha256'] for q in queries})])
    s.write_new_json(root/'roster.json',roster)
    source=dict(roster=dict(path=str(root/'roster.json')))
    cfg=dict(scope='e6_public_grounding_v1',source_commit='source',freeze_id='stage')
    monkeypatch.setattr(driver,'CONSTRUCTION_CODE',str(code));monkeypatch.setattr(s,'ROOT',root)
    monkeypatch.setattr(driver,'validate',lambda *a:(cfg,{}, {},source))
    monkeypatch.setattr(driver,'original_unit',lambda *a:(dict(stage_status='FAIL'),root/'origin',dict(terminal_manifest='original')))
    monkeypatch.setattr(driver,'commands',lambda *a:[])
    monkeypatch.setattr(driver.subprocess,'run',lambda *a,**k:pytest.fail('failed reconstruction must not invoke model'))
    result=driver.execute('config',stage,'scene','mild')
    assert len(result['queries'])==4 and result['planned_cohort_queries']==72 and result['stage_records']==[]
    assert result['discovered_objects'] is None and result['robot_frame'] is None
    assert all(q['roles']['target']['reason']=='upstream_reconstruction_rejected' for q in result['queries'])
    assert driver.validate_output('config',stage,'scene','mild')==result
    dest=stage/'audit/public_grounding/scene_mild'
    (dest/'unexpected.txt').write_text('tampered')
    with pytest.raises(ValueError,match='bytes changed'):driver.validate_output('config',stage,'scene','mild')
    with pytest.raises(FileExistsError):driver.execute('config',stage,'scene','mild')


def test_invalid_intrinsics_fail_closed():
    with pytest.raises(ValueError,match='intrinsics'):
        g.projected_instance(np.ones((10,3)),np.arange(10),np.zeros((3,3)),np.eye(4),np.ones((4,4)),np.ones((4,4)))


def test_virtual_transport_keeps_same_declaration_and_no_scale():
    declaration=dict(status='declared_virtual_frame',base_pos=[1.,2.,3.],base_yaw=.2)
    T=np.eye(4);T[:3,3]=[.5,.2,.1]
    a=dict(status='aligned_public_frame',transform=T.tolist(),scale_applied=1.)
    transported=g.transport_virtual_frame(declaration,a)
    assert np.allclose(np.asarray(transported['world_from_robot'])[:3,3],[.5,1.8,2.9])
    assert transported['physical_stability_known'] is False and transported['real_calibration_claimed'] is False
    assert g.transport_virtual_frame(declaration,dict(a,scale_applied=2))['world_from_robot'] is None
    assert g.transport_virtual_frame(None,a)['reason']=='clean_virtual_declaration_unavailable'
