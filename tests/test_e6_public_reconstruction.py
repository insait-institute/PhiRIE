import copy
import json
from pathlib import Path
import numpy as np
import pytest
from run.icra2027 import e6_public_reconstruction as e6


@pytest.fixture
def roster(tmp_path):
    if not e6.PUBLIC.is_dir():
        pytest.skip('public RGB scene tree is unavailable (set SIMANY_ROOT)')
    sources=e6.read(e6.CODE/'configs/experiments/icra2027/public_task_queries/sources.json')
    queries=e6.read(e6.CODE/'configs/experiments/icra2027/public_task_queries/queries.json')['queries']
    index={s['source_id']:s for row in sources['scenes'] for s in row['sources']}
    public=tmp_path/'public';public.mkdir()
    for scene in e6.SCENES:
        q=next(q for q in queries if q['scene_id']==scene)
        anchor=Path(index[q['source_state_anchor']]['relative_path']).name
        for c in e6.CONDITIONS:
            d=public/(scene+('' if c=='clean' else '_'+c))/'dslr/resized_undistorted_images';d.mkdir(parents=True)
            for n in [anchor,'zz_other.jpg']:(d/n).write_bytes(b'public-only-rgb')
    r=e6.public_roster(public_root=public)
    return public,r


def test_rgb_roster_retains18conditions72queries(roster):
    public,r=roster
    assert len(r['rows'])==18 and r['planned_query_rows']==72
    assert e6.validate_roster(r,public_root=public)==r
    assert not r['legacy_depth_pose_splat_inputs']
    assert all(len(x['query_hashes'])==4 for x in r['rows'])
    assert all('/dslr/resized_undistorted_images/' in f['path'] for r in r['rows'] for f in r['frames'])


def test_uniform48_is_frozen_and_preserves_only_existing_anchor():
    names=[f'{i:03}.jpg' for i in range(100)]
    selected=e6.select_frames(names,'099.jpg')
    assert len(selected)==48 and selected==e6.select_frames(names,'099.jpg')
    assert '037.jpg' in e6.select_frames(names,'037.jpg')
    assert 'missing.jpg' not in e6.select_frames(names,'missing.jpg')
    for bad in [names[::-1],names+[names[-1]],['../gt.jpg','x.jpg']]:
        with pytest.raises(ValueError):e6.select_frames(bad,'x.jpg')
    with pytest.raises(ValueError):e6.select_frames(names,'037.jpg',47)


@pytest.mark.parametrize('kind',['drop_condition','depth_path','changed_query','removed_frame','changed_selection','anchor','promote'])
def test_roster_fails_closed(roster,kind):
    public,original=roster;r=copy.deepcopy(original);row=r['rows'][0]
    if kind=='drop_condition':r['rows'].pop()
    elif kind=='depth_path':row['frames'][0]['path']=str(public/'gt/depth.png')
    elif kind=='changed_query':row['query_hashes']['e6_public_q01']='0'*64
    elif kind=='removed_frame':row['frames'].pop();row['frame_count']-=1
    elif kind=='changed_selection':row['frames'][0]['selected']=False
    elif kind=='anchor':row['source_frame']='different.jpg'
    else:r['legacy_depth_pose_splat_inputs']=True
    with pytest.raises(ValueError):e6.validate_roster(r,public_root=public)


def test_roster_rejects_symlink_to_forbidden_tree(roster):
    public,r=roster;f=Path(r['rows'][0]['frames'][0]['path']);f.unlink()
    secret=public/'gt';secret.mkdir();(secret/'depth.jpg').write_bytes(b'not-public')
    f.symlink_to(secret/'depth.jpg')
    with pytest.raises(e6.sealed.sealed_cpu.PilotGateError,match='symlink'):
        e6.public_roster(public_root=public)


def runtime():
    return dict(python='/python',gs_python='/gs-python',omega_source='/omega',omega_checkpoint='/omega/model.pt',
                da3_checkpoint='/da3-model',da3_source='/da3-source',da3_overlay='/overlay',gs_overlay='/gs-overlay',
                torch_extensions_root='/torch-cache')


def test_commands_only_invoke_existing_rgb_producers_and_separate_condition_paths(tmp_path):
    rows=e6.commands({},runtime(),tmp_path/'scene_clean',tmp_path/'scene_clean/frames')
    assert [r[0] for r in rows]==['omega','metricize','scene','gaussian']
    assert [r[2][1] for r in rows]==['models.vggt_scene','agents.recon.metricize','agents.recon.make_scene_dir','agents.recon.gsplat_train']
    assert rows[0][2][rows[0][2].index('--backend')+1]=='omega'
    assert '--checkpoint' in rows[1][2] and '--init-ply' in rows[3][2]
    assert all('gt/' not in s and 'run_behavior_recon' not in s for _,_,args in rows for s in args)
    other=e6.commands({},runtime(),tmp_path/'scene_mild',tmp_path/'scene_mild/frames')
    assert rows[0][2]!=other[0][2]


def test_environment_removes_legacy_inputs_and_user_site(monkeypatch,tmp_path):
    monkeypatch.setenv('SIMANY_MESH_SRC','/old/gt/mesh.ply')
    monkeypatch.setenv('SIMANY_SCANNETPP_ROOT','/old/data')
    monkeypatch.setenv('PYTHONPATH','/unsealed')
    env=e6.environment(runtime(),tmp_path,'omega')
    assert env['SIMANY_MESH_SRC']=='derived' and 'SIMANY_SCANNETPP_ROOT' not in env
    assert env['PYTHONNOUSERSITE']=='1' and '/unsealed' not in env['PYTHONPATH']
    assert env['HF_HUB_OFFLINE']=='1' and env['TRANSFORMERS_OFFLINE']=='1'


def products(tmp_path):
    names=['a.jpg','b.jpg'];row={'frames':[{'name':n,'selected':True} for n in names]}
    p=dict(names=names,w2c=np.repeat(np.eye(4)[None],2,axis=0),K=np.eye(3),points=np.ones((2,3)),
           depth=np.ones((2,2,2)))
    np.savez(tmp_path/'recon.npz',**p)
    np.savez(tmp_path/'recon_metric.npz',**p,metric_scale=1.,T_align=np.eye(4))
    (tmp_path/'gaussian').mkdir();(tmp_path/'gaussian/scene.ply').write_bytes(b'ply')
    e6.write_new_json(tmp_path/'gaussian/train_report.json',dict(seed=0,iters=15000,independent_heldout_evaluation=False,
        gradient_train_frames=['b.jpg'],internal_diagnostic_frames=['a.jpg']))
    return row


def test_actual_npz_schema_and_split_verification(tmp_path):
    row=products(tmp_path);e6.validate_products(tmp_path,row)
    with np.load(tmp_path/'recon.npz') as x:d=dict(x)
    d['points'][0,0]=float('nan');np.savez(tmp_path/'recon.npz',**d)
    with pytest.raises(ValueError,match='nonfinite'):e6.validate_products(tmp_path,row)


def test_missing_gaussian_and_fabricated_heldout_claim_rejected(tmp_path):
    row=products(tmp_path);p=tmp_path/'gaussian/train_report.json';r=e6.read(p)
    r['independent_heldout_evaluation']=True;p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='split/RNG'):e6.validate_products(tmp_path,row)
    r['independent_heldout_evaluation']=False;p.write_text(json.dumps(r));(tmp_path/'gaussian/scene.ply').unlink()
    with pytest.raises(ValueError,match='missing'):e6.validate_products(tmp_path,row)


def test_nonzero_producer_preserved_as_typed_failure_no_fallback(roster,tmp_path,monkeypatch):
    from types import SimpleNamespace
    public,r=roster;row=r['rows'][0];stage=tmp_path/'stage'
    config={'scope':'e6_rgb_only_reconstruction_pilot','source_commit':'a'*40,'freeze_id':'stage','execution_scene':e6.SCENES[0]}
    monkeypatch.setattr(e6,'ROOT',tmp_path)
    monkeypatch.setattr(e6,'validate',lambda *a,**k:(config,runtime(),row))
    calls=[]
    def failure(command,**kwargs):
        calls.append(command);kwargs['stdout'].write('deliberate missing runtime\n');return SimpleNamespace(returncode=7)
    monkeypatch.setattr(e6.subprocess,'run',failure)
    report=e6.run('config',stage,'clean')
    assert report['stage_status']=='FAIL' and report['failure_type']=='producer_nonzero_exit'
    assert report['failed_stage']=='omega' and len(calls)==1
    assert report['feature_rows_written']==0 and report['paper_ready'] is False
    assert e6.validate_output('config',stage,'clean')==report
    with pytest.raises(FileExistsError):e6.run('config',stage,'clean')
    assert len(calls)==1


@pytest.fixture
def e0_case(roster,tmp_path,monkeypatch):
    public,r=roster
    monkeypatch.setattr(e6,'ROOT',tmp_path);monkeypatch.setattr(e6,'PUBLIC',public)
    monkeypatch.setattr(e6,'verify_gs_runtime',lambda *a:None)
    commit='a'*40
    monkeypatch.setattr(e6,'git_snapshot',lambda path:{'commit':commit,'dirty':False})
    stage=tmp_path/'outputs/icra2027/20260906-aaaaaaa-v1';stage.mkdir(parents=True)
    rp=stage/'roster.json';e6.write_new_json(rp,r)
    rt=runtime();rt['files']=[];rt['sources']=[];rt['checkpoints']=[]
    for key in ['python','gs_python']:
        p=stage/key;p.write_text(key);rt[key]=str(p);rt['files'].append(e6.identity(p))
    for name in ['omega','da3']:
        d=stage/name;d.mkdir();rt[name+'_source']=str(d)
        rt['sources'].append({'id':name,'path':str(d),'commit':commit})
        weight=d/'model.safetensors';weight.write_bytes(b'weights')
        rt[name+'_checkpoint']=str(weight if name=='omega' else d)
        rt['checkpoints'].append({'id':name+'_checkpoint',**e6.identity(weight),'mtime_ns':weight.stat().st_mtime_ns})
    runtime_path=stage/'runtime.json';e6.write_new_json(runtime_path,rt)
    config=dict(scope='e6_rgb_only_reconstruction_pilot',tier='pilot',source_commit=commit,freeze_id=stage.name,
        max_frames=48,seed=0,gs_iters=15000,execution_scene=e6.SCENES[0],fallback_allowed=False,
        roster=e6.identity(rp),runtime=e6.identity(runtime_path))
    cp=stage/'execution.json';e6.write_new_json(cp,config)
    resources=[]
    for name,ref in [('e6_reconstruction_config',e6.identity(cp)),('e6_rgb_roster',config['roster']),('e6_runtime',config['runtime'])]:
        resources.append({'id':name,'resolved_path':ref['path'],'sha256':ref['sha256']})
    for f in rt['checkpoints']:resources.append({'id':f['id'],'resolved_path':f['path'],'sha256':f['sha256']})
    contract=dict(freeze_id=stage.name,code={'commit':commit,'dirty':False},resource_inventory=resources)
    contract['contract_sha256']=e6.canonical_hash(contract)
    (stage/'contract').mkdir();e6.write_new_json(stage/'contract/freeze_manifest.json',contract)
    return cp,stage,rt


def test_exact_e0_binds_roster_runtime_checkpoint_and_selected_rgb(e0_case):
    cp,stage,rt=e0_case
    config,runtime,row=e6.validate(cp,stage,selected='clean')
    assert row['scene_id']==e6.SCENES[0] and runtime==rt


@pytest.mark.parametrize('kind',['source','e0','config','runtime','weight','rgb','unbound_model'])
def test_e0_rejects_source_config_checkpoint_and_rgb_drift(e0_case,monkeypatch,kind):
    cp,stage,rt=e0_case
    if kind=='source':monkeypatch.setattr(e6,'git_snapshot',lambda path:{'commit':'b'*40,'dirty':False})
    elif kind=='e0':
        p=stage/'contract/freeze_manifest.json';x=e6.read(p);x['resource_inventory'].pop();p.write_text(json.dumps(x))
    elif kind=='config':
        x=e6.read(cp);x['seed']=1;cp.write_text(json.dumps(x))
    elif kind=='runtime':Path(rt['python']).write_text('changed runtime')
    elif kind=='weight':Path(rt['omega_checkpoint']).write_text('changed checkpoint')
    elif kind=='rgb':
        x=e6.read(stage/'roster.json');Path(x['rows'][0]['frames'][0]['path']).write_text('changed RGB')
    else:
        p=stage/'runtime.json';x=e6.read(p);x['omega_checkpoint']='/gt/private.pt';p.write_text(json.dumps(x))
    with pytest.raises(ValueError):e6.validate(cp,stage,selected='clean')
