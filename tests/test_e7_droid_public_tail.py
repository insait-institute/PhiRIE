"""Public tail preserves the original population, source closure and missing units."""
import copy
import json
import os
import sys
from pathlib import Path
import numpy as np
import pytest
from run.icra2027 import e7_droid_public_tail as m
from agents.recon.droid_extract import input_identity


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v));return input_identity(p)


def write_mesh(dest):
    (dest/'derived_mesh.ply').write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n')


@pytest.fixture
def source(tmp_path,monkeypatch):
    old=tmp_path/'old';old.mkdir();stage=tmp_path/'gs-freeze';stage.mkdir()
    ids=['droid_'+str(i) for i in range(10)]
    config=write(old/'config.json',{'workspace_ids':ids,'pilot_workspace':ids[0]})
    e={'freeze_id':stage.name,'code':{'commit':'old','dirty':False},'resource_inventory':[{'id':'e7_gaussian_config','sha256':config['sha256']}]}
    e['contract_sha256']=m.canonical_hash(e)
    contract=write(stage/'contract/freeze_manifest.json',e)
    monkeypatch.setattr(m,'git_snapshot',lambda _:{'commit':'old','dirty':False})
    return {'gaussian_source':{'code_root':str(old),'stage_root':str(stage),'commit':'old','config':config,'contract':contract,'python':sys.executable},'workspace_ids':ids,'pilot_workspace':ids[0],'evidence_root':str(tmp_path)}


def test_exact_original_population(source):
    assert len(m.validate_source(source)['workspace_ids'])==10


@pytest.mark.parametrize('change',['drop','order','pilot','commit','config','e0'])
def test_source_mutations_fail(source,change):
    c=copy.deepcopy(source)
    if change=='drop':c['workspace_ids']=c['workspace_ids'][:-1]
    if change=='order':c['workspace_ids']=c['workspace_ids'][::-1]
    if change=='pilot':c['pilot_workspace']=c['workspace_ids'][1]
    if change=='commit':c['gaussian_source']['commit']='new'
    if change in {'config','e0'}:Path(c['gaussian_source']['config' if change=='config' else 'contract']['path']).write_text('{}')
    with pytest.raises((ValueError,RuntimeError)):m.validate_source(c)


def test_original_probe_binds_e0(source,monkeypatch):
    e=json.loads(Path(source['gaussian_source']['contract']['path']).read_text())
    r={'source_commit':'old','workspace_id':source['pilot_workspace'],'contract_sha256':e['contract_sha256'],'available':False}
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'unrelated log\nE7_GS_SOURCE_JSON='+json.dumps(r))
    assert m.original_gaussian(source,source['pilot_workspace'])==r
    r['contract_sha256']='wrong'
    with pytest.raises(ValueError,match='wrong GS'):m.original_gaussian(source,source['pilot_workspace'])


def test_probe_no_reference_inputs():
    compile(m.SOURCE_PROBE,'source_probe','exec')
    for forbidden in ('result.json','held_out_reference','alignment_evaluation','evaluate_sealed_fit'):
        assert forbidden not in m.SOURCE_PROBE
    assert m.SOURCE_PROBE.index('validate_prepared(c,w,d,p,code)')<m.SOURCE_PROBE.index("r=json.loads((d/'train_receipt.json')")


@pytest.mark.parametrize('path',['/data/ScanNetpp/scene.ply','/a/held_out_reference.json','/a/alignment_evaluation.json','/a/result.json','/a/vault/ref.ply','/a/full_cpu_summary/rows.json',str(Path(__file__).resolve().parents[1]/'data/recon_scenes/data/x.ply')])
def test_reference_reads_rejected(path):
    with pytest.raises(PermissionError):m.enforce_public_read('open',(path,'r'))


def test_public_auto_contract_name_allowed():
    m.enforce_public_read('open',('/a/public_tail/objects/000/gt_points.ply','r'))
    m.enforce_public_read('open',(3,'r'))
    m.enforce_public_read('other',('/a/result.json',))


@pytest.fixture
def products(tmp_path):
    sd=tmp_path/'scene';names=[f'frame_{i:06}.jpg' for i in range(7)]
    write(sd/'dslr/nerfstudio/transforms_undistorted.json',{'h':720,'frames':[{'file_path':'images/'+n} for n in names]})
    d=tmp_path/'tail';(d/'mesh_derive').mkdir(parents=True)
    write_mesh(d)
    for i,n in enumerate(names[::3]):np.savez(d/'mesh_derive'/f'view_{i:04d}.npz',fname=n)
    write(d/'object_read_frames.json',names[:2]);write(d/'objects/objects.json',[{'gt_object_id':1001,'index':0,'mask_source':'derived_projection'}])
    np.savez(d/'auto_instances.npz',labels=np.array(['box','mug']))
    return {'scene_dir':str(sd),'workspace_id':'fixed','scene_root':str(tmp_path),'gaussian':{'path':'/sealed/scene.ply'}},d


def test_real_camera_roster_and_canonical_denominator(products):
    s,d=products;m.validate_public_products(s,d);r=m.object_contracts(d)
    assert r['discovered_instances']==2 and r['prepared_instances']==1
    assert r['jobs'][0]['failure_reason']=='filtered_by_existing_preparation_gates'
    assert r['jobs'][1]['prepared']


@pytest.mark.parametrize('change',['extra','missing','wrong_camera','outside_read','duplicate_read'])
def test_public_population_mutations_fail(products,change):
    s,d=products
    if change=='extra':np.savez(d/'mesh_derive/view_9999.npz',fname='extra')
    if change=='missing':(d/'mesh_derive/view_0000.npz').unlink()
    if change=='wrong_camera':np.savez(d/'mesh_derive/view_0000.npz',fname='wrong.jpg')
    if change=='outside_read':write(d/'object_read_frames.json',['outside.jpg'])
    if change=='duplicate_read':write(d/'object_read_frames.json',['frame_000000.jpg']*2)
    with pytest.raises(ValueError):m.validate_public_products(s,d)


def test_commands_reuse_existing_producers(products):
    s,d=products;r={'render_python':'render','sam3_python':'sam'};c={'cpu_python':'cpu'}
    rows=m.phase_commands(c,r,s,d)
    assert [x[0] for x in rows]==['render','fuse','discover','prepare']
    assert rows[0][3][-2:]==['--frame-stride','3'] and rows[2][3][-2:]==['--frame-stride','3']
    assert rows[0][3][3:5]==['--splat-ply','/sealed/scene.ply']
    assert all('s3_lift' not in x[2] for x in rows)


def test_worker_environment_isolation_and_metadata_scale(products,monkeypatch):
    s,d=products;r={'sam3_checkpoint':{'path':'checkpoint'},'hf_home':'hf','renderer_dependency_root':'overlay','sam3_source':{'path':'sam'}}
    c={'torch_extensions_root':'extensions'}
    monkeypatch.setenv('SIMANY_AUTO','0');monkeypatch.setenv('PYTHONHOME','bad')
    env=m.environment(c,r,s,d,'prepare')
    assert env['SIMANY_AUTO']=='1' and env['SIMANY_NO_GT']=='1' and env['SIMANY_FULL']=='0'
    assert env['SIMANY_MIN_BBOX_PX']=='30' and env['SIMANY_MIN_MASK_PX']=='152'
    assert 'PYTHONHOME' not in env
    monkeypatch.setattr(m.os,'environ',env)
    m.validate_worker_environment(c,r,s,d,'prepare',sys.executable)
    env['SIMANY_MESH_SRC']='gt'
    with pytest.raises(ValueError,match='environment'):m.validate_worker_environment(c,r,s,d,'prepare',sys.executable)


def setup_run(tmp_path,monkeypatch,available=False):
    c={'workspace_ids':['fixed'],'pilot_workspace':'fixed','freeze_id':'stage'};p=tmp_path/'config.json';write(p,c)
    code={'commit':'exact','dirty':False};e={'contract_sha256':'e0'}
    monkeypatch.setattr(m,'context',lambda *a,**k:(c,code,e,{}))
    src={'available':available,'reason':'sealed_TRAIN_fit_unavailable'}
    monkeypatch.setattr(m,'original_gaussian',lambda *a:src)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('SLURMD_NODENAME','hala');monkeypatch.delenv('SLURM_ARRAY_JOB_ID',raising=False)
    return p,tmp_path/'stage',src


def test_missing_unit_no_fake_execution_and_no_overwrite(tmp_path,monkeypatch):
    p,s,src=setup_run(tmp_path,monkeypatch)
    r=m.execute(p,s,'fixed');assert r['status']=='NOT_RUN' and r['jobs'] is None and r['stages']==[]
    assert m.validate_output(p,s,'fixed')==r
    with pytest.raises(FileExistsError):m.execute(p,s,'fixed')
    (s/'public_tail/fixed/extra').write_text('unlisted')
    with pytest.raises(ValueError,match='inventory'):m.validate_output(p,s,'fixed')


def test_ordinary_jobs_only(tmp_path,monkeypatch):
    p,s,_=setup_run(tmp_path,monkeypatch);monkeypatch.setenv('SLURM_ARRAY_JOB_ID','1')
    with pytest.raises(ValueError,match='ordinary'):m.execute(p,s,'fixed')
    assert not (s/'public_tail').exists()


def test_failed_producer_preserved(tmp_path,monkeypatch):
    p,s,_=setup_run(tmp_path,monkeypatch,True)
    monkeypatch.setattr(m,'phase_commands',lambda *a:[('render','python','canonical',[])])
    monkeypatch.setattr(m,'environment',lambda *a:{})
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:type('Result',(),{'returncode':9})())
    m.plan(p,s,'fixed');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(RuntimeError,match='producer failed'):m.run_phase(p,s,'fixed','render')
    r=m.validate_phase(p,s,'fixed','render');assert r['status']=='FAIL' and r['execution']['returncode']==9
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    with pytest.raises(ValueError,match='failed producer'):m.finalize(p,s,'fixed')
    assert m.validate_output(p,s,'fixed')['jobs'] is None


def test_runtime_alias_preserves_legacy(monkeypatch,tmp_path):
    import importlib,importlib.metadata,site
    from agents.recon.droid_extract import runtime_manifest
    from types import SimpleNamespace
    record=Path('name.dist-info/RECORD');module=Path('module.py')
    (tmp_path/record).parent.mkdir();(tmp_path/record).write_text('record');(tmp_path/module).write_text('source')
    calls=[]
    def distribution(name):
        calls.append(name);return SimpleNamespace(files=[record,module],version='1',locate_file=lambda p:tmp_path/p)
    monkeypatch.setattr(importlib.metadata,'distribution',distribution)
    monkeypatch.setattr(importlib,'import_module',lambda _:SimpleNamespace(__file__=str(tmp_path/module)))
    monkeypatch.setattr(site,'ENABLE_USER_SITE',False)
    r=runtime_manifest(['legacy',('PIL','Pillow')])
    assert calls==['legacy','Pillow'] and set(r['packages'])=={'legacy','PIL'}


def test_success_replay_excludes_only_receipt_and_keeps_rejected_instances(products,monkeypatch,tmp_path):
    source,d=products
    stage=tmp_path/'stage';unit=stage/'public_tail/fixed';unit.parent.mkdir(parents=True);d.rename(unit)
    p=tmp_path/'config.json';write(p,{})
    c={'workspace_ids':['fixed'],'freeze_id':'stage'};code={'commit':'exact'};e={'contract_sha256':'e0'}
    runtime={'render_python':'render','sam3_python':'sam'};c['cpu_python']='cpu'
    monkeypatch.setattr(m,'context',lambda *a,**k:(c,code,e,runtime));source['available']=True
    monkeypatch.setattr(m,'original_gaussian',lambda *a:source)
    stages=[]
    for n,py,module,args in m.phase_commands(c,runtime,source,unit):
        stages.append({'stage':n,'returncode':0,'wall_s':1,'producer_module':module,'producer_args':args,'command':[py,'-m','run.icra2027.e7_droid_public_tail','--config',str(p),'--stage-root',str(stage),'--workspace','fixed','--worker',n]})
    r={'scope':m.SCOPE,'code':code,'config':input_identity(p),'freeze_id':'stage','contract_sha256':'e0','workspace_id':'fixed','planned_workspaces':10,'paper_ready':False,'full_build':False,'original_gaussian':source,'status':'PASS','stages':stages,**m.object_contracts(unit),'artifacts_without_receipt':m.tree(unit)}
    write(unit/'tail_receipt.json',r)
    assert m.validate_output(p,stage,'fixed')==r
    r['prepared_instances']=2;write(unit/'tail_receipt.json',r)
    with pytest.raises(ValueError,match='denominator'):m.validate_output(p,stage,'fixed')
    r['prepared_instances']=1;r['stages'][0]['producer_args']=['altered'];write(unit/'tail_receipt.json',r)
    with pytest.raises(ValueError,match='invocation'):m.validate_output(p,stage,'fixed')


def test_worker_refuses_second_producer_invocation(tmp_path,monkeypatch):
    p,s,source=setup_run(tmp_path,monkeypatch,True)
    d=s/'public_tail/fixed';d.mkdir(parents=True)
    monkeypatch.setattr(m,'phase_commands',lambda *a:[('render',sys.executable,'canonical',[])])
    monkeypatch.setattr(m,'validate_worker_environment',lambda *a:None)
    monkeypatch.setattr(m.sys,'addaudithook',lambda *a:None)
    calls=[];monkeypatch.setattr(m.runpy,'run_module',lambda *a,**k:calls.append(a))
    m.worker(p,s,'fixed','render')
    with pytest.raises(FileExistsError):m.worker(p,s,'fixed','render')
    assert len(calls)==1


def test_separated_chain_requires_predecessors_and_replays_all_receipts(tmp_path,products,monkeypatch):
    import shutil
    p,stage,source=setup_run(tmp_path,monkeypatch,True)
    actual,template=products;source.update(actual)
    c={'workspace_ids':['fixed'],'pilot_workspace':'fixed','freeze_id':'stage','cpu_python':sys.executable}
    runtime={'render_python':sys.executable,'sam3_python':sys.executable}
    monkeypatch.setattr(m,'context',lambda *a,**k:(c,{'commit':'exact','dirty':False},{'contract_sha256':'e0'},runtime))
    monkeypatch.setattr(m,'environment',lambda *a:{})
    destination=stage/'public_tail/fixed'
    def producer(command,**kwargs):
        if command[-1]=='fuse':write_mesh(destination)
        if command[-1]=='prepare':shutil.copytree(template,destination,dirs_exist_ok=True)
        return type('Result',(),{'returncode':0})()
    monkeypatch.setattr(m.subprocess,'run',producer)
    m.plan(p,stage,'fixed')
    with pytest.raises(FileNotFoundError):m.run_phase(p,stage,'fixed','fuse')
    for phase in m.PHASES:
        if phase in {'render','discover'}:monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
        else:monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
        assert m.run_phase(p,stage,'fixed',phase)['status']=='PASS'
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    r=m.finalize(p,stage,'fixed');assert r['status']=='PASS' and r['prepared_instances']==1
    assert m.validate_output(p,stage,'fixed')==r
    with pytest.raises(FileExistsError):m.run_phase(p,stage,'fixed','prepare')
    (destination/'render.log').write_text('tampered')
    with pytest.raises((RuntimeError,ValueError)):m.validate_phase(p,stage,'fixed','render')


@pytest.mark.parametrize('gpu,visible',[(False,'0'),(True,''),(True,'0,1')])
def test_cpu_gpu_resource_separation(tmp_path,monkeypatch,gpu,visible):
    setup_run(tmp_path,monkeypatch);monkeypatch.setenv('CUDA_VISIBLE_DEVICES',visible)
    with pytest.raises(ValueError):m.scheduler_guard(gpu)


@pytest.mark.parametrize('key',['SLURM_JOB_GPUS','SLURM_STEP_GPUS','SLURM_GPUS_ON_NODE'])
@pytest.mark.parametrize('visible',['','-1'])
def test_hidden_gpu_allocation_rejected_on_cpu(tmp_path,monkeypatch,key,visible):
    setup_run(tmp_path,monkeypatch);monkeypatch.setenv('CUDA_VISIBLE_DEVICES',visible)
    monkeypatch.setenv(key,'1')
    with pytest.raises(ValueError,match='must not reserve'):m.scheduler_guard(False)


@pytest.mark.parametrize('key',['SLURM_JOB_GPUS','SLURM_STEP_GPUS'])
def test_gpu_index_zero_is_an_allocation(tmp_path,monkeypatch,key):
    setup_run(tmp_path,monkeypatch);monkeypatch.setenv(key,'0')
    with pytest.raises(ValueError,match='must not reserve'):m.scheduler_guard(False)


@pytest.mark.parametrize('damage',['missing','empty','nonfinite','indices','no_faces'])
def test_fused_mesh_postcondition(damage,tmp_path):
    write_mesh(tmp_path);p=tmp_path/'derived_mesh.ply'
    if damage=='missing':p.unlink()
    elif damage=='empty':p.write_bytes(b'')
    elif damage=='nonfinite':p.write_text(p.read_text().replace('1 0 0\n','nan 0 0\n'))
    elif damage=='indices':p.write_text(p.read_text().replace('3 0 1 2','3 0 1 5'))
    else:p.write_text(p.read_text().replace('element face 1','element face 0').replace('3 0 1 2\n',''))
    with pytest.raises(ValueError):m.validate_fused_mesh(tmp_path)


def test_successful_exit_without_mesh_stays_failed_and_blocks_gpu(tmp_path,monkeypatch):
    p,s,_=setup_run(tmp_path,monkeypatch,True)
    monkeypatch.setattr(m,'phase_commands',lambda *a:[('fuse',sys.executable,'canonical',[])])
    monkeypatch.setattr(m,'environment',lambda *a:{})
    original=m.validate_phase
    monkeypatch.setattr(m,'validate_phase',lambda *a: {'status':'PASS'} if a[-1]=='render' else original(*a))
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:type('Result',(),{'returncode':0})())
    m.plan(p,s,'fixed')
    with pytest.raises(ValueError,match='missing or empty fused mesh'):m.run_phase(p,s,'fixed','fuse')
    r=original(p,s,'fixed','fuse')
    assert r['status']=='FAIL' and r['execution']['returncode']==0
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError,match='predecessor'):m.run_phase(p,s,'fixed','discover')
