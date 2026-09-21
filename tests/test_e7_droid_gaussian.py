"""GS continuation preserves source, TRAIN boundary, missing slots and bytes."""
import copy
import json
from pathlib import Path
import pytest
from run.icra2027 import e7_droid_gaussian as m
from agents.recon.droid_extract import input_identity


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return input_identity(path)


def test_tree_tamper_and_symlink(tmp_path):
    d=tmp_path/'artifacts';d.mkdir();(d/'a').write_text('original')
    tree=m.tree(d);m.verify_tree(d,tree)
    (d/'a').write_text('tamper')
    with pytest.raises(ValueError,match='inventory'):m.verify_tree(d,tree)
    (d/'b').symlink_to(d/'a')
    with pytest.raises(ValueError,match='symlink'):m.tree(d)


def test_tree_symlink_root(tmp_path):
    d=tmp_path/'original';d.mkdir();p=tmp_path/'link';p.symlink_to(d)
    with pytest.raises(ValueError,match='canonical'):m.tree(p)


def test_source_probe_has_no_outcome_inputs():
    compile(m.SOURCE_PROBE,'original_source_probe','exec')
    for forbidden in ('result.json','held_out_reference','alignment_evaluation','evaluate_sealed_fit'):
        assert forbidden not in m.SOURCE_PROBE
    assert m.SOURCE_PROBE.index('validate_train_fit(u/') < m.SOURCE_PROBE.index("ex=json.loads")
    assert "sys.exit(0)" in m.SOURCE_PROBE  # missing fit is explicit, never a fabricated successful fit


@pytest.fixture
def source(tmp_path,monkeypatch):
    old=tmp_path/'old';old.mkdir();stage=tmp_path/'cpu-freeze';stage.mkdir()
    ids=['droid_'+str(i) for i in range(10)]
    cpu=write(old/'execution.json',{'captures':[{'workspace_id':i} for i in ids]})
    e={'freeze_id':stage.name,'code':{'commit':'old','dirty':False},'configs':[{'field':'real_world_config','source_content_sha256':cpu['sha256']}]}
    e['contract_sha256']=m.canonical_hash(e)
    contract=write(stage/'contract/freeze_manifest.json',e)
    rule=write(tmp_path/'rule.json',{'planned_workspace_ids':ids,'training_iterations':30000,'all_original_slots_retained':True,'threshold_or_view_policy_tuning':False,'config':{k:cpu[k] for k in ('path','sha256')}})
    monkeypatch.setattr(m,'git_snapshot',lambda _: {'commit':'old','dirty':False})
    c={'cpu_source':{'code_root':str(old),'stage_root':str(stage),'commit':'old','config':cpu,'contract':contract},'workspace_ids':ids,'pilot_workspace':ids[0],'continuation_rule':rule}
    return c


def test_original_roster_and_rule_authenticated(source):
    c,e=m.source_config(source);assert len(c['captures'])==10


@pytest.mark.parametrize('mutation',['drop','reorder','pilot','source','rule'])
def test_original_source_roster_rule_rejected(source,mutation):
    c=copy.deepcopy(source)
    if mutation=='drop':c['workspace_ids'].pop()
    if mutation=='reorder':c['workspace_ids'].reverse()
    if mutation=='pilot':c['pilot_workspace']=c['workspace_ids'][1]
    if mutation=='source':c['cpu_source']['commit']='new'
    if mutation=='rule':Path(c['continuation_rule']['path']).write_text('{}')
    with pytest.raises((ValueError,RuntimeError)):m.source_config(c)


def test_cpu_terminal_not_quality_gate(source,tmp_path,monkeypatch):
    c=source;c['cpu_submissions']={}
    for i,w in enumerate(c['workspace_ids']):
        c['cpu_submissions'][w]=write(tmp_path/f'{i}.json',{'job_id':str(i),'source_commit':'old','freeze_id':'cpu-freeze','workspace_id':w})
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'\n'.join(f'{i}|FAILED|' for i in range(10)))
    assert len(m.require_cpu_terminal(c))==10  # scientific result never determines whether the gate may be inspected
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'\n'.join(f'{i}|RUNNING|' for i in range(10)))
    with pytest.raises(ValueError,match='still running'):m.require_cpu_terminal(c)
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'0|FAILED|\n'+'\n'.join(f'{i}|RUNNING|' for i in range(1,10)))
    m.require_cpu_terminal(c,c['workspace_ids'][0])
    with pytest.raises(ValueError,match='still running'):m.require_cpu_terminal(c,c['workspace_ids'][1])


def test_original_numerical_replay_host_is_enforced(source,monkeypatch):
    source['cpu_source']['validation_host']='original-host'
    monkeypatch.setenv('SLURMD_NODENAME','different-host')
    with pytest.raises(ValueError,match='original CPU host'):m.train_source(source,source['pilot_workspace'])


def test_missing_fit_preserved_without_scene_or_metrics(tmp_path,monkeypatch):
    c={'workspace_ids':['fixed'],'pilot_workspace':'fixed','freeze_id':'new'}
    config=tmp_path/'config.json';write(config,c)
    monkeypatch.setattr(m,'context',lambda *a:(c,{'commit':'current'}, {'contract_sha256':'e0'}))
    monkeypatch.setattr(m,'require_cpu_terminal',lambda *args:None)
    monkeypatch.setattr(m,'train_source',lambda *a:{'eligible':False,'reason':'sealed_TRAIN_fit_unavailable'})
    r=m.run(config,tmp_path/'new','fixed','prepare')
    assert r['status']=='NOT_RUN' and r['full_build'] is False
    assert not (tmp_path/'new/gaussian/fixed/scene').exists()
    assert 'training_report' not in r
    with pytest.raises(FileExistsError):m.run(config,tmp_path/'new','fixed','prepare')


def test_real_scene_export_once(tmp_path,monkeypatch):
    import numpy as np
    from PIL import Image
    workspace='fixed';cpu=tmp_path/'cpu';u=cpu/'real_world/workspaces'/workspace
    frames=u/'extraction/wrist_frames';frames.mkdir(parents=True)
    Image.new('RGB',(16,16),'red').save(frames/'frame_000001.jpg')
    image=input_identity(frames/'frame_000001.jpg')
    rec=u/'recon_base.npz'
    np.savez(rec,names=np.array(['frame_000001.jpg']),w2c=np.eye(4)[None],K=np.array([[10,0,8],[0,10,8],[0,0,1]]),points=np.array([[0,0,1],[.1,0,1]]),points_rgb=np.array([[255,0,0],[255,0,0]],dtype=np.uint8))
    c={'workspace_ids':[workspace],'pilot_workspace':workspace,'freeze_id':'new','cpu_source':{'stage_root':str(cpu)}}
    config=tmp_path/'config.json';write(config,c)
    monkeypatch.setattr(m,'context',lambda *a:(c,{'commit':'current'}, {'contract_sha256':'e0'}))
    monkeypatch.setattr(m,'require_cpu_terminal',lambda *args:None)
    monkeypatch.setattr(m,'train_source',lambda *a:{'eligible':True,'fit':{'aligned':input_identity(rec)},'extraction':{'frames':[image]}})
    r=m.run(config,tmp_path/'new',workspace,'prepare')
    assert r['status']=='PASS' and len(r['public_rgb'])==1
    scene=tmp_path/'new/gaussian/fixed/scene'
    m.verify_tree(scene,r['scene_artifacts'])
    assert (scene/'data/fixed/init_points.ply').is_file()
    assert not any('held_out_reference' in k for k in r['scene_artifacts'])
    with pytest.raises(FileExistsError):m.run(config,tmp_path/'new',workspace,'prepare')


def test_runtime_env_excludes_uncontrolled_python(monkeypatch):
    monkeypatch.setenv('PYTHONHOME','/bad');monkeypatch.setenv('LD_PRELOAD','/bad');monkeypatch.setenv('SIMANY_GT','bad')
    c={'training_dependency_root':'/overlay','torch_extensions_root':'/extensions','evidence_root':'/evidence'}
    env=m.training_environment(c,'fixed')
    assert 'PYTHONHOME' not in env and 'LD_PRELOAD' not in env and 'SIMANY_GT' not in env
    assert env['PYTHONNOUSERSITE']=='1' and env['SIMANY_AUTO']=='1'


@pytest.mark.parametrize('mutation',[None,'iters','heldout','population','overlap','nonfinite','counts'])
def test_training_report_recipe_and_diagnostic_scope(mutation):
    r={'iters':30000,'seed':0,'independent_heldout_evaluation':False,'bitwise_determinism_claimed':False,
       'n_gaussians':10,'wall_s':1.,'psnr_internal_diagnostic':20.,'gradient_train_frames':['a'],
       'internal_diagnostic_frames':['b'],'n_train':1,'n_holdout':1}
    if mutation=='iters':r['iters']=100
    if mutation=='heldout':r['independent_heldout_evaluation']=True
    if mutation=='population':r['gradient_train_frames']=['different']
    if mutation=='overlap':r['gradient_train_frames']=['b']
    if mutation=='nonfinite':r['wall_s']=float('nan')
    if mutation=='counts':r['n_train']=2
    if mutation:
        with pytest.raises(ValueError):m.validate_training_report(r,{'public_rgb':{'a':{},'b':{}}},30000)
    else:m.validate_training_report(r,{'public_rgb':{'a':{},'b':{}}},30000)


@pytest.mark.parametrize('mutation',[None,'embedded_fit','embedded_extraction','different_slot','wrong_path','changed_bytes'])
def test_prepared_source_payload_and_paths(tmp_path,monkeypatch,mutation):
    u=tmp_path/'cpu/real_world/workspaces/fixed'
    plan=write(tmp_path/'plan.json',{'scope':'train'})
    train=write(u/'extraction/train_trajectory.json',{'role':'train_only'})
    recon=write(u/'recon.npz',{})
    aligned=write(u/'alignment/recon_base.npz',{})
    fit={'plan':plan,'train':train,'recon':recon,'aligned':aligned,'solution':{'scale':1.}}
    fit_id=write(u/'alignment/fit.json',fit)
    seal=write(u/'alignment/seal.json',{'fit_sha256':fit_id['sha256'],'fit_digest':m.canonical_hash(fit)})
    rgb=write(u/'extraction/wrist_frames/a.jpg',{})
    extraction={'plan':plan,'train_trajectory':train,'frames':[rgb]}
    extraction_id=write(u/'extraction/extraction_manifest.json',extraction)
    cap={'workspace_id':'fixed','plan':plan}
    c={'cpu_source':{'commit':'old','stage_root':str(tmp_path/'cpu')}}
    src={'capture':cap,'source_commit':'old','contract_sha256':'e0','eligible':True,'fit':copy.deepcopy(fit),
         'extraction':copy.deepcopy(extraction),'fit_identity':fit_id,'seal':seal,'extraction_identity':extraction_id}
    monkeypatch.setattr(m,'source_config',lambda _ :({'captures':[cap]},{'contract_sha256':'e0'}))
    if mutation=='embedded_fit':src['fit']['solution']['scale']=2.
    if mutation=='embedded_extraction':src['extraction']['frames']=[]
    if mutation=='different_slot':src['capture']={'workspace_id':'other','plan':plan}
    if mutation=='wrong_path':src['fit_identity']=write(tmp_path/'other_fit.json',fit)
    if mutation=='changed_bytes':Path(fit_id['path']).write_text('{}')
    if mutation:
        with pytest.raises((ValueError,RuntimeError)):m.validate_source_receipt(c,'fixed',src)
    else:m.validate_source_receipt(c,'fixed',src)


@pytest.mark.parametrize('fail_check',[None,1,2])
def test_runtime_revalidated_after_trainer_before_success(tmp_path,monkeypatch,fail_check):
    c={'workspace_ids':['fixed'],'pilot_workspace':'fixed','freeze_id':'new','smoke_iters':16,
       'training_python':'/pinned/python','training_dependency_root':'/overlay',
       'torch_extensions_root':'/extensions','evidence_root':str(tmp_path)}
    config=tmp_path/'config.json';write(config,c)
    monkeypatch.setattr(m,'context',lambda *a:(c,{'commit':'current'}, {'contract_sha256':'e0'}))
    monkeypatch.setattr(m,'validate_prepared',lambda *a:{'public_rgb':{'a':{},'b':{}}})
    monkeypatch.setenv('SLURMD_NODENAME','sof1-test');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'test allocated GPU')
    events=[]
    def verify(*args):
        events.append('runtime')
        if events.count('runtime')==fail_check:raise ValueError('GS runtime closure changed')
    monkeypatch.setattr(m,'verify_training_runtime',verify)
    def trainer(command,**kwargs):
        events.append('trainer')
        out=Path(command[command.index('--out')+1]);out.write_bytes(b'preserved model')
        write(out.parent/'train_report.json',{'iters':16,'seed':0,'independent_heldout_evaluation':False,
          'bitwise_determinism_claimed':False,'n_gaussians':10,'wall_s':1.,'psnr_internal_diagnostic':20.,
          'gradient_train_frames':['a'],'internal_diagnostic_frames':['b'],'n_train':1,'n_holdout':1})
    monkeypatch.setattr(m.subprocess,'run',trainer)
    if fail_check:
        with pytest.raises(ValueError,match='runtime closure changed'):m.run(config,tmp_path/'new','fixed','train-smoke')
    else:m.run(config,tmp_path/'new','fixed','train-smoke')
    receipt=json.loads((tmp_path/'new/gaussian/fixed/train-smoke_receipt.json').read_text())
    assert receipt['status']==('FAIL' if fail_check else 'PASS')
    assert events==(['runtime'] if fail_check==1 else ['runtime','trainer','runtime'])
    if fail_check==2:
        assert (tmp_path/'new/gaussian/fixed/train-smoke/scene.ply').read_bytes()==b'preserved model'
        assert 'training_report' not in receipt
