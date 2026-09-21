"""Stage formatting authenticates real files and retains missing/failed slots."""
import copy
import json
from pathlib import Path
import pytest
import yaml
from robo.eval import paper_droid_stages as d, paper_pipeline as p
from robo.manifest.hash import canonical_hash
from agents.recon.droid_extract import input_identity
from tests.test_paper_droid_cpu import fixture as cpu_fixture
from tests.test_paper_pipeline_audit import fixture_config


def put(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data))
    return input_identity(path)


def fixture(tmp_path,monkeypatch):
    cpu=cpu_fixture(tmp_path)
    ids=[r['workspace_id'] for r in json.loads(Path(cpu['path']).read_text())['rows']]
    code=dict(repository=str(tmp_path/'code'),commit='a'*40,dirty=False)
    monkeypatch.setattr(d,'git_snapshot',lambda _:code)
    roots=[tmp_path/'gs',tmp_path/'tail'];audits=[];configs=[]
    for root,resource,scope in zip(roots,['e7_gaussian_config','e7_tail_config'],
            ['complete_original_droid_gaussian_integrity','complete_original_droid_public_tail_integrity']):
        config=dict(workspace_ids=ids)
        if resource=='e7_gaussian_config':
            cpu_root=Path(cpu['path']).parent.parent
            config['cpu_source']=dict(stage_root=str(cpu_root),commit=code['commit'],contract=input_identity(cpu_root/'contract/freeze_manifest.json'))
        if resource=='e7_tail_config':
            config['gaussian_source']=dict(config=audits[0]['config'],commit=code['commit'],stage_root=str(roots[0]),
                contract=input_identity(roots[0]/'contract/freeze_manifest.json'))
        ref=put(root/'config.json',config);configs.append(config)
        e=dict(freeze_id=root.name,code=code,resource_inventory=[dict(id=resource,sha256=ref['sha256'])])
        e['contract_sha256']=canonical_hash(e);put(root/'contract/freeze_manifest.json',e)
        audits.append(dict(schema_version=1,scope=scope,status='PASS',freeze_id=root.name,
            config=ref,contract_sha256=e['contract_sha256'],source_commit=code['commit'],planned_workspaces=10,
            paper_ready=False,full_build=False,full_e7_complete=False,independent_image_fidelity=False,rows=[]))
    gs,tail=audits
    for i,wid in enumerate(ids):
        gd=roots[0]/'gaussian'/wid;td=roots[1]/'public_tail'/wid;td.mkdir(parents=True)
        eligible=i<5
        base=dict(code=code,config=gs['config'],workspace_id=wid)
        prep=dict(**base,status='PASS' if eligible else 'NOT_RUN',public_rgb={'a':{},'b':{}})
        gr=dict(workspace_id=wid,preparation=put(gd/'prepare_receipt.json',prep),gaussian_status='PASS' if eligible else 'NOT_RUN',
            train_fit_available=eligible,full_build=False,independent_image_fidelity=False)
        if eligible:
            report=dict(iters=30000,seed=0,independent_heldout_evaluation=False,bitwise_determinism_claimed=False,
                n_gaussians=1,wall_s=1.,psnr_internal_diagnostic=10.,gradient_train_frames=['a'],internal_diagnostic_frames=['b'],n_train=1,n_holdout=1)
            put(gd/'train/train_report.json',report);(gd/'train/scene.ply').write_bytes(b'fixture Gaussian bytes')
            train=dict(**base,status='PASS',training_report=report,artifacts=d.tree(gd/'train'))
            gr.update(training_receipt=put(gd/'train_receipt.json',train),n_gaussians=1)
        gs['rows'].append(gr)
        status='FAIL' if i==4 else 'PASS' if eligible else 'NOT_RUN'
        n=(2 if i==3 else 0) if status=='PASS' else None
        stages=[dict(stage=s,returncode=0) for s in ['render','fuse','discover','prepare']]
        original=dict(source_commit=gs['source_commit'],contract_sha256=gs['contract_sha256'],available=eligible)
        if eligible:original['training']=gr['training_receipt']
        r=dict(code=code,config=tail['config'],workspace_id=wid,status=status,discovered_instances=n,prepared_instances=n,
            jobs=[dict(prepared=True) for _ in range(n)] if n is not None else None,
            full_build=False,paper_ready=False,original_gaussian=original,stages=stages,reason=None if status=='PASS' else 'unavailable',
            artifacts_without_receipt=d.tree(td))
        ref=put(td/'tail_receipt.json',r)
        tail['rows'].append({k:r[k] for k in ['workspace_id','status','discovered_instances','prepared_instances','jobs','full_build','paper_ready','reason']})
        tail['rows'][-1].update(receipts={'tail_receipt.json':ref},phase_executions=stages)
    gs.update(valid_train_fits=5,gaussian_completed=5,gaussian_failed=0,gaussian_not_run=5)
    tail.update(processing_complete=4,processing_failed=1,not_run=5,discovered_instances=2,prepared_instances=2,
        null_instance_count_workspaces=6,qualification_or_policy_success=None)
    gp=roots[0]/'full_gaussian_completion_audit.json';tp=roots[1]/'full_public_tail_completion_audit.json'
    spec=put(tp,tail);spec['completion_audit']=copy.deepcopy(spec);spec['gaussian_completion_audit']=put(gp,gs)
    return cpu,spec,tp,tail,gp,gs,ids


def test_stages_preserve_all_ten_and_null_counts(tmp_path,monkeypatch):
    cpu,spec,tp,t,gp,g,ids=fixture(tmp_path,monkeypatch)
    rows=d.validate_stages(tp,t,gp,g,ids)['rows']
    assert len(rows)==10 and sum(r['status']=='FAIL' for r in rows)==1
    assert [r['prepared_instances'] for r in rows]==[0,0,0,2,None,None,None,None,None,None]
    path,c=fixture_config(tmp_path);c['engineering_appendix']={'droid_public_construction':spec,'droid_cpu_alignment':cpu}
    path.write_text(yaml.safe_dump(c));out=tmp_path/'paper';p.generate(path,out)
    tex=(out/'generated_tables/droid_cpu_alignment_engineering.tex').read_text()
    assert 'GS / prep.' in tex and 'C / 0' in tex and 'C / --' in tex and '-- / --' in tex
    assert 'not verified distinct objects' in tex
    assert 'rows[9].prepared_instances' in (out/'claim_ledger.csv').read_text()


@pytest.mark.parametrize('kind',['roster','scope','claim','policy','missing_gs','changed_gs','train_roster','count','null','source','tail_bytes','aggregate','availability'])
def test_stages_reject_tampering(tmp_path,monkeypatch,kind):
    _,_,tp,t,gp,g,ids=fixture(tmp_path,monkeypatch)
    if kind=='roster':ids.reverse()
    elif kind=='scope':g['scope']=t['scope']
    elif kind=='claim':g['rows'][0]['full_build']=True
    elif kind=='policy':t['qualification_or_policy_success']=1
    elif kind in {'missing_gs','changed_gs'}:
        path=gp.parent/'gaussian'/ids[0]/'train/scene.ply'
        if kind=='missing_gs':path.unlink()
        else:path.write_bytes(b'changed')
    elif kind=='train_roster':
        # Even a rehashed, mutually consistent receipt cannot redefine the recipe.
        gd=gp.parent/'gaussian'/ids[0];tr=json.loads((gd/'train_receipt.json').read_text())
        tr['training_report']['gradient_train_frames']=['b'];put(gd/'train/train_report.json',tr['training_report'])
        tr['artifacts']=d.tree(gd/'train');g['rows'][0]['training_receipt']=put(gd/'train_receipt.json',tr)
    elif kind=='count':t['rows'][0]['prepared_instances']=1
    elif kind=='null':t['rows'][4]['prepared_instances']=0
    elif kind=='source':g['source_commit']='b'*40
    elif kind=='tail_bytes':(tp.parent/'public_tail'/ids[0]/'injected.json').write_text('{}')
    elif kind=='aggregate':t['processing_complete']=5
    elif kind=='availability':g['rows'][5]['train_fit_available']=True
    with pytest.raises((ValueError,FileNotFoundError)):d.validate_stages(tp,t,gp,g,ids)


def test_stages_require_cpu_cohort():
    with pytest.raises(ValueError,match='complete CPU'):p._engineering_sources({'droid_public_construction':{}})


def test_gaussian_cannot_be_rebound_to_another_cpu_source(tmp_path,monkeypatch):
    _,_,tp,t,gp,g,ids=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='another CPU source'):
        d.validate_stages(tp,t,gp,g,ids,dict(stage_root=str(tmp_path/'another'),commit='a'*40))
