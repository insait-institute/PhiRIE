"""Paper coverage never converts missing prerequisites into measured failures."""
import copy
import json
import pytest
from robo.eval import paper_full_qualification as paper
from run.icra2027 import e4_planning_terminal as terminal


@pytest.fixture
def cohort(monkeypatch):
    scenes={};tasks=[];cells=[]
    for i in range(50):
        sid=f'{i:010d}';n=(7 if i<23 else 6) if i<41 else 0
        selected=[f'{sid}__obj_{j}_to_region' for j in range(n)]
        status=('EXPORT_REJECTED' if i==0 else 'PLANNING_UNAVAILABLE' if i==1
                else 'QUALIFICATION_COMPLETE' if n else 'SOURCE_NO_QUERIES')
        scenes[sid]=dict(status=status,selected_task_ids=selected,planned_cells=n*10,
            actual_900_step_cells=0,qualified_cells=0,strict_pass_task_ids=[],
            policy_executed=0,policy_success=None)
        for task in selected:
            tasks.append(dict(task_id=task,scene_id=sid))
            for arm in ('A0','A4'):
                for ep in range(5):
                    cid=f'{arm.lower()}__{task}__seed0__ep{ep}'
                    measured=status=='QUALIFICATION_COMPLETE'
                    cells.append(dict(scene_id=sid,task_id=task,policy_id=arm,episode=ep,cell_id=cid,
                        qualification_state='FAIL' if measured else 'NOT_RUN',
                        source_kind='canonical_qualifier' if measured else status.lower(),
                        qualification_evidence=dict(cell_id=cid,passed=False) if measured else None,
                        policy_executed=False,policy_success=None,reset_definition=None,camera=None,physics=None))
    source={k:v for k,v in paper.FIXED.items() if k!='policy_target_episodes'}
    source['source']=dict(scenes={s:{} for s in scenes})
    protocol=dict(qualification_tasks=tasks)
    observed=dict(source_config=source,protocol=protocol,scenes=scenes,cells=cells)
    config=dict(jobs={s:dict(elapsed_seconds=1) for s in scenes},source={},freeze_id='fresh',
                prior_attempts={next(iter(scenes)):[dict(elapsed_seconds=3)]})
    monkeypatch.setattr(terminal,'_full_replay',lambda _:copy.deepcopy(observed))
    gate,scenes,cells=terminal._full_payload(config,{})
    return gate,scenes,cells,protocol,config


def test_metadata_counts_keep_missing_cells_and_null_policy(cohort):
    paper.validate_metadata(*cohort)
    gate=cohort[0]
    assert gate['planned_qualification_cells']==2690 and gate['prerequisite_checked_cells']==2550
    assert gate['scheduler_elapsed_seconds']==53 and gate['policy_success'] is None


@pytest.mark.parametrize('change',['drop','duplicate','replace','drop_scene','drop_task','fake_policy',
    'fake_reset','fake_pass','fake_steps','fake_summary','fake_strict','fake_source_only',
    'drop_input','drop_budget','wrong_cell_id','wrong_scene_status','hide_prior_runtime',
    'measured_missing','promote','success_zero'])
def test_metadata_rejects_unsupported_resealed_claims(cohort,change):
    gate,scenes,cells,protocol,config=cohort;r=cells[0];scene=scenes[r['scene_id']]
    if change=='drop':cells.pop()
    elif change=='duplicate':cells[-1]=copy.deepcopy(r)
    elif change=='replace':r['task_id']='new_task'
    elif change=='drop_scene':scenes.pop('0000000049')
    elif change=='drop_task':protocol['qualification_tasks'].pop()
    elif change=='fake_policy':r['policy_executed']=True
    elif change=='fake_reset':r['reset_definition']={}
    elif change=='fake_pass':r['qualification_state']='PASS'
    elif change=='fake_steps':scene['actual_900_step_cells']=1
    elif change=='fake_summary':gate['qualified_cells']=1
    elif change=='fake_strict':scene['strict_pass_task_ids']=[r['task_id']]
    elif change=='fake_source_only':scene['status']='SOURCE_NO_QUERIES'
    elif change=='drop_input':gate['planned_objects']=399
    elif change=='drop_budget':gate['budget_exclusions']=0
    elif change=='wrong_cell_id':r['cell_id']='new'
    elif change=='wrong_scene_status':scene['status']='PASS'
    elif change=='hide_prior_runtime':gate['scheduler_elapsed_seconds']=50
    elif change=='measured_missing':r['qualification_state']='FAIL'
    elif change=='promote':gate['paper_ready']=True
    elif change=='success_zero':gate['policy_success']=0
    with pytest.raises(ValueError):paper.validate_metadata(*cohort)


def test_hash_and_alias_fail_before_reading_source(tmp_path):
    p=tmp_path/'gate.json';p.write_text('{}')
    ref=terminal.api.identity(p)
    assert paper.checked(ref)==p
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='bytes changed'):paper.checked(ref)
    alias=tmp_path/'alias.json';alias.symlink_to(p)
    with pytest.raises(ValueError,match='aliased'):paper.checked({**ref,'path':str(alias)})


def test_publisher_binds_table_values_to_original_gate(cohort,tmp_path,monkeypatch):
    # The complete authentication entrypoint is separately required; this test
    # isolates existing paper formatter integration and source-field lineage.
    from tests.test_paper_pipeline_audit import fixture_config
    from robo.eval.paper_pipeline import generate
    import yaml
    path,config=fixture_config(tmp_path)
    gate=cohort[0];g=tmp_path/'gate.json';g.write_text(json.dumps(gate))
    qa=tmp_path/'completion_audit.json';qa.write_text('{}')
    config['engineering_appendix']={'full_qualification':{
        **terminal.api.identity(g),'completion_audit':terminal.api.identity(qa)}}
    path.write_text(yaml.safe_dump(config))
    calls=[]
    monkeypatch.setattr(paper,'validate_source',lambda *a:calls.append(a) or {})
    out=tmp_path/'out';generate(path,out,tmp_path/'paper')
    assert len(calls)==1
    text=(out/'generated_tables/full_qualification_engineering.tex').read_text()
    assert '50 & 1871 & 269/6155 & 2690 & 2550 & 0 & 0 & 0' in text
    ledger=(out/'claim_ledger.csv').read_text()
    assert 'input_semantic_queries' in ledger
    assert 'planned_qualification_cells' in ledger and terminal.api.identity(g)['sha256'] in ledger
    with pytest.raises(FileExistsError):generate(path,out,tmp_path/'paper')


@pytest.fixture
def sealed(cohort,tmp_path,monkeypatch):
    gate,scenes,cells,protocol,config=cohort
    stage=tmp_path/'fresh';stage.mkdir();out=stage/'full_qualification'
    terminal.screen._publish_bundle(out,manifest_kind=terminal.FULL_SCOPE,payloads={
        'gate.json':terminal.screen._json_bytes(gate),
        'scene_receipts.json':terminal.screen._json_bytes(scenes),
        'planned_qualification_cells.jsonl':(''.join(json.dumps(r)+'\n' for r in cells)).encode()},
        manifest_fields={'code':{},'freeze_id':'fresh'})
    ref=terminal.api.identity
    qa=dict(schema_version=1,status='PASS',scope=terminal.FULL_SCOPE,full_output=str(out),
        paper_ready=False,headline_eligible=False,claim_gate='NOT_RUN',
        members={n:ref(out/n) for n in paper.MEMBERS},manifest=ref(out/'manifest.json'),seal=ref(out/'seal.json'),
        validated_gate={**gate,'manifest_sha256':ref(out/'manifest.json')['sha256'],
                        'seal_sha256':ref(out/'seal.json')['sha256']},
        source={'commit':'a'*40},command=['python','validate-full',str(out),'a'*40],
        job_id='123',hostname='synthetic',runtime_seconds=1,config={},E0={})
    driver=stage/'driver.py';driver.write_text('# synthetic metadata fixture')
    process=stage/'validation_process.json'
    process.write_text(json.dumps(dict(returncode=0,command=qa['command'],runtime_seconds=1,
                                      stdout=json.dumps(qa['validated_gate']))))
    qa.update(driver=ref(driver),validation_process=ref(process))
    audit=stage/'completion_audit.json';audit.write_text(json.dumps(qa))
    spec={**ref(out/'gate.json'),'completion_audit':ref(audit)}
    monkeypatch.setattr(paper,'_contexts',lambda *_:(config,protocol,stage))
    monkeypatch.setattr(terminal.screen,'evidence_root',lambda:tmp_path)
    return spec,out/'gate.json',gate,audit,qa


def test_pinned_audit_and_canonical_bundle_are_required(sealed):
    result=paper.validate_source(*sealed)
    assert 'metadata-only' in result['verification_scope']


@pytest.mark.parametrize('change',['member','added_member','seal','audit','replayed_gate',
    'audit_promotion','receipt_missing_command','receipt_nonfinite','missing_member','alias','driver','process'])
def test_full_entrypoint_rejects_tamper(sealed,change):
    spec,path,gate,audit,qa=sealed
    if change=='member':(path.parent/'planned_qualification_cells.jsonl').write_text('')
    elif change=='added_member':(path.parent/'extra.json').write_text('{}')
    elif change=='seal':(path.parent/'seal.json').write_text('{}')
    elif change=='audit':audit.write_text('{}')
    elif change=='replayed_gate':qa['validated_gate']['qualified_cells']=1
    elif change=='audit_promotion':qa['paper_ready']=True
    elif change=='receipt_missing_command':qa['command']=[]
    elif change=='receipt_nonfinite':qa['runtime_seconds']=float('nan')
    elif change=='missing_member':qa['members'].pop('gate.json')
    elif change=='driver':paper.checked(qa['driver']).write_text('modified driver')
    elif change=='process':paper.checked(qa['validation_process']).write_text('{}')
    else:
        alias=path.parent.parent/'alias.json';alias.symlink_to(path)
        spec['path']=str(alias)
    if change in ['replayed_gate','audit_promotion','receipt_missing_command','receipt_nonfinite','missing_member']:
        audit.write_text(json.dumps(qa));spec['completion_audit']=terminal.api.identity(audit)
    with pytest.raises((ValueError,terminal.screen.CandidateScreenError)):
        paper.validate_source(spec,path,gate,audit,qa)
