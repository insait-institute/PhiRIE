"""TRAIN structural admission cannot become outcome-based room filtering."""
import copy
import json
from pathlib import Path
import pytest
from run.icra2027 import e2_full_preparation as p


def report(*,accept=True,plane='FIT',views=None,pixels=None):
    row=dict(object_slot='obj_1000',terminal_action='accept' if accept else 'abstain',plane_status=plane,
             selected_frames=['TRAIN.JPG'] if views is None else views,
             projected_mask_pixels=[20] if pixels is None else pixels)
    return dict(objects=[row],planned_objects=1,accepted_objects=int(accept),official_test_images_read=0)

@pytest.mark.parametrize('r,state,reason',[
    (report(),'STRUCTURALLY_FILLABLE',None),
    (report(accept=False),'NO_ACCEPTED_OBJECTS',None),
    (report(plane='NO_PLANE'),'BLOCKED_UNFILLABLE_ACCEPTED','NO_PLANE'),
    (report(views=[]),'BLOCKED_UNFILLABLE_ACCEPTED','NO_VIEW'),
    (report(pixels=[0]),'BLOCKED_UNFILLABLE_ACCEPTED','EMPTY_PRIMARY_PROJECTED_MASK')])
def test_predeclared_structural_gate(r,state,reason):
    observed,blocked=p.structural_state(r);assert observed==state
    assert (blocked[0]['reasons'][0] if blocked else None)==reason


def test_one_missing_required_object_blocks_whole_scene():
    r=report();r['objects']+=report(plane='NO_PLANE')['objects']
    assert p.structural_state(r)[0]=='BLOCKED_UNFILLABLE_ACCEPTED'


@pytest.fixture
def cohort(tmp_path,monkeypatch):
    scenes=[f'{i:010x}' for i in range(50)];root=tmp_path/'fidelity';root.mkdir()
    config=dict(freeze_id='20260906-abcdef1-v1',scene_ids=scenes[1:],contexts={},pilot_preparation={'scene':scenes[0]},full_protocol={},background_admission={})
    contract=dict(code={'commit':'a'*40});protocol=dict(scene_ids=scenes,pilot_scene=scenes[0],population=dict.fromkeys(scenes,1))
    reports={s:report() for s in scenes};reports[scenes[0]]=report(plane='NO_PLANE');reports[scenes[1]]=report(accept=False)
    for s in scenes[1:]:
        context_path=tmp_path/(s+'.json');context_path.write_text(json.dumps(dict(scene_id=s,freeze_id=config['freeze_id'])))
        config['contexts'][s]=p._identity(context_path)
        d=root/'removal'/s/'inpaint';d.mkdir(parents=True);(d/'seal.json').write_text('{}')
    cp=tmp_path/'config.json';ep=tmp_path/'e0.json';cp.write_text(json.dumps(config));ep.write_text(json.dumps(contract))
    monkeypatch.setattr(p,'context',lambda *_:(config,contract,protocol,root))
    monkeypatch.setattr(p,'preparation_spec',lambda cp,*_:{'scene':cp.stem})
    monkeypatch.setattr(p,'_preparation',lambda spec,scene:dict(report=copy.deepcopy(reports[scene]),jobs=[{}]))
    return cp,ep,root,scenes,reports,config


def test_full_inventory_retains_all_scenes_and_chooses_first_supported(cohort):
    cp,ep,root,scenes,reports,c=cohort;r=p.inventory(cp,ep)
    assert len(r['rows'])==50 and r['planned_test_views']==400
    assert r['rows'][0]['status']=='BLOCKED_UNFILLABLE_ACCEPTED'
    assert r['rows'][0]['missing_test_views']==8
    assert r['model_pilot_scene']==scenes[2]
    assert r['rows'][1]['accepted_objects']==0
    with pytest.raises(FileExistsError):p.inventory(cp,ep)


def test_incomplete_inventory_waits_without_publishing(cohort):
    cp,ep,root,scenes,*_=cohort
    (root/'removal'/scenes[-1]/'inpaint/seal.json').unlink()
    assert p.inventory(cp,ep)['status']=='WAITING'
    assert not (root/'preparation_inventory').exists()


def test_no_fillable_scene_keeps_full_denominator(cohort):
    cp,ep,root,scenes,reports,_=cohort
    for s in scenes:reports[s]=report(plane='NO_PLANE')
    r=p.inventory(cp,ep);assert r['model_pilot_scene'] is None and len(r['rows'])==50


def test_existing_preparation_cannot_create_failure_receipt(cohort):
    cp,ep,root,scenes,*_=cohort
    with pytest.raises(FileExistsError):p.run(cp,ep,scenes[1])
    assert not (root/'preparation_failures').exists()


def test_pilot_prefix_need_not_wait_for_later_scenes(cohort):
    cp,ep,root,scenes,*_=cohort
    (root/'removal'/scenes[-1]/'inpaint/seal.json').unlink()
    r=p.inventory(cp,ep,pilot_only=True)
    assert r['model_pilot_scene']==scenes[2] and len(r['authenticated_prefix'])==3
    assert r['full_inventory_complete'] is False and r['planned_full_test_views']==400


def test_pilot_prefix_cannot_skip_missing_earlier_scene(cohort):
    cp,ep,root,scenes,*_=cohort
    (root/'removal'/scenes[1]/'inpaint/seal.json').unlink()
    r=p.inventory(cp,ep,pilot_only=True)
    assert r['status']=='WAITING' and r['missing_prefix_scene']==scenes[1]
    assert not (root/'model_pilot_admission').exists()


def test_config_preparation_waits_without_allocating_a_freeze(tmp_path,monkeypatch):
    cp=tmp_path/'pilot.json';bp=tmp_path/'admission.yaml';mp=tmp_path/'extra.yaml'
    bp.write_text('test');mp.write_text('{"freeze_id":"20260906-abcdef1-v1","scene_ids":[]}')
    pilot='0000000000';scenes=[f'{i:010x}' for i in range(50)]
    cp.write_text(json.dumps(dict(status='PASS',source_commit=p.PILOT_SOURCE,planned_objects=1)))
    monkeypatch.setattr(p,'ADMISSION_SHA',p._identity(bp)['sha256'])
    monkeypatch.setattr(p.original,'validate_protocol',lambda _:dict(scene_ids=scenes,pilot_scene=pilot,population=dict.fromkeys(scenes,1),factory_freeze_root=str(tmp_path/'old')))
    target=tmp_path/'newconfigs'
    r=p.build_configs(target,full_protocol=tmp_path/'protocol',background_admission=bp,missing_factory_config=mp,pilot_gate=cp)
    assert r['status']=='WAITING' and len(r['missing_scenes'])==49 and not target.exists()

@pytest.mark.parametrize('change',['none','factory_source','factory_slots','seed','admission','pilot_source','batch_population','snapshot','missing_roster'])
def test_full_context_refuses_treatment_or_constructor_drift(cohort,tmp_path,monkeypatch,change):
    cp,ep,root,scenes,reports,c=cohort
    c.update(schema_version=1,scope=p.SCOPE,paper_ready=False)
    proto=tmp_path/'protocol.yaml';proto.write_text('{}');c['full_protocol']=p._identity(proto)
    admission=dict(scope='full_e2_training_structural_background_admission',all_planned_scenes=50,all_planned_objects=1871,
                   all_planned_test_views=400,TEST_quality_used_for_admission=False,
                   required_accepted_object_inputs=['support_plane_available','at_least_one_selected_training_view','primary_projected_mask_nonempty'],
                   mask_fill_pilot_rule='first_lexicographic_scene_with_at_least_one_accepted_object_and_all_required_inputs_available')
    ap=tmp_path/'admission.json';ap.write_text(json.dumps(admission));c['background_admission']=p._identity(ap)
    monkeypatch.setattr(p,'ADMISSION_SHA',c['background_admission']['sha256'])
    c['pilot_preparation']['producer_commit']=p.PILOT_SOURCE
    reports[scenes[0]]['source_validation']={'e3_producer_commit':p.original.CONSTRUCTION_COMMIT}
    manifests={}
    for s in scenes[1:]:
        mp=tmp_path/(s+'_materialized.json');manifest=dict(e3_code_commit=p.original.CONSTRUCTION_COMMIT,e3_freeze_id=p.original.CONSTRUCTION_FREEZE,scene_id=s,policy_id='A4',roster=dict(job_count=1,object_slots=['obj_1000']))
        mp.write_text(json.dumps(manifest));manifests[s]=mp
        local=dict(scene_id=s,freeze_id=c['freeze_id'],policy_id='A4',scope='automatic_train_only_removal_preparation',algorithm=p.PUBLIC_ALGORITHM,seed=0,materialization_manifest=p._identity(mp))
        lp=Path(c['contexts'][s]['path']);lp.write_text(json.dumps(local));c['contexts'][s]=p._identity(lp)
    c.update(unavailable_factory_scenes=[],all_planned_scenes=50,all_planned_objects=1871,all_planned_test_views=400,
             batch_rule='all_currently_available_factories_before_preparation_outcomes',
             availability_snapshot={s:p._identity(manifests[s]) for s in scenes[1:]})
    sid=scenes[1];lp=Path(c['contexts'][sid]['path']);local=json.loads(lp.read_text())
    if change in {'factory_source','factory_slots'}:
        mp=manifests[sid];m=json.loads(mp.read_text())
        if change=='factory_source':m['e3_code_commit']='b'*40
        else:m['roster']['object_slots']=['obj_9999']
        mp.write_text(json.dumps(m));local['materialization_manifest']=p._identity(mp)
    elif change=='seed':local['seed']=1
    elif change=='admission':c['background_admission']['sha256']='b'*64
    elif change=='pilot_source':c['pilot_preparation']['producer_commit']='b'*40
    elif change=='batch_population':c['all_planned_objects']=1
    elif change=='snapshot':c['availability_snapshot'][sid]=None
    elif change=='missing_roster':c['unavailable_factory_scenes']=[scenes[-1]]
    lp.write_text(json.dumps(local));c['contexts'][sid]=p._identity(lp)
    proto_obj=dict(scene_ids=scenes,pilot_scene=scenes[0],population=dict.fromkeys(scenes,1),resolved_jobs={'path':'jobs'})
    monkeypatch.setattr(p.original,'validate_protocol',lambda _:proto_obj)
    monkeypatch.setattr(p.original,'read',lambda _:dict(scenes=[dict(scene_id=s,jobs=[dict(object_slot='obj_1000')]) for s in scenes]))
    monkeypatch.setattr(p,'_bound_contract',lambda *_:(c,{'code':{'commit':'a'*40}}))
    monkeypatch.setattr(p.e3,'_validate_cli_execution',lambda *_,**__:None)
    # Undo the lightweight context double installed by the inventory fixture.
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('full_preparation_context_under_test',Path(p.__file__));actual=module_from_spec(spec);spec.loader.exec_module(actual)
    actual._bound_contract=p._bound_contract;actual._preparation=p._preparation;actual.ADMISSION_SHA=p.ADMISSION_SHA
    if change=='none':assert actual.context(cp,ep)[0]==c
    else:
        with pytest.raises(ValueError):actual.context(cp,ep)


def test_ready_batch_cannot_skip_unavailable_earlier_scene(cohort):
    cp,ep,root,scenes,reports,c=cohort
    del c['contexts'][scenes[1]];c['scene_ids'].remove(scenes[1])
    c['unavailable_factory_scenes']=[scenes[1]]
    result=p.inventory(cp,ep,pilot_only=True)
    assert result==dict(status='WAITING',missing_prefix_scene=scenes[1],paper_ready=False)
    assert not (root/'model_pilot_admission').exists()
    assert p.inventory(cp,ep)['missing_scenes']==[scenes[1]]


def test_ready_batch_releases_complete_prefix_only(cohort):
    cp,ep,root,scenes,reports,c=cohort
    del c['contexts'][scenes[-1]];c['scene_ids'].remove(scenes[-1])
    c['unavailable_factory_scenes']=[scenes[-1]]
    result=p.inventory(cp,ep,pilot_only=True)
    assert result['model_pilot_scene']==scenes[2]
    assert result['planned_full_objects']==1871 and result['planned_full_test_views']==400
    assert result['full_inventory_complete'] is False


def test_failed_prefix_cannot_select_later_model_pilot(cohort):
    cp,ep,root,scenes,reports,c=cohort;scene=scenes[1]
    (root/'removal'/scene/'inpaint/seal.json').unlink()
    failure=root/'preparation_failures'/f'{scene}.json';failure.parent.mkdir()
    failure.write_text(json.dumps(dict(status='FAILED_PREPARATION',scene_id=scene,context=c['contexts'][scene],
        code_commit='a'*40,freeze_id=c['freeze_id'],planned_objects=1,config=p._identity(cp),contract=p._identity(ep))))
    r=p.inventory(cp,ep,pilot_only=True)
    assert r['status']=='BLOCKED' and r['failed_prefix_scene']==scene
    assert not (root/'model_pilot_admission').exists()
    full=p.inventory(cp,ep)
    assert len(full['rows'])==50 and full['rows'][1]['status']=='FAILED_PREPARATION'
    assert full['model_pilot_scene'] is None and full['failed_model_pilot_prefix']==[scene]


def test_inventory_reuses_prior_preparations_without_rerunning(cohort):
    cp,ep,root,scenes,reports,c=cohort
    prior=scenes[1]
    del c['contexts'][prior];c['scene_ids'].remove(prior)
    c['prior_preparations']={prior:{'scene':prior}}
    (root/'removal'/prior/'inpaint/seal.json').unlink()
    result=p.inventory(cp,ep)
    assert len(result['rows'])==50 and result['rows'][1]['preparation']=={'scene':prior}
    assert result['model_pilot_scene']==scenes[2]


@pytest.mark.parametrize('change',['none','source','missing_row','unit_context','seal_bytes','gate_contract','planned_count','test_read'])
def test_prior_batch_source_roster_and_bytes_bound(tmp_path,monkeypatch,change):
    scene='0000000001';cp=tmp_path/'old.yaml';ep=tmp_path/'e0.json';unit=tmp_path/'unit.yaml';seal=tmp_path/'seal.json';gp=tmp_path/'gate.json'
    for path in (cp,ep,unit,seal):path.write_text('{}')
    ref=dict(config=p._identity(cp),contract=p._identity(ep))
    spec=dict(context=p._identity(unit),contract=ref['contract'],seal=p._identity(seal),producer_commit=p.PRIOR_BATCH_SOURCE)
    pc=dict(scope=p.SCOPE,scene_ids=[scene],contexts={scene:spec['context']})
    g=dict(scope='authenticated_full_e2_preparation_batch',status='PASS',source_commit=p.PRIOR_BATCH_SOURCE,config=ref['config'],contract=ref['contract'],planned_batch_scenes=1,planned_full_scenes=50,planned_full_objects=1871,planned_full_views=400,official_test_images_read=0,paper_ready=False,rows=[dict(scene_id=scene,preparation=spec)])
    if change=='source':g['source_commit']='f'*40
    elif change=='missing_row':g['rows']=[]
    elif change=='unit_context':spec['context']=ref['config']
    elif change=='seal_bytes':seal.write_text('changed')
    elif change=='gate_contract':g['contract']=ref['config']
    elif change=='planned_count':g['planned_full_objects']=1
    elif change=='test_read':g['official_test_images_read']=8
    gp.write_text(json.dumps(g));ref['integrity']=p._identity(gp)
    monkeypatch.setattr(p,'_bound_contract',lambda *_args,**_kwargs:(pc,{}))
    if change=='none':assert p.prior_batch_preparations(ref)=={scene:spec}
    else:
        with pytest.raises(ValueError):p.prior_batch_preparations(ref)
