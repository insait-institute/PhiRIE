"""Admission/reuse/failure tests for the unchanged measurement producer bridge."""
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
from run.icra2027 import e6_public_reconstruction_full as full


def old(tier='pilot'):
    d=dict(schema_version=1,tier=tier,scope='e6_rgb_only_reconstruction_'+tier,freeze_id='original',
        source_commit=full.PRODUCER_SHA,max_frames=48,seed=0,gs_iters=15000,
        execution_scene='behavior_task0020',fallback_allowed=False,roster={'path':'roster'},runtime={'path':'runtime'})
    if tier=='smoke':d['selection_rationale']='smallest_positive_clean_frame_count_then_frozen_scene_order'
    return d


def test_measurement_recipe_exhaustive_partition():
    api=full.shared; a=old(); b=old('smoke'); b['execution_scene']='behavior_task0023'; b['freeze_id']='other'
    runtime=dict(python='python',gs_python='gs',omega_source='omega',omega_checkpoint='w',da3_checkpoint='da3')
    assert full.recipe(a,runtime,api)==full.recipe(b,runtime,api)
    for key,value in [('seed',1),('max_frames',6),('gs_iters',14000),('fallback_allowed',True),('source_commit','other')]:
        changed=copy.deepcopy(a); changed[key]=value
        with pytest.raises(ValueError):full.recipe(changed,runtime,api)
    a['unknown_affecting_option']='anything'
    with pytest.raises(ValueError,match='unknown'):full.recipe(a,runtime,api)


def test_source_relabel_and_dirty_rejected(monkeypatch,tmp_path):
    with pytest.raises(ValueError,match='pinned'):full.producer({'path':str(tmp_path),'commit':'new'})
    monkeypatch.setattr(full.shared.sealed.sealed_cpu,'_inside',lambda *a,**k:None)
    monkeypatch.setattr(full.shared,'git_snapshot',lambda p:dict(commit=full.PRODUCER_SHA,dirty=True))
    with pytest.raises(ValueError,match='source changed'):full.producer({'path':str(tmp_path),'commit':full.PRODUCER_SHA})


def fixture(monkeypatch,tmp_path,reuse=False):
    stage=tmp_path/'full';stage.mkdir(); public=tmp_path/'public'; public.mkdir()
    image=public/'000.jpg';image.write_bytes(b'actual RGB fixture bytes')
    row=dict(scene_id='behavior_task0023',condition_id='mild',frames=[dict(name='000.jpg',selected=True,**full.shared.identity(image))])
    cfg=dict(scope='e6_public_rgb_full18',source_commit='wrapper',measurement_producer={'path':'oldcode','commit':full.PRODUCER_SHA},
        recipe_sha256='recipe',freeze_id=stage.name,reuse=[])
    calls=[]
    def commands(c,r,d,f):
        return [(n,'python',['-m','existing.'+n,'--out',str(d/n)]) for n in ('omega','metricize','scene','gaussian')]
    api=SimpleNamespace(CODE=tmp_path/'oldcode',commands=commands,environment=lambda *a:{'PINNED':'true'},
                        validate_products=lambda *a:calls.append('products'))
    if reuse:
        origin=tmp_path/'original';unit=origin/'audit/public_reconstruction/behavior_task0023_mild'
        (unit/'terminal').mkdir(parents=True)
        full.shared.write_new_json(unit/'terminal/gate.json',dict(stage_status='PASS',stage_records=[{'original':'yes'}],wall_s=123.))
        cfg['reuse']=[dict(scene_id=row['scene_id'],condition_id=row['condition_id'],config={'path':str(origin/'execution.json')},
                          terminal_manifest={'sha256':'originalmanifest'},terminal_seal={'sha256':'originalseal'})]
        monkeypatch.setattr(full,'replay',lambda *a:dict(returncode=0,command=['original validator']))
    monkeypatch.setattr(full,'validate',lambda *a,**k:(cfg,{},dict(rows=[row]),api))
    monkeypatch.setattr(full.shared,'ROOT',tmp_path)
    return stage,cfg,row,api,calls


def test_fresh_uses_pinned_cwd_and_preserves_typed_failure(monkeypatch,tmp_path):
    stage,cfg,row,api,calls=fixture(monkeypatch,tmp_path)
    def fail(cmd,**kwargs):
        assert kwargs['cwd']==api.CODE and kwargs['env']=={'PINNED':'true'}
        calls.append(cmd);return SimpleNamespace(returncode=9)
    monkeypatch.setattr(full.subprocess,'run',fail)
    gate=full.execute('config',stage,row['scene_id'],row['condition_id'])
    assert gate['stage_status']=='FAIL' and gate['failed_stage']=='omega'
    assert gate['failure_type']=='producer_nonzero_exit' and len(gate['stage_records'])==1
    assert gate['feature_rows_written']==0 and gate['paper_ready'] is False
    assert len(calls)==1
    assert full.validate_output('config',stage,row['scene_id'],row['condition_id'])==gate
    with pytest.raises(FileExistsError):full.execute('config',stage,row['scene_id'],row['condition_id'])


def test_reuse_is_reference_not_new_measurement(monkeypatch,tmp_path):
    stage,cfg,row,api,calls=fixture(monkeypatch,tmp_path,reuse=True)
    monkeypatch.setattr(full.subprocess,'run',lambda *a,**k:pytest.fail('no measurement invocation allowed'))
    gate=full.execute('config',stage,row['scene_id'],row['condition_id'])
    assert gate['measurement_wall_s']==123 and gate['stage_records']==[{'original':'yes'}]
    assert gate['measurement_producer']['commit']==full.PRODUCER_SHA and gate['wrapper_source_commit']=='wrapper'
    assert full.validate_output('config',stage,row['scene_id'],row['condition_id'])==gate
    dest=stage/'audit/public_reconstruction/behavior_task0023_mild'
    (dest/'input_manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='content differs'):
        full.validate_output('config',stage,row['scene_id'],row['condition_id'])


def test_reuse_original_validator_failure_stays_unpublished(monkeypatch,tmp_path):
    stage,cfg,row,api,calls=fixture(monkeypatch,tmp_path,reuse=True)
    def failure(*a):raise ValueError('original artifact tampered')
    monkeypatch.setattr(full,'replay',failure)
    with pytest.raises(ValueError,match='tampered'):full.execute('config',stage,row['scene_id'],row['condition_id'])
    assert not (stage/'audit/public_reconstruction/behavior_task0023_mild/terminal').exists()


def test_unknown_unit_is_not_substituted():
    with pytest.raises(ValueError,match='outside'):full.unit_row({'rows':[]},'scene','clean')


def test_cohort_preserves_all_unrun_and_partial_queries(monkeypatch,tmp_path):
    stage,cfg,row,api,calls=fixture(monkeypatch,tmp_path)
    rows=[dict(scene_id=s,condition_id=c,source_frame_present=False) for s in full.shared.SCENES for c in full.shared.CONDITIONS]
    monkeypatch.setattr(full,'validate',lambda *a,**k:(cfg,{},dict(rows=rows),api))
    (stage/'audit/public_reconstruction/behavior_task0020_clean').mkdir(parents=True)
    report=full.publish_cohort('config',stage,stage/'snapshot1')
    assert report['planned_conditions']==18 and report['planned_query_rows']==72
    assert len(report['units'])==18 and sum(r['planned_queries'] for r in report['units'])==72
    assert report['counts']==dict(PASS=0,FAIL=0,INCOMPLETE=1,NOT_RUN=17)
    assert all(r['terminal_manifest'] is None for r in report['units'])
    with pytest.raises(FileExistsError):full.publish_cohort('config',stage,stage/'snapshot1')


def admission(monkeypatch,tmp_path):
    root=tmp_path;stage=root/'outputs/icra2027/20260906-abcdef0-v1';stage.mkdir(parents=True)
    runtime=dict(python='python',gs_python='gs',omega_source='omega',omega_checkpoint='w',da3_checkpoint='da3',checkpoints=[])
    full.shared.write_new_json(root/'runtime.json',runtime); rt=full.shared.identity(root/'runtime.json')
    full.shared.write_new_json(root/'roster.json',dict(rows=[]));roster=full.shared.identity(root/'roster.json')
    refs=[]
    for tier,scene in [('smoke','behavior_task0023'),('pilot','behavior_task0020')]:
        original=root/tier;original.mkdir(); c=old(tier);c.update(runtime=rt,roster=roster,execution_scene=scene)
        full.shared.write_new_json(original/'execution.json',c);refs.append(full.shared.identity(original/'execution.json'))
    reuse=[]
    for scene,condition in full.REUSED:
        ref=refs[0 if scene=='behavior_task0023' else 1]
        terminal=Path(ref['path']).parent/'audit/public_reconstruction'/(scene+'_'+condition)/'terminal'
        terminal.mkdir(parents=True)
        for name in ('manifest.json','seal.json'):(terminal/name).write_text('{}')
        reuse.append(dict(scene_id=scene,condition_id=condition,config=ref,
                          terminal_manifest=full.shared.identity(terminal/'manifest.json'),terminal_seal=full.shared.identity(terminal/'seal.json')))
    api=SimpleNamespace(SCENES=full.shared.SCENES,RULE=full.shared.RULE,commands=full.shared.commands,validate_roster=lambda r:r)
    cfg=dict(schema_version=1,scope='e6_public_rgb_full18',freeze_id=stage.name,source_commit='wrapper',
        measurement_producer=dict(path='oldcode',commit=full.PRODUCER_SHA),reference_configs=refs,roster=roster,runtime=rt,
        recipe_sha256=full.shared.canonical_hash(full.recipe(full.shared.read(refs[0]['path']),runtime,api)),reuse=reuse)
    path=stage/'execution.json';full.shared.write_new_json(path,cfg)
    bindings={'e6_full_config':full.shared.identity(path),'e6_rgb_roster':roster,'e6_runtime':rt,
              'e6_original_smoke':refs[0],'e6_original_pilot':refs[1]}
    contract=dict(freeze_id=stage.name,code=dict(commit='wrapper',dirty=False),
                  resource_inventory=[dict(id=k,resolved_path=v['path'],sha256=v['sha256']) for k,v in bindings.items()])
    contract['contract_sha256']=full.shared.canonical_hash(contract)
    (stage/'contract').mkdir();full.shared.write_new_json(stage/'contract/freeze_manifest.json',contract)
    monkeypatch.setattr(full.shared,'ROOT',root);monkeypatch.setattr(full.shared,'git_snapshot',lambda *a:dict(commit='wrapper',dirty=False))
    monkeypatch.setattr(full,'producer',lambda *a:api);monkeypatch.setattr(full,'replay',lambda *a:dict(returncode=0))
    return stage,path,cfg


def test_full_e0_binding_accepts_exact_originals(monkeypatch,tmp_path):
    stage,path,cfg=admission(monkeypatch,tmp_path)
    assert full.validate(path,stage)[0]==cfg


@pytest.mark.parametrize('drift',['config','seal','runtime','roster','source','recipe','reuse_subset','unknown'])
def test_full_admission_rejects_any_unbound_drift(monkeypatch,tmp_path,drift):
    stage,path,cfg=admission(monkeypatch,tmp_path)
    if drift=='config':cfg['freeze_id']='another'
    elif drift=='seal':Path(cfg['reuse'][0]['terminal_seal']['path']).write_text('changed')
    elif drift in {'runtime','roster'}:Path(cfg[drift]['path']).write_text('{}')
    elif drift=='source':cfg['source_commit']='different'
    elif drift=='recipe':cfg['recipe_sha256']='different'
    elif drift=='reuse_subset':cfg['reuse'].pop()
    elif drift=='unknown':cfg['unbound_flag']=True
    path.write_text(full.json.dumps(cfg))
    with pytest.raises(ValueError):full.validate(path,stage)
