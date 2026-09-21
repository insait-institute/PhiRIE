import json
from pathlib import Path
import pytest
import yaml
from PIL import Image
from interface import demo_agentic as d


def paper():
    return {'sources':{'agentic':{'payload':{'rows':[{'coverage':.5}]}}}}


def shot(source='still',role='capture',duration=90,**extra):
    return dict(shot_id='one',source_id=source,role=role,duration_s=duration,overlay_text=['Measured evidence'],**extra)


def test_exact_three_profiles_and_static_loop():
    sources={'still':{'kind':'image'}}
    for name,seconds in d.FINAL_DURATIONS.items():
        assert d._final_timeline([shot(duration=seconds)],sources,paper(),name)[0]['duration_s']==seconds
    with pytest.raises(d.DemoError,match='duration differs'):
        d._final_timeline([shot(duration=89)],sources,paper(),'hero')
    with pytest.raises(d.DemoError,match='cannot loop'):
        d._final_timeline([shot(duration=8)],{'still':{'kind':'video'}},paper(),'loop')


def test_numeric_overlays_only_from_exact_machine_fields():
    field={'source':'agentic','pointer':['rows',0,'coverage'],'label':'Coverage','format':'.1%'}
    assert d._result_text(field,paper())=='Coverage 50.0%'
    with pytest.raises(d.DemoError,match='numeric text'):
        d._final_timeline([{**shot(),'overlay_text':['Coverage 100 percent']}],{'still':{'kind':'image'}},paper(),'hero')
    with pytest.raises(d.DemoError,match='outside frozen results'):
        d._final_timeline([{**shot(),'metric_fields':[field]}],{'still':{'kind':'image'}},paper(),'hero')
    with pytest.raises(d.DemoError,match='rounding'):
        d._result_text({**field,'format':'.10e'},paper())


def test_episode_segments_cannot_splice_skip_reverse_or_fake_success():
    sources={'a':{'kind':'synchronized_episode','episode':{'episode_id':'a','frame_hz':15,'ticks':1500,'success':False}},
             'b':{'kind':'synchronized_episode','episode':{'episode_id':'b','frame_hz':15,'ticks':1500,'success':True}}}
    first=shot('a','synchronized_episode',45,source_start_s=0)
    second={**shot('a','synchronized_episode',45,source_start_s=45),'shot_id':'two'}
    d._final_timeline([first,second],sources,paper(),'hero')
    for changed,match in [({**second,'source_id':'b'},'multiple episodes'),
                          ({**second,'source_start_s':44},'skip, overlap'),
                          ({**second,'overlay_text':['Successful placement']},'claims success'),
                          ({**second,'source_start_s':90},'extends past')]:
        with pytest.raises(d.DemoError,match=match):d._final_timeline([first,changed],sources,paper(),'hero')


def final_fixture(tmp_path,monkeypatch):
    image=tmp_path/'actual.png';Image.new('RGB',(1920,1080),(20,60,80)).save(image)
    e3={'completion_audit':{'sha256':'bound'}}
    monkeypatch.setattr(d,'full_e3_selection',lambda spec,root:{'candidates':[{'scene_id':'s','accepted_objects':5}]})
    monkeypatch.setattr(d,'_paper_source',lambda spec,root:{**paper(),'provenance':{'sources':{'agentic':{'completion_audit':{'sha256':'bound'}}}},'submission_pass':False})
    roles=sorted(d.FINAL_ROLES)
    sources=[{'id':r,'kind':('synchronized_episode' if r=='synchronized_episode' else 'selection' if r in {'selection','retry'} else 'result_card' if r=='results' else 'image'),
              'decision':r,'path':None if r in {'selection','retry','synchronized_episode','results'} else str(image),
              'sha256':d.sha256_file(image)} for r in roles]
    hero=[dict(shot_id=r,source_id=r,role=r,duration_s=9,overlay_text=['Actual archived evidence']) for r in roles]
    cfg=dict(schema_version=2,mode='final',freeze_id='new-freeze',source_root=str(tmp_path),resolution=[1920,1080],fps=30,
             e3=e3,e9={},sources=sources,timelines={'hero':hero,'teaser':[shot(roles[0],duration=30)],'loop':[shot(roles[0],duration=8)]},harmonizer_gate=None)
    p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(cfg));return p,cfg


def test_plan_reports_missing_gates_and_render_writes_nothing(tmp_path,monkeypatch):
    p,cfg=final_fixture(tmp_path,monkeypatch)
    result=d.resolve_final(p,require_ready=False)
    assert not result['full_demo_ready']
    assert {'genuine_synchronized_policy_episode','E5_preservation_certified_frames','E9_common_submission_freeze','D0_exact_source_config_E0'}<=set(result['missing_prerequisites'])
    output=tmp_path/cfg['freeze_id']/'demo'
    with pytest.raises(d.DemoError,match='final demo blocked'):d.render_final(p,output)
    assert not output.exists() and not output.parent.exists()


def test_tampered_or_missing_final_footage_fails(tmp_path,monkeypatch):
    p,cfg=final_fixture(tmp_path,monkeypatch)
    cfg['sources'][0]['sha256']='0'*64;p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(d.DemoError,match='SHA256'):d.resolve_final(p,require_ready=False)


def test_missing_source_never_replaced_by_a_card(tmp_path,monkeypatch):
    p,cfg=final_fixture(tmp_path,monkeypatch)
    cfg['sources'][0]['path']=None;p.write_text(yaml.safe_dump(cfg))
    assert 'missing_source:'+cfg['sources'][0]['id'] in d.resolve_final(p,require_ready=False)['missing_prerequisites']
    with pytest.raises(d.DemoError,match='missing_source'):d.render_final(p,tmp_path/'new-freeze'/'demo')


def test_incomplete_hero_storyboard_rejected(tmp_path,monkeypatch):
    p,cfg=final_fixture(tmp_path,monkeypatch)
    cfg['timelines']['hero'][0]['role']='reconstruction';p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(d.DemoError,match='storyboard'):d.resolve_final(p,require_ready=False)


def test_prospective_storyboard_durations():
    cfg=yaml.safe_load(Path('configs/demo/icra2027_final_storyboard.yaml').read_text())
    assert sum(cfg['hero_durations_s'].values())==90
    assert sum(cfg['teaser_durations_s'].values())==30
    assert cfg['loop']['duration_s']==8 and cfg['hero_rule']['physical_success_required'] is False


def test_e9_artifact_and_result_source_hashes(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'git_snapshot',lambda path:{'commit':'formatter','dirty':False})
    tables=tmp_path/'paper_tables';(tables/'generated_tables').mkdir(parents=True)
    table=tables/'generated_tables/results.tex';table.write_text('machine-generated')
    config=tmp_path/'paper.yaml';config.write_text('config')
    source=tmp_path/'results.json';source.write_text(json.dumps({'rows':[{'coverage':.5}]}))
    manifest=tables/'paper_table_provenance.json';manifest.write_text(json.dumps(dict(freeze_id='p',config=str(config),config_sha256=d.sha256_file(config),
        formatter_commit='formatter',tables={'results.tex':{'sha256':d.sha256_file(table)}},sources={'agentic':d._identity(source)},paper_ready=False)))
    audit=tables/'submission_audit.json';audit.write_text(json.dumps({'submission_freeze':'FAIL'}))
    spec={'freeze_id':'p','provenance':d._identity(manifest),'submission_audit':d._identity(audit)}
    assert not d._paper_source(spec,tmp_path)['submission_pass']
    table.write_text('hand-edited')
    with pytest.raises(d.DemoError,match='SHA256'):d._paper_source(spec,tmp_path)


def test_actual_final_encoder_profile_package_and_no_overwrite(tmp_path,monkeypatch):
    # Synthetic one-second-per-deliverable encoder smoke only. Production
    # durations cannot be changed by a configuration (they are constants).
    image=tmp_path/'synthetic.png';Image.new('RGB',(1920,1080),(28,44,63)).save(image)
    config=tmp_path/'synthetic.yaml';config.write_text('synthetic encoder fixture')
    manifest={'schema_version':2,'mode':'final','freeze_id':'synthetic-encoder',
        'config_path':str(config),'full_demo_ready':True,'sources':[{'id':'still','kind':'image',
        'path':str(image),'sha256':d.sha256_file(image)}],
        'timelines':{name:[shot(duration=1)] for name in d.FINAL_DURATIONS}}
    monkeypatch.setattr(d,'resolve_final',lambda _:d.copy.deepcopy(manifest))
    monkeypatch.setattr(d,'FINAL_DURATIONS',dict.fromkeys(d.FINAL_DURATIONS,1))
    output=tmp_path/'synthetic-encoder/demo';d.render_final(config,output)
    qa=json.loads((output/'qa.json').read_text())
    assert all(p['frames']==30 and p['width']==1920 and p['height']==1080 for p in qa['profiles'].values())
    assert (output/'master/simanyroom_hero_90s_master.mov').is_file()
    assert (output/'loop/simanyroom_loop_8s.webp').is_file()
    assert (output/'poster/simanyroom_poster.png').is_file()
    for path in d.FINAL_VIDEO_PATHS.values():assert (output/path).with_suffix('.srt').is_file()
    assert d.assemble(output)['full_demo_ready']
    with pytest.raises(d.DemoError,match='overwrite'):d.render_final(config,output)


def episode_fixture(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from robo.eval import harness_spec,harness_validation,episode_log
    from robo.manifest.hash import canonical_hash
    from robo.envs import pi05_env
    def js(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return d._identity(p)
    image=tmp_path/'view.png';Image.new('RGB',(40,40),(50,70,90)).save(image);iref=d._identity(image)
    config=js('harness.json',{'contract':{'policy':{'kind':'real'}}})
    reset=js('reset.json',{})
    treatment=SimpleNamespace(scene='agentic',collision='full_room')
    monkeypatch.setattr(harness_spec,'load_harness_spec',lambda _:SimpleNamespace(treatments={'a':treatment}))
    monkeypatch.setattr(harness_validation,'validate_records',lambda *a,**k:{'ok':True})
    monkeypatch.setattr(harness_validation,'validate_saved_treatment_records',lambda *a,**k:{'ok':True})
    monkeypatch.setattr(episode_log,'load_reset_states',lambda _: [SimpleNamespace(reset_state_id='r')])
    ticks=[{'t':i,'objects':{'object':{'pos':[0,0,0],'quat_wxyz':[1,0,0,0]}}} for i in range(2)]
    trace=tmp_path/'trace.json.gz';episode_log.write_timeseries(trace,ticks);tref=d._identity(trace)
    factory=tmp_path/'factory';(factory/'inpaint').mkdir(parents=True)
    clean=factory/'inpaint/clean_background.ply';clean.write_text('synthetic clean geometry')
    manifest=js('episode.json',{'artifacts':{'trace_path':str(trace)},'factory_dir':str(factory)})
    row=dict(episode_id='a__r',scene_id='scene',treatment_id='a',outcome='task_failure',success=False,
             ticks=2,manifest_path=manifest['path'],trace_path=str(trace),video_path=str(image))
    ledger=js('ledger.jsonl',row)
    frames=[{'tick':r['t'],'state_sha256':canonical_hash(r),'views':{'mujoco':iref,'photoreal':iref}} for r in ticks]
    producer=tmp_path/'renderer.py';producer.write_text('# synthetic source fixture')
    source={'commit':'synthetic','dirty':False,'branch':'test'}
    monkeypatch.setattr(d,'git_snapshot',lambda _:source)
    execution=js('render_receipt.json',{'status':'PASS','source':source,'producer':d._identity(producer),
         'config':config,'ledger':ledger,'trace':tref,'frames_sha256':canonical_hash(frames)})
    sync=js('sync.json',{'schema_version':1,'scope':'canonical_harness_render_replay','episode_id':'a__r','trace':tref,'ledger':ledger,'episode_manifest':manifest,
           'collision_mode':'full_room','simulation_reexecuted':False,'producer':d._identity(producer),'config':config,
           'execution_receipt':execution,'clean_background':d._identity(clean),'frames':frames,
           'frame_hz':pi05_env.rig.CONTROL_HZ,'video':iref})
    record={**iref,'episode_id':'a__r','harness_config':config,'reset_bank':reset,'ledger':ledger,
            'episode_manifest':manifest,'trace':tref,'synchronized_frames':sync}
    return record,treatment


def test_same_state_failed_episode_is_admissible_as_failed_footage(tmp_path,monkeypatch):
    record,_=episode_fixture(tmp_path,monkeypatch)
    result=d._episode_source(record,tmp_path)
    assert result['outcome']=='task_failure' and result['success'] is False and result['ticks']==2
    assert result['recorded_candidates']==[{'scene_id':'scene','episode_id':'a__r'}]


@pytest.mark.parametrize('change',['state','ticks','rerun','hz','clean','ledger','shims','scripted'])
def test_pair_evidence_rejects_mismatched_or_uninvoked_sources(tmp_path,monkeypatch,change):
    record,treatment=episode_fixture(tmp_path,monkeypatch)
    sync=Path(record['synchronized_frames']['path']);payload=json.loads(sync.read_text())
    if change=='state':payload['frames'][0]['state_sha256']='0'*64
    if change=='ticks':payload['frames'].pop()
    if change=='rerun':payload['simulation_reexecuted']=True
    if change=='hz':payload['frame_hz']=1
    if change=='clean':payload['clean_background']=record['trace']
    if change=='ledger':payload['ledger']=record['trace']
    if change=='shims':treatment.collision='private_shims'
    if change=='scripted':
        p=Path(record['harness_config']['path']);p.write_text(json.dumps({'contract':{'policy':{'kind':'scripted_smoke'}}}));record['harness_config']=d._identity(p)
    sync.write_text(json.dumps(payload));record['synchronized_frames']=d._identity(sync)
    with pytest.raises(d.DemoError):d._episode_source(record,tmp_path)


def engineering_fixture(tmp_path,monkeypatch):
    from robo.manifest.hash import canonical_hash
    image=tmp_path/'figure.png';Image.new('RGB',(1920,1080),(25,40,60)).save(image)
    event=tmp_path/'event.json';event.write_text('{}')
    evidence={'record':{'path':str(event)},'comparison':{}}
    monkeypatch.setattr(d,'full_e3_selection',lambda spec,root:{'selection':evidence,'retry':evidence})
    monkeypatch.setattr(d,'_paper_source',lambda spec,root:{'provenance':{'formatter_commit':'source',
        'sources':{'agentic':{'completion_audit':{'sha256':'audit'}}}},'artifacts':{'figures/result.png':d._identity(image)}})
    monkeypatch.setattr(d,'git_snapshot',lambda path:{'commit':'source','dirty':False})
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps({'freeze_id':'paper','producer_commit':'source',
        'table_provenance_sha256':'paper-sha','full_e3_evidence_freeze':d.FULL_E3_FREEZE}))
    contract=tmp_path/'contract.json';config=tmp_path/'config.yaml'
    cfg={'schema_version':2,'mode':'engineering_evidence_clips','freeze_id':'new',
         'source_root':str(tmp_path),'e3':{'completion_audit':{'sha256':'audit'}},
         'e9':{'freeze_id':'paper','provenance':{'sha256':'paper-sha'}},
         'publication_receipt':d._identity(receipt),'execution_contract':str(contract),
         'result_figure':'figures/result.png'}
    config.write_text(yaml.safe_dump(cfg))
    c={'freeze_id':'new','code':{'commit':'source','dirty':False},'resource_inventory':[
       {'resolved_path':str(config),'sha256':d.sha256_file(config)}]}
    c['contract_sha256']=canonical_hash(c);contract.write_text(json.dumps(c))
    return config,cfg,receipt


def test_engineering_scope_and_changed_transfer_receipt(tmp_path,monkeypatch):
    config,cfg,receipt=engineering_fixture(tmp_path,monkeypatch)
    result=d.resolve_evidence_clips(config)
    assert result['full_demo_ready'] is False and result['paper_ready'] is False
    record=json.loads(receipt.read_text());record['freeze_id']='wrong';receipt.write_text(json.dumps(record))
    with pytest.raises(d.DemoError,match='SHA256'):d.resolve_evidence_clips(config)


def test_engineering_no_output_with_changed_config_or_missing_source(tmp_path,monkeypatch):
    config,cfg,receipt=engineering_fixture(tmp_path,monkeypatch)
    cfg['result_figure']='figures/hand-edited.png';config.write_text(yaml.safe_dump(cfg))
    out=tmp_path/'outputs/icra2027/new/demo'
    with pytest.raises(d.DemoError,match='config absent'):d.render_evidence_clips(config,out)
    assert not out.exists()


def test_engineering_actual_encoder_and_no_overwrite(tmp_path,monkeypatch):
    config,cfg,receipt=engineering_fixture(tmp_path,monkeypatch)
    def renderer(source):
        source['comparison']['rendering_checks']=[{'clipping':False}]
        return Image.new('RGB',(1920,1080),(25,40,60))
    monkeypatch.setattr(d,'evidence_comparison_image',renderer)
    out=tmp_path/'outputs/icra2027/new/demo'
    d.render_evidence_clips(config,out)
    qa=json.loads((out/'qa.json').read_text())
    assert len(qa['clips'])==3 and qa['full_demo_ready'] is False
    assert qa['rendering_checks']=={'selection':[{'clipping':False}],'retry':[{'clipping':False}]}
    assert all('rendering_checks' not in row['comparison'] for row in json.loads((out/'source_manifest.json').read_text())['sources'])
    assert len(list(out.rglob('*.srt')))==3
    with pytest.raises(d.DemoError,match='already exists'):d.render_evidence_clips(config,out)
