import copy
from pathlib import Path
import pytest
from run.icra2027 import e1_full_drop as f


@pytest.fixture
def context(tmp_path,monkeypatch):
    monkeypatch.setattr(f.base.shared.e3,'REPOSITORY_ROOT',tmp_path)
    monkeypatch.setattr(f.base.metrics,'REPOSITORY_ROOT',tmp_path)
    monkeypatch.setattr(f.base,'_exact_code',lambda *_:None)
    monkeypatch.delenv('SIMANY_EVIDENCE_ROOT',raising=False)
    config=dict(scope=f.SCOPE,freeze_id='new-freeze',regime=f.base.metrics.REQUIRED_REGIMES[3],
        source={'bound':'original source'},measurement_scope=f.base.metrics.MEASUREMENT_SCOPE,planned_scenes=['s'],pilot_scenes=f.base.PILOT)
    code=dict(commit='a'*40,dirty=False,code_root=str(f.CODE))
    unit=dict(population=dict(input_instances=12,controller_accepted_instances=2,accepted_slots=['obj_1000','obj_1001']),
        mode='original_paired_room_rejection',anchors={},original_factory='source',e2_materialization=None)
    return config,code,unit,tmp_path


def test_paired_room_rejection_is_not_isolated_body_export_success(context):
    config,code,unit,out=context
    report=f._terminal(config,code,unit,'s',out,'original_paired_room_rejection',f.REJECTION)
    assert report['record']['input_instances']==12 and report['record']['controller_accepted_instances']==2
    assert report['record']['accepted_instances']==0 and report['attempted_bodies']==0
    assert report['measurements']==[] and report['authenticated_export'] is None
    assert report['full_room_export_status']=='BLOCKED_BY_OBSERVED_PAIRED_STATIC_REJECTION'
    assert report['record']['scene_status']=='failed' and report['record']['f1_20'] is None
    assert f.base.metrics.aggregate([report['record']],allow_preliminary=True,smoke=True)[0]['yield']==0


def test_zero_accepted_scene_has_explicit_no_body_closure(context):
    config,code,unit,out=context;unit['population']['controller_accepted_instances']=0;unit['population']['accepted_slots']=[]
    report=f._terminal(config,code,unit,'s',out,'no_accepted_bodies','original A4 accepted no bodies')
    assert report['record']['scene_status']=='empty' and report['record']['accepted_instances']==0
    assert report['attempted_bodies']==0 and report['record']['tested_instances']==0
    assert report['full_room_export_status']=='NOT_APPLICABLE_NO_ACCEPTED_BODIES'
    assert report['record']['runtime_minutes'] is None


def test_unavailable_export_retains_unknown_numerator_and_no_measurements(context):
    config,code,unit,out=context
    report=f._terminal(config,code,unit,'s',out,'export_or_measurement_unavailable','missing file')
    assert report['record']['accepted_instances'] is None and report['record']['tested_instances'] is None
    assert report['measurements']==[] and report['record']['input_instances']==12
    assert f.base.metrics.aggregate([report['record']],allow_preliminary=True,smoke=True)[0]['yield'] is None


def test_terminal_output_cannot_overwrite(context):
    config,code,unit,out=context
    f._terminal(config,code,unit,'s',out,'original_paired_room_rejection',f.REJECTION)
    with pytest.raises(FileExistsError):f._terminal(config,code,unit,'s',out,'original_paired_room_rejection',f.REJECTION)


def test_wrong_population_is_not_silently_replaced(tmp_path,monkeypatch):
    monkeypatch.setattr(f.base,'source_contract',lambda c:({}, {'source':{'scenes':{}}}))
    with pytest.raises(ValueError,match='all50/1871'):f._source_population({'planned_scenes':[]})


def test_original_rejection_must_replay_not_just_parse_claim(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(f.screen,'evidence_root',lambda:tmp_path)
    monkeypatch.setattr(f.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout='',stderr='claim did not replay'))
    with pytest.raises(ValueError,match='exact-source replay'):
        f._replay_original_rejection({'source':{'e0':{'path':str(tmp_path/'contract/freeze_manifest.json')}}},{},'s',tmp_path,{'python':'python'},{'code_root':str(tmp_path)})
    assert (tmp_path/'original_rejection.stderr').read_text()=='claim did not replay'


def test_no_model_or_GT_producers_in_full_wrapper():
    import ast
    tree=ast.parse(Path(f.__file__).read_text())
    names={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
    assert not names & {'load_gt_instances','run_inference','generate_mesh','plan_automatic_population_tasks','qualify_scene'}
    assert {'materialize_factory_variant','_run_full_room_export','publish_authenticated_drop'} <= names


def test_unit_mode_cannot_hide_source_rejection(context):
    config,code,unit,root=context
    stage=root/'original';(stage/'dispatch').mkdir(parents=True)
    rejection=stage/'dispatch/s_export_rejection.json';rejection.write_text('{}')
    unit.update(mode='fresh_export',original_factory=str(stage/'automatic_candidates/s/materialized/A4'))
    config['units']={'s':unit}
    with pytest.raises(ValueError,match='hide original paired-room rejection'):
        f._validate_units(config,{'s':unit['population']},stage)


def test_no_body_mode_cannot_drop_accepted_objects(context):
    config,code,unit,root=context;unit.update(mode='no_accepted_bodies',original_factory=str(root/'automatic_candidates/s/materialized/A4'))
    config['units']={'s':unit}
    with pytest.raises(ValueError,match='hides accepted objects'):
        f._validate_units(config,{'s':unit['population']},root)


def test_full_roster_cannot_omit_or_reorder_scene(context):
    config,code,unit,root=context;config.update(planned_scenes=['s','other'],units={'s':unit})
    with pytest.raises(ValueError,match='unit roster'):
        f._validate_units(config,{'s':unit['population']},root)


def test_original_factory_cannot_be_replaced(context):
    config,code,unit,root=context;config['units']={'s':unit}
    with pytest.raises(ValueError,match='factory path'):
        f._validate_units(config,{'s':unit['population']},root)


def test_full_aggregate_preserves_250_cells_with_four_unexecuted_regimes(context,monkeypatch):
    config,code,unit,root=context;scenes=list(f.base.metrics.PAPER_SCENE_IDS)
    config.update(planned_scenes=scenes,units={})
    for i,sid in enumerate(scenes):
        u=copy.deepcopy(unit);u['population'].update(input_instances=1822 if i==0 else 1,controller_accepted_instances=0,accepted_slots=[])
        u['mode']='no_accepted_bodies';config['units'][sid]=u
        out=root/'construction_drop'/sid;out.mkdir(parents=True)
        f._terminal(config,code,u,sid,out,'no_accepted_bodies','no accepted bodies')
    config_path=root/'execution.json';f.base._write(config_path,config)
    monkeypatch.setattr(f,'validate_stage',lambda *_:(config,code,{},{}))
    result=f.aggregate_full(config_path,root,code['commit'])
    records=f.api.read(root/'construction/scene_records.json')
    assert len(records)==250 and len(result['rows'])==5 and result['paper_ready'] is False
    measured=[r for r in result['rows'] if r['regime']==config['regime']][0]
    assert measured['instances']==1871 and measured['scenes']==50 and measured['accepted_instances']==0
    missing=[r for r in result['rows'] if r['regime']!=config['regime']]
    assert all(r['scenes']==50 and r['instances'] is None and r['yield'] is None for r in missing)
    assert all(r['f1_20'] is None and r['runtime_minutes'] is None for r in result['rows'])
    with pytest.raises(FileExistsError):f.aggregate_full(config_path,root,code['commit'])
    # Updating a self-authored seal cannot promote invented evaluation values.
    report_path=root/'construction_drop'/scenes[0]/'drop_report.json'
    report=f.api.read(report_path);report['record']['f1_20']=.8
    import json
    report_path.write_text(json.dumps(report))
    seal_path=report_path.parent/'seal.json';seal=f.api.read(seal_path)
    seal['members']['drop_report.json']=f.api.identity(report_path);seal_path.write_text(json.dumps(seal))
    with pytest.raises(ValueError,match='promoted unmeasured'):
        f.aggregate_full(config_path,root,code['commit'])


@pytest.mark.parametrize('mutation',['omitted','aliased','other_scene'])
def test_full_aggregate_rejects_unbound_consumed_unit_member(context,monkeypatch,mutation):
    import json
    config,code,unit,root=context
    unit['mode']='no_accepted_bodies';unit['population'].update(controller_accepted_instances=0,accepted_slots=[])
    config['units']={'s':unit}
    out=root/'construction_drop/s';out.mkdir(parents=True)
    f._terminal(config,code,unit,'s',out,'no_accepted_bodies','no accepted bodies')
    config_path=root/'execution.json';f.base._write(config_path,config)
    monkeypatch.setattr(f,'validate_stage',lambda *_:(config,code,{},{}))
    seal_path=out/'seal.json';seal=f.api.read(seal_path)
    if mutation=='omitted':del seal['members']['drop_report.json']
    else:
        alias=out/'alias.json' if mutation=='aliased' else root/'construction_drop/other/drop_report.json'
        alias.parent.mkdir(parents=True,exist_ok=True)
        alias.write_bytes((out/'drop_report.json').read_bytes())
        seal['members']['drop_report.json']=f.api.identity(alias)
    seal_path.write_text(json.dumps(seal))
    with pytest.raises(ValueError,match='unit seal member'):
        f.aggregate_full(config_path,root,code['commit'])
    assert not (root/'construction').exists()


@pytest.mark.parametrize('mutation',[None,'native_threads','isa','missing_library'])
def test_export_environment_requires_actual_fixed_runtime(monkeypatch,mutation):
    for key,value in f.EXPORT_FIXED_ENVIRONMENT.items():monkeypatch.setenv(key,value)
    monkeypatch.setenv('LD_LIBRARY_PATH','/approved/osmesa/lib')
    if mutation=='native_threads':monkeypatch.setenv('OPENBLAS_NUM_THREADS','8')
    if mutation=='isa':monkeypatch.delenv('NPY_DISABLE_CPU_FEATURES')
    if mutation=='missing_library':monkeypatch.delenv('LD_LIBRARY_PATH')
    if mutation:
        with pytest.raises(ValueError,match='full E1 requires'):f.export_environment()
    else:
        assert f.export_environment()==dict(f.EXPORT_FIXED_ENVIRONMENT,LD_LIBRARY_PATH='/approved/osmesa/lib')


@pytest.mark.parametrize('supplied',[False,True])
def test_explicit_export_environment_reaches_both_real_subprocesses(tmp_path,monkeypatch,supplied):
    import json
    import subprocess
    from run.icra2027 import e4_automatic_export_reuse as reuse
    screen=f.screen
    factories=[tmp_path/'A0',tmp_path/'A4']
    for path in factories:path.mkdir()
    for key in ('TMPDIR','XDG_CACHE_HOME'):monkeypatch.delenv(key,raising=False)
    monkeypatch.setenv('SLURM_CPUS_PER_TASK','12')
    monkeypatch.setenv('NPY_DISABLE_CPU_FEATURES','SHOULD_NOT_BE_IMPLICITLY_INHERITED')
    monkeypatch.setattr(screen,'_automatic_export_context',lambda *a,**k:None)
    monkeypatch.setattr(reuse,'resolve',lambda *a,**k:None)
    monkeypatch.setattr(screen,'_automatic_static_layout',lambda *a,**k:(tmp_path/'package',tmp_path/'spec.json'))
    monkeypatch.setattr(screen,'_automatic_static_spec',lambda *a:{'source':'fixture'})
    original_run=subprocess.run;seen=[]
    def observe(command,**kwargs):
        # Run a real CPU child through exactly the environment supplied to each
        # expensive exporter boundary; no model or synthetic physics is invoked.
        result=original_run([command[0],'-c','import json,os; print(json.dumps(dict(os.environ)))'],
            **kwargs,capture_output=True,text=True)
        seen.append(json.loads(result.stdout));return result
    monkeypatch.setattr(screen.subprocess,'run',observe)
    explicit=dict(f.EXPORT_FIXED_ENVIRONMENT,LD_LIBRARY_PATH='/approved/osmesa/lib')
    receipt=screen._run_full_room_export(factories[1],scene_id='s',root=tmp_path,
        common_carve_factories=factories,automatic=True,
        **({'subprocess_environment':explicit} if supplied else {}))
    assert len(seen)==2
    if supplied:
        assert receipt['requested_environment']==explicit and len(receipt['calls'])==2
        for observed,call in zip(seen,receipt['calls'],strict=True):
            assert all(observed[k]==v==call['environment'][k] for k,v in explicit.items())
            assert call['returncode']==0
    else:
        assert receipt is None
        assert all('NPY_DISABLE_CPU_FEATURES' not in env and env['OMP_NUM_THREADS']=='12' for env in seen)


@pytest.mark.parametrize('overrides',[{'PYTHONPATH':'evil'},{'SIMANY_NO_GT':'0'},{'OMP_NUM_THREADS':4},{'LD_LIBRARY_PATH':''}])
def test_explicit_export_environment_rejects_source_or_invalid_overrides(tmp_path,overrides):
    with pytest.raises(f.screen.CandidateScreenError,match='explicit exporter'):
        f.screen._run_full_room_export(tmp_path,scene_id='s',root=tmp_path,
            common_carve_factories=[],subprocess_environment=overrides)
