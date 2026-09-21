"""Scoped construction reporting retains failed rooms and unsupported fields."""
import copy
import json
from pathlib import Path
import pytest
from robo.eval import paper_full_construction as paper
from tests.test_e1_current_drop import exports


@pytest.fixture
def measured(exports):
    config=dict(scope=paper.producer.SCOPE,freeze_id='synthetic',regime=paper.metrics.REQUIRED_REGIMES[3],
        source={'original':'frozen'},planned_scenes=['scene'],measurement_scope=paper.metrics.MEASUREMENT_SCOPE)
    code=dict(commit='a'*40,dirty=False,code_root='synthetic')
    unit=dict(mode='original_export',population=dict(input_instances=3,controller_accepted_instances=1,accepted_slots=['obj_1000']))
    exports['materialization']['roster'].update(job_count=3)
    exports['full_room_admission']=dict(status='PASS',requirement=paper.producer.ROOM_SCOPE,source_mode=unit['mode'],source_unit=copy.deepcopy(unit))
    record=dict(scene_id='scene',freeze_id=config['freeze_id'],build_commit=code['commit'],regime=config['regime'],
        measurement_scope=config['measurement_scope'],input_instances=3,controller_accepted_instances=1,
        record_valid=False,geometry_reference_status='NOT_RUN',f1_20=None,f1_weight=0,runtime_minutes=None,
        validity_reasons=['independent_geometry_evaluation_not_run','full_runtime_not_measured'],
        scene_status='success',accepted_instances=1,tested_instances=1,stable_instances=0,
        source_artifact_hash=paper.api.canonical_hash(exports))
    report=dict(schema_version=1,scope=config['scope'],producer_code=code,paper_ready=False,source=config['source'],
        planned_scenes=config['planned_scenes'],measurement_scope=config['measurement_scope'],scene_id='scene',record=record,
        authenticated_export=exports,bodies=paper.base.export_bodies(exports),attempted_bodies=1,completed_bodies=1,status='PASS',
        measurements=[dict(object_slot='obj_1000',status='COMPLETE',error=None,runtime_seconds=.1,
                           measurement=dict(stable=False,sunk=False,drift_m=.2))])
    return report,unit,config,code


def test_verified_export_does_not_promote_negative_drop_or_missing_metrics(measured):
    paper.validate_unit(*measured)
    record=measured[0]['record']
    assert record['stable_instances']==0 and record['tested_instances']==1
    assert record['record_valid'] is False and record['f1_20'] is record['runtime_minutes'] is None


@pytest.mark.parametrize('change',['paper_promotion','record_promotion','geometry','runtime','limitations',
    'input','controller','source','body_missing','body_duplicate','attempt_missing','attempt_error',
    'fake_stability','nonfinite','negative_drift','bool_stable','scope','admission','source_hash',
    'wrong_roster','mesh','fresh_environment'])
def test_scoped_measurement_rejects_fabrication(measured,change):
    report,unit,config,code=measured;r=report['record']
    if change=='paper_promotion':report['paper_ready']=True
    elif change=='record_promotion':r['record_valid']=True
    elif change=='geometry':r['f1_20']=.9
    elif change=='runtime':r['runtime_minutes']=1
    elif change=='limitations':r['validity_reasons']=[]
    elif change=='input':r['input_instances']=1
    elif change=='controller':r['controller_accepted_instances']=2
    elif change=='source':report['producer_code']={**code,'commit':'b'*40}
    elif change=='body_missing':report['bodies']=[]
    elif change=='body_duplicate':report['bodies']*=2
    elif change=='attempt_missing':report['measurements']=[]
    elif change=='attempt_error':report['measurements'][0]['status']='ERROR'
    elif change=='fake_stability':r['stable_instances']=1
    elif change=='nonfinite':report['measurements'][0]['measurement']['drift_m']=float('nan')
    elif change=='negative_drift':report['measurements'][0]['measurement']['drift_m']=-1
    elif change=='bool_stable':report['measurements'][0]['measurement']['stable']=1
    elif change=='scope':report['measurement_scope']={}
    elif change=='admission':report['authenticated_export']['full_room_admission']['requirement']='isolated'
    elif change=='source_hash':r['source_artifact_hash']='0'*64
    elif change=='wrong_roster':unit['population']['accepted_slots']=[]
    elif change=='mesh':Path(report['bodies'][0]['members'][1]['path']).write_text('changed')
    elif change=='fresh_environment':
        unit['mode']='fresh_export';config['export_environment']={'OMP_NUM_THREADS':'4'}
        report['authenticated_export']['full_room_admission'].update(source_mode='fresh_export',source_unit=copy.deepcopy(unit))
    with pytest.raises(ValueError):paper.validate_unit(*measured)


@pytest.fixture
def terminal(measured,tmp_path):
    report,unit,config,code=measured
    unit['mode']='original_paired_room_rejection'
    stdout=tmp_path/'stdout';stdout.write_text('{}');stderr=tmp_path/'stderr';stderr.write_text('')
    report.update(authenticated_export=None,source_unit=copy.deepcopy(unit),room_export_requirement=paper.producer.ROOM_SCOPE,
        terminal_kind=unit['mode'],terminal_reason=paper.producer.REJECTION,status='NOT_RUN',bodies=[],measurements=[],
        attempted_bodies=0,completed_bodies=0,verification=dict(stdout=paper.api.identity(stdout),stderr=paper.api.identity(stderr),
            result=dict(status='PASS',observed_rejection=paper.producer.REJECTION,source_commit=paper.base.SOURCE_COMMIT,scene_id='scene')))
    report['record'].update(scene_status='failed',accepted_instances=0,tested_instances=0,stable_instances=0,
                            source_artifact_hash=paper.api.canonical_hash(unit))
    return report,unit,config,code


def test_rejected_room_keeps_inputs_without_drop_attempts(terminal):
    paper.validate_unit(*terminal)
    r=terminal[0]['record']
    assert r['input_instances']==3 and r['controller_accepted_instances']==1 and r['accepted_instances']==0


@pytest.mark.parametrize('change',['hide_rejection','fake_export','fake_attempt','fake_scene','fake_replay','changed_replay','missing_scope'])
def test_rejected_room_cannot_be_reclassified(terminal,change):
    report,unit,config,code=terminal
    if change=='hide_rejection':report['terminal_kind']='no_accepted_bodies'
    elif change=='fake_export':report['record']['accepted_instances']=1
    elif change=='fake_attempt':report['attempted_bodies']=1
    elif change=='fake_scene':report['record']['scene_status']='success'
    elif change=='fake_replay':report['verification']['result']['status']='FAIL'
    elif change=='changed_replay':Path(report['verification']['stdout']['path']).write_text('tampered')
    else:report['room_export_requirement']='isolated'
    with pytest.raises(ValueError):paper.validate_unit(*terminal)


def test_empty_scene_requires_no_controller_accepts(terminal):
    report,unit,config,code=terminal
    unit.update(mode='no_accepted_bodies');unit['population'].update(controller_accepted_instances=0,accepted_slots=[])
    report.update(source_unit=copy.deepcopy(unit),terminal_kind=unit['mode'],status='EMPTY')
    report['record'].update(scene_status='empty',controller_accepted_instances=0,source_artifact_hash=paper.api.canonical_hash(unit))
    paper.validate_unit(*terminal)
    unit['population']['controller_accepted_instances']=1
    with pytest.raises(ValueError):paper.validate_unit(*terminal)


def test_hash_and_alias_rejection(tmp_path):
    path=tmp_path/'table.json';path.write_text('{}');ref=paper.api.identity(path)
    assert paper.checked(ref)==path
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='bytes changed'):paper.checked(ref)
    alias=tmp_path/'alias';alias.symlink_to(path)
    with pytest.raises(ValueError,match='aliased'):paper.checked({**ref,'path':str(alias)})


@pytest.mark.parametrize('change',['table','audit','promotion','missing_member','wrong_path'])
def test_entrypoint_requires_pinned_aggregate_before_context(tmp_path,change):
    root=tmp_path/'icra2027/fresh/construction';(root/'table').mkdir(parents=True)
    path=root/'table/construction_table.json';path.write_text('{}');table={}
    audit=dict(paper_ready=False,planned_scene_regime_cells=250,measured_regimes=1,members={k:{} for k in paper.MEMBERS})
    audit_path=root/'seal.json';audit_path.write_text(json.dumps(audit))
    spec={**paper.api.identity(path),'completion_audit':paper.api.identity(audit_path)}
    if change=='table':path.write_text('{"changed":true}')
    elif change=='audit':audit_path.write_text('{}')
    elif change=='promotion':audit['paper_ready']=True
    elif change=='missing_member':audit['members'].pop('scene_records.json')
    elif change=='wrong_path':spec['path']=str(root/'other.json')
    with pytest.raises((ValueError,FileNotFoundError)):
        paper.validate_source(spec,path,table,audit_path,audit)
