"""No source/geometry generation is needed to test E1's typed adapter boundary."""
import json
from pathlib import Path
import pytest
from run.icra2027 import e1_current_drop as d


@pytest.fixture(autouse=True)
def local_output_boundary(tmp_path,monkeypatch):
    monkeypatch.setattr(d.shared.e3,'REPOSITORY_ROOT',tmp_path)


class Physics:
    DIRECT=1
    def __init__(self):self.closed=[]
    def connect(self,mode):assert mode==self.DIRECT;return 3
    def disconnect(self,connection):self.closed.append(connection)


@pytest.fixture
def exports(tmp_path):
    obj=tmp_path/'objects/obj_1000';obj.mkdir(parents=True)
    mesh=obj/'collision.obj';mesh.write_text('v 0 0 0\n')
    urdf=obj/'object.urdf';urdf.write_text('<robot><link><collision><geometry><mesh filename="collision.obj"/></geometry></collision></link></robot>')
    materialization=tmp_path/'materialization_manifest.json'
    materialization.write_text(json.dumps({'output_members':{str(p.relative_to(tmp_path)):
        {k:v for k,v in d.shared.identity(p).items() if k!='path'} for p in (mesh,urdf)}}))
    sim=tmp_path/'sim_export';sim.mkdir()
    manifest=sim/'isaac_manifest.json';manifest.write_text(json.dumps({'objects':[{'name':'obj_1000','urdf':str(urdf)}]}))
    return dict(factory=str(tmp_path),materialization={'manifest_sha256':d.shared.identity(materialization)['sha256'],
        'roster':{'accepted_slots':['obj_1000']}},export={'artifacts':{'isaac_manifest':d.shared.identity(manifest)}})


def test_exact_export_mesh_members_and_canonical_delegation(exports):
    bodies=d.export_bodies(exports);p=Physics();calls=[]
    def canonical(physics,urdf):
        assert physics is p;calls.append(str(urdf));return {'stable':False,'drift_m':.2,'sunk':False}
    result=d.measure_bodies(bodies,p,canonical)
    assert calls==[bodies[0]['urdf']] and result[0]['measurement']['stable'] is False
    assert p.closed==[3] and len(bodies[0]['members'])==2


def test_tampered_urdf_rejected_before_physics(exports):
    bodies=d.export_bodies(exports);Path(bodies[0]['urdf']).write_text('changed')
    with pytest.raises(ValueError,match='changed'):d.measure_bodies(bodies,Physics(),lambda *_:None)


def test_tampered_mesh_cannot_become_new_authenticated_input(exports):
    Path(exports['factory'],'objects/obj_1000/collision.obj').write_text('changed')
    with pytest.raises(ValueError,match='membership'):d.export_bodies(exports)


def test_changed_manifest_cannot_replace_body_roster(exports):
    Path(exports['factory'],'sim_export/isaac_manifest.json').write_text('{"objects":[]}')
    with pytest.raises(ValueError,match='Isaac manifest'):d.export_bodies(exports)


def test_drop_exception_retains_attempt_and_null_telemetry(exports):
    bodies=d.export_bodies(exports);p=Physics()
    def error(*_):raise RuntimeError('URDF load failed')
    result=d.measure_bodies(bodies,p,error)
    assert len(result)==1 and result[0]['status']=='ERROR' and result[0]['measurement'] is None
    assert result[0]['error']=={'type':'RuntimeError','message':'URDF load failed'} and p.closed==[3]


def test_duplicate_body_not_retried_or_counted_twice(exports):
    b=d.export_bodies(exports)
    with pytest.raises(ValueError,match='duplicate'):d.measure_bodies(b+b,Physics(),lambda *_:None)


def test_changed_mesh_during_measurement_rejected(exports):
    b=d.export_bodies(exports)
    def change(*_):
        Path(exports['factory'],'objects/obj_1000/collision.obj').write_text('changed')
        return {'stable':True,'drift_m':0.,'sunk':False}
    with pytest.raises(ValueError,match='changed'):d.measure_bodies(b,Physics(),change)


def test_out_of_roster_scene_rejected_before_source_replay(tmp_path):
    with pytest.raises(ValueError,match='fixed pilot'):d.authenticate_export({},'27dd4da69e',tmp_path)


def test_output_never_overwritten(tmp_path):
    path=tmp_path/'report.json';d._write(path,{'negative':True})
    with pytest.raises(FileExistsError):d._write(path,{'negative':False})
    assert json.loads(path.read_text())=={'negative':True}


def test_real_canonical_drop_smoke_never_reads_gt(tmp_path,monkeypatch):
    p=pytest.importorskip('pybullet')
    from agents.eval.factory_report import drop_test
    from agents.core import common
    def forbidden(*args,**kwargs):raise AssertionError('GT may not enter construction drop')
    monkeypatch.setattr(common,'load_gt_instances',forbidden)
    urdf=tmp_path/'box.urdf'
    urdf.write_text('''<robot name="box"><link name="body"><inertial><mass value="0.3"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
    <collision><geometry><box size="0.05 0.05 0.05"/></geometry></collision></link></robot>''')
    body={'object_slot':'obj_1000','urdf':str(urdf),'members':[d.shared.identity(urdf)]}
    result=d.measure_bodies([body],p,drop_test)
    assert result[0]['status']=='COMPLETE'
    measurement=result[0]['measurement']
    assert set(measurement)=={'drift_m','stable','sunk'}
    assert measurement['drift_m']>=0 and type(measurement['stable']) is bool


def test_execute_keeps_missing_geometry_separate_from_export_success(exports,tmp_path,monkeypatch):
    from agents.eval import factory_report
    stage=tmp_path/'stage';stage.mkdir()
    config={'scope':d.SCOPE,'freeze_id':'test-freeze','regime':d.metrics.REQUIRED_REGIMES[3],
        'measurement_scope':d.metrics.MEASUREMENT_SCOPE,'source':{'original':'bound fixture'},'planned_scenes':d.PILOT+['missing_scene']}
    code={'code_root':str(d.CODE),'commit':'a'*40,'dirty':False}
    exports['materialization']['roster'].update(job_count=3,accepted_count=1)
    monkeypatch.setattr(d,'validate_stage',lambda *_:(config,code))
    monkeypatch.setattr(d,'authenticate_export',lambda *_:exports)
    monkeypatch.setattr(d,'_exact_code',lambda *_:code)
    monkeypatch.setattr(d.metrics,'REPOSITORY_ROOT',tmp_path)
    monkeypatch.delenv('SIMANY_EVIDENCE_ROOT',raising=False)
    monkeypatch.setattr(factory_report,'drop_test',lambda *_:{'stable':False,'drift_m':.3,'sunk':False})
    result=d.execute('config',stage,'a'*40,d.PILOT[0])
    assert result['attempted_bodies']==1 and result['record']['accepted_instances']==1
    assert result['record']['scene_status']=='success' and result['record']['geometry_reference_status']=='NOT_RUN'
    assert result['record']['stable_instances']==0 and result['record']['tested_instances']==1
    aggregate=json.loads((stage/'construction_drop'/d.PILOT[0]/'construction_record_aggregate.json').read_text())[0]
    assert aggregate['yield']==1/3 and aggregate['failed_scenes']==0 and aggregate['f1_20'] is None
    assert aggregate['runtime_minutes'] is None and aggregate['valid_for_paper'] is False
    assert (stage/'construction_drop'/d.PILOT[0]/'seal.json').is_file()
    with pytest.raises(FileExistsError):d.execute('config',stage,'a'*40,d.PILOT[0])


def test_dirty_or_wrong_producer_rejected_before_any_measurement(tmp_path,monkeypatch):
    def wrong(*_):raise ValueError('terminal audit code is not exact and clean')
    monkeypatch.setattr(d,'_exact_code',wrong)
    with pytest.raises(ValueError,match='exact and clean'):
        d.validate_stage('does-not-exist',tmp_path,'a'*40)


def test_wrong_original_e0_path_cannot_relabel_source(tmp_path,monkeypatch):
    path=tmp_path/'e0.json';path.write_text('{}')
    monkeypatch.setattr(d.screen,'evidence_root',lambda:tmp_path)
    with pytest.raises(ValueError,match='source E0 path differs'):
        d.source_contract({'source':{'e0':d.shared.identity(path)}})
