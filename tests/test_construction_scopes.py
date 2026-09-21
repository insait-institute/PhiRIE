import copy
import json
from pathlib import Path
import pytest
from robo.eval import construction_metrics as m


@pytest.fixture
def row(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'REPOSITORY_ROOT',tmp_path)
    def create(scene='s',**changes):
        r=dict(freeze_id='f',regime=m.REQUIRED_REGIMES[3],scene_id=scene,scene_status='success',
            input_instances=4,controller_accepted_instances=3,accepted_instances=2,f1_20=.5,f1_weight=1,
            stable_instances=1,tested_instances=2,runtime_minutes=6,build_commit='a'*40,
            build_manifest_path=str(tmp_path/(scene+'.json')),failure_reason='',source_artifact_hash='b'*64,
            record_valid=True,validity_reasons=[],geometry_reference_status='independent_gt_evaluation',
            measurement_scope=copy.deepcopy(m.MEASUREMENT_SCOPE),runtime_components_seconds=dict.fromkeys(m.RUNTIME_COMPONENTS,60))
        r.update(changes);sync(r);return r
    return create


def sync(r):
    keys=('freeze_id','regime','scene_id','build_commit','source_artifact_hash','record_valid','validity_reasons')
    Path(r['build_manifest_path']).write_text(json.dumps({**{k:r[k] for k in keys},'record':r}))


def test_controller_and_export_yield_are_distinct(row):
    result=m.aggregate([row()],smoke=True)[0]
    assert result['controller_coverage']==.75 and result['yield']==.5
    assert result['verified_export_scene_count']==1 and result['verified_export_count_observed']==2
    assert result['runtime_minutes']==6 and result['runtime_complete_scenes']==1
    assert 'verified simulator-export' in m.render_latex([result],smoke=True)


def test_missing_exports_never_become_zero_or_complete_yield(row):
    a=row('a');b=row('b',accepted_instances=None,f1_20=None,f1_weight=0,stable_instances=None,tested_instances=None,
        scene_status='unavailable',failure_reason='export not run',record_valid=False,validity_reasons=['export_not_run'])
    result=m.aggregate([a,b],allow_preliminary=True,smoke=True)[0]
    assert result['instances']==8 and result['controller_coverage']==.75
    assert result['accepted_instances'] is None and result['yield'] is None
    assert result['verified_export_count_observed']==2 and result['verified_export_scene_count']==1
    assert not result['valid_for_paper'] and result['planned_scenes']==2


def test_unknown_inputs_preserved(row):
    r=row(input_instances=None,controller_accepted_instances=None,accepted_instances=None,f1_20=None,f1_weight=0,
        stable_instances=None,tested_instances=None,record_valid=False,validity_reasons=['not_run'],
        scene_status='unavailable',failure_reason='not_run')
    result=m.aggregate([r],allow_preliminary=True,smoke=True)[0]
    assert result['instances'] is None and result['controller_coverage'] is None and result['yield'] is None
    assert '--/1' in m.render_latex([result],smoke=True)


def test_partial_runtime_not_averaged_as_complete_scene_time(row):
    r=row('b',runtime_minutes=None,runtime_components_seconds={**dict.fromkeys(m.RUNTIME_COMPONENTS,60),'input_reconstruction':None})
    result=m.aggregate([row('a'),r],smoke=True)[0]
    assert result['runtime_minutes'] is None and result['runtime_complete_scenes']==1
    assert result['runtime_components_seconds']['input_reconstruction'] is None
    assert not result['valid_for_paper']


@pytest.mark.parametrize('field,value',[('physical_protocol','E3_isolated_probe'),('acceptance','controller_acceptance'),
    ('constructor_protocol','single_trellis'),('geometry','own_registration_surface')])
def test_protocol_mismatch_rejected(row,field,value):
    r=row();r['measurement_scope'][field]=value;sync(r)
    with pytest.raises(ValueError,match='scope/protocol'):m.aggregate([r],smoke=True)


def test_legacy_and_scoped_cannot_mix(row):
    a=row('a');b=row('b')
    for k in m.SCOPED_FIELDS:del b[k]
    sync(b)
    with pytest.raises(ValueError,match='mix legacy'):m.aggregate([a,b],smoke=True)


@pytest.mark.parametrize('changes,pattern',[
    ({'accepted_instances':4},'exceed controller'),
    ({'f1_weight':3},'f1_weight > accepted'),
    ({'runtime_minutes':10},'runtime differs'),
    ({'runtime_components_seconds':{}},'component roster'),
    ({'runtime_components_seconds':dict.fromkeys(m.RUNTIME_COMPONENTS,None)},'incomplete runtime'),
])
def test_inconsistent_scoped_evidence_rejected(row,changes,pattern):
    with pytest.raises(ValueError,match=pattern):m.aggregate([row(**changes)],smoke=True)


def test_scope_and_component_manifest_binding(row):
    r=row();p=Path(r['build_manifest_path']);manifest=json.loads(p.read_text())
    manifest['record']['controller_accepted_instances']=4;p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='controller_accepted_instances disagrees'):m.aggregate([r],smoke=True)


def test_missing_drop_is_not_paper_ready(row):
    result=m.aggregate([row(stable_instances=None,tested_instances=None)],smoke=True)[0]
    assert result['stability'] is None and not result['valid_for_paper']


def test_explicit_scoped_evidence_root_keeps_legacy_boundary(row,tmp_path,monkeypatch):
    r=row();root=Path(m.__file__).resolve().parents[2]
    assert m.aggregate([r],smoke=True)[0]['accepted_instances']==2
    monkeypatch.setenv('SIMANY_EVIDENCE_ROOT',str(root))
    assert m._resolve_build_manifest(str(root/'README.md'),record='test',scoped=True)==root/'README.md'
    with pytest.raises(ValueError,match='outside the repository'):
        m._resolve_build_manifest('/unapproved/manifest.json',record='test',scoped=True)
    monkeypatch.setenv('SIMANY_EVIDENCE_ROOT','invalid-relative')
    assert m._resolve_build_manifest(r['build_manifest_path'],record='legacy')==Path(r['build_manifest_path'])


def test_scoped_evidence_root_rejects_relative_override(row,monkeypatch):
    r=row();monkeypatch.setenv('SIMANY_EVIDENCE_ROOT','outputs')
    with pytest.raises(ValueError,match='absolute'):m.aggregate([r],smoke=True)


def test_scoped_csv_contains_machine_readable_scope(row):
    import csv,io
    result=m.aggregate([row()],smoke=True)
    parsed=next(csv.DictReader(io.StringIO(m._csv_text(result))))
    assert json.loads(parsed['measurement_scope'])==m.MEASUREMENT_SCOPE
    assert set(json.loads(parsed['runtime_components_seconds']))==set(m.RUNTIME_COMPONENTS)


def test_valid_export_with_measured_zero_independent_matches_retains_yield(row):
    r=row(f1_20=None,f1_weight=0,geometry_reference_status='independent_gt_evaluation_no_matches')
    result=m.aggregate([r],smoke=True)[0]
    assert result['yield']==.5 and result['accepted_instances']==2
    assert result['f1_20'] is None and result['f1_weight']==0
    assert result['valid_for_paper'] and result['stable_instances']==1


@pytest.mark.parametrize('changes',[
    {'f1_20':.5,'f1_weight':1,'geometry_reference_status':'independent_gt_evaluation_no_matches'},
    {'f1_20':None,'f1_weight':0,'geometry_reference_status':'NOT_RUN'},
    {'f1_20':None,'f1_weight':0,'geometry_reference_status':'independent_gt_evaluation'},
])
def test_missing_or_inconsistent_geometry_cannot_masquerade_as_measured_no_matches(row,changes):
    with pytest.raises(ValueError,match='no-matches|lacks independent F1'):
        m.aggregate([row(**changes)],smoke=True)


def test_legacy_valid_export_still_requires_independent_f1(row):
    r=row(f1_20=None,f1_weight=0,geometry_reference_status='independent_gt_evaluation_no_matches')
    for key in m.SCOPED_FIELDS:del r[key]
    sync(r)
    with pytest.raises(ValueError,match='lacks independent F1'):m.aggregate([r],smoke=True)
