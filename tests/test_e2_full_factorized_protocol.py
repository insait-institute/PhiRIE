"""Full-roster admission negatives, without images, generation, or evaluation."""
import copy
import json
from pathlib import Path
import pytest
import yaml
from run.icra2027 import e2_full_factorized_protocol as p

@pytest.fixture
def frozen(tmp_path, monkeypatch):
    original = Path(__file__).parents[1]/'configs/experiments/icra2027/e2_full_factorized/protocol.yaml'
    protocol = yaml.safe_load(original.read_text())
    roster = protocol['scene_ids']
    records = {protocol['construction_gate']['path']: dict(status='PASS',scope='complete_construction_integrity_only',source_commit=p.CONSTRUCTION_COMMIT,freeze_id=p.CONSTRUCTION_FREEZE,planned_scenes=50,planned_jobs=1871,evaluation_geometry_read=False,scenes=[{'scene_id':s} for s in roster]),
        protocol['resolved_jobs']['path']: dict(freeze_id=p.CONSTRUCTION_FREEZE,counts={'scenes':50,'jobs':1871,'policy_object_rows':9355},scenes=[dict(scene_id=s,jobs=[dict(object_slot=f'obj_{i+1000}') for i in range(protocol['population'][s])]) for s in roster]),
        protocol['camera_bank']['path']: dict(scope='exact_serialized_e2_camera_bank',scenes=[dict(scene_id=s,plan={'path':s}) for s in roster]),
        protocol['factory_execution']['path']: dict(freeze_id=Path(protocol['factory_freeze_root']).name,source=dict(control_audit=protocol['construction_gate'],e3_root=str(Path(protocol['resolved_jobs']['path']).parents[1]),scenes={s:{} for s in roster}))}
    for s in roster:
        records[s] = dict(scene_id=s,evaluation_images=[dict(frame=f,w2c=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) for f in protocol['evaluation_frames'][s]],optimization_input_frames=['TRAIN.JPG'],source_gaussian_training_provenance='FRESH_OFFICIAL_TRAIN_ONLY')
    monkeypatch.setattr(p,'read',lambda ref:copy.deepcopy(records[ref['path']]))
    path=tmp_path/'protocol.yaml'
    def write():path.write_text(yaml.safe_dump(protocol));return path
    return protocol,records,write


def test_exact_full_protocol_passes_without_image_read(frozen):
    protocol,_,write=frozen
    assert p.validate_protocol(write())['planned_views']==400
    assert protocol['pilot_scene']=='09c1414f1b'

@pytest.mark.parametrize('change',['roster','count','pilot','camera_order','camera_duplicate','train_leak','construction_gt','source_hash','factory_freeze','quality_rule'])
def test_full_admission_rejects_scope_drift(frozen,change):
    protocol,records,write=frozen;s=protocol['pilot_scene']
    if change=='roster':protocol['scene_ids'].reverse()
    elif change=='count':protocol['population'][s]-=1
    elif change=='pilot':protocol['pilot_scene']=protocol['scene_ids'][1]
    elif change=='camera_order':protocol['evaluation_frames'][s].reverse()
    elif change=='camera_duplicate':records[s]['evaluation_images'][1]=records[s]['evaluation_images'][0]
    elif change=='train_leak':records[s]['optimization_input_frames']=[protocol['evaluation_frames'][s][0]]
    elif change=='construction_gt':records[protocol['construction_gate']['path']]['evaluation_geometry_read']=True
    elif change=='source_hash':protocol['resolved_jobs']['sha256']='0'*64
    elif change=='factory_freeze':protocol['factory_freeze_root']+='_other'
    elif change=='quality_rule':protocol['quality_support']='only_successful_scenes'
    with pytest.raises(ValueError):p.validate_protocol(write())


def test_metadata_identity_accepts_existing_size_bytes(tmp_path):
    path=tmp_path/'input.json';path.write_text('{"input": 1}')
    ref=p._identity(path);ref['size_bytes']=ref.pop('bytes')
    assert p.read(ref)=={'input':1}
    path.write_text('{"input": 2}')
    with pytest.raises(ValueError):p.read(ref)


def test_preparation_pilot_requires_protocol_in_contract(frozen,tmp_path,monkeypatch):
    protocol,records,write=frozen;path=write();sid=protocol['pilot_scene']
    manifest=Path(protocol['factory_freeze_root'])/'automatic_candidates'/sid/'materialized/A4/materialization_manifest.json'
    c=dict(scene_id=sid,policy_id='A4',materialization_manifest={'path':str(manifest)})
    records[str(manifest)]=dict(e3_code_commit=p.CONSTRUCTION_COMMIT,e3_freeze_id=p.CONSTRUCTION_FREEZE,roster=dict(job_count=protocol['population'][sid],object_slots=[f'obj_{i+1000}' for i in range(protocol['population'][sid])]))
    contract=dict(resource_inventory=[dict(resolved_path=str(path),sha256=p._identity(path)['sha256'])])
    monkeypatch.setattr(p,'_bound_contract',lambda *_:(c,contract))
    monkeypatch.setattr(p,'_checked',lambda ref:Path(ref['path']))
    assert p.validate_prepare(tmp_path/'c',tmp_path/'e0',path)[0]==c
    records[str(manifest)]['roster']['object_slots'].reverse()
    with pytest.raises(ValueError,match='population'):p.validate_prepare(tmp_path/'c',tmp_path/'e0',path)
    records[str(manifest)]['roster']['object_slots'].reverse()
    contract['resource_inventory']=[]
    with pytest.raises(ValueError,match='E0-bound'):p.validate_prepare(tmp_path/'c',tmp_path/'e0',path)


def test_no_full_preparation_before_predeclared_pilot(frozen,tmp_path,monkeypatch):
    protocol,_,write=frozen;path=write()
    c=dict(scene_id=protocol['scene_ids'][1],policy_id='A4',materialization_manifest={'path':'not_accepted'})
    contract=dict(resource_inventory=[dict(resolved_path=str(path),sha256=p._identity(path)['sha256'])])
    monkeypatch.setattr(p,'_bound_contract',lambda *_:(c,contract))
    with pytest.raises(ValueError,match='only predeclared'):p.validate_prepare(tmp_path/'c',tmp_path/'e0',path)
