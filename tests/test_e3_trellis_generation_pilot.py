import importlib.util
import json
from pathlib import Path
import pytest

@pytest.fixture
def pilot():
    path=Path(__file__).resolve().parents[1]/'run/icra2027/e3_trellis_generation_pilot.py'
    spec=importlib.util.spec_from_file_location('e3_trellis_pilot',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def test_fixed_denominator_preserves_missing_preparation_and_incomplete_generation(pilot,tmp_path):
    jobs=[]
    for idx in range(6):
        jobs.append({'job_id':f'j{idx}','automatic_instance_id':1000+idx,'proposal_id':f'p{idx}','prepared':idx<3,'output_index':idx,'input':{'sha256':'frozen'},'frame':'train.jpg'})
    odir=tmp_path/'construction/objects/obj_00';odir.mkdir(parents=True)
    for name in pilot.OUTPUTS:(odir/name).write_bytes(b'complete')
    records=tmp_path/'producer_records';records.mkdir();(records/'object_00.json').write_text(json.dumps({'status':'generated','seed':42,'wall_s':1}))
    rows=pilot.collect_records(tmp_path,{'jobs':jobs},1)
    assert len(rows)==6
    assert [r['status'] for r in rows]==['available','generation_failed','generation_failed','unavailable','unavailable','unavailable']
    assert all(r['paper_ready'] is False for r in rows)
    assert all(r['reason']=='preparation_unavailable' for r in rows[3:])
    (odir/'trellis_gs.ply').unlink()
    assert pilot.collect_records(tmp_path,{'jobs':jobs},0)[0]['status']=='generation_failed'

def test_staged_crop_or_roster_tampering_fails(pilot,tmp_path):
    odir=tmp_path/'construction/objects/obj_00';odir.mkdir(parents=True)
    crop=odir/'rgba.png';crop.write_bytes(b'crop')
    obj={'index':0,'gt_object_id':1001,'frame':'train.jpg'}
    roster=odir.parent/'objects.json';roster.write_text(json.dumps([obj]))
    m={'jobs':[{'prepared':True,'output_index':0,'object_metadata':obj,'input':{'sha256':pilot.sha(crop)}}]}
    pilot.validate_staged(tmp_path,m)
    crop.write_bytes(b'changed')
    with pytest.raises(pilot.PilotError,match='crop'):pilot.validate_staged(tmp_path,m)
    crop.write_bytes(b'crop');roster.write_text('[]')
    with pytest.raises(pilot.PilotError,match='roster'):pilot.validate_staged(tmp_path,m)

def test_model_weight_hash_mismatch_fails_before_generation(pilot,tmp_path):
    weight=tmp_path/'weight';weight.write_bytes(b'weight')
    c={'models':{'weight':{'path':str(weight),'sha256':pilot.sha(weight)}}}
    pilot.validate_models(c)
    weight.write_bytes(b'changed')
    with pytest.raises(pilot.PilotError,match='model weight'):pilot.validate_models(c)

def test_real_sealed_six_job_input_manifest(pilot):
    source=Path('/group/worldcept/PhiRIE/code/SimAny/outputs/icra2027/20260904-07e8b05-v21/auto_discovery_pilot')
    if not source.is_dir():pytest.skip('local engineering pilot absent')
    rows=pilot.source_jobs(source)
    assert len(rows)==6 and sum(r['prepared'] for r in rows)==3
    assert {r['automatic_instance_id'] for r in rows}==set(range(1000,1006))

def test_runtime_drift_rejected_before_generation(pilot,tmp_path,monkeypatch):
    cfg=tmp_path/'config.json'
    cfg.write_text(json.dumps({'freeze_id':tmp_path.name,'paper_ready':False,'seed':42,'python':'python','runtime_sha256':'frozen'}))
    contract=tmp_path/'contract';contract.mkdir()
    (contract/'freeze_manifest.json').write_text(json.dumps({'code':{'dirty':False,'commit':'source'},'freeze_id':tmp_path.name,'resource_inventory':[{'id':'e3_trellis_config','sha256':pilot.sha(cfg)}]}))
    monkeypatch.setattr(pilot.subprocess,'check_output',lambda cmd,**kw:'' if 'status' in cmd else 'source')
    monkeypatch.setattr(pilot,'runtime_identity',lambda _:({},'frozen'))
    pilot.context(cfg,tmp_path)
    monkeypatch.setattr(pilot,'runtime_identity',lambda _:({},'changed'))
    with pytest.raises(pilot.PilotError,match='runtime differs'):pilot.context(cfg,tmp_path)

def test_manifest_cannot_reseal_changed_roster_or_crop(pilot,tmp_path,monkeypatch):
    import copy
    source=tmp_path/'source';source.mkdir()
    for name in ['pilot_summary.json','output_hashes.json','input_manifest.json']:(source/name).write_text('{}')
    c={'source_pilot':str(source),'source_summary_sha256':pilot.sha(source/'pilot_summary.json'),'source_output_hashes_sha256':pilot.sha(source/'output_hashes.json')}
    original=[{'job_id':'scene:auto:1001','automatic_instance_id':1001,'prepared':True,'input':{'sha256':'original_crop'}}]
    monkeypatch.setattr(pilot,'source_jobs',lambda _:copy.deepcopy(original))
    out=tmp_path/'freeze/trellis_initial';expected=copy.deepcopy(original)
    expected[0]['proposal_id']='freeze:scene:auto:1001:trellis:initial:seed42'
    m={'jobs':expected};pilot.validate_source_binding(c,out,m)
    for mutate in ['roster','crop','drop']:
        changed=copy.deepcopy(m)
        if mutate=='roster':changed['jobs'][0]['automatic_instance_id']=9999
        elif mutate=='crop':changed['jobs'][0]['input']['sha256']='changed_crop'
        else:changed['jobs']=[]
        with pytest.raises(pilot.PilotError,match='config-bound source'):pilot.validate_source_binding(c,out,changed)
