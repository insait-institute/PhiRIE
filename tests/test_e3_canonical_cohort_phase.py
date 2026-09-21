import copy
import json
import os
import sys

import pytest
import yaml

from run.icra2027 import e3_fresh_canonical_phase as launcher
from tests.test_e3_fresh_canonical import phase_config


@pytest.fixture
def cohort(phase_config):
    path,root,config,calls,jobs=phase_config
    config.pop('scene_id'); config.pop('object_slots')
    config.update(schema_version=2,planned_scenes=2,planned_jobs=2,planned_policy_object_rows=10,
        scene_ids=['09c1414f1b','38d58a7a31'],
        scene_object_slots={'09c1414f1b':['obj_1000'],'38d58a7a31':['obj_1000']})
    jobs['counts']={'scenes':2,'jobs':2,'policy_object_rows':10}
    jobs['scenes'].append({'scene_id':'38d58a7a31','jobs':[{'object_slot':'obj_1000'}]})
    path.write_text(yaml.safe_dump(config,sort_keys=False))
    return path,root,config,calls,jobs


def test_cohort_validate_requires_declared_scene_and_preserves_full_population(cohort):
    path,root,config,calls,jobs=cohort
    for scene in config['scene_ids']:
        launcher.validate(path,root,'control',scene)
        assert calls[-1]['require_inventory'] is True
        assert config['jobs_config'] not in calls[-1]['config_paths']
    with pytest.raises(ValueError,match='explicit frozen scene'):
        launcher.validate(path,root,'observe')
    with pytest.raises(ValueError,match='explicit frozen scene'):
        launcher.validate(path,root,'observe','abcdef1234')
    with pytest.raises(ValueError,match='global phase'):
        launcher.validate(path,root,'inventory',config['scene_ids'][0])


@pytest.mark.parametrize('change',['order','missing_scene','duplicate_slot','scene_count','job_count'])
def test_cohort_rejects_population_drift(cohort,change):
    path,root,config,calls,jobs=cohort
    if change=='order':config['scene_ids'].reverse()
    elif change=='missing_scene':config['scene_object_slots'].pop('38d58a7a31')
    elif change=='duplicate_slot':config['scene_object_slots']['38d58a7a31']=['obj_1000','obj_1000']
    elif change=='scene_count':config['planned_scenes']=1
    else:config['planned_jobs']=3;config['planned_policy_object_rows']=15
    path.write_text(yaml.safe_dump(config,sort_keys=False))
    with pytest.raises(ValueError):launcher.validate(path,root,'inventory')


def test_guard_blocks_other_scene_gt_as_well_as_selected_scene():
    guard=launcher.guard_for('observe',{'scene_id':'09c1414f1b'})
    for path in ('/data/ScanNetpp/data/38d58a7a31/scans/mesh.ply',
                 '/data/ScanNetpp/data/38d58a7a31/dslr/resized_undistorted_images/a.JPG'):
        with pytest.raises(ValueError):guard('open',(path,'r',0))


def test_control_receipts_are_per_scene_and_include_all_planned_rows(cohort, monkeypatch):
    from run.icra2027 import e3_trellis_generation_pilot
    path,root,config,calls,jobs=cohort
    config.update(control_python=sys.executable,control_runtime_sha256='a'*64)
    path.write_text(yaml.safe_dump(config,sort_keys=False))
    for scene in jobs['scenes']:
        scene['source_scene_gaussian']={'path':str(root/'gaussian.ply')}
    monkeypatch.setattr(launcher,'validate',lambda *args:(config,root,{'code':{'commit':'b'*40},'contract_sha256':'c'*64}))
    monkeypatch.setattr(e3_trellis_generation_pilot,'runtime_identity',lambda python:({},'a'*64))
    monkeypatch.setattr(sys,'addaudithook',lambda callback:None)
    commands=[]
    monkeypatch.setattr(launcher.e3,'main',lambda command:commands.append(command) or 0)
    def ledger(out,scene):
        directory=root/scene; directory.mkdir(exist_ok=True)
        (directory/'job_ledger.jsonl').write_text('{}\n'*5)
        return directory,{},{}
    monkeypatch.setattr(launcher.e3,'_load_control_scene',ledger)
    monkeypatch.setenv('SLURMD_NODENAME','sof1-h200-5')
    monkeypatch.setenv('SLURM_JOB_ID','123')
    monkeypatch.setenv('SLURM_GPUS_ON_NODE','0')
    monkeypatch.delenv('SLURM_ARRAY_JOB_ID',raising=False)
    for scene in config['scene_ids']:
        monkeypatch.setenv('SIMANY_SCENE',scene)
        assert launcher.run(path,root,'control',scene)==0
        receipt=json.loads((root/'execution_receipts'/scene/'control_execution_start.json').read_text())
        assert receipt['planned_jobs']==1 and receipt['planned_policy_object_rows']==5
        assert receipt['scene_id']==scene
        assert commands[-1][-2:]==['--scene-id',scene]
    monkeypatch.setenv('SIMANY_SCENE','wrong')
    with pytest.raises(ValueError,match='runtime scene environment'):
        launcher.run(path,root,'control',config['scene_ids'][0])


def test_array_rejected_before_execution(monkeypatch):
    monkeypatch.setenv('SLURM_ARRAY_JOB_ID','123')
    with pytest.raises(ValueError,match='ordinary individual jobs'):
        launcher.run('not-read','not-read','control','38d58a7a31')
