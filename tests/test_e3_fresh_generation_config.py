import json
from pathlib import Path

import pytest
import yaml

from run.icra2027 import e3_fresh_generation_config as builder
from run.icra2027.e3_auto_discovery_pilot import PilotError


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_waiting_discovery_never_writes_config_or_invents_count(tmp_path):
    output=tmp_path/'config.yaml'
    report=builder.prepare(tmp_path/'source','missing','1'*40,'missing','missing',output_config=output)
    assert report['status']=='WAITING_DISCOVERY' and len(report['missing_inputs'])==6
    assert report['planned_jobs'] is None and not report['config_written'] and not output.exists()


@pytest.mark.parametrize('count',[0,3,7])
def test_config_builder_uses_complete_dynamic_source_and_new_reservation(tmp_path,monkeypatch,count):
    root=tmp_path/'source';source=root/'auto_discovery_pilot'
    for name in builder.DISCOVERY_FILES|{'crop_hash_comparison.json'}:write(source/name,{})
    write(root/'contract/freeze_manifest.json',{})
    source_config=tmp_path/'source.yaml';source_config.write_text('{}')
    recipe=tmp_path/'recipe.yaml';recipe.write_text(yaml.safe_dump({'seed':42,'paper_ready':False,'models':{},'python':'python','runtime_sha256':'0'*64}))
    prior=tmp_path/'prior/trellis_initial'
    for name in builder.PRIOR_FILES:write(prior/name,{'producer_source_commit':'2'*40})
    write(prior.parent/'contract/freeze_manifest.json',{})
    jobs=[{'prepared':i<count-1} for i in range(count)]
    monkeypatch.setattr(builder,'discovery_binding',lambda c,_:(jobs,{'source_discovery_hashes':c['source_discovery_hashes']}))
    monkeypatch.setattr(builder,'validate_crop_comparison',lambda *_:None)
    monkeypatch.setattr(builder,'prior_trellis_source',lambda *_:{'reuse_eligible':False,'reuse_ineligible_reason':'runtime_bytes_unavailable'})
    monkeypatch.setattr(builder,'historical_algorithm_identity',lambda _:{'models/s4_trellis.py':'0'*64,'agents/core/common.py':'1'*64})
    args=(root,source_config,'1'*40,recipe,prior)
    report=builder.prepare(*args)
    assert report['status']=='READY_FOR_RESERVED_FREEZE' and report['planned_jobs']==count
    output=tmp_path/'new-config.yaml';freeze='20260904-07e8b05-v99'
    with pytest.raises(PilotError,match='reservation'):
        builder.prepare(*args,freeze_id=freeze,output_config=output)
    (tmp_path/'.freeze_ids'/freeze).mkdir(parents=True)
    report=builder.prepare(*args,freeze_id=freeze,output_config=output)
    assert report['config_written'] and report['execution_ready'] is False
    config=yaml.safe_load(output.read_text())
    assert config['source_gaussian_training_provenance']=='FRESH_OFFICIAL_TRAIN_ONLY'
    assert set(config['source_discovery_hashes'])==builder.DISCOVERY_FILES
    assert config['raw_reuse']['algorithm_files']['agents/core/common.py']=='1'*64
    assert 'planned_jobs' not in config  # Membership is always the sealed source.
    with pytest.raises(FileExistsError):builder.prepare(*args,freeze_id=freeze,output_config=output)
