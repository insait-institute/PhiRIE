import copy
from pathlib import Path
import pytest
from run.icra2027 import e3_pilot_evaluation as pilot


def test_gt_pinning_occurs_only_after_complete_construction_validation(monkeypatch):
    events=[]
    config={'scenes':[{'scene_id':'13c3e046d7'}]}
    monkeypatch.setattr(pilot.matching,'validate_construction',lambda c:events.append(('validated',copy.deepcopy(c))))
    def identity(path):
        assert events[0][0]=='validated'
        events.append(('GT',str(path)))
        return {'path':str(path),'size_bytes':5,'sha256':'a'*64}
    monkeypatch.setattr(pilot,'identity',identity)
    result=pilot.pin_gt_after_construction(config)
    assert len(events)==4
    assert set(result['scenes'][0]['gt_inputs'])==set(pilot.matching.GT_FILES)
    assert 'gt_inputs' not in config['scenes'][0]
    assert all(path.startswith('/data/ScanNetpp/data/13c3e046d7/scans/') for kind,path in events[1:])


def test_failed_construction_never_opens_gt(monkeypatch):
    def invalid(_):raise ValueError('unsealed construction')
    monkeypatch.setattr(pilot.matching,'validate_construction',invalid)
    monkeypatch.setattr(pilot,'identity',lambda _:pytest.fail('GT accessed before valid seal'))
    with pytest.raises(ValueError,match='unsealed'):
        pilot.pin_gt_after_construction({'scenes':[{'scene_id':'13c3e046d7'}]})


def test_identity_preserves_declared_dataset_path(monkeypatch):
    expected=Path('/data/ScanNetpp/data/13c3e046d7/scans/segments.json')
    monkeypatch.setattr(pilot.matching,'_absolute_identity',lambda path:{'path':str(path)})
    assert pilot.identity(expected)['path']==str(expected)


@pytest.fixture
def preparation(tmp_path,monkeypatch):
    import json,yaml
    code=tmp_path/'code';root=tmp_path/'evidence';code.mkdir();root.mkdir()
    monkeypatch.setattr(pilot,'CODE',code);monkeypatch.setattr(pilot.e3,'REPOSITORY_ROOT',root)
    construction=root/'outputs/icra2027/construction/agentic';directory=construction/'control/13c3e046d7'
    directory.mkdir(parents=True);(directory/'seal.json').write_text('{}')
    monkeypatch.setattr(pilot.e3,'checked_repo_path',lambda p,*a,**k:Path(p))
    monkeypatch.setattr(pilot.e3,'_load_control_scene',lambda *_:(directory,{'job_count':1},{}))
    gen=code/'gen.yaml';gen.write_text(yaml.safe_dump({k:'frozen' for k in pilot.DISCOVERY_FIELDS}))
    producer=code/'producer.json';producer.write_text(json.dumps({'resource_inventory':[{'kind':'experiment_config',
        'sha256':pilot.e3.sha256_file(gen),'resolved_path':str(gen)}]}))
    roster=code/'roster.yaml';roster.write_text('population: {scene_ids: [13c3e046d7]}')
    jobs=code/'jobs.yaml';jobs.write_text(yaml.safe_dump({'population':{'scene_roster_config':str(roster)},
        'automatic_sources':[{'scene_id':'13c3e046d7','planned_jobs':1,'initial_pools':{'trellis':{
            'producer_provenance':{'contract_path':str(producer),'config_sha256':pilot.e3.sha256_file(gen)}}}}]}))
    policies=code/'policies.yaml';policies.write_text('policies: frozen')
    contract={'code':{},'resource_inventory':[{'id':name,'resolved_path':str(path),
        'hash_method':'content_sha256','sha256':pilot.e3.sha256_file(path)} for name,path in
        [('agentic_fresh_jobs',jobs),('agentic_automatic_policies',policies)]]}
    (construction.parent/'contract').mkdir();(construction.parent/'contract/freeze_manifest.json').write_text(json.dumps(contract))
    execution=code/'execution.yaml';execution.write_text(yaml.safe_dump({'freeze_id':'construction',
        'pilot':{'scene_id':'13c3e046d7','planned_jobs':1}}))
    template=code/'configs/experiments/icra2027/e3_matching_pilot_38d/freeze.yaml'
    template.parent.mkdir(parents=True);template.write_text('schema_version: 1')
    (root/'outputs/icra2027/.freeze_ids/evaluation').mkdir(parents=True)
    def pin(config):
        config['scenes'][0]['gt_inputs']={'fixture':True}
        return config
    monkeypatch.setattr(pilot,'pin_gt_after_construction',pin)
    return execution,code/'new-config',jobs,policies,construction


def test_prepare_copies_exact_configs_and_refuses_overwrite(preparation):
    import yaml
    execution,destination,jobs,policies,_=preparation
    result=pilot.prepare(execution,'evaluation',destination)
    assert result['status']=='CONFIG_WRITTEN_E0_REQUIRED'
    assert (destination/'construction_jobs.yaml').read_bytes()==jobs.read_bytes()
    assert (destination/'construction_policies.yaml').read_bytes()==policies.read_bytes()
    config=yaml.safe_load((destination/'matching.yaml').read_text())
    assert config['planned_jobs']==1 and config['planned_scenes']==1 and config['paper_ready'] is False
    freeze=yaml.safe_load((destination/'freeze.yaml').read_text())
    assert {r['id'] for r in freeze['input_roots']}=={'e3_matching_config','e3_matching_jobs',
        'e3_matching_policies','e3_pilot_execution','e3_matching_roster'}
    with pytest.raises(FileExistsError):pilot.prepare(execution,'evaluation',destination)


def test_changed_construction_jobs_block_publication_before_gt(preparation,monkeypatch):
    execution,destination,jobs,_,_=preparation;jobs.write_text('changed')
    monkeypatch.setattr(pilot,'pin_gt_after_construction',lambda _:pytest.fail('GT read after job drift'))
    with pytest.raises(ValueError,match='jobs changed'):pilot.prepare(execution,'evaluation',destination)
    assert not destination.exists()


@pytest.mark.parametrize('bad_rows',[False,True])
@pytest.mark.parametrize('drift',[None,'external_evaluation_manifest_sha256','evaluation_freeze_id',
    'construction_freeze_id','freeze_id','evaluation_code_commit'])
def test_run_delegates_and_preserves_complete_pilot_population(tmp_path,monkeypatch,bad_rows,drift):
    import json,yaml
    root=tmp_path/'evidence';construction=root/'outputs/icra2027/construction/agentic'
    out=root/'outputs/icra2027/evaluation';out.mkdir(parents=True)
    config_path=tmp_path/'matching.yaml';config={'freeze_id':'evaluation','planned_jobs':1,
        'scenes':[{'scene_id':'13c3e046d7','construction_root':str(construction)}]}
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(pilot.e3,'REPOSITORY_ROOT',root)
    calls=[]
    monkeypatch.setattr(pilot.e3,'_validate_cli_execution',lambda *a,**k:calls.append('E0'))
    monkeypatch.setattr(pilot.matching,'validate_construction',lambda _:calls.append('construction'))
    def export(c,e,d):
        assert calls==['E0','construction'];d.mkdir();(d/'evaluation_references.json').write_text('{}');calls.append('matching')
    monkeypatch.setattr(pilot.matching,'export',export)
    monkeypatch.setattr(pilot.e3,'_load_control_scene',lambda *_:(construction,{'proposals':[{'proposal_id':'p'}]},
        {'members':{'controller_shard.json':'sealed'}}))
    monkeypatch.setattr(pilot.e3,'_load_inventory',lambda *a,**k:({'scenes':[{'scene_id':'13c3e046d7','jobs':[{'job_id':'j'}]}]},{}))
    monkeypatch.setattr(pilot.matching,'load_references',lambda *a:({}, {'config_sha256':pilot.e3.sha256_file(config_path),'code_commit':'a'*40}))
    def evaluate(argv):
        assert '--evaluate' in argv and '--aggregate' not in argv
        destination=out/'agentic/evaluation/13c3e046d7';destination.mkdir(parents=True)
        (destination/'eval_shard.json').write_text('{}');calls.append('evaluate');return 0
    monkeypatch.setattr(pilot.e3,'main',evaluate)
    def load(*a,**k):
        value={'job_count':1,'rows':[{}]*(4 if bad_rows else 5),
            'geometry_reference_jobs':0,'geometry_unmatched_jobs':1,'proposal_metrics':{'p':{}},
            'evaluation_freeze_id':'evaluation','construction_freeze_id':'construction','freeze_id':'construction',
            'evaluation_code_commit':'a'*40,'external_evaluation_manifest_sha256':
                pilot.e3.sha256_file(out/'evaluation_matching/evaluation_references.json')}
        if drift:value[drift]='changed'
        return value
    monkeypatch.setattr(pilot.e3,'_load_eval_scene',load)
    if drift:
        with pytest.raises(ValueError,match='identity differs'):pilot.run(config_path,tmp_path/'contract.json')
    elif bad_rows:
        with pytest.raises(ValueError,match='lost planned'):pilot.run(config_path,tmp_path/'contract.json')
    else:
        assert pilot.run(config_path,tmp_path/'contract.json')['policy_rows']==5
        assert pilot.run(config_path,tmp_path/'contract.json')['status']=='PASS'
        assert calls.count('matching')==1 and calls.count('evaluate')==1


def test_full_preparation_authenticates_execution_before_gt(preparation, monkeypatch):
    import yaml
    execution, destination, jobs, policies, _ = preparation
    events = []
    def binding(config, scenes):
        assert config['mode'] == 'pilot' and scenes == ['13c3e046d7']
        events.append('execution-bound')
    def pin(config):
        assert events == ['execution-bound']
        assert config['mode'] == 'full'
        events.append('GT')
        return config
    monkeypatch.setattr(pilot.matching, '_pilot_from_execution', binding)
    monkeypatch.setattr(pilot, 'pin_gt_after_construction', pin)
    result = pilot.prepare(execution, 'evaluation', destination, mode='full')
    assert result['mode'] == 'full' and events == ['execution-bound', 'GT']
    assert (destination/'construction_jobs.yaml').read_bytes() == jobs.read_bytes()


def test_full_execution_binding_failure_prevents_gt_and_publication(preparation, monkeypatch):
    execution, destination, *_ = preparation
    def fail(*_): raise ValueError('full execution differs')
    monkeypatch.setattr(pilot.matching, '_pilot_from_execution', fail)
    monkeypatch.setattr(pilot, 'pin_gt_after_construction', lambda _: pytest.fail('GT leaked'))
    with pytest.raises(ValueError, match='execution differs'):
        pilot.prepare(execution, 'evaluation', destination, mode='full')
    assert not destination.exists()


def test_missing_full_control_scene_prevents_gt(preparation, monkeypatch):
    import yaml
    execution, destination, jobs, *_ = preparation
    data = yaml.safe_load(jobs.read_text())
    Path(data['population']['scene_roster_config']).write_text('population: {scene_ids: [13c3e046d7, 27dd4da69e]}')
    original = pilot.e3._load_control_scene
    def control(root, scene):
        if scene == '27dd4da69e': raise FileNotFoundError('unsealed last scene')
        return original(root, scene)
    monkeypatch.setattr(pilot.e3, '_load_control_scene', control)
    monkeypatch.setattr(pilot, 'pin_gt_after_construction', lambda _: pytest.fail('GT leaked'))
    with pytest.raises(FileNotFoundError, match='unsealed'):
        pilot.prepare(execution, 'evaluation', destination, mode='full')
    assert not destination.exists()


def test_full_config_cannot_be_run_as_partial_pilot(tmp_path):
    import yaml
    config = tmp_path/'matching.yaml'
    config.write_text(yaml.safe_dump({'mode':'full', 'scenes':[{'scene_id':'13c3e046d7'}]}))
    with pytest.raises(ValueError, match='pilot-only'):
        pilot.run(config, tmp_path/'contract.json')
