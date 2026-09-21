"""Cache only authenticated original-scene replays; retain independent passes."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import pytest
from run.icra2027 import e4_terminal_cache as cache
from run.icra2027 import e4_planning_terminal as terminal


@pytest.fixture
def setup(tmp_path,monkeypatch):
    stage=tmp_path/'freeze';stage.mkdir()
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps({'qualification_tasks':[]}))
    source=tmp_path/'source.json';source.write_text(json.dumps({'source':{'protocol':cache.api.identity(protocol)}}))
    proof=tmp_path/'host.json';proof.write_text(json.dumps(dict(allowed_host='sof1-h200-0',source_commit='old',
        freeze_id='original',disabled_features=terminal.ISA_MODE,acceptance_tolerances_changed=False)))
    config=dict(replay_mode=cache.MODE,freeze_id='freeze',scene_cache_root=str(stage/'scene_validation'),
        source=dict(config=cache.api.identity(source),code_root=str(tmp_path),code_commit='old',freeze_id='original',
                    environment={'path':'env.sh'},runtime={'path':'python'}),jobs={'a':{'job_id':'11'},'b':{'job_id':'12'}},
        cache_hosts={'sof1-h200-0':cache.api.identity(proof)})
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(config))
    e0=tmp_path/'e0.json';e0.write_text('{}');code={'commit':'new','code_root':str(tmp_path),'dirty':False}
    context=(stage,cache.api.identity(config_path),cache.api.identity(e0))
    monkeypatch.setattr(cache,'_context',lambda c,k:context)
    monkeypatch.setattr(cache.screen,'_code_snapshot',lambda c:code)
    monkeypatch.setattr(cache.screen,'evidence_root',lambda:tmp_path)
    monkeypatch.setattr(cache.socket,'gethostname',lambda:'sof1-h200-0')
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('NPY_DISABLE_CPU_FEATURES',terminal.ISA_MODE)
    calls=[]
    def replay(c,*,scene_ids,process_receipt):
        sid=scene_ids[0];calls.append(sid)
        result=dict(source_config=json.loads(source.read_text()),protocol=json.loads(protocol.read_text()),
            scenes={sid:dict(job=config['jobs'][sid],evidence={})},cells=[])
        worker=Path(terminal.__file__).with_name('e4_full_terminal_worker.py')
        process_receipt.update(returncode=0,stdout=json.dumps(result),stderr='',runtime_seconds=1.,
            request_sha256=hashlib.sha256(json.dumps(dict(c,_scene_ids=[sid])).encode()).hexdigest(),
            cwd=c['source']['code_root'],worker=cache.api.identity(worker),command=terminal._full_worker_command(c,worker))
        return result
    monkeypatch.setattr(terminal,'_full_replay',replay)
    return stage,config_path,config,code,context,calls,replay


def run(setup,sid='a'):
    stage,path,*_=setup
    return cache.produce_scene(config_path=path,stage_root=stage,expected_code_commit='new',scene_id=sid)


def test_two_independent_passes_resume_without_reexecution(setup):
    assert run(setup)['status']=='PASS'
    assert setup[5]==['a','a']
    assert run(setup)['status']=='PASS' and setup[5]==['a','a']
    run(setup,'b')
    result=cache.compose(setup[2],setup[3])
    assert set(result['scenes'])=={'a','b'}
    assert set(result['validation_cache'])=={'a','b'} and len(setup[5])==4


def test_independent_failure_preserves_first_and_resumes_only_missing(setup,monkeypatch):
    original=setup[6];counter=[]
    def fail(c,**kw):
        counter.append(1)
        if len(counter)==2:raise ValueError('second process environment failure')
        return original(c,**kw)
    monkeypatch.setattr(terminal,'_full_replay',fail)
    with pytest.raises(ValueError,match='second process'):run(setup)
    root=setup[0]/'scene_validation/a'
    assert (root/'first/seal.json').exists() and not (root/'independent').exists()
    assert (setup[0]/'scene_validation_attempts/a/123-independent.json').exists()
    first=(root/'first/result.json').read_bytes()
    monkeypatch.setenv('SLURM_JOB_ID','456');monkeypatch.setattr(terminal,'_full_replay',original)
    run(setup)
    assert (root/'first/result.json').read_bytes()==first and setup[5]==['a','a']


@pytest.mark.parametrize('change',['missing_scene','extra_scene','missing_pass','extra_pass','tamper','symlink'])
def test_compose_rejects_incomplete_or_tampered_cache(setup,change):
    run(setup);run(setup,'b');root=setup[0]/'scene_validation'
    if change=='missing_scene':(root/'b').rename(setup[0]/'missing')
    elif change=='extra_scene':(root/'extra').mkdir()
    elif change=='missing_pass':(root/'a/independent').rename(setup[0]/'missing')
    elif change=='extra_pass':(root/'a/unreviewed').mkdir()
    elif change=='tamper':(root/'a/first/result.json').write_text('{}')
    else:
        original=root/'a/first/result.json';original.rename(setup[0]/'linked.json');original.symlink_to(setup[0]/'linked.json')
    with pytest.raises((ValueError,terminal.screen.CandidateScreenError)):cache.compose(setup[2],setup[3])


def reseal(path):
    manifest=json.loads((path/'manifest.json').read_text())
    manifest['files']={name:cache.screen._identity_without_path(path/name) for name in manifest['files']}
    (path/'manifest.json').write_bytes(cache.screen._json_bytes(manifest))
    seal=json.loads((path/'seal.json').read_text());seal['members']={'manifest.json':cache.screen._identity_without_path(path/'manifest.json')}
    (path/'seal.json').write_bytes(cache.screen._json_bytes(seal))


@pytest.mark.parametrize('change',['code','config','scene','process_exit','process_command','process_request',
                                  'process_worker','process_output','process_host','process_runtime','result_roster'])
def test_self_consistent_bundle_cannot_spoof_execution(setup,change):
    run(setup);path=setup[0]/'scene_validation/a/first'
    name='manifest.json' if change in {'code','config','scene'} else 'process.json'
    payload=json.loads((path/name).read_text())
    if change=='code':payload['code']['commit']='other'
    elif change=='config':payload['config']={}
    elif change=='scene':payload['scene_id']='b'
    elif change=='process_exit':payload['returncode']=1
    elif change=='process_command':payload['command']=['bash','-c','echo fabricated']
    elif change=='process_request':payload['request_sha256']='0'*64
    elif change=='process_worker':payload['worker']={}
    elif change=='process_output':payload['stdout']='{}'
    elif change=='process_host':payload['hostname']='unproven'
    elif change=='process_runtime':payload['runtime_seconds']=-1
    else:
        result=json.loads(payload['stdout']);result['scenes']={'b':result['scenes']['a']}
        payload['stdout']=json.dumps(result);(path/'result.json').write_bytes(cache.screen._json_bytes(result))
    (path/name).write_bytes(cache.screen._json_bytes(payload));reseal(path)
    with pytest.raises(ValueError):cache._pass(setup[2],setup[3],'a','first',setup[4])


def test_original_metadata_mutation_rejects_reuse(setup):
    run(setup)
    Path(setup[2]['source']['config']['path']).write_text('{}')
    with pytest.raises(ValueError,match='evidence changed'):run(setup)


def test_unknown_scene_or_unbound_runtime_rejected_before_process(setup,monkeypatch):
    with pytest.raises(ValueError,match='scene differs'):run(setup,'replacement')
    monkeypatch.setenv('NPY_DISABLE_CPU_FEATURES','')
    with pytest.raises(ValueError,match='ISA'):run(setup)
    assert setup[5]==[]
