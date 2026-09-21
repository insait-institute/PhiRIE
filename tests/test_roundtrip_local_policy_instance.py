import json
from pathlib import Path
import pytest
from robo.roundtrip.local_policy_instance import local_commands,validate_admission
from robo.roundtrip.matrix import sha,canonical_hash


def test_endpoint_only_override_does_not_change_planned_identity():
 original={'u':{'planned_unit_sha256':'frozen','argv':['python','-m','robo.roundtrip.paired','--config','frozen.json','--host','hala','--port','8017'],'requires_files':['unchanged']}}
 changed=local_commands(original,'127.0.0.1',30217)
 assert original['u']['argv'][-3:] == ['hala','--port','8017']
 assert changed['u']['argv'][-3:]==['127.0.0.1','--port','30217']
 assert changed['u']['planned_unit_sha256']=='frozen' and changed['u']['requires_files']==['unchanged']
 with pytest.raises(ValueError,match='exactly one'):local_commands({'u':{'argv':['--host','a','--host','b','--port','1']}},'localhost',1)


def test_admission_requires_actual_episode_and_exact_evidence(tmp_path):
 gate=tmp_path/'gate.json';gate.write_text(json.dumps({'passed':True}))
 episode=tmp_path/'result.json';episode.write_text(json.dumps({'executed':True,'success':False,'error':None}))
 metadata={'checkpoint':'pinned'}
 receipt={'kind':'local_policy_admission','passed':True,'cross_service_gate':str(gate),'cross_service_gate_sha256':sha(gate),'native_episode':str(episode),'native_episode_sha256':sha(episode),'policy_metadata_sha256':canonical_hash(metadata)}
 p=tmp_path/'admission.json';p.write_text(json.dumps(receipt))
 assert validate_admission(p,metadata)==receipt
 with pytest.raises(ValueError,match='identity'):validate_admission(p,{'checkpoint':'changed'})
 episode.write_text(json.dumps({'executed':False,'success':None,'error':'failure'}))
 with pytest.raises(ValueError,match='evidence changed'):validate_admission(p,metadata)
 receipt['native_episode_sha256']=sha(episode);p.write_text(json.dumps(receipt))
 with pytest.raises(ValueError,match='not validly executed'):validate_admission(p,metadata)


def test_port_readiness_requires_our_process():
 import os,socket
 from robo.roundtrip.local_policy_instance import owns_listening_port
 with socket.socket() as listener:
  listener.bind(('127.0.0.1',0));listener.listen();port=listener.getsockname()[1]
  assert owns_listening_port(os.getpid(),port)
  assert not owns_listening_port(99999999,port)


def test_engine_preflight_no_missing_arm_or_cross_process_resume(tmp_path,monkeypatch):
 from robo.roundtrip import local_policy_instance as m
 units=[];commands={};root=tmp_path/'workers'
 for method in ('REF_NATIVE','B0_FIXED_NATIVE'):
  cp=tmp_path/(method+'.json');cfg={'policy_engine_protocol':m.ENGINE_PROTOCOL};cp.write_text(json.dumps(cfg))
  unit={'unit_id':method,'controller_method':method,'reset_id':'r0','split':'development','config_path':str(cp),'config_sha256':canonical_hash(cfg),'result_path_planned':str(root/method/'runner/episode/result.json')}
  units.append(unit);commands[method]={'planned_unit_sha256':canonical_hash(unit),'requires_files':[]}
 m.preflight_engine(units,commands,root,['REF_NATIVE','B0_FIXED_NATIVE'],1,pilot=True)
 with pytest.raises(ValueError,match='builds must be resolved'):m.preflight_engine(units,{'REF_NATIVE':commands['REF_NATIVE']},root,['REF_NATIVE','B0_FIXED_NATIVE'],1,pilot=True)
 with pytest.raises(ValueError,match='complete ordered'):m.preflight_engine(units,commands,root,['REF_NATIVE'],1,pilot=True)
 (root/'REF_NATIVE').mkdir(parents=True)
 with pytest.raises(ValueError,match='partial runtime'):m.preflight_engine(units,commands,root,['REF_NATIVE','B0_FIXED_NATIVE'],1,pilot=True)
 (root/'REF_NATIVE/terminal.json').write_text('{}')
 monkeypatch.setattr(m,'validate_terminal',lambda u,t:{'terminal_status':'RECORDED'})
 with pytest.raises(ValueError,match='cannot resume'):m.preflight_engine(units,commands,root,['REF_NATIVE','B0_FIXED_NATIVE'],1,pilot=True)


def test_engine_admission_binds_actual_same_process_pair(tmp_path):
 from robo.roundtrip import local_policy_instance as m
 engine={'engine_id':'planned','engine_protocol':m.ENGINE_PROTOCOL,'canonical_instance_id':'native1','process_uuid':'process1','runtime_fingerprint':'sha1','runtime':{'source_hashes':{'server':'frozen'},'python':'3.11','packages':{'jax':'same'},'environment':{'CUDA_VISIBLE_DEVICES':'0'}}}
 metadata={'checkpoint':'pinned','policy_engine':engine}
 assert m.validate_engine(metadata,'native1')==engine
 with pytest.raises(ValueError,match='canonical'):m.validate_engine(metadata,'native2')
 evidence={}
 data={'endpoint':{'policy_metadata':metadata},'interleaving':{'passed':True,'metadata':metadata},'full_chunks':{'passed':True,'metadata':metadata,'repeat_reference':True,'model_chunk_requests':18,'full_actions_per_chunk':50,'endpoints':[{'host':'localhost','port':1}]*2}}
 for k in ('reference_episode','comparison_episode'):data[k]={'executed':True,'success':False,'error':None,'policy_engine':engine}
 for k,v in data.items():
  p=tmp_path/(k+'.json');p.write_text(json.dumps(v));evidence[k]={'path':str(p),'sha256':sha(p)}
 receipt={'kind':'per_canonical_engine_admission','passed':True,'engine_protocol':m.ENGINE_PROTOCOL,'policy_definition_sha256':m.policy_definition(metadata),'pilot_engine':engine,'evidence':evidence}
 path=tmp_path/'admission.json';path.write_text(json.dumps(receipt))
 # A different production process is allowed; its paired REF may never be reused.
 newmeta={**metadata,'policy_engine':{**engine,'process_uuid':'process2'}}
 m.validate_engine_admission(path,newmeta)
 changed_runtime={**engine['runtime'],'source_hashes':{'server':'changed'}}
 with pytest.raises(ValueError,match='source or package'):m.validate_engine_admission(path,{**newmeta,'policy_engine':{**engine,'runtime':changed_runtime}})
 with pytest.raises(ValueError,match='definition'):m.validate_engine_admission(path,{**newmeta,'checkpoint':'changed'})
 data['comparison_episode']['policy_engine']={**engine,'process_uuid':'process2'}
 cp=Path(evidence['comparison_episode']['path']);cp.write_text(json.dumps(data['comparison_episode']))
 evidence['comparison_episode']['sha256']=sha(cp);path.write_text(json.dumps(receipt))
 with pytest.raises(ValueError,match='within one process'):m.validate_engine_admission(path,newmeta)
