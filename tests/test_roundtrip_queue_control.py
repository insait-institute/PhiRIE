import pytest
from robo.roundtrip.queue_control import choose_releases


def test_capacity_prefers_ready_multiarm_and_never_running_or_external():
 records={str(j):{'executable_units':10 if j==4 else 50} for j in range(1,6)}
 states={'1':{'state':'RUNNING'},'2':{'state':'RUNNING'},'3':{'state':'COMPLETED'},'4':{'state':'PENDING','reason':'JobHeldUser'},'5':{'state':'PENDING','reason':'JobHeldUser'}}
 assert choose_releases(records,states,3)==(['5'],None)
 assert choose_releases(records,states,2)==([],None)
 states['3']={'state':'PENDING','reason':'Priority'}
 assert choose_releases(records,states,3)==([],None)
 with pytest.raises(ValueError,match='unregistered'):choose_releases(records,{**states,'99':{'state':'RUNNING'}},3)


def test_scheduler_failure_stops_release_without_using_policy_outcome():
 records={'1':{'executable_units':50},'2':{'executable_units':50}}
 states={'1':{'state':'FAILED'},'2':{'state':'PENDING','reason':'JobHeldUser'}}
 assert choose_releases(records,states,3)==([],'scheduler_failure')
 states['1']={'state':'COMPLETED'}
 assert choose_releases(records,states,3)==(['2'],None)


def test_missing_scheduler_state_never_opens_an_extra_gpu_slot():
 records={'1':{'executable_units':50},'2':{'executable_units':50}}
 assert choose_releases(records,{'2':{'state':'PENDING','reason':'JobHeldUser'}},3)==([],'scheduler_state_unknown')


def test_hardware_groups_do_not_consume_hala_preparation_slot():
 from robo.roundtrip.queue_control import choose_group_releases
 records={str(j):{'executable_units':50} for j in range(1,7)}
 states={str(j):{'state':'RUNNING'} if j<=3 else {'state':'PENDING','reason':'JobHeldUser'} for j in range(1,7)}
 route={'capacities':{'hala':3,'h200':1},'job_groups':{str(j):'h200' if j>=5 else 'hala' for j in range(1,7)}}
 assert choose_group_releases(records,states,route)==(['5'],None)
 states['5']={'state':'RUNNING'}
 assert choose_group_releases(records,states,route)==([],None)
 states['1']={'state':'COMPLETED'}
 assert choose_group_releases(records,states,route)==(['4'],None)
 states['1']={'state':'FAILED'}
 assert choose_group_releases(records,states,route)==([],'scheduler_failure')


def test_group_routing_rejects_missing_jobs_and_unknown_groups():
 from robo.roundtrip.queue_control import choose_group_releases
 r={'1':{'executable_units':50}};s={'1':{'state':'RUNNING'}}
 with pytest.raises(ValueError,match='every registered'):choose_group_releases(r,s,{'capacities':{'hala':3},'job_groups':{}})
 with pytest.raises(ValueError,match='unknown routing'):choose_group_releases(r,s,{'capacities':{'hala':3},'job_groups':{'1':'other'}})
 with pytest.raises(ValueError,match='positive integer'):choose_group_releases(r,s,{'capacities':{'hala':0},'job_groups':{'1':'hala'}})


def test_two_running_prep_jobs_block_fourth_hala_eligibility_until_done():
 from robo.roundtrip.queue_control import choose_group_releases
 records={str(j):{'executable_units':50} for j in range(1,7)}
 states={str(j):{'state':'RUNNING'} if j<=4 else {'state':'PENDING','reason':'JobHeldUser'} for j in range(1,7)}
 route={'capacities':{'policy':3,'prep':1,'h200':1},'job_groups':{'1':'policy','2':'policy','3':'prep','4':'prep','5':'policy','6':'h200'},'shared_capacity_pools':[{'groups':['policy','prep'],'capacity':4}]}
 assert choose_group_releases(records,states,route)==(['6'],None)
 states['3']={'state':'COMPLETED'}
 assert choose_group_releases(records,states,route)==(['6','5'],None)


def test_external_registry_only_gpu_receipts_and_retries_partial_json(tmp_path):
 import json
 from robo.roundtrip.queue_control import external_records
 (tmp_path/'one.json').write_text(json.dumps({'job_id':'3','command':['sbatch','--gres=gpu:a6000:1']}))
 (tmp_path/'cpu.json').write_text(json.dumps({'job_id':'4','command':['sbatch','--cpus-per-task=4']}))
 (tmp_path/'partial.json').write_text('{')
 records,groups,lineage=external_records({'external_submission_globs':[{'pattern':str(tmp_path/'*.json'),'group':'prep','require_a6000_gres':True}]})
 assert set(records)=={'3'} and groups=={'3':'prep'} and lineage['3'][0]['sha256']


def test_scope_gets_one_shared_policy_slot_and_declared_validation_priority():
 from robo.roundtrip.queue_control import choose_group_releases
 records={str(j):{'executable_units':50} for j in [1,2,3,90,99]};records['99']['scheduler_priority']=-100
 states={str(j):{'state':'RUNNING'} if j<3 else {'state':'PENDING','reason':'JobHeldUser'} for j in [1,2,3,90,99]}
 r={'capacities':{'core':3,'scope':1},'job_groups':{'1':'core','2':'core','3':'core','90':'scope','99':'scope'},'group_order':['scope','core'],'shared_capacity_pools':[{'groups':['core','scope'],'capacity':3}]}
 assert choose_group_releases(records,states,r)==(['99'],None)
 states['99']={'state':'RUNNING'}
 assert choose_group_releases(records,states,r)==([],None)
 states['1']={'state':'COMPLETED'}
 assert choose_group_releases(records,states,r)==(['3'],None)


def test_manual_resolution_preserves_failed_state_and_requires_evidence(tmp_path):
 import json,hashlib
 from robo.roundtrip.queue_control import resolved_scheduler_states
 evidence=tmp_path/'failure.log';evidence.write_text('missing library')
 path=tmp_path/'resolution.json';r={'job_id':'7','allow_other_jobs':True,'classification':'environment_issue','resolution':'terminal failure audited, independent core jobs may proceed','scheduler_state':'FAILED','failure_evidence_path':str(evidence),'failure_evidence_sha256':hashlib.sha256(evidence.read_bytes()).hexdigest()};path.write_text(json.dumps(r))
 states={'7':{'state':'FAILED'},'8':{'state':'RUNNING'}}
 effective,lineage=resolved_scheduler_states(states,{'manual_resolution_glob':str(path)})
 assert states['7']['state']=='FAILED' and effective['7']['state']=='RESOLVED_SCHEDULER_FAILURE' and lineage['7']
 evidence.write_text('drift')
 with pytest.raises(ValueError,match='audited evidence'):resolved_scheduler_states(states,{'manual_resolution_glob':str(path)})


def test_zero_active_gap_waits_for_dynamic_producer_completion(tmp_path):
 import json
 from robo.roundtrip.queue_control import dynamic_producers_done,choose_group_releases
 route={'capacities':{'core':3},'job_groups':{'1':'core'},'external_submission_globs':[{'pattern':'/approved/held-submissions/*.json','group':'core'}]}
 assert choose_group_releases({'1':{'executable_units':50}},{'1':{'state':'COMPLETED'}},route)==([],None)
 assert not dynamic_producers_done(route)
 receipt=tmp_path/'complete.json';route['producer_completion_receipt']=str(receipt)
 assert not dynamic_producers_done(route)
 receipt.write_text(json.dumps({'completed':True,'submission_patterns':['/approved/held-submissions/*.json']}))
 assert dynamic_producers_done(route)
 receipt.write_text(json.dumps({'completed':True,'submission_patterns':[]}))
 with pytest.raises(ValueError,match='every dynamic'):dynamic_producers_done(route)
 assert dynamic_producers_done(None)


def loan_fixture(core_states,scope_states,prep_states):
 groups={};states={};records={};j=0
 for group,values in [('core',core_states),('scope',scope_states),('prep',prep_states)]:
  for state in values:
   j+=1;job=str(j);groups[job]=group;records[job]={'executable_units':50};states[job]={'state':state}
   if state=='PENDING':states[job]['reason']='JobHeldUser'
 route={'capacities':{'core':3,'scope':1,'prep':1},'job_groups':groups,'group_order':['scope','core','prep'],
   'shared_capacity_pools':[{'groups':['core','scope','prep'],'capacity':4}],
   'work_conserving_policy_pool':{'core_group':'core','scope_group':'scope','prep_group':'prep','total_capacity':4,'scope_reservation':1}}
 return records,states,route


def test_scope_borrows_drained_core_slots_and_idle_prep_slot():
 from robo.roundtrip.queue_control import choose_group_releases
 r,s,c=loan_fixture(['COMPLETED'],['PENDING']*5,[])
 assert choose_group_releases(r,s,c)==(['2','3','4','5'],None)
 r,s,c=loan_fixture(['COMPLETED'],['RUNNING','PENDING','PENDING','PENDING'],['PENDING'])
 assert choose_group_releases(r,s,c)==(['3','4','6'],None)
 assert c['capacities']['scope']==1  # immutable input policy


def test_pending_core_preserves_one_scope_reservation_and_prep():
 from robo.roundtrip.queue_control import choose_group_releases
 r,s,c=loan_fixture(['RUNNING','PENDING','PENDING'],['PENDING','PENDING'],['PENDING'])
 assert choose_group_releases(r,s,c)==(['4','2','6'],None)
 r,s,c=loan_fixture(['RUNNING'],['RUNNING','PENDING','PENDING'],['RUNNING'])
 assert choose_group_releases(r,s,c)==(['3'],None)
 # A returning prep job cannot interrupt four running loaned policy engines.
 r,s,c=loan_fixture([],['RUNNING']*4,['PENDING'])
 assert choose_group_releases(r,s,c)==([],None)


def test_h200_refill_only_after_all_target_jobs_drain_and_only_primary_held():
 from robo.roundtrip.queue_control import choose_hardware_refill
 r={j:{'executable_units':50} for j in ['1','2','3','4']}
 c={'capacities':{'core':3,'h200':1},'job_groups':{'1':'h200','2':'core','3':'core','4':'core'},'hardware_refill':{'target_group':'h200','source_group':'core'}}
 s={'1':{'state':'COMPLETED'},'2':{'state':'PENDING','reason':'JobHeldUser'},'3':{'state':'RUNNING'},'4':{'state':'PENDING','reason':'JobHeldUser'}}
 assert choose_hardware_refill(r,s,{'1','3','4'},c)=='4' # external2 never migrates
 s['1']={'state':'PENDING','reason':'JobHeldUser'}
 assert choose_hardware_refill(r,s,set(r),c) is None
 s['1']={'state':'COMPLETED'};s['2']={'state':'PENDING','reason':'Priority'}
 assert choose_hardware_refill(r,s,set(r),c)=='4'
 s['1']={'state':'FAILED'}
 assert choose_hardware_refill(r,s,set(r),c) is None


def test_refill_preserves_submission_and_only_updates_after_unstarted_preflight(tmp_path,monkeypatch):
 import json,hashlib
 from types import SimpleNamespace
 from robo.roundtrip import queue_control,local_policy_instance
 directory=tmp_path/'instance';directory.mkdir();record={'job_id':'4'}
 for name,value in [('submission.json',json.dumps(record)),('planned_units.jsonl','{}\n'),('commands.json','{}'),('engine.sbatch','original73-worker')]:
  (directory/name).write_text(value)
 digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 gate=tmp_path/'gate.json';gate.write_text('{}')
 route={'job_groups':{'4':'core'},'original_submissions':{'4':{'path':str(directory/'submission.json'),'sha256':digest(directory/'submission.json')}},'hardware_admission':str(gate),'hardware_admission_sha256':digest(gate),'same_device_identity':str(gate),'same_device_identity_sha256':digest(gate),
  'hardware_refill':{'source_group':'core','target_group':'h200','worker_root':'/frozen/workers','partition':'batch','qos':'normal','gres':'gpu:h200:1','node':'sof1-h200-3','input_hashes':{'4':{n:digest(directory/n) for n in ['planned_units.jsonl','commands.json','engine.sbatch']}}}}
 calls=[]
 def runner(cmd,**kwargs):
  calls.append(cmd)
  if cmd[0]=='squeue':return SimpleNamespace(stdout='PENDING|JobHeldUser\n')
  return SimpleNamespace(returncode=0,stdout='JobState=PENDING StartTime=Unknown Partition=batch QOS=normal ReqNodeList=sof1-h200-3 TresPerNode=gres/gpu:h200:1',stderr='')
 monkeypatch.setattr(queue_control.subprocess,'run',runner)
 def deny(*args,**kwargs):raise ValueError('partial runtime shard cannot resume')
 monkeypatch.setattr(local_policy_instance,'preflight_engine',deny)
 with pytest.raises(ValueError,match='partial runtime'):queue_control.refill_hardware('4',{'4':record},route,tmp_path/'failed')
 assert calls==[]
 monkeypatch.setattr(local_policy_instance,'preflight_engine',lambda *a,**k:None)
 assert queue_control.refill_hardware('4',{'4':record},route,tmp_path/'passed')
 assert route['job_groups']['4']=='h200'
 assert [x for x in calls if x[:2]==['scontrol','update']]==[['scontrol','update','JobId=4','Partition=batch','QOS=normal','Gres=gpu:h200:1','NodeList=sof1-h200-3']]
 assert (directory/'engine.sbatch').read_text()=='original73-worker'
 assert json.loads((tmp_path/'passed/routing_after_refill.json').read_text())['job_groups']['4']=='h200'
 (directory/'engine.sbatch').write_text('changed')
 with pytest.raises(ValueError,match='inputs changed'):queue_control.refill_hardware('4',{'4':record},route,tmp_path/'drift')
