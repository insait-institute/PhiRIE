import copy
import json
from pathlib import Path
import pytest
import yaml
from robo.roundtrip import matrix as m
from robo.manifest.hash import canonical_hash


def config():return yaml.safe_load(Path('configs/experiments/sim_recon_sim/scale_up/dev.yaml').read_text())


def unit():
 c=config();_,rows=m.enumerate_plan(c);r=rows[0];r.update(canonical_instance_id='native-'+('a'*64),unit_id='unit-a',terminal_status='NOT_SCHEDULED');return r


def terminal(u,success=False):
 return {**{k:u[k] for k in m.IDENTITY_FIELDS},'unit_id':u['unit_id'],'planned_unit_sha256':canonical_hash(u),
 'terminal_status':'RECORDED','executed':True,'success':success,'result':{'executed':True,'success':success}}


def test_dev_exact_8_builds_40_units_per_arm():
 slots,rows=m.enumerate_plan(config());assert len(slots)==8 and len(rows)==80
 assert len({s['instance_slot_id'] for s in slots})==8
 assert len({r['plan_unit_id'] for r in rows})==80
 assert all(r['canonical_instance_id'] is None for r in rows)
 assert sum(r['controller_method']=='REF_NATIVE' for r in rows)==40
 assert {r['native_horizon'] for r in rows if r['task_id']=='PickPlaceSinkToCounter'}=={900}


def test_test_budget_480_per_arm_without_models():
 c=config();c.update(split='test',layout_ids=list(range(20,28)),tasks={**c['tasks'],'ThirdTaskResolved':750})
 c['reset_perturbations']=[{'reset_id':f'r{i}','translation_world_m':[0,0,0],'policy_rng_seed':i} for i in range(10)]
 slots,rows=m.enumerate_plan(c);assert len(slots)==48 and len(rows)==960


@pytest.mark.parametrize('field',['layout_ids','instance_generation_seeds','controller_methods','reset_perturbations'])
def test_duplicate_roster_rejected(field):
 c=config();c[field].append(c[field][0])
 with pytest.raises(ValueError,match='duplicate'):m.enumerate_plan(c)


def test_atomic_no_overwrite(tmp_path):
 p=tmp_path/'receipt.json';m.save_new(p,{'a':1})
 with pytest.raises(FileExistsError):m.save_new(p,{'a':2})
 assert json.loads(p.read_text())=={'a':1}
 assert not list(tmp_path.glob('*.partial.*'))


def test_collect_retains_pending_and_native_failure(tmp_path):
 u=unit();v={**u,'unit_id':'unit-b','controller_method':'B0_FIXED_NATIVE'}
 m.save_new(tmp_path/'shards'/'a'/'terminal.json',terminal(u,False))
 rows=m.collect([u,v],tmp_path/'shards',tmp_path/'merge')
 assert rows[0]['executed'] is True and rows[0]['success'] is False
 assert rows[1]['terminal_status']=='NOT_SCHEDULED' and rows[1]['success'] is None
 status=json.loads((tmp_path/'merge/status.json').read_text());assert status['complete'] is False


def test_collect_external_error_is_not_measured_zero(tmp_path):
 u=unit();t=terminal(u);t.update(terminal_status='RESOURCE_FAILED',executed=False,success=None)
 m.save_new(tmp_path/'shards/a/terminal.json',t);m.collect([u],tmp_path/'shards',tmp_path/'merged')
 s=json.loads((tmp_path/'merged/status.json').read_text())['groups'][0]
 assert s['unmeasured']==1 and s['success_per_planned'] is None


@pytest.mark.parametrize('change',['duplicate','identity','config','false_execution'])
def test_collect_rejects_corrupted_or_duplicate_shards(tmp_path,change):
 u=unit();t=terminal(u)
 if change=='identity':t['reset_id']='wrong'
 if change=='config':t['planned_unit_sha256']='wrong'
 if change=='false_execution':t['executed']=False
 m.save_new(tmp_path/'shards/a/terminal.json',t)
 if change=='duplicate':m.save_new(tmp_path/'shards/b/terminal.json',t)
 with pytest.raises(ValueError):m.collect([u],tmp_path/'shards',tmp_path/'merged')
 assert not (tmp_path/'merged').exists()


def test_dispatch_no_array_and_resume_no_duplicate(tmp_path,monkeypatch):
 u=unit();commands={u['unit_id']:{'planned_unit_sha256':canonical_hash(u),'argv':['python','existing_runner.py']}}
 with pytest.raises(ValueError,match='arrays'):m.dispatch([u],commands,tmp_path/'jobs',sbatch_args=['--array=1-4'],max_jobs=1)
 calls=[]
 monkeypatch.setattr(m,'git_snapshot',lambda:{'commit':'a'*40,'dirty':False})
 class Result:returncode=0;stdout='1234\n';stderr=''
 monkeypatch.setattr(m.subprocess,'run',lambda argv,**kwargs:(calls.append(argv) or Result()))
 r=m.dispatch([u],commands,tmp_path/'jobs',sbatch_args=['--partition=debug'],max_jobs=1)
 assert r[0]['job_id']=='1234' and len(calls)==1 and '--array' not in str(calls)
 assert m.dispatch([u],commands,tmp_path/'jobs',sbatch_args=[],max_jobs=1)==[] and len(calls)==1


def test_worker_delegates_and_publishes_terminal_once(tmp_path,monkeypatch):
 u=unit();out=tmp_path/'unit'
 class Result:returncode=0
 def run(argv,**kwargs):
  assert argv==['existing_harness_runner'];m.save_new(out/'episode/result.json',{'executed':True,'success':False});return Result()
 monkeypatch.setattr(m.subprocess,'run',run)
 r=m.worker(u,['existing_harness_runner'],out);assert r['terminal_status']=='RECORDED' and r['success'] is False
 with pytest.raises(FileExistsError):m.worker(u,['existing_harness_runner'],out)


def test_worker_crash_has_no_invented_native_outcome(tmp_path,monkeypatch):
 class Result:returncode=1
 monkeypatch.setattr(m.subprocess,'run',lambda *a,**kw:Result())
 r=m.worker(unit(),['existing_harness_runner'],tmp_path/'unit')
 assert r['terminal_status']=='CODE_FAILED' and r['executed'] is False and r['success'] is None


def test_acquisition_uses_one_generated_instance_all_resets_and_public_train(tmp_path,monkeypatch):
 import numpy as np
 c=config();slots,_=m.enumerate_plan(c);slot=slots[-1];seen=[]
 class Adapter:
  def __init__(self,cfg):
   self.cfg=cfg;self.rendered=0;self.closed=False;self.imported=False;seen.append(self)
   self.state={'qpos':[0.],'qvel':[0.],'act':[],'time':0.,'integration_state':[0.], 'native_metadata':{'lang':'Pick the cup and place it in the bowl.'},'native_private':'opaque-private-asset'}
  def reset_from_spec(self,state):self.reset=state
  def export_reference_for_evaluator(self,path):path.mkdir();(path/'scene.xml').write_text('<mujoco/>');(path/'canonical_state.json').write_text(json.dumps(self.state))
  def get_state(self):return self.state
  def import_xml(self,xml,*,canonical_state):self.state=canonical_state;self.imported=True
  def native_success(self):return False
  def get_named_body_state(self):return {'private_target':[0,0,1,1,0,0,0]}
  def get_policy_observation(self):return {'annotation.human.task_description':'Pick the cup and place it in the bowl.'}
  def render_capture(self,camera,*,width,height):
   self.rendered+=1;return {'rgb':np.zeros((height,width,3),np.uint8),'depth_m':np.ones((height,width),np.float32),'K':np.array([[3,0,1],[0,3,1],[0,0,1.]]),'T_world_from_camera':np.array(camera['T_world_from_camera']),'timestamp':0.}
  def close(self):self.closed=True
 def poses(_):
  from robo.roundtrip.capture_native import NATIVE_CAMERAS
  l=np.eye(4);r=np.eye(4);r[0,3]=1;return dict(zip(NATIVE_CAMERAS,[l,r]))
 monkeypatch.setattr(m,'git_snapshot',lambda:{'commit':'c'*40,'dirty':False})
 binding=m.acquire_instance(c,slot['instance_slot_id'],tmp_path/'capture',adapter_factory=Adapter,
  identity_factory=lambda b,c:{'canonical_instance_id':'native-'+('a'*64)},
  reset_factory=lambda i,r:{'resets':r},camera_pose_reader=poses,width=4,height=3)
 assert seen[0].reset=={'seed':1} and seen[0].cfg['horizon']==900
 assert seen[0].rendered==8 and seen[0].closed and seen[0].imported
 bank=json.loads(Path(binding['reset_bank']).read_text());assert len(bank['resets'])==5
 assert bank['resets'][1]['delta_world'][0][3]==.01
 assert binding['policy_invoked'] is False
 public=Path(binding['capture_public'])
 for f in public.rglob('*'):
  if f.suffix=='.json':assert 'opaque-private-asset' not in f.read_text()
 assert not (public/'test').exists()


def test_acquisition_rejects_unknown_slot_before_native_loading(tmp_path):
 with pytest.raises(ValueError,match='slot not'):m.acquire_instance(config(),'unplanned',tmp_path/'out')
 assert not (tmp_path/'out').exists()


def bindings_for_plan(tmp_path,c):
 import numpy as np
 from robo.roundtrip.identity import create_canonical_instance,create_reset_bank
 slots,_=m.enumerate_plan(c);bindings=[]
 for n,slot in enumerate(slots):
  d=tmp_path/str(n);d.mkdir();b=d/'canonical';b.mkdir()
  state={k:[] for k in ['integration_state','controller_state','qpos','qvel']}
  state.update(observable_timing={},native_metadata={'object_id':f'object-{n}'},object_states={})
  (b/'canonical_state.json').write_text(json.dumps(state));(b/'scene.xml').write_text('<mujoco/>')
  base=yaml.safe_load(Path(c['base_config']).read_text());base['instance'].update(task_id=slot['task_id'],layout_id=slot['layout_id'],style_id=slot['style_id'])
  base['horizon']=slot['native_horizon']
  base['policy']['checkpoint_receipt_sha256']=c['checkpoint_receipt_sha256']
  m.save_new(d/'native_config.json',base)
  manifest=create_canonical_instance(b,base);resets=[]
  for r in c['reset_perturbations']:
   delta=np.eye(4);delta[:3,3]=r['translation_world_m'];resets.append({'reset_id':r['reset_id'],'delta_world':delta.tolist(),'policy_rng_seed':r['policy_rng_seed']})
  m.save_new(d/'manifest.json',manifest);m.save_new(d/'reset.json',create_reset_bank(manifest,resets))
  bindings.append({'instance_slot_id':slot['instance_slot_id'],'canonical_manifest':str(d/'manifest.json'),'reset_bank':str(d/'reset.json'),'bundle_dir':str(b),'native_config':str(d/'native_config.json')})
 return bindings


def test_resolve_actual_n0_bindings_80_unique_units(tmp_path):
 c=config();bindings=bindings_for_plan(tmp_path,c);rows=m.resolve_plan(c,bindings)
 assert len(rows)==80 and len({r['unit_id'] for r in rows})==80
 assert len({r['canonical_instance_id'] for r in rows})==8
 assert all(r['terminal_status']=='NOT_SCHEDULED' for r in rows)


def test_resolve_rejects_different_native_layout_despite_same_seed(tmp_path):
 c=config();bindings=bindings_for_plan(tmp_path,c);slots,_=m.enumerate_plan(c)
 bindings[0]['canonical_manifest']=bindings[-1]['canonical_manifest'];bindings[0]['bundle_dir']=bindings[-1]['bundle_dir'];bindings[0]['reset_bank']=bindings[-1]['reset_bank']
 with pytest.raises(ValueError,match='native instance differs'):m.resolve_plan(c,bindings)


def test_resolve_rejects_valid_but_unplanned_reset_bank(tmp_path):
 from robo.roundtrip.identity import create_reset_bank
 c=config();bindings=bindings_for_plan(tmp_path,c);b=bindings[0];manifest=json.loads(Path(b['canonical_manifest']).read_text())
 bank=json.loads(Path(b['reset_bank']).read_text());bank['resets'][1]['delta_world'][0][3]=.02
 Path(b['reset_bank']).write_text(json.dumps(create_reset_bank(manifest,bank['resets'])))
 with pytest.raises(ValueError,match='predeclared perturbation'):m.resolve_plan(c,bindings)


def test_resolved_configs_preserve_native_horizon_and_pending_builds(tmp_path):
 c=config();bindings=bindings_for_plan(tmp_path,c);rows=m.resolve_plan(c,bindings)
 emitted,commands=m.write_resolved_configs(rows,bindings,tmp_path/'resolved',worker_root=tmp_path/'workers')
 assert len(emitted)==80 and len(commands)==40
 assert all('--object-dir' not in cmd['argv'] for cmd in commands.values())
 for r in emitted:
  cfg=json.loads(Path(r['config_path']).read_text())
  assert cfg['schema_version']==2 and cfg['horizon']==r['native_horizon']
  assert cfg['policy_rng_seed']==r['policy_rng_seed']
  assert cfg['policy_id']==r['policy_id'] and cfg['renderer']=='native'
  assert cfg['canonical_manifest_sha256']==r['canonical_manifest_sha256']


def test_dispatch_waits_for_reference_and_build_without_invented_terminal(tmp_path,monkeypatch):
 u=unit();commands={u['unit_id']:{'planned_unit_sha256':canonical_hash(u),'argv':['runner'],'requires_files':[str(tmp_path/'missing-reference.json')]}}
 monkeypatch.setattr(m.subprocess,'run',lambda *a,**kw:pytest.fail('must not submit missing dependency'))
 assert m.dispatch([u],commands,tmp_path/'dispatch',sbatch_args=[],max_jobs=1)==[]
 assert not list((tmp_path/'dispatch').rglob('terminal.json'))


def test_sealed_acquisition_resume_keeps_native_identity_and_detects_config_drift(tmp_path):
 import shutil
 c=config();bindings=bindings_for_plan(tmp_path,c);slot=m.enumerate_plan(c)[0][0];b=bindings[0];d=Path(b['native_config']).parent
 (d/'private').mkdir();shutil.copytree(b['bundle_dir'],d/'private/canonical')
 shutil.copyfile(b['canonical_manifest'],d/'private/canonical_instance.json');shutil.copyfile(b['reset_bank'],d/'private/reset_bank.json')
 m.save_new(d/'acquisition_manifest.json',{'config_sha256':canonical_hash(c),'slot':slot})
 loaded=m.load_acquisition_source(c,slot,d)
 assert loaded['manifest']['canonical_instance_id']==json.loads(Path(b['canonical_manifest']).read_text())['canonical_instance_id']
 changed=copy.deepcopy(c);changed['reset_perturbations'][0]['policy_rng_seed']=100
 with pytest.raises(ValueError,match='slot/config'):m.load_acquisition_source(changed,slot,d)


def test_instance_runner_sequential_five_reset_resume(tmp_path,monkeypatch):
 c=config();bindings=bindings_for_plan(tmp_path,c);rows=m.resolve_plan(c,bindings)
 rows,commands=m.write_resolved_configs(rows,bindings,tmp_path/'resolved',worker_root=tmp_path/'workers')
 instance=rows[0]['canonical_instance_id'];calls=[]
 def fake_worker(u,argv,out,**kwargs):
  calls.append(u['reset_id']);t=terminal(u,False);t['result']['config_sha256']=u['config_sha256'];m.save_new(out/'terminal.json',t);return t
 monkeypatch.setattr(m,'worker',fake_worker)
 r=m.run_instance(rows,commands,instance_id=instance,method='REF_NATIVE',out=tmp_path/'workers')
 assert calls==['r0','r1','r2','r3','r4'] and r['terminal']==5 and r['executed']==5
 assert m.run_instance(rows,commands,instance_id=instance,method='REF_NATIVE',out=tmp_path/'workers')['terminal']==5
 assert len(calls)==5
 missing=m.run_instance(rows,commands,instance_id=instance,method='B0_FIXED_NATIVE',out=tmp_path/'workers')
 assert len(missing['pending_unit_ids'])==5 and missing['terminal']==0
 with pytest.raises(ValueError,match='reset roster'):m.run_instance(rows,commands,instance_id=instance,method='REF_NATIVE',out=tmp_path/'workers',expected_resets=4)


def test_late_build_commands_preserve_existing_unit_configs(tmp_path):
 c=config();bindings=bindings_for_plan(tmp_path,c);rows=m.resolve_plan(c,bindings)
 rows,_=m.write_resolved_configs(rows,bindings,tmp_path/'resolved',worker_root=tmp_path/'workers')
 before=[m.sha(r['config_path']) for r in rows]
 builds=[{'canonical_instance_id':rows[0]['canonical_instance_id'],'controller_method':'B0_FIXED_NATIVE','object_dir':str(tmp_path/'build')}]
 commands=m.commands_for_units(rows,builds)
 assert len(commands)==45
 assert before==[m.sha(r['config_path']) for r in rows]
 assert all('--reference-episode' in commands[r['unit_id']]['argv'] for r in rows if r['controller_method']=='B0_FIXED_NATIVE' and r['canonical_instance_id']==rows[0]['canonical_instance_id'])


def test_extend_method_reuses_exact_reference_configs(tmp_path):
 c=config();bindings=bindings_for_plan(tmp_path,c);rows=m.resolve_plan(c,bindings)
 rows,_=m.write_resolved_configs(rows,bindings,tmp_path/'resolved',worker_root=tmp_path/'workers')
 before=m.canonical_hash(rows);hashes=[m.sha(r['config_path']) for r in rows]
 builds=[{'canonical_instance_id':rows[0]['canonical_instance_id'],'controller_method':'B3_AGENT_NATIVE','object_dir':str(tmp_path/'b3')}]
 combined,commands=m.extend_methods(rows,['B3_AGENT_NATIVE'],bindings,tmp_path/'extra',worker_root=tmp_path/'new-workers',builds=builds)
 assert len(combined)==120 and len(commands)==5
 assert combined[:80]==rows and m.canonical_hash(rows)==before
 assert hashes==[m.sha(r['config_path']) for r in rows]
 assert all(r['controller_method']=='B3_AGENT_NATIVE' for r in combined[80:])
 for command in commands.values():
  assert '--reference-episode' in command['argv']
  assert '/workers/' in command['requires_files'][0]
 with pytest.raises(ValueError,match='already exists'):m.extend_methods(rows,['B0_FIXED_NATIVE'],bindings,tmp_path/'bad',worker_root=tmp_path/'new-workers')
 with pytest.raises(ValueError,match='duplicate'):m.extend_methods(rows,['B3_AGENT_NATIVE']*2,bindings,tmp_path/'bad',worker_root=tmp_path/'new-workers')


def test_abstained_build_with_asset_is_never_dispatched(tmp_path):
 c=config();bindings=bindings_for_plan(tmp_path,c);rows=m.resolve_plan(c,bindings)
 rows,_=m.write_resolved_configs(rows,bindings,tmp_path/'resolved',worker_root=tmp_path/'workers')
 b={'canonical_instance_id':rows[0]['canonical_instance_id'],'controller_method':'B0_FIXED_NATIVE','object_dir':str(tmp_path/'diagnostic-only'),'accepted':False,'terminal_status':'ABSTAINED'}
 commands=m.commands_for_units(rows,[b]);assert len(commands)==40
 assert all('--object-dir' not in command['argv'] for command in commands.values())
 b['terminal_status']='BUILD_FAILED';assert len(m.commands_for_units(rows,[b]))==40
 b['accepted']=True
 with pytest.raises(ValueError,match='contradicts'):m.commands_for_units(rows,[b])
 b.pop('terminal_status');b['accepted']=False
 with pytest.raises(ValueError,match='explicit terminal'):m.commands_for_units(rows,[b])


def test_test_admission_requires_complete_dev_and_unchanged_evidence(tmp_path,monkeypatch):
 c=config();c['split']='test'
 status=tmp_path/'status.json';status.write_text(json.dumps({'groups':[{'controller_method':mth,'planned':40,'terminal':40,'unmeasured':0} for mth in ['REF_NATIVE','B0_FIXED_NATIVE']]}))
 gate=tmp_path/'identity.json';gate.write_text(json.dumps({'passed':True}));frozen=tmp_path/'settings.yaml';frozen.write_text('fixed: true\n')
 item=lambda p:{'path':str(p),'sha256':m.sha(p)}
 r={'kind':'native_test_admission','capture_ready':True,'source_commit':'fixed','config_sha256':m.canonical_hash(c),'core_status':item(status),'identity_gates':{task:item(gate) for task in c['tasks']},'frozen_inputs':[item(frozen)]}
 p=tmp_path/'admission.json';p.write_text(json.dumps(r));monkeypatch.setattr(m,'git_snapshot',lambda:{'commit':'fixed','dirty':False})
 assert m.validate_test_capture_admission(c,p)==r
 with pytest.raises(ValueError,match='requires admission'):m.validate_test_capture_admission(c,None)
 frozen.write_text('fixed: false\n')
 with pytest.raises(ValueError,match='evidence changed'):m.validate_test_capture_admission(c,p)
 r['frozen_inputs']=[item(frozen)];r['identity_gates'].pop(next(iter(c['tasks'])));p.write_text(json.dumps(r))
 with pytest.raises(ValueError,match='missing native task'):m.validate_test_capture_admission(c,p)
 r['identity_gates']={task:item(gate) for task in c['tasks']};status.write_text(json.dumps({'groups':[{'controller_method':'REF_NATIVE','planned':40,'terminal':39,'unmeasured':1}]}));r['core_status']=item(status);p.write_text(json.dumps(r))
 with pytest.raises(ValueError,match='not complete'):m.validate_test_capture_admission(c,p)


def test_test_unit_configs_require_resolved_roster_admission(tmp_path):
 c=config();c['split']='test';bindings=bindings_for_plan(tmp_path,c);rows=m.resolve_plan(c,bindings)
 with pytest.raises(ValueError,match='execution admission required'):
  m.write_resolved_configs(rows,bindings,tmp_path/'blocked',worker_root=tmp_path/'workers')
 admission={'kind':'resolved_test_execution_admission','ready':True,'roster_sha256':m.canonical_reset_roster_hash(rows),'dev_gate_sha256':'a'*64,'thresholds_sha256':'b'*64}
 emitted,_=m.write_resolved_configs(rows,bindings,tmp_path/'test',worker_root=tmp_path/'workers',test_admission=admission)
 assert all(json.loads(Path(r['config_path']).read_text())['test_admission']['roster_sha256']==admission['roster_sha256'] for r in emitted)
 extended,_=m.extend_methods(emitted,['B3_AGENT_NATIVE'],bindings,tmp_path/'extra-test',worker_root=tmp_path/'extra-workers')
 assert m.canonical_reset_roster_hash(extended)==admission['roster_sha256']
 assert json.loads(Path(extended[-1]['config_path']).read_text())['test_admission']['thresholds_sha256']=='b'*64
 changed={**admission,'roster_sha256':'c'*64}
 with pytest.raises(ValueError,match='roster differs'):
  m.write_resolved_configs(rows,bindings,tmp_path/'wrong',worker_root=tmp_path/'wrong-workers',test_admission=changed)
