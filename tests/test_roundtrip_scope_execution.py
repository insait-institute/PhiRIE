import copy
import json
from pathlib import Path
import pytest
from robo.manifest.hash import canonical_hash
from robo.roundtrip.local_policy_instance import preflight_scope_engine,SCOPE_BLOCKS
from robo.roundtrip.scope_execution import emit


def test_scope_blocks_require_fresh_full_same_reset_controls(tmp_path):
 rows=[];commands={};worker=tmp_path/'workers'
 for i,(method,scope) in enumerate(SCOPE_BLOCKS['l1_b3_v1']):
  c={'scope_engine_protocol':'l1_b3_v1','scope_run_id':'run','policy_engine_protocol':'per_canonical_engine_v1'}
  p=tmp_path/f'{i}.json';p.write_text(json.dumps(c))
  u={'unit_id':str(i),'controller_method':method,'scope':scope,'reset_id':'r0','config_path':str(p),'config_sha256':canonical_hash(c),'result_path_planned':str(worker/str(i)/'runner/episode/result.json')}
  rows.append(u);commands[str(i)]={'planned_unit_sha256':canonical_hash(u),'requires_files':[]}
 assert len(preflight_scope_engine(rows,commands,worker,'l1_b3_v1',1))==3
 source=tmp_path/'artifact';source.write_text('sealed')
 from robo.roundtrip.matrix import sha
 commands['0']['scope_input_hashes']={str(source):sha(source)}
 source.write_text('changed')
 with pytest.raises(ValueError,match='command artifact'):preflight_scope_engine(rows,commands,worker,'l1_b3_v1',1)
 commands['0'].pop('scope_input_hashes')
 with pytest.raises(ValueError,match='block'):preflight_scope_engine(rows[:-1],commands,worker,'l1_b3_v1',1)
 (worker/'0').mkdir(parents=True)
 with pytest.raises(ValueError,match='partial'):preflight_scope_engine(rows,commands,worker,'l1_b3_v1',1)


def test_emit_uses_fresh_control_paths_and_same_target(tmp_path,monkeypatch):
 import robo.roundtrip.spec as spec
 monkeypatch.setattr(spec,'validate_spec',lambda c:None)
 c={'scope':'L0_target_only','controller_method':'REF_NATIVE','policy_engine_id':'old','instance':{'split':'development'}}
 p=tmp_path/'native.json';p.write_text(json.dumps(c))
 ref=dict(cohort_id='dev',canonical_instance_id='instance',reset_id='r0',policy_id='p',controller_method='REF_NATIVE',scope='L0_target_only',sensor_regime='ideal',renderer='native',execution_protocol='primary_native',unit_id='oldref',config_path=str(p),bundle_dir='bundle',canonical_manifest='manifest',reset_bank='bank')
 obj=tmp_path/'same_target';(obj/'collision').mkdir(parents=True)
 for name in ['aligned.json','physics.json','mesh_sim.obj','collision/part_0.obj']:(obj/name).write_text('{}')
 build=tmp_path/'build';build.write_text('{}');pool=tmp_path/'pool';pool.write_text('{}')
 t=[{'canonical_instance_id':'instance','controller_method':'B3_AGENT_NATIVE','object_dir':str(obj)}]
 d=[{'canonical_instance_id':'instance','build_manifest':str(build),'methods':{'B3_AGENT_NATIVE':{'candidate_pool':str(pool)}}}]
 rows,cmd=emit([ref],t,d,tmp_path/'out',worker_root=tmp_path/'workers',protocol='l1_b3_v1',run_id='new')
 assert len(rows)==3 and len(cmd)==3
 assert all(r['unit_id']!='oldref' for r in rows)
 l0=next(r for r in rows if r['controller_method']=='B3_AGENT_NATIVE' and r['scope']=='L0_target_only')
 l1=next(r for r in rows if r['scope']=='L1_target_destination')
 assert l0['result_path_planned'] in cmd[l1['unit_id']]['requires_files']
 assert str(obj) in cmd[l0['unit_id']]['argv'] and str(obj) in cmd[l1['unit_id']]['argv']
 assert len(l1['scope_input_hashes'])==6
 assert all('old' not in json.loads(Path(r['config_path']).read_text()).get('policy_engine_id','') for r in rows)


def test_plan_before_assets_preserves_every_unit_when_bound(tmp_path,monkeypatch):
 import robo.roundtrip.spec as spec
 monkeypatch.setattr(spec,'validate_spec',lambda c:None)
 c={'scope':'L0_target_only','controller_method':'REF_NATIVE','instance':{'split':'test'}}
 p=tmp_path/'native.json';p.write_text(json.dumps(c))
 refs=[]
 for cid in ['ready','unresolved']:
  refs.append(dict(cohort_id='test',canonical_instance_id=cid,reset_id='r0',policy_id='p',controller_method='REF_NATIVE',scope='L0_target_only',sensor_regime='ideal',renderer='native',execution_protocol='primary_native',unit_id='old-'+cid,config_path=str(p),bundle_dir='bundle',canonical_manifest='manifest',reset_bank='bank'))
 worker=tmp_path/'workers';plan,cmd=emit(refs,[],[{'canonical_instance_id':c} for c in ['ready','unresolved']],tmp_path/'plan',worker_root=worker,protocol='l1_b3_v1',run_id='phase',plan_only=True)
 assert len(plan)==6 and cmd=={} and all(r['terminal_status']=='NOT_SCHEDULED' for r in plan)
 before=(tmp_path/'plan/planned_units.jsonl').read_bytes()
 obj=tmp_path/'obj';(obj/'collision').mkdir(parents=True)
 for name in ['aligned.json','physics.json','mesh_sim.obj','collision/part_0.obj']:(obj/name).write_text('{}')
 build=tmp_path/'build';build.write_text('{}');pool=tmp_path/'pool';pool.write_text('{}')
 t=[{'canonical_instance_id':'ready','controller_method':'B3_AGENT_NATIVE','object_dir':str(obj)}]
 d=[{'canonical_instance_id':'ready','build_manifest':str(build),'methods':{'B3_AGENT_NATIVE':{'candidate_pool':str(pool)}}}]
 rows,cmd=emit(refs,t,d,tmp_path/'admission',worker_root=worker,protocol='l1_b3_v1',run_id='phase',frozen_plan=plan)
 assert rows==[r for r in plan if r['canonical_instance_id']=='ready']
 assert before==(tmp_path/'plan/planned_units.jsonl').read_bytes()
 assert len(next(v for v in cmd.values() if str(pool) in v['requires_files'])['scope_input_hashes'])==6
 assert all(v['planned_unit_sha256']==canonical_hash(next(u for u in plan if u['unit_id']==key)) for key,v in cmd.items())
 bad=copy.deepcopy(plan);bad[0]['config_sha256']='changed'
 with pytest.raises(ValueError,match='plan/config'):emit(refs,t,d,tmp_path/'bad',worker_root=worker,protocol='l1_b3_v1',run_id='phase',frozen_plan=bad)
