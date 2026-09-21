import json
import pytest
from robo.roundtrip.cohort_dispatch import b0_bindings,read_context_bindings,nonrollout,METHODS
from robo.roundtrip.matrix import sha


def fixture(tmp_path):
 binding={'instance_slot_id':'slot1','canonical_instance_id':'native1','capture_manifest_sha256':'capture','cohort_id':'cohort'}
 root=tmp_path/'slot1';root.mkdir();(root/'build_config.json').write_text(json.dumps(binding))
 return binding,root


def test_only_proven_shared_discovery_failure_propagates(tmp_path):
 b,r=fixture(tmp_path)
 assert b0_bindings(b,tmp_path)==[]
 (r/'build_failure.json').write_text(json.dumps({'status':'construction_unavailable','phase':'segment'}))
 (r/'segment_runtime.json').write_text(json.dumps({'exit_code':1}))
 (r/'segment.log').write_text('RuntimeError: CUDA unavailable')
 assert b0_bindings(b,tmp_path)==[]
 (r/'segment.log').write_text('ValueError: no automatic target mask with usable TRAIN depth')
 rows=b0_bindings(b,tmp_path)
 assert [x['controller_method'] for x in rows]==METHODS[1:]
 assert all(x['terminal_status']=='BUILD_FAILED' and x['accepted'] is False and x['object_dir'] is None for x in rows)
 assert len(rows[0]['source_evidence'])==4
 (r/'segment_runtime.json').write_text(json.dumps({'exit_code':0}))
 assert b0_bindings(b,tmp_path)==[]


def test_b0_rejects_wrong_capture_and_changed_artifact(tmp_path):
 b,r=fixture(tmp_path)
 with pytest.raises(ValueError,match='acquisition'):b0_bindings({**b,'capture_manifest_sha256':'other'},tmp_path)
 obj=r/'construction/objects/obj_00';obj.mkdir(parents=True)
 for name in ('aligned.json','physics.json'):(obj/name).write_text('{}')
 m={**b,'status':'BUILT','source_hashes':{'construction/objects/obj_00/physics.json':sha(obj/'physics.json')}}
 (r/'build_manifest.json').write_text(json.dumps(m))
 assert b0_bindings(b,tmp_path)[0]['accepted'] is True
 (obj/'physics.json').write_text('{"changed":true}')
 with pytest.raises(ValueError,match='artifact hash'):b0_bindings(b,tmp_path)


def test_context_receipt_requires_exact_manifest_bytes(tmp_path):
 p=tmp_path/'manifest.json';p.write_text('{}')
 row={'controller_method':'B3_AGENT_NATIVE','build_manifest':str(p),'build_manifest_sha256':sha(p)}
 f=tmp_path/'bindings.jsonl';f.write_text(json.dumps(row)+'\n')
 assert read_context_bindings([f])==[row]
 p.write_text('{"changed":true}')
 with pytest.raises(ValueError,match='manifest changed'):read_context_bindings([f])


def test_new_engine_can_be_submitted_held_and_never_duplicated(tmp_path,monkeypatch):
 from types import SimpleNamespace
 from robo.roundtrip import cohort_dispatch as m
 rows=[{'canonical_instance_id':'native1','controller_method':method,'unit_id':method+str(i)} for method in METHODS for i in range(10)]
 binding={'canonical_instance_id':'native1','instance_slot_id':'slot1'}
 monkeypatch.setattr(m,'b0_bindings',lambda *a:[{'controller_method':method,'terminal_status':'READY','accepted':True} for method in METHODS[1:]])
 monkeypatch.setattr(m,'commands_for_units',lambda *a,**kw:{'u':{'argv':['unchanged']}})
 monkeypatch.setattr(m,'preflight_engine',lambda *a,**kw:None)
 monkeypatch.setattr(m,'git_snapshot',lambda:{'commit':'frozen','dirty':False})
 calls=[]
 def submit(argv,**kw):calls.append(argv);return SimpleNamespace(returncode=0,stdout='123\n',stderr='')
 monkeypatch.setattr(m.subprocess,'run',submit)
 kwargs=dict(worker_root=tmp_path/'workers',worker_source=tmp_path,admission=tmp_path/'admission',hold_new=True)
 result=m.dispatch_ready(rows,[binding],tmp_path,[],tmp_path/'dispatch',**kwargs)
 assert result[0]['job_id']=='123' and calls[0][:2]==['sbatch','--hold']
 assert not any('--array' in x for x in calls[0])
 assert m.dispatch_ready(rows,[binding],tmp_path,[],tmp_path/'dispatch',**kwargs)==[]
 assert len(calls)==1


def test_metadata_override_rejects_transform_or_geometry_changes(tmp_path):
 from robo.roundtrip.cohort_dispatch import corrected_b0
 b={'canonical_instance_id':'native','capture_manifest_sha256':'capture','instance_slot_id':'slot'}
 old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
 for root in (old,new):
  for n in ('physics.json','mesh_sim.obj','mesh_sim.ply'):(root/n).write_text('frozen')
  (root/'aligned.json').write_text(json.dumps({'world_dims':[1,2,3] if root==old else [1,2,2.9],'transform':'same'}))
 parent=tmp_path/'parent.json';parent.write_text('{}');child=tmp_path/'child.json';child.write_text('{}')
 proofpath=tmp_path/'proof.json'
 proof={'kind':'implementation_metadata_correction','changed_files':['aligned.json'],'changed_fields':['world_dims'],
        'referenced_surface_equal':True,'geometry_transform_collision_physics_unchanged':True,'outcome_input_used':False,
        'parent_hashes':{p.name:sha(p) for p in old.iterdir()},'child_hashes':{p.name:sha(p) for p in new.iterdir()}}
 proofpath.write_text(json.dumps(proof))
 row={**b,'controller_method':'B0_FIXED_NATIVE','accepted':True,'object_dir':str(new),
      'parent_build_manifest':str(parent),'parent_build_manifest_sha256':sha(parent),
      'build_manifest':str(child),'build_manifest_sha256':sha(child),
      'metadata_repair_receipt':str(proofpath),'metadata_repair_receipt_sha256':sha(proofpath)}
 original={'object_dir':str(old),'build_manifest_sha256':sha(parent)}
 assert corrected_b0(row,original,b)==row
 (new/'aligned.json').write_text(json.dumps({'world_dims':[1,2,2.9],'transform':'different'}))
 with pytest.raises(ValueError,match='alignment or transform'):corrected_b0(row,original,b)
 (new/'aligned.json').write_text(json.dumps({'world_dims':[1,2,2.9],'transform':'same'}))
 (new/'mesh_sim.obj').write_text('changed geometry')
 with pytest.raises(ValueError,match='artifact changed'):corrected_b0(row,original,b)
