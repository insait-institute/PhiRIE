"""Exercise the new commands against actual repository contracts, without policy inference."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from robo.roundtrip import mechanism_followup as follow
from robo.roundtrip import object_fidelity as appearance
from robo.roundtrip.matrix import canonical_hash, read_rows, save_new, sha
from robo.roundtrip.spec import validate_spec
from robo.roundtrip.native_policy import CHECKPOINT_REVISION, POLICY_CONFIG, UPSTREAM_COMMIT


def native_config(method='REF_NATIVE'):
    return dict(schema_version=2,benchmark='robocasa_native',execution_ready=True,
        platform={'robocasa_commit':'a'*40,'robosuite_commit':'b'*40,'mujoco_version':'3.3.1'},
        instance={'task_id':'PickPlaceCounterToSink','layout_id':1,'style_id':1,'split':'test'},
        horizon=600,reset_seeds=[0],sensor_regime='ideal_rgbd_posed',cohort_id='old',
        canonical_instance_id='i0',reset_id='r0',policy_rng_seed=123,
        canonical_manifest_sha256='a'*64,reset_contract_sha256='b'*64,
        scope='L0_target_only',replacement_scope='target_only',oracle_context=True,
        controller_method=method,execution_protocol='primary_native',renderer='native',
        test_admission={'roster_sha256':'a'*64,'dev_gate_sha256':'b'*64,'thresholds_sha256':'c'*64},
        policy={'checkpoint_revision':CHECKPOINT_REVISION,'config':POLICY_CONFIG,'code_commit':UPSTREAM_COMMIT,
                'replan_steps':5,'checkpoint_receipt_sha256':'c'*64})


@pytest.mark.parametrize('method',follow.METHODS)
def test_complete_new_method_contract(method):
    c=native_config(method);c['mechanism_followup']=follow.PROTOCOL
    validate_spec(c)


@pytest.mark.parametrize('method',['B1_FIXED_PRIORITY','B2_EVIDENCE'])
def test_new_methods_not_silently_admitted_into_old_experiments(method):
    with pytest.raises(ValueError,match='follow-up'):validate_spec(native_config(method))


def test_mechanism_followup_rejects_renderer_change():
    c=native_config('B1_FIXED_PRIORITY');c.update(mechanism_followup=follow.PROTOCOL,renderer='gaussian')
    with pytest.raises(ValueError):validate_spec(c)


def fake_preparation(tmp_path,monkeypatch):
    import trimesh
    source=tmp_path/'source';source.mkdir()
    cfg=native_config();cfg_path=source/'ref.json';save_new(cfg_path,cfg)
    for name in ('scene.xml','canonical_state.json','canonical.json','bank.json','admission.json'):
        (source/name).write_text('{}')
    unit=dict(cohort_id='old',canonical_instance_id='i0',reset_id='r0',policy_id='p',
        controller_method='REF_NATIVE',scope='L0_target_only',sensor_regime='ideal_rgbd_posed',renderer='native',
        execution_protocol='primary_native',policy_rng_seed=123,layout_id=1,style_id=1,task_id='PickPlaceCounterToSink',
        split='test',native_horizon=600,config_path=str(cfg_path),config_sha256=canonical_hash(cfg),
        bundle_dir=str(source),canonical_manifest=str(source/'canonical.json'),reset_bank=str(source/'bank.json'))
    save_new(source/'planned.jsonl',[unit],jsonl=True)
    save_new(source/'bindings.jsonl',[dict(canonical_instance_id='i0',capture_manifest_sha256='capture')],jsonl=True)
    obj=source/'selected_00';(obj/'collision').mkdir(parents=True)
    mesh=trimesh.creation.box();mesh.export(obj/'mesh_sim.obj');mesh.export(obj/'collision/part_00.obj')
    (obj/'aligned.json').write_text('{}');(obj/'physics.json').write_text('{}')
    hashes={str(p.relative_to(obj)):sha(p) for p in obj.rglob('*') if p.is_file()}
    pool=dict(schema_version=2,canonical_instance_id='i0',capture_manifest_sha256='capture',
              initial_candidates=[{'proposal_id':'p0'}],outcomes=[dict(native_method=m,selected_proposal_id='p0',
                object_dir=str(obj),artifact_hashes=hashes) for m in follow.METHODS[2:4]])
    save_new(source/'candidate_pool.json',pool)
    manifest=source/'build.json';save_new(manifest,{'source_hashes':hashes})
    builds=[dict(canonical_instance_id='i0',controller_method=m,accepted=True,terminal_status='READY',
                 build_manifest=str(manifest),build_manifest_sha256=sha(manifest),object_dir=str(obj))
            for m in ('B0_FIXED_NATIVE','B3_AGENT_NATIVE')]
    save_new(source/'build_bindings.jsonl',builds,jsonl=True)
    config=dict(cohort_id='new',source_planned_units=str(source/'planned.jsonl'),
        source_bindings=str(source/'bindings.jsonl'),candidate_pool_roots=[str(source/'candidate_pool.json')],
        source_build_bindings=[str(source/'build_bindings.jsonl')],engine_admission=str(source/'admission.json'),
        native_python='/usr/bin/python3',worker_source=str(tmp_path),sbatch_args=['--nodes=1','--ntasks=1'])
    monkeypatch.setattr(follow,'worker_identity',lambda path:{'root':str(tmp_path),'commit':'a'*40})
    return config,obj


def test_prepare_real_matrix_contract(tmp_path,monkeypatch):
    config,obj=fake_preparation(tmp_path,monkeypatch)
    out=tmp_path/'prepared';units=follow.prepare(config,out)
    assert len(units)==5 and len(read_rows(out/'planned_units.jsonl'))==5
    commands=json.loads((out/'commands.json').read_text())
    assert len(commands)==5
    for u in units:
        c=json.loads(Path(u['config_path']).read_text());validate_spec(c)
        assert c['cohort_id']=='new'
        assert c['mechanism_followup']==follow.PROTOCOL
        assert commands[u['unit_id']]['planned_unit_sha256']==canonical_hash(u)
    from robo.roundtrip.local_policy_instance import preflight_engine
    preflight_engine(units,commands,out/'workers',list(follow.METHODS),1)
    report=follow.launch(out,max_jobs=1)
    assert len(report)==1 and report[0]['submitted'] is False
    assert not (out/'jobs').exists()
    (obj/'physics.json').write_text('{"tampered":true}')
    with pytest.raises(ValueError,match='input changed'):follow.launch(out)


def test_prepare_does_not_infer_failure_from_missing_pool(tmp_path,monkeypatch):
    config,_=fake_preparation(tmp_path,monkeypatch)
    p=Path(config['candidate_pool_roots'][0]);d=json.loads(p.read_text());d['canonical_instance_id']='another';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='NOT automatically'):follow.prepare(config,tmp_path/'no-plan')


def test_object_metrics_delegate_to_existing_formula_and_preserve_missing(tmp_path):
    bindings=tmp_path/'bindings.jsonl';save_new(bindings,[{'canonical_instance_id':'i0'},{'canonical_instance_id':'i1'}],jsonl=True)
    target=np.zeros((40,40,3),np.uint8);pred=target.copy();pred[12:28,12:28]=128
    target_path=tmp_path/'target.png';pred_path=tmp_path/'pred.png'
    Image.fromarray(target).save(target_path);Image.fromarray(pred).save(pred_path)
    mask=np.zeros((40,40),bool);mask[12:28,12:28]=True
    mask_path=tmp_path/'mask.png';Image.fromarray(mask.astype(np.uint8)*255).save(mask_path)
    records=[dict(canonical_instance_id='i0',frame_id='f0',status='READY',reference_rgb=str(target_path),
                  reference_sha256=sha(target_path),mask=str(mask_path),mask_sha256=sha(mask_path),
                  width=40,height=40,crop_xyxy=list(appearance.crop_from_mask(mask)),mask_pixels=int(mask.sum())),
             dict(canonical_instance_id='i1',frame_id=None,status='MASK_INSTANCE_UNAVAILABLE')]
    masks=tmp_path/'mask_manifest.json';save_new(masks,dict(bindings=str(bindings),bindings_sha256=sha(bindings),
        planned_instances=2,shard_count=1,shard_index=0,records=records))
    render=tmp_path/'appearance_render.json';save_new(render,dict(canonical_instance_id='i0',
        native_import_render_identity=[{'byte_exact':True}],records=[dict(method=m,frame_id='f0',pred_rgb=str(pred_path),
        pred_sha256=sha(pred_path),reference_rgb=str(target_path),reference_sha256=sha(target_path)) for m in appearance.METHODS]))
    result=appearance.metrics([masks],[render],tmp_path/'metrics',skip_lpips=True)
    assert len(result)==4 and all(r['planned_instances']==2 and r['common_instances']==1 for r in result)
    assert all(r['crop_lpips'] is None for r in result)
    expected=-10*np.log10((128/255)**2)
    assert all(abs(r['masked_psnr']-expected)<1e-5 for r in result)
    observed=read_rows(tmp_path/'metrics/object_region_records.jsonl')
    assert sum(r['status']=='MASK_INSTANCE_UNAVAILABLE' for r in observed)==4
    with pytest.raises(FileExistsError):appearance.metrics([masks],[render],tmp_path/'metrics',skip_lpips=True)
