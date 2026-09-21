"""The preparer writes definitions/configs; the canonical harness owns all rollouts."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from run.icra2027 import e4_compact_policy as preparer
from robo.eval import harness_runner
from robo.eval import e4_reset_eligibility
from tests.test_e4_automatic_eligibility import fixture, adapter_spec


def setup(tmp_path,monkeypatch,*,all_rejected=False):
    config,spec,states,chain,checked=fixture(tmp_path,monkeypatch)
    chain['menagerie']={'root':str(tmp_path/'menagerie'),'commit':'d'*40}
    if all_rejected:
        for row in chain['rows']:
            row.update(passed=False,checks={'accepted':False})
        for cell in checked['cells']:
            cell.update(executed=False,passed=False,outcome='build_failure',checks=None,reset_provenance=None)
    real_contract=copy.deepcopy(config['contract'])
    real_contract['policy']['checkpoint_path']=str(tmp_path/'checkpoint')
    real_contract['runtime_dependencies']={'openpi':{'root':str(tmp_path/'openpi'),'commit':'e'*40},'mujoco_menagerie':{'root':str(tmp_path/'menagerie'),'commit':'d'*40},'checkpoint_cache_root':'/scratch/fixture'}
    monkeypatch.setattr(preparer.camera,'_evidence_root',lambda:tmp_path)
    monkeypatch.setattr(preparer.camera,'_git_snapshot',lambda commit:{'commit':commit,'dirty':False})
    monkeypatch.setattr(preparer,'_policy_contract',lambda *a,**kw:copy.deepcopy(real_contract))
    monkeypatch.setattr(preparer,'load_harness_spec',adapter_spec)
    monkeypatch.setattr(preparer,'expected_server_identity_from_config',lambda config:{'expected_only':True})
    reference=config['cpu_reset_eligibility']['camera_config']
    kwargs=dict(camera_config_path=reference['path'],camera_config_sha256=reference['sha256'],
        camera_output=config['cpu_reset_eligibility']['bundle'],expected_code_commit='a'*40,
        out=tmp_path/'outputs/icra2027/fixture-stage-v1/harness/compact_scripted_smoke',freeze_id='fixture-stage-v1',
        openpi_root=tmp_path/'openpi',openpi_commit='e'*40,mode='scripted')
    monkeypatch.setattr(preparer,'_validate_stage_e0',lambda *a:None)
    return kwargs,config,states,chain,checked


def test_prepare_persists_only_twenty_canonical_definitions_and_full_suites(tmp_path,monkeypatch):
    kwargs,original,states,chain,checked=setup(tmp_path,monkeypatch)
    receipt=preparer.prepare(**kwargs)
    out=kwargs['out'];config=yaml.safe_load((out/'harness_config.yaml').read_text())
    assert receipt['planned_episodes']==40 and receipt['eligible_episodes']==20
    assert not receipt['policy_invoked'] and not receipt['measured_reset_telemetry']
    assert config['policy']=='scripted_sinusoid'
    assert config['study_scope']=='e4_compact_scripted_smoke'
    assert json.loads((out/'reset_states.json').read_text())==[s.to_dict() for s in states]
    assert all(set(s.to_dict())=={'reset_state_id','scene_id','task_id','ep','base_seed','reset_seed'} for s in states)
    assert [s['tasks_json'] for s in config['scenes']]==[s['tasks_json'] for s in original['scenes']]
    assert not (out/harness_runner.LEDGER_NAME).exists()
    with pytest.raises(preparer.camera.CameraScorerGateError,match='reuse'):
        preparer.prepare(**kwargs)


@pytest.mark.parametrize('damage',['camera_sha','outside_output','missing_camera','upstream','denominator'])
def test_preparation_failure_leaves_no_partial_output(tmp_path,monkeypatch,damage):
    kwargs,config,states,chain,checked=setup(tmp_path,monkeypatch)
    if damage=='camera_sha':kwargs['camera_config_sha256']='0'*64
    elif damage=='outside_output':kwargs['out']=tmp_path/'wrong'
    elif damage=='missing_camera':Path(kwargs['camera_config_path']).unlink()
    elif damage=='upstream':
        def fail(*a,**kw):raise preparer.camera.CameraScorerGateError('changed upstream')
        monkeypatch.setattr(preparer.camera,'validate_automatic_chain',fail)
    else:chain['rows'].pop()
    with pytest.raises((ValueError,OSError,preparer.camera.CameraScorerGateError)):
        preparer.prepare(**kwargs)
    assert not kwargs['out'].exists()


def test_eligible_real_preparation_requires_authentic_scripted_task_smoke(tmp_path,monkeypatch):
    kwargs,*_=setup(tmp_path,monkeypatch)
    kwargs.update(mode='real',out=kwargs['out'].with_name('compact_policy'))
    with pytest.raises(ValueError,match='requires sealed canonical scripted smoke'):
        preparer.prepare(**kwargs)
    assert not kwargs['out'].exists()


def test_all_rejected_real_dispatches_to_canonical_harness_without_policy_service(tmp_path,monkeypatch):
    kwargs,*_=setup(tmp_path,monkeypatch,all_rejected=True)
    kwargs.update(mode='real',out=kwargs['out'].with_name('compact_policy'))
    receipt=preparer.prepare(**kwargs)
    calls=[]
    def canonical(config,out,resume):
        calls.append((config,out,resume));return {'canonical_harness_called':True}
    monkeypatch.setattr(harness_runner,'run_matrix',canonical)
    assert preparer.execute(out=kwargs['out'],expected_code_commit='a'*40,mode='prebuild-only',contract=kwargs['out'].parent.parent/'contract')=={'canonical_harness_called':True}
    assert len(calls)==1 and receipt['eligible_episodes']==0
    assert calls[0][0]['policy']=='pi05_droid_jointpos'
    assert not (kwargs['out']/harness_runner.LEDGER_NAME).exists()  # This unit never fabricates rows.


@pytest.mark.parametrize('name',['reset_states.json','harness_config.yaml','preparation_receipt.json'])
def test_changed_staged_artifact_blocks_harness_dispatch(tmp_path,monkeypatch,name):
    kwargs,*_=setup(tmp_path,monkeypatch,all_rejected=True)
    preparer.prepare(**kwargs)
    (kwargs['out']/name).write_text('{}')
    monkeypatch.setattr(harness_runner,'run_matrix',lambda *a,**kw:pytest.fail('must not execute'))
    with pytest.raises(ValueError,match='changed'):
        preparer.execute(out=kwargs['out'],expected_code_commit='a'*40,mode='prebuild-only',contract=kwargs['out'].parent.parent/'contract')


def test_prebuild_only_refuses_eligible_cells(tmp_path,monkeypatch):
    kwargs,*_=setup(tmp_path,monkeypatch)
    preparer.prepare(**kwargs)
    monkeypatch.setattr(harness_runner,'run_matrix',lambda *a,**kw:pytest.fail('must not execute'))
    with pytest.raises(ValueError,match='cannot launch eligible'):
        preparer.execute(out=kwargs['out'],expected_code_commit='a'*40,mode='prebuild-only',contract=kwargs['out'].parent.parent/'contract')


@pytest.mark.parametrize('damage',['policy','scope','kind'])
def test_scripted_adapter_permission_is_narrow(tmp_path,monkeypatch,damage):
    config,spec,states,*_=fixture(tmp_path,monkeypatch)
    config.update(policy='scripted_sinusoid',study_scope='e4_compact_scripted_smoke')
    config['contract']['policy']={'id':'scripted_sinusoid','kind':'scripted_smoke','checkpoint_hash':None}
    assert len(e4_reset_eligibility.load_reset_eligibility(config,spec,states,root=tmp_path))==40
    if damage=='policy':config['policy']='arbitrary_controller'
    elif damage=='scope':config['study_scope']='paper_main'
    else:config['contract']['policy']['kind']='real'
    with pytest.raises(ValueError,match='frozen engineering'):
        e4_reset_eligibility.load_reset_eligibility(config,spec,states,root=tmp_path)


def test_scripted_noninvocation_marker_does_not_invent_server(tmp_path,monkeypatch):
    config,spec,states,*_=fixture(tmp_path,monkeypatch)
    config.update(policy='scripted_sinusoid',study_scope='e4_compact_scripted_smoke')
    config['contract']['policy']={'id':'scripted_sinusoid','kind':'scripted_smoke','checkpoint_hash':None}
    result=harness_runner._runtime_contract(config=config,state=states[0],controller_hash='controller',
        camera_hash='camera',action_convention='absolute_joint_position',action_dim=8,
        policy_hash=None,task_instruction='place',policy_not_invoked=True)
    assert result['policy_execution']=='not_invoked_prebuild'
    assert result['policy_checkpoint_hash'] is None
    assert result['policy'].get('server_identity') is None


@pytest.mark.parametrize('damage',['gate','geometry','reset','checkpoint_input'])
def test_real_smoke_binding_rejects_drift(tmp_path,monkeypatch,damage):
    kwargs,*_=setup(tmp_path,monkeypatch)
    preparer.prepare(**kwargs)
    out=kwargs['out'];config=yaml.safe_load((out/'harness_config.yaml').read_text())
    real=copy.deepcopy(config);real['policy']='pi05_droid_jointpos'
    real['contract']['policy']={'id':'pi05_droid_jointpos','kind':'real'}
    receipt={'fixture':'already validated canonical runtime'}
    gate=out/'scripted_smoke_gate.json';gate.write_text(json.dumps(receipt))
    digest=preparer.screen._sha256(gate)
    monkeypatch.setattr(preparer,'_scripted_receipt',lambda *a:receipt)
    assert preparer.validate_scripted_smoke(out,'a'*40,expected_sha256=digest,real_config=real)==receipt
    if damage=='gate':digest='0'*64
    elif damage=='geometry':real['scenes'][0]['construction_variants']['agentic']['scene_xml']='changed'
    elif damage=='reset':real['contract']['reset_ids'].pop()
    else:real['contract']['runtime_dependencies']={'changed':True}
    with pytest.raises(ValueError):
        preparer.validate_scripted_smoke(out,'a'*40,expected_sha256=digest,real_config=real)


def test_scripted_gate_accepts_zero_success_but_rejects_incomplete_runtime(tmp_path,monkeypatch):
    from robo.eval import harness_validation
    kwargs,original,states,chain,checked=setup(tmp_path,monkeypatch)
    preparer.prepare(**kwargs)
    out=kwargs['out']
    records=[]
    for state in states:
        for arm in ('a0','a4'):
            record={'treatment_id':f'{arm}_raster','reset_state_id':state.reset_state_id,
                'task_id':state.task_id,'scene_id':state.scene_id,'success':False,
                'outcome':'build_failure' if arm=='a0' else 'task_failure','ticks':0 if arm=='a0' else 2}
            if arm=='a4':
                directory=out/'episodes'/f'{arm}__{state.reset_state_id}';directory.mkdir(parents=True)
                manifest=directory/'manifest.json';trace=directory/'timeseries.json.gz'
                manifest.write_text(json.dumps({'git':{'commit':'a'*40,'dirty':False},
                    'outcome':'task_failure','contract':{'policy':{'id':'scripted_sinusoid','kind':'scripted_smoke'}}}))
                preparer.episode_log.write_timeseries(trace,[{'t':i,'stages':{state.task_id:{'grasp':False}}} for i in range(2)])
                record.update(manifest_path=str(manifest),trace_path=str(trace))
            records.append(record)
    ledger=out/harness_runner.LEDGER_NAME
    ledger.write_text(''.join(json.dumps(row)+'\n' for row in records))
    # This boundary consumes the canonical validators; their own geometric/source
    # correctness is covered by existing harness tests, not a second evaluator.
    monkeypatch.setattr(harness_validation,'validate_saved_treatment_records',lambda *a,**kw:{'ok':True})
    monkeypatch.setattr(harness_runner,'_preflight_resolved_scenes',lambda *a:({}, {}, {}))
    monkeypatch.setattr(harness_runner,'_validate_record_inputs_against_current',lambda *a:[])
    gate=preparer._scripted_receipt(out,'a'*40)
    assert gate['executed_episodes']==20 and gate['success_required'] is False
    assert gate['runtime_scorer_integrity_passed']
    records[1]['outcome']='env_crash'
    ledger.write_text(''.join(json.dumps(row)+'\n' for row in records))
    with pytest.raises(ValueError,match='lacks completed runtime'):
        preparer._scripted_receipt(out,'a'*40)
    records.pop()
    ledger.write_text(''.join(json.dumps(row)+'\n' for row in records))
    with pytest.raises(ValueError,match='denominator'):
        preparer._scripted_receipt(out,'a'*40)


def test_policy_contract_binds_git_rig_bytes_and_camera_openpi(tmp_path,monkeypatch):
    from types import SimpleNamespace
    entry=SimpleNamespace(id='pi05_droid_jointpos',client_kind='pi05_server',status='verified',
        checkpoint_path=str(tmp_path/'checkpoint'),checkpoint_hash='ab'*32,training_config='pi05_droid_jointpos',
        image_preprocessing=SimpleNamespace(model_dump=lambda **kw:{'mode':'resize_with_pad','resize_hw':[224,224]}))
    registry=SimpleNamespace(get=lambda _:entry,verify_checkpoint_hash=lambda _:None,validate_control_contract=lambda _:[])
    monkeypatch.setattr(preparer.PolicyRegistry,'from_config_dir',lambda:registry)
    monkeypatch.setattr(preparer,'verify_clean_git_checkout',lambda *a,**kw:{})
    files=[{'path':'franka_emika_panda/rig.xml','sha256':'aa'},{'path':'robotiq_2f85/rig.xml','sha256':'bb'}]
    monkeypatch.setattr(preparer.camera,'_tree_inventory',lambda path,**kw:[files[0 if path.name=='franka_emika_panda' else 1]])
    chain={'menagerie':{'root':str(tmp_path/'runtime_copy'),'source_root':str(tmp_path/'git_rig'),
            'source_commit':'d'*40,'files':copy.deepcopy(files)},
        'summary':{'openpi_resize_identity':{'root':str(tmp_path/'openpi'),'commit':'e'*40}},
        'scenes':{'scene':{'suites':{'A0':{'robot':{'base_pos':[0,0,0]},'ext_cam':{'pos':[0,0,1]}}}}}}
    args=dict(openpi_root=tmp_path/'openpi',openpi_commit='e'*40,cache_root=Path('/scratch/fixture'))
    contract=preparer._policy_contract(chain,**args)
    assert contract['runtime_dependencies']['mujoco_menagerie']['root']==str(tmp_path/'git_rig')
    assert contract['horizon_s']==32
    with pytest.raises(ValueError,match='preprocessing'):
        preparer._policy_contract(chain,**{**args,'openpi_commit':'f'*40})
    chain['menagerie']['files'][0]['sha256']='changed'
    with pytest.raises(ValueError,match='runtime rig bytes'):
        preparer._policy_contract(chain,**args)


def test_policy_stage_e0_authenticates_actual_config_bank_and_runtime_inputs(tmp_path,monkeypatch):
    from robo.eval import freeze
    from robo.manifest.hash import canonical_hash
    validate=preparer._validate_stage_e0
    kwargs,*_=setup(tmp_path,monkeypatch,all_rejected=True)
    receipt=preparer.prepare(**kwargs)
    out=kwargs['out'];config=yaml.safe_load((out/'harness_config.yaml').read_text())
    for name in ('openpi','menagerie'):
        (tmp_path/name).mkdir();(tmp_path/name/'runtime.txt').write_text('runtime')
    (tmp_path/'checkpoint').write_text('fixture checkpoint')
    spec=preparer._stage_freeze_config(config,out,receipt)
    inventory=[freeze._inventory_resource(entry,category=category,repo_root=preparer.CODE_ROOT)
               for category in ('input_roots','checkpoint_roots') for entry in spec[category]]
    contract=out.parent.parent/'contract';contract.mkdir()
    manifest={'freeze_id':config['freeze_id'],'code':{'commit':'a'*40,'dirty':False},
              'paper_ready':False,'resource_inventory':inventory}
    manifest['contract_sha256']=canonical_hash(manifest)
    (contract/'freeze_manifest.json').write_text(json.dumps(manifest))
    (contract/'preflight.log').write_text('PREFLIGHT=PASS\n')
    validate(contract,config,out,'a'*40)
    (tmp_path/'openpi/runtime.txt').write_text('runtime changed')
    with pytest.raises(ValueError,match='runtime identity differs: openpi'):
        validate(contract,config,out,'a'*40)


def test_same_camera_freeze_cannot_be_reused_for_changed_policy_config(tmp_path,monkeypatch):
    kwargs,*_=setup(tmp_path,monkeypatch)
    kwargs['freeze_id']='fixture-v1'
    with pytest.raises(ValueError,match='separately reserved'):
        preparer.prepare(**kwargs)
