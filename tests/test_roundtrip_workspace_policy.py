import copy,json
from pathlib import Path
import pytest
from robo.roundtrip.spec import load_spec,validate_spec
from robo.roundtrip.identity import file_hash
from robo.roundtrip.scope_bundle import artifact_hashes
from robo.roundtrip.workspace_policy import seal,validate_workspace_bundle
from robo.manifest.hash import canonical_hash


def write(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))


@pytest.fixture
def inputs(tmp_path):
    c=load_spec(Path(__file__).parents[1]/'configs/experiments/sim_recon_sim/reference.yaml')
    c.update(schema_version=2,cohort_id='dev',canonical_instance_id='native-a',reset_id='r0',policy_rng_seed=3,
        canonical_manifest_sha256='a'*64,reset_contract_sha256='b'*64,scope='L0_target_only',controller_method='B3_AGENT_NATIVE',execution_protocol='primary_native',horizon=900)
    c['instance']['task_id']='PickPlaceSinkToCounter';c['policy']['checkpoint_receipt_sha256']='c'*64
    config=tmp_path/'baseline.json';write(config,c);ep=tmp_path/'baseline/episode'
    write(ep/'result.json',dict(config_sha256=canonical_hash(c),executed=True,error=None,success=False))
    entities=[]
    for oid,role,body,kind in [('target','target','obj_main',None),('destination','receptacle','container_main',None),('source_support','support','sink_main','sink_basin'),('destination_support','support','counter_main','counter_top')]:
        d=tmp_path/oid
        for name in ('aligned.json','physics.json','mesh_sim.obj','collision/part_00.obj'):
            p=d/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('frozen')
        e=dict(object_id=oid,role=role,body_name=body,object_dir=str(d),artifact_hashes=artifact_hashes(d))
        if kind:e['component_kind']=kind
        entities.append(e)
    coverage=tmp_path/'coverage.json';write(coverage,dict(native_asset_access=False,unknown_samples=999999,admission='ABSTAIN_UNRESOLVED_INVENTORY'))
    engineering=tmp_path/'engineering.json';write(engineering,dict(tier='DEV',canonical_instance_id='native-a',coverage=str(coverage),entities=entities,input_files={str(coverage):file_hash(coverage)}))
    return engineering,config,ep,tmp_path/'sealed'


def test_failed_baseline_and_unknown_space_do_not_veto_finite_diagnostic(inputs):
    b=seal(*inputs);config=json.loads((inputs[3]/'config.json').read_text())
    assert config['scope']=='DEV_partial_observed_workspace' and b['L2_READY'] is False
    assert b['controller_decision'] is None
    assert validate_workspace_bundle(b,config,inputs[2],inputs[1])==b
    with pytest.raises(FileExistsError):seal(*inputs)


@pytest.mark.parametrize('change',['test','policy_seed','geometry','native_role','whole_l2'])
def test_partial_workspace_rejects_contract_tampering(inputs,change):
    b=seal(*inputs);config=json.loads((inputs[3]/'config.json').read_text())
    if change=='test':config['instance']['split']='test'
    if change=='policy_seed':config['policy_rng_seed']+=1
    if change=='geometry':Path(b['entities'][0]['object_dir'],'physics.json').write_text('changed')
    if change=='native_role':b['entities'][2]['native_role']='other_sink'
    if change=='whole_l2':b['L2_READY']=True
    with pytest.raises(ValueError):validate_workspace_bundle(b,config,inputs[2],inputs[1])


def test_partial_scope_reaches_canonical_harness_without_settle_acceptance(inputs,tmp_path,monkeypatch):
    import numpy as np
    from types import SimpleNamespace as NS
    from robo.roundtrip.identity import create_canonical_instance,create_reset_bank
    import robo.roundtrip.paired as paired
    import robo.eval.harness_runner as harness
    engineering,config_path,ep,out=inputs;c=json.loads(config_path.read_text())
    directory=tmp_path/'canonical';directory.mkdir();(directory/'scene.xml').write_text('<mujoco/>')
    write(directory/'canonical_state.json',dict(integration_state=[],controller_state=[],observable_timing={},native_metadata={},qpos=[],qvel=[],object_states={}))
    manifest=create_canonical_instance(directory,c);bank=create_reset_bank(manifest,[dict(reset_id='r0',delta_world=np.eye(4).tolist(),policy_rng_seed=3)])
    c.update(canonical_instance_id=manifest['canonical_instance_id'],canonical_manifest_sha256=manifest['manifest_sha256'],reset_contract_sha256=bank['reset_contract_sha256']);write(config_path,c)
    source=json.loads(engineering.read_text());source['canonical_instance_id']=c['canonical_instance_id'];write(engineering,source)
    mpath=tmp_path/'manifest.json';bpath=tmp_path/'bank.json';write(mpath,manifest);write(bpath,bank)
    binding=dict(canonical_instance_id=manifest['canonical_instance_id'],canonical_manifest_sha256=manifest['manifest_sha256'],scene_xml_sha256=manifest['identity']['scene_xml_sha256'],state_sha256=manifest['identity']['state_sha256'],asset_closure_sha256=manifest['identity']['asset_closure_sha256'])
    metadata=dict(checkpoint_receipt_sha256=c['policy']['checkpoint_receipt_sha256'])
    baseline=json.loads((ep/'result.json').read_text());baseline.update(config_sha256=canonical_hash(c),policy_identity=metadata,canonical_reference=binding);write(ep/'result.json',baseline);write(ep.parent/'import_receipt.json',{})
    seal(*inputs);config=json.loads((out/'config.json').read_text());calls=[]
    adapter=NS(apply_paired_reset=lambda reset,role: calls.append('reset') or {})
    monkeypatch.setattr(paired,'prepare_scope_adapter',lambda *a: calls.append(a[2]['scope']) or {})
    def run(adapter,policy,**kw):
        calls.append('canonical_harness');kw['out_dir'].mkdir();r=dict(executed=True,success=False,error=None);write(kw['out_dir']/'result.json',r);return r
    monkeypatch.setattr(harness,'run_native_episode',run)
    result=paired.run_resolved_episode(adapter,NS(metadata=metadata),config=config,canonical_reference=directory,canonical_manifest=mpath,reset_bank=bpath,out_dir=tmp_path/'run',reference_episode=ep,baseline_config=config_path,scope_bundle=out/'scope_bundle.json')
    assert result['success'] is False and calls==['DEV_partial_observed_workspace','reset','canonical_harness']
    assert set(adapter.scope_binding['scope_role_constructors'])=={'target','destination','source_support','destination_support'}
