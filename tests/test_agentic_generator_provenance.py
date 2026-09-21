import copy
import json
from pathlib import Path
import numpy as np
import pytest

from robo.eval import agentic_ablation as e3
from tests.test_agentic_ablation import _proposal, _fake_runtime
from agents.orchestrator import runtime
from tests.test_agentic_missing_observations import automatic_sources


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    mesh=tmp_path/'raw.ply';mesh.write_bytes(b'raw bytes')
    proposal=_proposal(mesh)
    proposal['raw_generator_provenance']={
        'generator_commit':'c'*40,'artifact_bytes_hash_frozen':True,
        'checkpoint_identity_recorded':True,'runtime_manifest_recorded':True,
        'claim_status':'engineering_only'}
    root=tmp_path/'freeze';inventory=root/'input_inventory';inventory.mkdir(parents=True)
    (inventory/'resolved_proposals.json').write_text('sealed inventory bytes')
    (inventory/'seal.json').write_text(json.dumps({'controller_members':{
        'resolved_jobs.json':'a'*64,'resolved_proposals.json':'b'*64}}))
    final=root/'control'/proposal['scene_id'];staging=tmp_path/'staging';staging.mkdir()
    sealed=copy.deepcopy(proposal)
    monkeypatch.setattr(e3,'_load_inventory',lambda path,**kw:
        ({'freeze_id':'freeze'},{'proposals':[copy.deepcopy(sealed)]}))
    monkeypatch.setattr(runtime,'align_and_probe',_fake_runtime)
    return proposal,final,staging


@pytest.mark.parametrize('retry',[False,True])
def test_registered_candidate_propagates_sealed_generator_not_executor(recorded,retry):
    proposal,final,staging=recorded
    result=e3._process_candidate(proposal,{'observation_sha256':'a'*64,
        'observation_mask_fraction':0.8},np.zeros((60,3)),staging,final,'b'*40,
        retry=retry,parent_proposal_id=proposal['proposal_id'] if retry else None)
    assert result['raw_generator_commit']=='c'*40
    assert result['tool_commit']==('b'*40 if retry else 'c'*40)
    assert result['registration_executor_commit']=='b'*40
    assert result['raw_generator_provenance_sha256']==e3._canonical_digest(proposal['raw_generator_provenance'])
    assert result['raw_generator_inventory_sha256']==e3.sha256_file(final.parent.parent/'input_inventory/resolved_proposals.json')


@pytest.mark.parametrize('change',['invented_commit','malformed_commit','missing_runtime','changed_mesh_hash','different_freeze','wrong_destination'])
def test_unsealed_generator_claim_fails_before_registration(recorded,monkeypatch,change):
    proposal,final,staging=recorded
    if change=='invented_commit':proposal['raw_generator_provenance']['generator_commit']='f'*40
    elif change=='malformed_commit':proposal['raw_generator_provenance']['generator_commit']='invented'
    elif change=='missing_runtime':proposal['raw_generator_provenance']['runtime_manifest_recorded']=False
    elif change=='changed_mesh_hash':proposal['artifact_hashes']['raw_mesh']='f'*64
    elif change=='different_freeze':proposal['freeze_id']='other'
    else:final=final.parent.parent/'other'/proposal['scene_id']
    def forbidden(*a,**kw):raise AssertionError('registration must not run')
    monkeypatch.setattr(runtime,'align_and_probe',forbidden)
    with pytest.raises(ValueError):
        e3._process_candidate(proposal,{},np.zeros((60,3)),staging,final,'b'*40)
    assert not list(staging.iterdir())


def test_unrecorded_legacy_provenance_is_never_inferred_from_executor():
    assert e3._recorded_generator_identity({},Path('/legacy'))=={'raw_generator_commit':'legacy-unrecorded'}
    with pytest.raises(ValueError):
        e3._recorded_generator_identity({'raw_generator_provenance':[]},Path('/legacy'))


@pytest.fixture
def real_recorded(automatic_sources,tmp_path,monkeypatch):
    """Use the real publisher, loader, schema and file seals on synthetic data."""
    config,contract=automatic_sources
    out=tmp_path/'published';e3.run_inventory(config,contract,'freeze',out)
    _,inventory=e3._load_inventory(out,controller_safe=True)
    proposal=next(p for p in inventory['proposals'] if p['availability']=='available')
    staging=tmp_path/'real_staging';staging.mkdir()
    final=out/'control'/proposal['scene_id']
    monkeypatch.setattr(runtime,'align_and_probe',_fake_runtime)
    return proposal,final,staging,out


@pytest.mark.parametrize('retry',[False,True])
def test_real_inventory_seal_preserves_generator_and_retry_executor(real_recorded,retry):
    proposal,final,staging,_=real_recorded
    result=e3._process_candidate(proposal,{'observation_sha256':'a'*64,
        'observation_mask_fraction':.8},np.zeros((60,3)),staging,final,'b'*40,
        retry=retry,parent_proposal_id=proposal['proposal_id'] if retry else None)
    assert result['raw_generator_commit']=='d'*40
    assert result['registration_executor_commit']=='b'*40
    assert result['tool_commit']==('b'*40 if retry else 'd'*40)


@pytest.mark.parametrize('omitted',['resolved_jobs.json','resolved_proposals.json'])
def test_omitted_controller_seal_member_cannot_authenticate_invented_commit(real_recorded,monkeypatch,omitted):
    proposal,final,staging,out=real_recorded
    directory=out/'input_inventory'
    proposals_path=directory/'resolved_proposals.json'
    inventory=json.loads(proposals_path.read_text())
    changed=next(p for p in inventory['proposals'] if p['proposal_id']==proposal['proposal_id'])
    changed['raw_generator_provenance']['generator_commit']='f'*40
    proposals_path.write_text(json.dumps(inventory))
    seal_path=directory/'seal.json';seal=json.loads(seal_path.read_text())
    # Every remaining declared member is hash-valid. The missing required
    # member itself must fail, not a coincidental mismatch in another file.
    seal['controller_members']['resolved_proposals.json']=e3.sha256_file(proposals_path)
    del seal['controller_members'][omitted]
    seal_path.write_text(json.dumps(seal))
    def forbidden(*args,**kwargs):raise AssertionError('registration must not run')
    monkeypatch.setattr(runtime,'align_and_probe',forbidden)
    with pytest.raises(ValueError,match='exactly both sanitized inventory members'):
        e3._process_candidate(changed,{},np.zeros((60,3)),staging,final,'b'*40)
    assert not list(staging.iterdir())


def test_real_inventory_loader_rejects_changed_proposal_bytes(real_recorded):
    proposal,final,_,out=real_recorded
    path=out/'input_inventory/resolved_proposals.json'
    with path.open('a') as f:f.write(' ')
    with pytest.raises(ValueError,match='sealed member hash mismatch'):
        e3._recorded_generator_identity(proposal,final)
