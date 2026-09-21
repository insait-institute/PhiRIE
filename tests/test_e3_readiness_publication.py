"""Receipt-bound metadata publication never substitutes for canonical inventory."""
import json
import hashlib
import pytest
import yaml
from run.icra2027 import e3_fresh_canonical_config as builder


@pytest.fixture
def receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(builder.e3, 'CODE_ROOT', tmp_path)
    monkeypatch.setattr(builder.e3, 'REPOSITORY_ROOT', tmp_path)
    scenes = [f'{i:010x}' for i in range(50)]
    roster = tmp_path/'roster.yaml'
    roster.write_text(yaml.safe_dump({'population': {'scene_ids': scenes}}))
    override = tmp_path/'override.yaml'; override.write_text('{}')
    anchor = dict(path=str(override), sha256=builder.e3.sha256_file(override))
    monkeypatch.setattr(builder, 'load_pool_overrides', lambda *a: ({}, anchor))
    pools = {}
    for scene in scenes:
        d = tmp_path/'discovery'/scene; d.mkdir(parents=True)
        n = 1822 if scene == scenes[0] else 1
        rows = [dict(automatic_instance_id=1000+i) for i in range(n)]
        ident = dict(path=str(d/'scene.ply'), bytes=20, sha256='a'*64)
        meta = dict(gaussian=ident, metadata={k:ident for k in ('nerfstudio/transforms_undistorted.json','colmap/images.txt')})
        for name in builder.DISCOVERY_FILES:
            value = meta if name == 'input_manifest.json' else dict(scene_id=scene, planned_jobs=n, rows=rows)
            (d/name).write_text(json.dumps(value))
        hashes = {name:builder.e3.sha256_file(d/name) for name in sorted(builder.DISCOVERY_FILES)}
        p = tmp_path/'pools'/scene; p.mkdir(parents=True)
        for name in ('proposal_pool.json','proposal_records.jsonl','contract.json','audit.json'):
            (p/name).write_text('{}')
        (p/'input_manifest.json').write_text(json.dumps(dict(source_discovery_hashes=hashes)))
        provenance = dict(input_manifest_sha256=builder.e3.sha256_file(p/'input_manifest.json'),
            proposal_records_sha256=builder.e3.sha256_file(p/'proposal_records.jsonl'),
            contract_path=str(p/'contract.json'), contract_file_sha256=builder.e3.sha256_file(p/'contract.json'),
            completion_audit_path=str(p/'audit.json'), completion_audit_sha256=builder.e3.sha256_file(p/'audit.json'))
        pools[scene] = {'trellis':dict(path=str(p/'proposal_pool.json'), sha256=builder.e3.sha256_file(p/'proposal_pool.json'), producer_provenance=provenance)}
    value = dict(status='READY_FOR_RESERVED_FREEZE', config_written=False, paper_ready=False,
        planned_scenes=50, planned_jobs=1871, planned_policy_object_rows=9355,
        population=dict(scene_roster_config=str(roster), scene_roster_config_file_sha256=builder.e3.sha256_file(roster),
            scene_roster_sha256=hashlib.sha256(('\n'.join(scenes)+'\n').encode()).hexdigest(),
            planned_scenes=50,planned_jobs_per_policy=1871,planned_policy_object_rows=9355),
        initial_pools=pools, pool_override_manifest=anchor, runtime_accounting=dict(schema_version=1,process_histories=[]))
    path=tmp_path/'receipt.json';path.write_text(json.dumps(value))
    captured=[]
    def writer(payload,bundles,result,*a):
        captured.append((payload,bundles));return dict(result,config_written=True)
    monkeypatch.setattr(builder,'_write_configs',writer)
    monkeypatch.setattr(builder,'authenticate_pool',lambda *a,**k:pytest.fail('do not repeat preparation artifact scan'))
    return path,value,roster,captured


def publish(fixture):
    path,_,roster,_=fixture
    return builder.publish_readiness(path,builder.e3.sha256_file(path),freeze_id='reserved',
        config_directory=path.parent/'config',discovery_root=path.parent/'discovery',roster_config=roster)


def test_publication_preserves_population_and_requires_real_inventory(receipt):
    result=publish(receipt)
    payload,bundles=receipt[3][0]
    assert len(bundles)==50 and sum(len(j) for _,j in bundles)==1871
    assert payload['population']==receipt[1]['population']
    assert payload['runtime_accounting']==receipt[1]['runtime_accounting']
    assert result['canonical_inventory_validation_required'] is True
    assert result['paper_ready'] is False


@pytest.mark.parametrize('change',['receipt_hash','incomplete','roster','discovery','pool','audit','contract','histories','count'])
def test_publication_rejects_metadata_drift_without_writing(receipt,change):
    path,value,roster,captured=receipt
    if change=='receipt_hash':
        with pytest.raises(ValueError,match='receipt hash'):
            builder.publish_readiness(path,'f'*64,freeze_id='reserved',config_directory=path.parent/'config',
                discovery_root=path.parent/'discovery',roster_config=roster)
        return
    if change=='incomplete':value['status']='WAITING_REAL_INITIAL_POOLS'
    elif change=='roster':roster.write_text(roster.read_text()+'\n')
    elif change=='discovery':(path.parent/'discovery/0000000000/all_jobs_manifest.json').write_text('{}')
    elif change in ('pool','audit','contract'):
        name={'pool':'proposal_pool.json','audit':'audit.json','contract':'contract.json'}[change]
        (path.parent/'pools/0000000000'/name).write_text('{"changed":true}')
    elif change=='histories':value['runtime_accounting']['process_histories']=[dict(scene_id='0000000000')]
    elif change=='count':value['planned_jobs']=1870
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):publish(receipt)
    assert not captured
