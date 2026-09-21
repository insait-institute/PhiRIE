"""Executing-source identity stays separate from the shared artifact checkout."""
import copy
import json
import subprocess
from pathlib import Path

import pytest

from robo.eval import agentic_ablation as e3
from robo.manifest.hash import canonical_hash


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),'-c','user.name=Fixture',
        '-c','user.email=fixture@example.invalid','-c','commit.gpgsign=false',*args],text=True).strip()


def seal_contract(path,value):
    value['contract_sha256']=canonical_hash({k:v for k,v in value.items()
        if k not in {'created_utc','environment','contract_sha256'}})
    path.write_text(json.dumps(value))
    return value


@pytest.fixture
def checkout_pair(tmp_path,monkeypatch):
    # main(control) defensively sets this process flag; restore it after each CLI fixture.
    monkeypatch.setenv("SIMANY_NO_GT", "0")
    evidence=tmp_path/'artifacts';evidence.mkdir()
    git(evidence,'init','-q')
    (evidence/'.gitignore').write_text('outputs/\n')
    (evidence/'configs').mkdir();(evidence/'configs/jobs.yaml').write_text('scope: old\n')
    (evidence/'configs/policies.yaml').write_text('policy: old\n')
    git(evidence,'add','.');git(evidence,'commit','-qm','artifact checkout')
    code=tmp_path/'source';git(evidence,'worktree','add','-qb','execution',str(code))
    (code/'configs/jobs.yaml').write_text('scope: executing\n')
    (code/'configs/policies.yaml').write_text('policy: executing\n')
    git(code,'add','.');git(code,'commit','-qm','executing source')
    monkeypatch.setattr(e3,'CODE_ROOT',code)
    monkeypatch.setattr(e3,'REPOSITORY_ROOT',evidence)
    path=evidence/'outputs/icra2027/freeze/contract/freeze_manifest.json'
    path.parent.mkdir(parents=True)
    value=dict(freeze_id='freeze',code={'commit':git(code,'rev-parse','HEAD'),'dirty':False},
        resource_inventory=[{'resolved_path':str(code/'configs'/name),'hash_method':'content_sha256',
            'sha256':e3.sha256_file(code/'configs'/name)} for name in ['jobs.yaml','policies.yaml']])
    seal_contract(path,value)
    return code,evidence,path,value


def test_sibling_checkout_is_allowed_but_artifact_and_config_roots_stay_separate(checkout_pair):
    code,evidence,path,contract=checkout_pair
    assert git(code,'rev-parse','HEAD')!=git(evidence,'rev-parse','HEAD')
    assert e3._validated_evidence_root(str(evidence))==evidence
    assert e3._validated_evidence_root(None)==code
    assert e3._load_yaml_code('configs/jobs.yaml','jobs')['scope']=='executing'
    assert e3._load_yaml_repo('configs/jobs.yaml','legacy helper')['scope']=='old'
    assert e3._validate_cli_execution(path,'freeze',path.parent.parent/'agentic',
        config_paths=['configs/jobs.yaml','configs/policies.yaml'])==contract
    with pytest.raises(ValueError,match='repository root'):
        e3.checked_repo_path(code/'configs/jobs.yaml','artifact escape')
    with pytest.raises(ValueError,match='executing code root'):
        e3._checked_code_path(evidence/'configs/jobs.yaml','wrong source config')


@pytest.mark.parametrize('damage',['relative','symlink','nested','foreign','missing'])
def test_evidence_override_requires_nonsymlink_same_git_checkout(checkout_pair,tmp_path,damage):
    code,evidence,*_=checkout_pair
    if damage=='relative':candidate='artifacts'
    elif damage=='symlink':
        candidate=tmp_path/'alias';candidate.symlink_to(evidence,target_is_directory=True)
    elif damage=='nested':candidate=evidence/'configs'
    elif damage=='foreign':
        candidate=tmp_path/'foreign';candidate.mkdir();git(candidate,'init','-q')
    else:candidate=tmp_path/'absent'
    with pytest.raises(ValueError):e3._validated_evidence_root(str(candidate))


@pytest.mark.parametrize('damage',['evidence_commit','wrong_freeze','dirty_contract','digest','dirty_code','stale_config','wrong_config_root'])
def test_cli_rejects_source_e0_or_config_swap(checkout_pair,damage):
    code,evidence,path,contract=checkout_pair
    if damage=='evidence_commit':contract['code']['commit']=git(evidence,'rev-parse','HEAD')
    elif damage=='wrong_freeze':contract['freeze_id']='different'
    elif damage=='dirty_contract':contract['code']['dirty']=True
    elif damage=='dirty_code':(code/'uncommitted').write_text('source changed')
    elif damage=='stale_config':
        (code/'configs/policies.yaml').write_text('policy: changed\n')
        git(code,'add','.');git(code,'commit','-qm','new source')
        contract['code']['commit']=git(code,'rev-parse','HEAD')
    elif damage=='wrong_config_root':
        contract['resource_inventory'][0]['resolved_path']=str(evidence/'configs/jobs.yaml')
    seal_contract(path,contract)
    if damage=='digest':
        contract['contract_sha256']='0'*64;path.write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        e3._validate_cli_execution(path,'freeze',path.parent.parent/'agentic',
            config_paths=['configs/jobs.yaml','configs/policies.yaml'])


def empty_inventory(out,contract):
    from tests.test_agentic_missing_observations import inventory
    jobs,proposals=inventory()
    jobs['scenes'][0]['jobs']=[]
    jobs['counts']={'scenes':1,'jobs':0,'policy_object_rows':0}
    jobs['source_contract'].update(code_commit=contract['code']['commit'],
        contract_sha256=contract['contract_sha256'])
    proposals['counts']={'jobs':0,'trellis_available':0,'reconviagen_available':0,'reconviagen_typed_unavailable':0}
    proposals['proposals']=[]
    directory=out/'input_inventory';directory.mkdir(parents=True)
    for name,data in [('resolved_jobs.json',jobs),('resolved_proposals.json',proposals)]:
        (directory/name).write_text(json.dumps(data))
    members={name:e3.sha256_file(directory/name) for name in ['resolved_jobs.json','resolved_proposals.json']}
    (directory/'seal.json').write_text(json.dumps({'members':members,'controller_members':members}))
    return jobs


def test_observe_cli_uses_sealed_inventory_without_opening_jobs_yaml(checkout_pair,monkeypatch):
    code,evidence,path,contract=checkout_pair;out=path.parent.parent/'agentic'
    empty_inventory(out,contract)
    # Neither CLI config is read by observe; both can deliberately be absent.
    assert e3.main(['--jobs','never-read-jobs.yaml','--policies','never-read-policies.yaml',
        '--contract-manifest',str(path),'--freeze-id','freeze','--out',str(out),
        '--scene-id','09c1414f1b','--observe'])==0
    manifest=json.loads((out/'observations/09c1414f1b/manifest.json').read_text())
    assert manifest['code_commit']==git(code,'rev-parse','HEAD')
    assert manifest['renderer_runtime'] is None and manifest['records']==[]
    assert not (code/'outputs').exists()


@pytest.mark.parametrize('damage',['code','contract','freeze'])
def test_resealed_inventory_cannot_bind_another_executor(checkout_pair,damage):
    code,evidence,path,contract=checkout_pair;out=path.parent.parent/'agentic'
    jobs=empty_inventory(out,contract)
    key={'code':'code_commit','contract':'contract_sha256','freeze':'contract_freeze_id'}[damage]
    jobs['source_contract'][key]='wrong'
    directory=out/'input_inventory';p=directory/'resolved_jobs.json';p.write_text(json.dumps(jobs))
    seal=json.loads((directory/'seal.json').read_text())
    seal['members'][p.name]=seal['controller_members'][p.name]=e3.sha256_file(p)
    (directory/'seal.json').write_text(json.dumps(seal))
    with pytest.raises(ValueError):e3._validate_cli_execution(path,'freeze',out,require_inventory=True)


def test_control_cli_uses_executing_policy_and_shared_empty_scene(checkout_pair):
    code,evidence,path,contract=checkout_pair;out=path.parent.parent/'agentic'
    real_policy=Path(e3.__file__).resolve().parents[2]/'configs/experiments/icra2027/agentic_automatic_policies.yaml'
    policy=code/'configs/policies.yaml';policy.write_bytes(real_policy.read_bytes())
    git(code,'add','.');git(code,'commit','-qm','frozen real policy')
    contract['code']['commit']=git(code,'rev-parse','HEAD')
    for row in contract['resource_inventory']:
        row['sha256']=e3.sha256_file(row['resolved_path'])
    seal_contract(path,contract);empty_inventory(out,contract)
    args=['--jobs','never-read-jobs.yaml','--policies','configs/policies.yaml',
        '--contract-manifest',str(path),'--freeze-id','freeze','--out',str(out),
        '--scene-id','09c1414f1b']
    assert e3.main([*args,'--observe'])==0
    assert e3.main([*args,'--control'])==0
    directory=out/'control/09c1414f1b'
    shard=json.loads((directory/'controller_shard.json').read_text())
    assert shard['job_count']==0 and shard['ledger_row_count']==0
    assert {p.name for p in (directory/'selected_assets').iterdir()}=={'A0','A1','A2','A3','A4'}
    assert not (code/'outputs').exists()
