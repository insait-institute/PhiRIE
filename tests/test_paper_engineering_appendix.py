import copy
import json
from pathlib import Path

import pytest
import yaml

from robo.eval import paper_pipeline as p
from robo.manifest.hash import canonical_hash
from tests.test_paper_pipeline_audit import fixture_config


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    return dict(path=str(path),sha256=p._sha256(path),size_bytes=path.stat().st_size)


def t2_fixture(tmp_path):
    root=tmp_path/'t2';path=root/'trellis2_geometry/geometry.json'
    rows=[dict(job_id=f'38d58a7a31/obj_{i}',geometry_status='matched' if i in (1001,1006,1008) else 'unmatched',
               matched_gt_id=i if i in (1001,1006,1008) else None,
               cd_cm=3. if i in (1001,1006,1008) else None,f1_20=.2 if i in (1001,1006,1008) else None,
               collapse=False if i in (1001,1006,1008) else None,psnr=None,ssim=None,lpips=None) for i in range(1000,1015)]
    payload=dict(schema_version=1,scope='trellis2_mesh_only_independent_geometry_engineering_pilot',freeze_id='t2',
                 source_commit='a'*40,native_gaussian=False,full_twin_ready=False,paper_ready=False,headline_eligible=False,
                 claim_gate='NOT_RUN',planned_jobs=15,generated_jobs=15,geometry_evaluated_jobs=3,unmatched_jobs=12,cd_cm=3.,f1_20=.2,
                 catastrophic_collapses=0,rows=rows)
    spec=put(path,payload)
    checks=('source_train_registration_before_GT','original_source_validators','complete15_roster','three_frozen_matches',
            'canonical_geometry_replay_exact','aggregate_arithmetic_exact','CSV_JSON_exact','unmatched_null_geometry',
            'appearance_null','sealed_outputs','nonheadline_gates')
    audit=dict(schema_version=1,scope='trellis2_mesh_geometry_output_integrity',status='PASS',source_commit='a'*40,
               freeze_id='t2',native_gaussian=False,full_twin_ready=False,paper_ready=False,headline_eligible=False,
               claim_gate='NOT_RUN',checks={k:'PASS' for k in checks},evidence_hashes={str(path):spec})
    spec['completion_audit']=put(root/'independent_geometry_completion_audit.json',audit)
    return spec


def e4_fixture(tmp_path):
    from run.icra2027.e4_compact_harness import checked_protocol,definitions
    primary=tmp_path/'primary';root=primary/'outputs/icra2027/e4';directory=root/'planned_coverage'
    protocol_path=Path(__file__).resolve().parents[1]/'configs/experiments/icra2027/e4_compact_canonical/protocol.yaml'
    protocol={'path':str(protocol_path),'sha256':p._sha256(protocol_path)}
    parsed=checked_protocol(**dict(path=protocol['path'],expected_sha256=protocol['sha256']))
    nulls=('outcome','success','score','grasp','lift','place','ticks','task_definition','reset_provenance','camera_diagnostics','policy_latency_ms','observation_latency_ms')
    cells=[]
    for reset in definitions(parsed):
        for arm in ('A0','A4'):
            cells.append(dict(scene_id=reset.scene_id,task_id=reset.task_id,reset_state_id=reset.reset_state_id,
                              reset_seed=reset.reset_seed,construction_policy=arm,episode_id=f'{arm}-{reset.reset_state_id}',
                              execution_status='NOT_RUN',policy_execution='not_invoked_prebuild',**{k:None for k in nulls}))
    old=primary/'outputs/icra2027/old';summary=dict(planned_cells=100,passed_cells=0,fixed_selected_cells=20,fixed_passed_cells=0)
    for key in ('gate','manifest','metrics','seal'):
        spec=put(old/f'qualifier/{key}.json',{});spec['path']=str(Path(spec['path']).relative_to(primary));summary[key]=spec
    terminal=put(root/'planning_terminal.json',dict(applicability_status='FAIL'))
    payload=dict(scope='compact_harness_planning_unqualified_coverage',paper_ready=False,policy_launch_allowed=False,
                 execution_status='NOT_RUN',applicability_gate='FAIL',rollout_ledger=None,actual_reset_bank=None,
                 planned_episode_cells=40,planned_semantic_pairs=28,planned_qualification_cells=280,recorded_rollout_episodes=0,
                 source_commit='b'*40,planning_terminal=terminal,source_qualification_summaries={'27dd4da69e':None,'40aec5fffa':summary},
                 protocol=protocol,planning_applicability_source={'e0':{'path':str(old/'contract/freeze_manifest.json')}})
    spec=put(directory/'handoff.json',payload)
    put(directory/'planned_episode_cells.json',cells);put(directory/'planned_reset_definitions.json',[])
    manifest=dict(code={'commit':'b'*40},planning_terminal=terminal,files={name:dict(sha256=p._sha256(directory/name),size_bytes=(directory/name).stat().st_size)
                  for name in ('handoff.json','planned_episode_cells.json','planned_reset_definitions.json')})
    m=put(directory/'manifest.json',manifest);seal=put(directory/'seal.json',dict(members={'manifest.json':m}))
    contract=dict(freeze_id='e4',code={'commit':'b'*40,'dirty':False});contract['contract_sha256']=canonical_hash(contract)
    put(root/'contract/freeze_manifest.json',contract)
    receipt=dict(freeze_id='e4',source_commit='b'*40,planned_cells=40,executed_policy_cells=0,applicability_gate='FAIL',manipulation_claim_gate='NOT_RUN',
                 coverage_manifest_sha256=m['sha256'],coverage_seal_sha256=seal['sha256'],terminal_sha256=terminal['sha256'],
                 source_qualification_summaries=payload['source_qualification_summaries'],contract_sha256=contract['contract_sha256'])
    spec['publication_receipt']=put(root/'publication_receipt.json',receipt)
    spec['completion_audit']=put(root/'independent_qa.json',dict(integrity='PASS',planned_episode_cells=40,policy_executed_cells=0,
                        manipulation_claim_gate='NOT_RUN',manifest_sha256=m['sha256'],null_measurement_fields=list(nulls)))
    return spec


def test_optional_engineering_tables_keep_main_rosters_and_trace_top_fields(tmp_path):
    path,config=fixture_config(tmp_path)
    config['engineering_appendix']={'trellis2_geometry':t2_fixture(tmp_path),'e4_planning':e4_fixture(tmp_path)}
    path.write_text(yaml.safe_dump(config));out=tmp_path/'paperout'
    result=p.generate(path,out)
    assert len(result['tables'])==5 and result['paper_ready'] is False
    t=(out/'generated_tables/trellis2_geometry_engineering.tex').read_text()
    assert '15 & 3 & 12 & 3.000 & 0.200 & 0' in t and 'mesh-only' in t
    e=(out/'generated_tables/compact_applicability_engineering.tex').read_text()
    assert '40 & 20 & 0 & 0 & --' in e and 'Prereq. checked' in e and 'No simulator or policy rollout' in e
    assert 'trellis2_geometry' not in (out/'generated_tables/agentic_ablation_main.tex').read_text()
    ledger=(out/'claim_ledger.csv').read_text()
    assert 'source_qualification_summaries.40aec5fffa.fixed_selected_cells' in ledger
    assert (out/'sources/trellis2_geometry.csv').exists() and (out/'sources/e4_planning.csv').exists()


@pytest.mark.parametrize('change',['scope','denominator','generated','matched','appearance','unmatched','claim','arithmetic','source_bytes','missing_audit_gate'])
def test_t2_integrity_negatives(tmp_path,change):
    spec=t2_fixture(tmp_path);path=Path(spec['path']);d=json.loads(path.read_text())
    if change=='scope':d['scope']='full_twin'
    elif change=='denominator':d['planned_jobs']=14
    elif change=='generated':d['generated_jobs']=14
    elif change=='matched':d['rows'][0]['geometry_status']='matched'
    elif change=='appearance':d['rows'][0]['psnr']=20
    elif change=='unmatched':d['rows'][0]['f1_20']=.5
    elif change=='claim':d['paper_ready']=True
    elif change=='arithmetic':d['cd_cm']=4
    elif change=='source_bytes':path.write_text('{}')
    else:
        ap=Path(spec['completion_audit']['path']);a=json.loads(ap.read_text());a['checks'].pop('source_train_registration_before_GT')
        spec['completion_audit']=put(ap,a)
    if change not in ('source_bytes','missing_audit_gate'):
        fresh=put(path,d);ap=Path(spec['completion_audit']['path']);a=json.loads(ap.read_text());a['evidence_hashes'][str(path)]=fresh
        spec.update(fresh);spec['completion_audit']=put(ap,a)
    with pytest.raises(ValueError):p._engineering_sources({'trellis2_geometry':spec})


@pytest.mark.parametrize('change',['scope','rollout','policy','denominator','success','reset','qualifier_bytes','manifest','null_contract','receipt'])
def test_e4_integrity_negatives(tmp_path,change):
    spec=e4_fixture(tmp_path);path=Path(spec['path']);d=json.loads(path.read_text())
    if change in ('success','reset'):
        f=path.parent/'planned_episode_cells.json';cells=json.loads(f.read_text());cells[0]['success' if change=='success' else 'reset_provenance']=False
        put(f,cells)
    elif change=='qualifier_bytes':
        f=Path(d['planning_applicability_source']['e0']['path']).parents[4]/d['source_qualification_summaries']['40aec5fffa']['metrics']['path'];f.write_text('changed')
    elif change=='manifest':(path.parent/'manifest.json').write_text('{}')
    elif change=='null_contract':
        f=Path(spec['completion_audit']['path']);a=json.loads(f.read_text());a['null_measurement_fields'].remove('success');spec['completion_audit']=put(f,a)
    elif change=='receipt':
        f=Path(spec['publication_receipt']['path']);a=json.loads(f.read_text());a['executed_policy_cells']=1;spec['publication_receipt']=put(f,a)
    else:
        if change=='scope':d['scope']='learned_rollouts'
        elif change=='rollout':d['rollout_ledger']='invented.jsonl'
        elif change=='policy':d['policy_launch_allowed']=True
        else:d['planned_episode_cells']=39
        spec.update(put(path,d))
    with pytest.raises(ValueError):p._engineering_sources({'e4_planning':spec})


def test_unknown_engineering_source_is_rejected():
    with pytest.raises(ValueError):p._engineering_sources({'invented':{}})


@pytest.mark.parametrize('kind',['receipt_freeze','contract_freeze','member_symlink','receipt_symlink','manifest_freeze'])
def test_e4_stage_identity_and_symlink_negatives(tmp_path,kind):
    spec=e4_fixture(tmp_path);path=Path(spec['path']);root=path.parent.parent
    if kind=='receipt_freeze':
        f=Path(spec['publication_receipt']['path']);d=json.loads(f.read_text());d['freeze_id']='another';spec['publication_receipt']=put(f,d)
    elif kind=='contract_freeze':
        f=root/'contract/freeze_manifest.json';d=json.loads(f.read_text());d['freeze_id']='another';d.pop('contract_sha256')
        d['contract_sha256']=canonical_hash(d);put(f,d)
        rp=Path(spec['publication_receipt']['path']);receipt=json.loads(rp.read_text());receipt['contract_sha256']=d['contract_sha256'];spec['publication_receipt']=put(rp,receipt)
    elif kind=='member_symlink':
        f=path.parent/'planned_episode_cells.json';other=tmp_path/'real-cells.json';f.rename(other);f.symlink_to(other)
    elif kind=='receipt_symlink':
        f=Path(spec['publication_receipt']['path']);other=tmp_path/'real-receipt.json';f.rename(other);f.symlink_to(other)
    else:
        f=path.parent/'manifest.json';d=json.loads(f.read_text());d['freeze_id']='another';put(f,d)
    with pytest.raises(ValueError):p._engineering_sources({'e4_planning':spec})
