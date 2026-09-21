"""Preregister E4 reset identities and authenticate CPU qualification; never launch.

The output is preparation metadata, not a rollout ledger or measured reset bank.
Planning failures retain their full declared population without invented tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

from robo.eval import episode_log as elog

SCENES = ('27dd4da69e', '40aec5fffa')
TASKS = (
    (SCENES[0], 'obj_1001_to_region', 'object_to_region'),
    (SCENES[0], 'obj_1001_to_obj_1000', 'object_to_receptacle'),
    (SCENES[1], 'obj_1001_to_region', 'object_to_region'),
    (SCENES[1], 'obj_1001_to_obj_1005', 'object_to_receptacle'),
)
ARMS = {'A0': 'a0_raster', 'A4': 'a4_raster'}
BLOCKERS = (
    'Automatic camera/scorer producer must authenticate these paired task bundles and every selected executable reset.',
    'Canonical CPU eligibility and prebuild certification must support the authenticated automatic qualifier schema for real pi05.',
    'Canonical real-policy execution must preserve failed selected cells and require camera/scorer plus exact policy identity before eligible rollouts.',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_protocol(path, expected_sha256):
    path = Path(path).resolve(strict=True)
    if sha(path) != expected_sha256:
        raise ValueError('preregistered protocol SHA256 differs')
    p = yaml.safe_load(path.read_text())
    fixed = [(x['scene_id'], x['task_id'], x['task_family']) for x in p['fixed_manipulation_tasks']]
    expected = [(scene, f'{scene}__{task}', family) for scene, task, family in TASKS]
    endpoints = [(t.get('target'), t.get('receptacle')) for t in p['fixed_manipulation_tasks']]
    if endpoints != [('obj_1001',None),('obj_1001','obj_1000'),('obj_1001',None),('obj_1001','obj_1005')]:
        raise ValueError('fixed task endpoints changed')
    if fixed != expected:
        raise ValueError('fixed task roster changed; no replacement after qualification')
    frozen = dict(schema_version=1, study_scope='e4_compact_canonical_engineering',
        paper_ready=False, no_substitution_after_qualification=True,
        no_policy_outcome_selection=True, no_full_cohort_replacement=True,
        episodes_per_task_arm=5, reset_base_seed=0, jitter_xy_m=.01,
        planned_manipulation_episodes=40, construction_arms=['A0','A4'],
        first_real_policy='pi05_droid_jointpos', planned_qualification_queries=28,
        planned_qualification_cells=280)
    if any(p.get(k) != value for k, value in frozen.items()):
        raise ValueError('compact protocol frozen scope/reset/policy fields differ')
    if p.get('population') != dict(scene_ids=list(SCENES), planned_objects=17,
                                    planned_policy_object_rows=85):
        raise ValueError('canonical object population differs')
    queries = p['selected_semantic_queries']
    ids = [x['task_id'] for x in queries]
    if len(ids) != 28 or len(set(ids)) != 28:
        raise ValueError('complete qualification population differs')
    for task in p['fixed_manipulation_tasks']:
        candidates = [q for q in queries if q['scene_id'] == task['scene_id']
                      and q['task_family'] == task['task_family']]
        if not candidates or min(candidates, key=lambda q:q['task_id']) != task:
            raise ValueError('fixed task is not the predeclared first semantic query')
    return p


def definitions(protocol):
    """Use the canonical identity/seed type; supply no unmeasured pose or jitter."""
    return [elog.ResetState(reset_state_id=f"{t['task_id']}__seed0__ep{ep}",
                scene_id=t['scene_id'], task_id=t['task_id'], ep=ep, base_seed=0,
                reset_seed=elog.derive_reset_seed(0, t['task_id'], ep))
            for t in protocol['fixed_manipulation_tasks'] for ep in range(5)]


def qualifier_handoff(protocol, *, screen_id, expected_commit, root):
    """Delegate full source, artifact, geometry and denominator replay to E4.

    Execute from the qualifier's exact source worktree; no foreign-source alias.
    Never select from strict_pass_task_ids or change a failed selected task.
    """
    from robo.eval import e4_candidate_screen as screen
    screen._code_snapshot(expected_commit)
    selected = {t['task_id'] for t in protocol['fixed_manipulation_tasks']}
    result, sources = [], {}
    for scene in SCENES:
        checked = screen._validate_qualifier_output(root=Path(root).resolve(),
            screen_id=screen_id, scene_id=scene, expected_commit=expected_commit)
        prepared = screen._load_prepare(root=Path(root).resolve(), screen_id=screen_id,
            scene_id=scene, expected_commit=expected_commit)
        if 'automatic_population' not in prepared['gate']:
            raise ValueError('compact qualification requires automatic source construction')
        all_expected = {(q['task_id'],arm,ep) for q in protocol['selected_semantic_queries']
            if q['scene_id']==scene for arm in ARMS for ep in range(5)}
        all_observed = [(r['task_id'],r['policy_id'],r['episode']) for r in checked['metric_rows']]
        if len(all_observed) != len(set(all_observed)) or set(all_observed) != all_expected:
            raise ValueError('complete preregistered qualification population differs')
        bundle = checked['bundle']['directory']
        sources[scene] = {name: {'path':str(bundle/name), 'sha256':sha(bundle/name)}
                         for name in ('gate.json','manifest.json','seal.json','metrics.jsonl')}
        # The full canonical validator above sees every prospective task, including failures.
        result.extend(row for row in checked['metric_rows'] if row['task_id'] in selected)
    expected = {(s.scene_id,s.task_id,arm,s.ep,s.reset_seed)
                for s in definitions(protocol) for arm in ARMS}
    observed = [(r['scene_id'],r['task_id'],r['policy_id'],r['episode'],r['reset_seed']) for r in result]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError('selected qualification cells missing, duplicated or replaced')
    return result, sources


def _planning_terminal(path, expected_sha256, *, protocol_path, protocol_sha256,
                       protocol, expected_commit, evidence_root):
    """Authenticate an applicability audit; it cannot authorize a rollout."""
    from robo.eval import e4_candidate_screen as screen
    from run.icra2027.e4_planning_terminal import validate_terminal_receipt
    if not expected_commit or not expected_sha256:
        raise ValueError('planning-terminal coverage requires exact source and receipt hashes')
    producer = screen._code_snapshot(expected_commit)
    path = screen.sealed_cpu._regular_file(Path(path), root=Path(evidence_root).resolve(),
                                            label='planning terminal receipt')
    if sha(path) != expected_sha256:
        raise ValueError('planning terminal receipt hash differs')
    receipt = validate_terminal_receipt(path, expected_producer_commit=expected_commit)
    expected_header = {'schema_version':1,'scope':'automatic_compact_planning_applicability_audit',
        'status':'PASS','applicability_status':'FAIL','execution_status':'NOT_RUN',
        'paper_ready':False,'headline_eligible':False,'planned_semantic_pairs':28,'planned_episodes':40}
    if any(type(receipt.get(k)) is not type(v) or receipt[k] != v for k,v in expected_header.items()):
        raise ValueError('planning terminal scope/claim/denominator differs')
    if receipt.get('producer_code') != producer:
        raise ValueError('planning terminal producer source differs')
    expected_protocol = {'path':str(Path(protocol_path).resolve()),'sha256':protocol_sha256}
    if receipt.get('protocol') != expected_protocol:
        raise ValueError('planning terminal protocol differs')
    scenes=receipt.get('scenes',{})
    if set(scenes) != set(SCENES):
        raise ValueError('planning terminal scene roster differs')
    for scene,count in zip(SCENES,(18,10),strict=True):
        if (type(scenes[scene].get('semantic_pairs')) is not int
                or scenes[scene]['semantic_pairs'] != count
                or sum(q['scene_id']==scene for q in protocol['selected_semantic_queries']) != count):
            raise ValueError('planning terminal semantic population differs')
        status=scenes[scene].get('planning_status')
        reason=scenes[scene].get('reason')
        if ((status=='FAIL' and reason!='no_prepared_size_admissible_target')
                or (status=='PASS' and reason is not None) or status not in {'PASS','FAIL'}):
            raise ValueError('planning terminal scene result differs')
    if not any(s['planning_status']=='FAIL' for s in scenes.values()):
        raise ValueError('planning terminal lacks an observed planning failure')
    return receipt, {'path':str(path),'sha256':expected_sha256}, producer


def prepare(protocol_path, expected_sha256, out, *, screen_id=None,
            expected_commit=None, evidence_root='/group/worldcept/PhiRIE/code/SimAny',
            planning_terminal_path=None, planning_terminal_sha256=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite compact handoff: {out}')
    if planning_terminal_path is not None:
        if screen_id is not None or not expected_commit or not planning_terminal_sha256:
            raise ValueError('planning-terminal mode requires receipt hash/source and cannot mix qualifier mode')
        from robo.eval import e4_candidate_screen as screen
        out = screen.sealed_cpu._inside(out, root=Path(evidence_root).resolve(),
                                       label='planning coverage output')
    elif planning_terminal_sha256 is not None or bool(screen_id) != bool(expected_commit):
        raise ValueError('qualification requires screen ID and exact producer commit together')
    protocol = checked_protocol(protocol_path, expected_sha256)
    states = definitions(protocol)
    terminal = None
    if planning_terminal_path is not None:
        terminal,terminal_identity,producer = _planning_terminal(planning_terminal_path,
            planning_terminal_sha256,protocol_path=protocol_path,protocol_sha256=expected_sha256,
            protocol=protocol,expected_commit=expected_commit,evidence_root=evidence_root)
    rows, sources = (qualifier_handoff(protocol,screen_id=screen_id,
        expected_commit=expected_commit,root=evidence_root) if screen_id else ([],{}))
    index = {(r['policy_id'],r['task_id'],r['episode']):r for r in rows}
    cells = []
    for state in states:
        for arm,treatment in ARMS.items():
            row = index.get((arm,state.task_id,state.ep))
            cells.append({**state.to_dict(), 'construction_policy':arm,
                'treatment_id':treatment, 'episode_id':elog.episode_id_for(treatment,state.reset_state_id),
                'rollout_state':'NOT_RUN', 'outcome':None,
                'qualification_state':'NOT_RUN' if row is None else ('PASS' if row['passed'] else 'FAIL'),
                'source_qualifier_cell_id':None if row is None else row['cell_id'],
                'source_failure_type':None if row is None else row.get('failure_type'),
                'source_failed_checks':None if row is None else sorted(k for k,v in row['checks'].items() if not v)})
    report = {'schema_version':1,'scope':'compact_harness_preparation_only',
        'protocol':{'path':str(Path(protocol_path).resolve()),'sha256':expected_sha256},
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],
                       cwd=Path(__file__).resolve().parents[2],text=True).strip(),
        'planned_reset_definitions':len(states),'planned_episode_cells':len(cells),
        'actual_reset_bank':None,'measured_initial_poses':None,'harness_config':None,
        'rollout_ledger':None,'policy_id':protocol['first_real_policy'],
        'qualification_sources':sources,'qualification_replay':'PASS' if screen_id else 'NOT_RUN',
        'policy_launch_allowed':False,'paper_ready':False,'claim_gate':'NOT_RUN',
        'blockers':list(BLOCKERS),
        'reset_contract':'Definitions are IDs/seeds only. Materialize geometry and jitter through the existing common observed-frame task/reset producer; never independently transform A0/A4.'}
    if terminal is not None:
        for cell in cells:
            scene_result = terminal['scenes'][cell['scene_id']]
            failed = scene_result['planning_status']=='FAIL'
            cell.update(execution_status='NOT_RUN',policy_execution='not_invoked_prebuild',
                scene_planning_status=scene_result['planning_status'],
                task_applicability_status='FAIL' if failed else 'NOT_RUN',
                source_failure_type='planning_unqualified' if failed else 'paired_matrix_prerequisite_unavailable',
                source_failed_checks=['prepared_size_admissible_target_exists'] if failed else None,
                planning_terminal=terminal_identity,
                task_definition=None,reset_provenance=None,camera_diagnostics=None,
                success=None,score=None,grasp=None,lift=None,place=None,ticks=None,
                policy_latency_ms=None,observation_latency_ms=None)
        report.update(scope='compact_harness_planning_unqualified_coverage',
            producer_code=producer,planning_terminal=terminal_identity,
            planning_applicability_source=terminal['source'],
            planned_semantic_pairs=28,planned_qualification_cells=280,
            semantic_pairs_by_scene={s:terminal['scenes'][s]['semantic_pairs'] for s in SCENES},
            source_qualification_summaries={s:terminal['scenes'][s].get('qualifier') for s in SCENES},
            qualification_replay='SOURCE_SUMMARY_ONLY',
            qualification_cell_note='Per-cell qualification_state is NOT_RUN because this preparation artifact does not ingest qualifier rows; authenticated source summaries are retained separately.',
            execution_status='NOT_RUN',applicability_gate='FAIL',claim_gate='NOT_RUN',
            recorded_rollout_episodes=0,
            blockers=['Observed source-geometry planning failure prevents a complete paired task definition.',
                      'This manifest records planned coverage only; no canonical rollout ledger or policy success measurement exists.'])
    out.parent.mkdir(parents=True,exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.compact-handoff-',dir=out.parent))
    try:
        elog.save_reset_states(states,stage/'planned_reset_definitions.json')
        for name,value in (('planned_episode_cells.json',cells),('handoff.json',report)):
            (stage/name).write_text(json.dumps(value,indent=2)+'\n')
        if screen_id:
            (stage/'selected_qualifier_cells.json').write_text(json.dumps(rows,indent=2)+'\n')
        if terminal is not None:
            from robo.eval import e4_candidate_screen as screen
            # Reauthenticate at publication; changed source/receipt fails closed.
            after,after_identity,after_producer = _planning_terminal(planning_terminal_path,
                planning_terminal_sha256,protocol_path=protocol_path,protocol_sha256=expected_sha256,
                protocol=protocol,expected_commit=expected_commit,evidence_root=evidence_root)
            if (after,after_identity,after_producer) != (terminal,terminal_identity,producer):
                raise ValueError('planning terminal source changed during preparation')
            screen._publish_bundle(out,manifest_kind='e4_planning_unqualified_coverage',
                payloads={p.name:p.read_bytes() for p in stage.iterdir()},
                manifest_fields={'code':producer,'planning_terminal':terminal_identity,
                                 'scope':report['scope'],'paper_ready':False})
            shutil.rmtree(stage)
        else:
            stage.rename(out)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--qualification-screen-id')
    parser.add_argument('--expected-code-commit')
    parser.add_argument('--evidence-root',default='/group/worldcept/PhiRIE/code/SimAny')
    parser.add_argument('--planning-terminal')
    parser.add_argument('--planning-terminal-sha256')
    a=parser.parse_args()
    print(json.dumps(prepare(a.protocol,a.protocol_sha256,a.out,
        screen_id=a.qualification_screen_id,expected_commit=a.expected_code_commit,
        evidence_root=a.evidence_root,planning_terminal_path=a.planning_terminal,
        planning_terminal_sha256=a.planning_terminal_sha256),indent=2))


if __name__=='__main__':
    main()
