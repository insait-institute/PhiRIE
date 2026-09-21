"""Preregister a compact qualified-task matrix from all fifty discovery inputs.

This module reads no controller decisions, policy outcomes, GT, or evaluation
metrics. Existing E4 producers remain responsible for qualification and rollouts.
"""
from pathlib import Path
import argparse
import json
import yaml
from run.icra2027 import e4_compact_canonical as original

CODE = Path(__file__).resolve().parents[2]
FAMILIES = ('object_to_region', 'object_to_receptacle')


def protocol_from_population(populations, *, references):
    scenes = [r['scene_id'] for r in populations]
    if len(scenes) != 50 or len(set(scenes)) != 50:
        raise ValueError('all fifty original scenes required')
    if sum(r['planned_objects'] for r in populations) != 1871:
        raise ValueError('all 1871 discovered jobs required')
    queries = [q for r in populations for q in r['queries']]
    ids = [q['task_id'] for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate task identity')
    for r in populations:
        if any(q['scene_id'] != r['scene_id'] or q['task_family'] not in FAMILIES for q in r['queries']):
            raise ValueError('invalid scene or task family')
    planned = [q for r in populations for family in FAMILIES
               for q in sorted((q for q in r['queries'] if q['task_family']==family),
                               key=lambda q:q['task_id'])[:4]]
    return dict(schema_version=1, scope='e4_full_discovery_qualified_task_protocol', paper_ready=False,
        protocol_timing='Declared after the failed two-room compact pilot, before full-cohort qualification or learned-policy rollouts; not a replacement of that pilot.',
        input_references=references, planned_scenes=50, planned_objects=1871,
        source_populations=populations, input_semantic_queries=len(queries),
        qualification_input_selection='First four task_id values per family per scene, using only original discovery labels/IDs before full qualification; no replacement after failure.',
        qualification_tasks=planned, planned_semantic_queries=len(planned),
        unselected_input_semantic_queries=len(queries)-len(planned),
        qualification_cells_per_query=10, planned_qualification_cells=10*len(planned),
        arms=['A0','A4'], treatment_axis='construction', observation='mujoco_raster',
        qualification_implementation='robo.eval.e4_candidate_screen',
        camera_implementation='robo.eval.e4_camera_scorer_gate',
        rollout_implementation='robo.eval.harness_runner',
        role_vocabulary_changed=False, geometry_thresholds_changed=False,
        scene_selection='First four scene_id values with at least two fully qualified tasks in each family; lexicographic, independent of learned-policy outcomes.',
        task_selection='First two task_id values per family in each selected scene, after identical five-reset qualification in both arms and camera/scripted-stage gates.',
        pilot_selection='First two selected scenes and first selected task in each family; five matched resets per arm. These cells are retained in the full matrix with identical policy and contracts.',
        minimum_qualified_rooms=4, tasks_per_room=4, resets_per_task=5,
        planned_policy_episodes_per_arm_if_qualified=80,
        first_policy='pi05_droid_jointpos', multi_policy_claim=False,
        second_checkpoint='NOT_RUN unless independently available and frozen before its first episode',
        random_seed=0, reset_jitter_xy_m=0.01,
        failed_qualification='Keep every original scene, semantic query, reset/arm qualification cell and reason; do not fabricate a policy rollout.',
        insufficient_qualified_matrix='NOT_RUN paper-target matrix; publish complete qualification coverage, no relaxed threshold or substitute population.',
        qualification_coverage_denominator='All fifty scenes and every preselected semantic query; additionally report the complete input semantic population and fixed budget exclusions before conditional policy performance.',
        policy_coverage_denominator='All 80 pre-frozen reset cells per arm within the qualified cohort; later build/crash/timeout failures remain terminal rows.',
        no_post_policy_substitution=True, no_reference_or_policy_metric_selection=True,
        source_full_control_gate='All50/1871/9355 exact construction integrity PASS before qualification execution',
        full_release_gate='Exact-source E0 plus qualified frozen pilot, scripted predicates, real-policy smoke, camera/reset/control/checkpoint identities PASS',
        old_compact_experiment_retained=True)


def select_matrix(protocol, authenticated_qualifications):
    """Deterministic roster choice, only after external canonical authentication.

    Input is a complete boolean view of canonical qualification AND camera gates,
    with no policy metrics. This function computes no new qualification metric.
    """
    queries = protocol['qualification_tasks']
    expected = {q['task_id']:q for q in queries}
    rows = {}
    for row in authenticated_qualifications:
        if set(row) != {'task_id','all_frozen_qualification_gates_pass'} or type(row['all_frozen_qualification_gates_pass']) is not bool:
            raise ValueError('only authenticated qualification booleans allowed, no policy outcomes')
        tid = row['task_id']
        if tid not in expected or tid in rows:
            raise ValueError('unknown or duplicate qualification task')
        rows[tid] = row['all_frozen_qualification_gates_pass']
    if set(rows) != set(expected):
        raise ValueError('incomplete qualification population')
    available = []
    for scene in sorted(r['scene_id'] for r in protocol['source_populations']):
        groups = {family: sorted((q for q in queries if q['scene_id']==scene and q['task_family']==family and rows[q['task_id']]),key=lambda q:q['task_id']) for family in FAMILIES}
        if all(len(group)>=2 for group in groups.values()):
            available.append((scene,[q for family in FAMILIES for q in groups[family][:2]]))
    selected=available[:4] if len(available)>=4 else []
    return dict(status='READY_FOR_FROZEN_PILOT' if selected else 'NOT_RUN',
        qualified_scenes=len(available), planned_scenes=50, planned_semantic_queries=len(queries),
        qualification_pass_queries=sum(rows.values()), qualification_failed_queries=len(rows)-sum(rows.values()),
        scene_ids=[scene for scene,_ in selected], tasks=[q for _,tasks in selected for q in tasks],
        planned_episodes_per_arm=80 if selected else 0, policy_executed=0, paper_ready=False)


def build_protocol():
    old=original.make_protocol()  # Existing TRAIN discovery hashes, full roster and role definition.
    populations=[]
    for row in old['source_populations']:
        ref=row['all_jobs_manifest'];p=Path(ref['path'])
        if original.sha(p)!=ref['sha256']:
            raise ValueError('original discovery input changed')
        report=json.loads(p.read_text())
        populations.append(dict(scene_id=row['scene_id'],planned_objects=row['planned_objects'],
            all_jobs_manifest=ref,discovery_audit=row['discovery_audit'],
            queries=original.semantic_pairs(row['scene_id'],report['rows'])))
    return protocol_from_population(populations,references={k:old[k] for k in ('source_cohort','role_definition')})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    args=parser.parse_args();payload=build_protocol();p=Path(args.out)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:yaml.safe_dump(payload,f,sort_keys=False)
    print(json.dumps({k:payload[k] for k in ('planned_scenes','planned_objects','planned_semantic_queries','planned_qualification_cells','paper_ready')}))

if __name__=='__main__':main()
