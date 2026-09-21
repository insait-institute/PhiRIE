import copy
import pytest
from run.icra2027.e4_full_protocol import protocol_from_population,select_matrix


def fixture():
    populations=[]
    for i in range(50):
        scene=f'{i:010d}'
        populations.append(dict(scene_id=scene,planned_objects=58 if i==0 else 37,
            queries=[dict(scene_id=scene,task_id=f'{scene}_{family}_{j}',task_family=family)
                     for family in ('object_to_region','object_to_receptacle') for j in range(3)]))
    p=protocol_from_population(populations,references={})
    rows=[dict(task_id=q['task_id'],all_frozen_qualification_gates_pass=True) for r in populations for q in r['queries']]
    return p,rows


def test_exact_roster_ties_and_denominators():
    p,rows=fixture();r=select_matrix(p,list(reversed(rows)))
    assert p['planned_objects']==1871 and p['planned_qualification_cells']==3000
    assert r['scene_ids']==[f'{i:010d}' for i in range(4)]
    assert len(r['tasks'])==16 and r['planned_episodes_per_arm']==80 and r['policy_executed']==0
    assert all(q['task_id'].endswith(('_0','_1')) for q in r['tasks'])


@pytest.mark.parametrize('damage',['missing','duplicate','foreign','policy','integer'])
def test_bad_or_policy_driven_selection_rejected(damage):
    p,rows=fixture()
    if damage=='missing':rows.pop()
    elif damage=='duplicate':rows.append(rows[0])
    elif damage=='foreign':rows[0]['task_id']='unknown'
    elif damage=='policy':rows[0]['success']=True
    else:rows[0]['all_frozen_qualification_gates_pass']=1
    with pytest.raises(ValueError):select_matrix(p,rows)


def test_three_qualified_scenes_does_not_relax_matrix_or_erase_failures():
    p,rows=fixture()
    for row in rows:row['all_frozen_qualification_gates_pass']=int(row['task_id'][:10])<3
    r=select_matrix(p,rows)
    assert r['status']=='NOT_RUN' and r['tasks']==[] and r['qualified_scenes']==3
    assert r['qualification_failed_queries']==282 and r['planned_semantic_queries']==300


@pytest.mark.parametrize('damage',['scene','job','task'])
def test_population_drift_rejected(damage):
    p,_=fixture();r=copy.deepcopy(p['source_populations'])
    if damage=='scene':r.pop()
    elif damage=='job':r[0]['planned_objects']+=1
    else:r[1]['queries'][0]['task_id']=r[0]['queries'][0]['task_id']
    with pytest.raises(ValueError):protocol_from_population(r,references={})


def test_budget_is_fixed_before_qualification_and_does_not_replace_failed_slots():
    p,rows=fixture();population=copy.deepcopy(p['source_populations'])
    scene=population[0]['scene_id']
    for i in range(3,8):
        population[0]['queries'].append(dict(scene_id=scene,task_id=f'{scene}_object_to_region_{i}',task_family='object_to_region'))
    p=protocol_from_population(population,references={})
    assert p['input_semantic_queries']==305 and p['planned_semantic_queries']==301
    ids={q['task_id'] for q in p['qualification_tasks']}
    assert f'{scene}_object_to_region_3' in ids and f'{scene}_object_to_region_4' not in ids
    rows=[dict(task_id=t,all_frozen_qualification_gates_pass=False) for t in ids]
    r=select_matrix(p,rows)
    assert r['status']=='NOT_RUN' and r['qualification_failed_queries']==301
