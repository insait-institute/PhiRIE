import json
import pytest
from robo.eval.native_scale_tables import geometry_table,_sha
from robo.eval.native_scale_construction import construction_decisions
from robo.eval.native_scale_evidence import conclusions

METHODS=('B0_FIXED_NATIVE','B3_AGENT_NATIVE','B4_ROOM_REPAIR_NATIVE','BM_BUDGET_MATCHED_NATIVE')


def write(path,d):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(d));return path


@pytest.fixture
def support(tmp_path):
    units=[dict(cohort_id='c',scope='L0_target_only',sensor_regime='ideal_rgbd_posed',split='test',canonical_instance_id=i,controller_method=m) for i in ('a','b','missing') for m in METHODS]
    records=[]
    for iid,methods in [('a',METHODS),('b',METHODS[:2])]:
        root=tmp_path/iid;mesh=root/'mesh_sim.obj';mesh.parent.mkdir();mesh.write_text('bound')
        rows=[dict(method=m,object_dir=str(root),cd_cm=1. if iid=='a' else 3.,f1_20=.8 if iid=='a' else .2,collapse=False) for m in methods]
        record=dict(geometry_producer='robo.eval.fidelity_metrics.geometry_metrics',evaluator_alignment='NONE',cohort_id='c',scope='L0_target_only',sensor_regime='ideal_rgbd',tier='TEST',canonical_instance_id=iid,canonical_manifest_sha256='canonical-'+iid,reference_points_sha256='reference-'+iid,rows=rows,frozen_estimated_artifacts={m:{'mesh_sim.obj':_sha(mesh)} for m in methods})
        records.append(write(root/'geometry.json',record))
    original=[]
    for u in units:
        row=dict(canonical_instance_id=u['canonical_instance_id'],controller_method=u['controller_method'])
        if u['canonical_instance_id']=='missing':row.update(terminal_status='NOT_READY',accepted=None,object_dir=None)
        else:
            accepted=u['canonical_instance_id']=='a' or u['controller_method'] in METHODS[:2]
            m=write(tmp_path/'manifests'/(u['canonical_instance_id']+u['controller_method']+'.json'),dict(accepted=accepted,canonical_instance_id=u['canonical_instance_id']))
            row.update(terminal_status='READY' if accepted else 'ABSTAINED',accepted=accepted,object_dir=str(tmp_path/u['canonical_instance_id']) if accepted else None,build_manifest=str(m),build_manifest_sha256=_sha(m))
        original.append(row)
    source=tmp_path/'source.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in original))
    snapshot=tmp_path/'snapshot.jsonl';snapshot.write_text(''.join(json.dumps(dict(r,binding_source=str(source),binding_source_sha256=_sha(source)) if r['accepted'] is not None else r)+'\n' for r in original))
    return units,records,snapshot


def test_available_and_common_geometry_have_separate_explicit_denominators(support):
    u,r,s=support;rows,_=geometry_table(u,r,[s]);b0=next(x for x in rows if x['method']==METHODS[0]);b4=next(x for x in rows if x['method']==METHODS[2])
    assert b0['planned_objects']==3 and b0['matched_objects']==2 and b0['cd_cm']==2.
    assert b0['common_matched_objects']==1 and b0['common_cd_cm']==1.
    assert b0['common_matched_instance_ids']==['a'] and b0['accepted_objects'] is None
    assert b0['accepted_objects_observed']==2 and b0['build_decisions_unmeasured']==1
    assert b4['abstained_objects']==1 and b4['accepted_objects_observed']==1


def test_geometry_rejects_changed_common_reference_samples(support):
    u,r,s=support;d=json.loads(r[0].read_text());one=dict(d,rows=[d['rows'][0]],reference_points_sha256='wrong');d['rows']=d['rows'][1:];write(r[0],d);p=write(r[0].parent/'wrong.json',one)
    with pytest.raises(ValueError,match='reference surfaces'):geometry_table(u,[*r,p],[s])


@pytest.mark.parametrize('bad',['duplicate','accepted','pending','source'])
def test_constructor_join_rejects_duplicate_forged_and_measured_pending(support,bad):
    u,_,s=support;rows=[json.loads(l) for l in s.read_text().splitlines()]
    if bad=='duplicate':rows.append(rows[0])
    if bad=='accepted':rows[0]['accepted']=False
    if bad=='pending':next(r for r in rows if r['terminal_status']=='NOT_READY')['accepted']=False
    if bad=='source':rows[0]['binding_source_sha256']='0'*64
    s.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):construction_decisions(u,[s])


def test_test_conclusions_never_treat_nonsignificance_as_preservation():
    arm=dict(split='test',scope='L0_target_only',method='B3_AGENT_NATIVE',executed=2,planned=10,successes_observed=0,unmeasured=8,complete=False)
    pair=dict(split='test',method='B3_AGENT_NATIVE',reference='REF_NATIVE',paired_population_complete=False,reference_has_success=False,delta_pp=None,ci95_pp=[None,None])
    rows=conclusions([arm],[pair]);assert rows[0]['claim_id'].startswith('test_') and rows[1]['gate']=='NOT_RUN'
    pair.update(paired_population_complete=True,delta_pp=0.,ci95_pp=[0.,0.]);rows=conclusions([arm],[pair]);r=rows[1]
    assert r['paired_direction']=='INCONCLUSIVE' and r['preservation_claim']=='NOT_RUN' and r['outcome_benefit']=='NOT_RUN'
    assert r['reference_has_observed_success'] is False
