"""Automatic candidate population uses real descriptors and preserves failed roles."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from robo.eval import e4_candidate_screen as screen
from robo.eval import e3_factory_materializer as materializer
from robo.eval import agentic_ablation as e3
from tests.test_e4_automatic_materializer import automatic_factory_input
from tests.test_e3_factory_materializer import _write_json, _reseal_inventory
from tests.test_agentic_missing_observations import inventory


@pytest.fixture
def population(automatic_factory_input, monkeypatch):
    root, old, descriptor, _, _ = automatic_factory_input
    source=root/'discovery';construction=source/'construction'
    labels=['plant pot','plant pot','book','book','book']
    np.savez(construction/'auto_instances.npz',labels=np.array(labels),scores=np.full(5,.8),
             **{f'vert_idx_{i}':np.arange(4) for i in range(5)})
    identities=json.loads((source/'output_hashes.json').read_text())
    identities['auto_instances.npz']={'path':str(construction/'auto_instances.npz'),
        'bytes':(construction/'auto_instances.npz').stat().st_size,
        'sha256':e3.sha256_file(construction/'auto_instances.npz')}
    _write_json(source/'output_hashes.json',identities)
    _write_json(source/'pilot_summary.json',{'stages':[{'exit_code':0}],
        'rows':[{'automatic_instance_id':1000+i,'prepared':False} for i in range(5)]})
    _write_json(source/'postrun_audit.json',{'summary_sha256':e3.sha256_file(source/'pilot_summary.json'),
        'output_hashes_sha256':e3.sha256_file(source/'output_hashes.json')})
    d=json.loads(descriptor.read_text());d['discovery_hashes']={n:e3.sha256_file(source/n) for n in d['discovery_hashes']}
    _write_json(descriptor,d)
    jobs,proposals=inventory();one=jobs['scenes'][0]['jobs'][0];pool=proposals['proposals']
    jobs['scenes'][0]['jobs']=[json.loads(json.dumps(one).replace('1000',str(1000+i))) for i in range(5)]
    jobs['counts'].update(jobs=5,policy_object_rows=25)
    proposals['proposals']=[json.loads(json.dumps(p).replace('1000',str(1000+i))) for i in range(5) for p in pool]
    proposals['counts'].update(jobs=5,reconviagen_typed_unavailable=5)
    audit={'paper_ready':False,'source_discovery':d['discovery_hashes'],'source_gaussian_training_provenance':'UNKNOWN',
        'jobs':[{'job_id':j['job_id'],'source_job_id':f'09c1414f1b:auto:{1000+i}',
                 'automatic_instance_id':1000+i,'prepared_output_index':None}
                for i,j in enumerate(jobs['scenes'][0]['jobs'])]}
    out=root/'complete-agentic';(out/'input_inventory').mkdir(parents=True)
    for name,value in [('resolved_jobs.json',jobs),('resolved_proposals.json',proposals),('inventory_audit.json',audit)]:
        _write_json(out/'input_inventory'/name,value)
    (out/'input_inventory/artifact_hashes.sha256').write_text('');_reseal_inventory(out)
    e3.run_observe('freeze',out,'09c1414f1b')
    policies=yaml.safe_load((materializer.CODE_ROOT/'configs/experiments/icra2027/agentic_policies.yaml').read_text())
    e3.run_control(policies,{'code':{'commit':'d'*40}},'freeze',out,'09c1414f1b')
    monkeypatch.setattr(screen,'evidence_root',lambda:root)
    monkeypatch.setattr(screen,'_code_snapshot',lambda _:dict(code_root=str(screen.CODE_ROOT),commit='d'*40,dirty=False))
    def no_gt(*a,**k):raise AssertionError('GT external source must not run')
    monkeypatch.setattr(screen,'external_geometry_identities',no_gt)
    return root,out,descriptor


def test_real_automatic_descriptor_keeps_nine_pairs_and_all_failed_jobs(population):
    root,out,descriptor=population
    result=screen.prepare_automatic_candidates(screen_id='automatic-nine',scene_id='09c1414f1b',
        e3_root=out,automatic_scene_descriptor=descriptor,expected_commit='d'*40)
    assert result['counts']==dict(planned_objects=5,semantic_pairs=9,region_pairs=3,receptacle_pairs=6)
    assert result['planned_pair_arm_rows']==18 and result['qualified_tasks'] is None
    for pair in result['pairs']:
        assert pair['qualified'] is None
        assert pair['construction_by_policy']['A0']['state']=='failed_build'
        assert pair['construction_by_policy']['A4']['state']=='failed_build'
        assert {r['terminal_action'] for r in pair['construction_by_policy']['A0']['failed_objects']}=={'reject'}
        assert {r['terminal_action'] for r in pair['construction_by_policy']['A4']['failed_objects']}=={'abstain'}
        assert all(v['workspace'] is None and v['manipulation_success'] is None for v in pair['construction_by_policy'].values())
    assert not result['gpu_launch_allowed'] and not result['paper_ready']
    assert result['materializations']['A0']['roster']['job_count']==5
    assert result['materializations']['A4']['roster']['job_count']==5
    with pytest.raises(FileExistsError):
        screen.prepare_automatic_candidates(screen_id='automatic-nine',scene_id='09c1414f1b',
            e3_root=out,automatic_scene_descriptor=descriptor,expected_commit='d'*40)


@pytest.mark.parametrize('mutation',['scene','hash','gt_source'])
def test_automatic_candidate_rejects_source_drift_before_export(population, mutation):
    root,out,descriptor=population
    d=json.loads(descriptor.read_text())
    if mutation=='scene':d['scene_id']='wrong'
    elif mutation=='hash':d['discovery_hashes']['output_hashes.json']='0'*64
    else:d['discovery_directory']=str(root/'scans')
    _write_json(descriptor,d)
    with pytest.raises((ValueError,FileNotFoundError)):
        screen.prepare_automatic_candidates(screen_id='bad',scene_id='09c1414f1b',
            e3_root=out,automatic_scene_descriptor=descriptor,expected_commit='d'*40,export=True)
    assert not (root/'outputs/icra2027/bad').exists()


def test_automatic_export_route_keeps_source_namespace_and_empty_accepted_union(population,monkeypatch):
    root,out,descriptor=population
    # This fixture replaces the evidence root; inherited process caches belong to the real checkout.
    for name in ('TMPDIR','XDG_CACHE_HOME'):
        cache=root/name.lower();cache.mkdir()
        monkeypatch.setenv(name,str(cache))
    screen.prepare_automatic_candidates(screen_id='auto-route',scene_id='09c1414f1b',
        e3_root=out,automatic_scene_descriptor=descriptor,expected_commit='d'*40)
    factories={p:root/f'outputs/icra2027/auto-route/automatic_candidates/09c1414f1b/materialized/{p}' for p in screen.POLICIES}
    context=screen._automatic_export_context(factories,scene_id='09c1414f1b',root=root)
    assert context['accepted_union']==[] and len(context['object_slots'])==5
    calls=[]
    monkeypatch.setattr(screen.subprocess,'run',lambda *a,**kw:calls.append((a,kw)))
    screen._run_full_room_export(factories['A0'],scene_id='09c1414f1b',root=root,
                                common_carve_factories=list(factories.values()),automatic=True)
    args,kwargs=calls[0]
    assert 'robo.sim.export_mjcf' in args[0]
    assert {key:kwargs['env'][key] for key in ('SIMANY_AUTO','SIMANY_NO_GT','SIMANY_MESH_SRC')} == {
        'SIMANY_AUTO':'1','SIMANY_NO_GT':'1','SIMANY_MESH_SRC':'derived'}
    assert args[0].count('--background-carve-factory')==2


@pytest.mark.parametrize('mutation',[None,'carve_rejected','gt_clip','wrong_mesh','empty_roster'])
def test_auto_export_validation_checks_real_descriptor_and_rejected_static_union(population, mutation):
    import shutil
    from tests.test_e4_collision_repair import _write_minimal_repaired_export
    root,out,descriptor=population
    screen.prepare_automatic_candidates(screen_id='export-check',scene_id='09c1414f1b',
        e3_root=out,automatic_scene_descriptor=descriptor,expected_commit='d'*40)
    factories={p:root/f'outputs/icra2027/export-check/automatic_candidates/09c1414f1b/materialized/{p}' for p in screen.POLICIES}
    context=screen._automatic_export_context(factories,scene_id='09c1414f1b',root=root)
    synthetic,collision=_write_minimal_repaired_export(root/'synthetic')
    destination=factories['A0']
    for name in ('sim','sim_export'):shutil.copytree(synthetic/name,destination/name)
    xml=destination/'sim_export/scene.xml';xml.write_text(xml.read_text().replace(str(synthetic),str(destination)))
    carve=collision['background_carve']
    carve.update(support_clip_source='automatic_instance_aabb_bottom_plus_5mm',
        discovered_slots=context['object_slots'],carved_slots=[],hull_count=0,
        source_mesh_sha256=context['source_mesh_sha256'])
    collision['collision_exclusion']['hull_count']=0
    if mutation=='carve_rejected':carve['carved_slots']=['obj_1000']
    if mutation=='gt_clip':carve['support_clip_source']='discovered_scan_aabb_bottom_plus_5mm'
    if mutation=='wrong_mesh':carve['source_mesh_sha256']='0'*64
    if mutation=='empty_roster':carve['discovered_slots']=[]
    _write_json(destination/'sim_export/room_collision_report.json',collision)
    kwargs=dict(factory_dir=destination,scene_id='09c1414f1b',policy='A0',root=root,
        expected_object_slots=[],expected_discovered_slots=context['object_slots'],automatic_factories=factories)
    if mutation is None:
        result=screen.validate_full_room_export(**kwargs)
        assert result['background_carve']['hull_count']==0
    else:
        with pytest.raises(screen.CandidateScreenError):screen.validate_full_room_export(**kwargs)


def test_declared_denominator_survives_failure_and_only_missing_arm_resumes(population,monkeypatch):
    root,out,descriptor=population
    original=materializer.materialize_factory_variant
    calls=[]
    def interrupted(**kwargs):
        calls.append(kwargs['policy_id'])
        if kwargs['policy_id']=='A4':raise RuntimeError('injected interruption')
        return original(**kwargs)
    monkeypatch.setattr(materializer,'materialize_factory_variant',interrupted)
    kwargs=dict(screen_id='partial',scene_id='09c1414f1b',e3_root=out,
                automatic_scene_descriptor=descriptor,expected_commit='d'*40)
    with pytest.raises(RuntimeError,match='interruption'):screen.prepare_automatic_candidates(**kwargs)
    directory=root/'outputs/icra2027/partial/automatic_candidates/09c1414f1b'
    assert len(json.loads((directory/'declaration/population.json').read_text())['pairs'])==9
    old=(directory/'materialized/A0/materialization_manifest.json').read_bytes()
    calls.clear()
    def resumed(**kwargs):calls.append(kwargs['policy_id']);return original(**kwargs)
    monkeypatch.setattr(materializer,'materialize_factory_variant',resumed)
    result=screen.prepare_automatic_candidates(**kwargs)
    assert calls==['A4'] and result['planned_pair_arm_rows']==18
    assert (directory/'materialized/A0/materialization_manifest.json').read_bytes()==old
