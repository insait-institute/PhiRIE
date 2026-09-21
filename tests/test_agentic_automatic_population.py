"""Canonical multi-scene inventories retain empty discoveries and missing jobs."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from robo.eval import agentic_ablation as e3
from run.icra2027.e3_auto_discovery_pilot import summarize_instances


def _dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return e3.sha256_file(path)


def _source(root, scene, count):
    source=root/scene;construction=source/'construction';construction.mkdir(parents=True)
    labels=['unprepared']*count
    np.savez(construction/'auto_instances.npz', labels=np.asarray(labels,dtype='U16'))
    _dump(construction/'objects/objects.json', [])
    identities={role:{'path':f'/never-open/{scene}/{role}','bytes':1,'sha256':ch*64}
                for role,ch in zip(('gaussian','intrinsics','poses'),'abc')}
    hashes={}
    hashes['input_manifest.json']=_dump(source/'input_manifest.json',{
        'scene_id':scene,'source_gaussian_training_provenance':'FRESH_OFFICIAL_TRAIN_ONLY',
        'boundary':{'training_frames':['train.jpg']},'planned_stages':['discover','prepare'],
        'gaussian':identities['gaussian'],'metadata':{'colmap/images.txt':identities['poses'],
            'nerfstudio/transforms_undistorted.json':identities['intrinsics']}})
    rows=summarize_instances(labels,[])
    hashes['pilot_summary.json']=_dump(source/'pilot_summary.json',{
        'scene_id':scene,'freeze_id':'discovery','code_commit':'d'*40,'paper_ready':False,
        'source_gaussian_training_provenance':'FRESH_OFFICIAL_TRAIN_ONLY',
        'rows':rows,'discovered_instances':count,'prepared_instances':0,
        'stages':[{'stage':s,'exit_code':0} for s in ('discover','prepare')]})
    hashes['output_hashes.json']=_dump(source/'output_hashes.json',{
        str(p.relative_to(construction)):{'sha256':e3.sha256_file(p)}
        for p in construction.rglob('*') if p.is_file()})
    hashes['all_jobs_manifest.json']=_dump(source/'all_jobs_manifest.json',{
        'schema_version':1,'kind':'complete_automatic_discovery_jobs','scene_id':scene,
        'freeze_id':'discovery','code_commit':'d'*40,'paper_ready':False,'planned_jobs':count,
        'rows':rows,'source_gaussian_training_provenance':'FRESH_OFFICIAL_TRAIN_ONLY',
        **{('summary_sha256' if name=='pilot_summary.json' else name.replace('.json','_sha256')):value for name,value in hashes.items()}})
    hashes['postrun_audit.json']=_dump(source/'postrun_audit.json',{
        ('summary_sha256' if name=='pilot_summary.json' else name.replace('.json','_sha256')):value for name,value in hashes.items()})
    return {'scene_id':scene,'planned_jobs':count,'discovery_directory':str(source),
        'discovery_hashes':hashes,'scene_sources':{
            k:{'path':v['path'],'size_bytes':v['bytes'],'sha256':v['sha256']} for k,v in identities.items()},
        'initial_pools':{tool:{'status':'not_run','reason_code':'initial_tool_not_run'}
                         for tool in ('trellis','reconviagen')}}


@pytest.fixture
def population(tmp_path):
    scenes=['09c1414f1b','825d228aec']
    roster=tmp_path/'roster.yaml';roster.write_text(yaml.safe_dump({'population':{'scene_ids':scenes}}))
    jobs={'study_scope':'automatic_training_only_engineering','paper_ready':False,
        'automatic_sources':[_source(tmp_path,scenes[0],3),_source(tmp_path,scenes[1],0)],
        'population':{'scene_roster_config':str(roster),'scene_roster_config_file_sha256':e3.sha256_file(roster),
            'scene_roster_sha256':hashlib.sha256(('\n'.join(scenes)+'\n').encode()).hexdigest(),
            'planned_scenes':2,'planned_jobs_per_policy':3,'planned_policy_object_rows':15}}
    return jobs,{'freeze_id':'freeze','contract_sha256':'e'*64,'code':{'commit':'f'*40}}


def test_complete_population_reaches_canonical_observe_and_control_including_empty_scene(population,tmp_path):
    jobs,contract=population;out=tmp_path/'agentic'
    audit=e3.run_inventory(jobs,contract,'freeze',out)
    resolved,proposals=e3._load_inventory(out,controller_safe=True)
    assert resolved['counts']=={'scenes':2,'jobs':3,'policy_object_rows':15}
    assert len(proposals['proposals'])==6 and len(audit['scene_audits'])==2
    assert resolved['source_contract']['jobs_sha256']==e3._canonical_digest(jobs)
    policies=yaml.safe_load((e3.REPOSITORY_ROOT/'configs/experiments/icra2027/agentic_automatic_policies.yaml').read_text())
    for scene,count in zip(audit['scene_roster'],[3,0]):
        observed=e3.run_observe('freeze',out,scene)
        assert observed['renderer_runtime'] is None
        controlled=e3.run_control(policies,contract,'freeze',out,scene)
        assert controlled['job_count']==count and controlled['ledger_row_count']==5*count
    assert audit['paper_ready'] is False and audit['evaluation_references']==[]


@pytest.mark.parametrize('damage',['drop_empty_scene','duplicate_scene','reorder','job_count','row_count','roster_bytes','lost_hash'])
def test_population_drift_rejected_before_inventory_publication(population,tmp_path,damage):
    jobs,contract=copy.deepcopy(population)
    if damage=='drop_empty_scene':jobs['automatic_sources'].pop()
    elif damage=='duplicate_scene':jobs['automatic_sources'][1]=copy.deepcopy(jobs['automatic_sources'][0])
    elif damage=='reorder':jobs['automatic_sources'].reverse()
    elif damage=='job_count':jobs['population']['planned_jobs_per_policy']=2
    elif damage=='row_count':jobs['population']['planned_policy_object_rows']=14
    elif damage=='roster_bytes':Path(jobs['population']['scene_roster_config']).write_text('changed')
    else:jobs['population'].pop('scene_roster_config_file_sha256')
    with pytest.raises(ValueError):
        e3.run_inventory(jobs,contract,'freeze',tmp_path/'agentic')
    assert not (tmp_path/'agentic/input_inventory').exists()
