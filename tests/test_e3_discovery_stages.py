"""Stage resource separation keeps the original actions and validates dependency bytes."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from run.icra2027 import e3_auto_discovery_pilot as auto


@pytest.fixture
def staged(tmp_path, monkeypatch):
    c=yaml.safe_load((auto.CODE/'configs/experiments/icra2027/e3_discovery_cohort.yaml').read_text())
    c['scene_id']='09c1414f1b'
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(c))
    destination=auto.destination_for(c,tmp_path);destination.mkdir(parents=True)
    manifest={'code_commit':'source','config_sha256':auto.sha(config),'sam3_source':{'tree_sha256':'source-tree'},
        'boundary':{},'input_images':{},'metadata':{},'gaussian_provenance':{'status':'PASS'},
        'planned_stages':['render','fuse','discover','prepare','refine']}
    auto.write_new(destination/'input_manifest.json',manifest)
    monkeypatch.setattr(auto,'load_context',lambda *a:(c,'source'))
    monkeypatch.setattr(auto,'checked_manifest',lambda *a:manifest)
    monkeypatch.setattr(auto,'bind_cohort_gaussian',lambda c:c)
    monkeypatch.setattr(auto,'validate_cohort_runtime',lambda *a:None)
    monkeypatch.setattr(auto,'validate_input_stats',lambda *a:None)
    monkeypatch.setattr(auto,'validate_staged_inputs',lambda *a:None)
    monkeypatch.setattr(auto,'validate_gaussian_provenance',lambda *a:{'status':'PASS'})
    monkeypatch.setattr(auto,'validate_gpu_memory',lambda *a:None)
    from run.icra2027 import e3_gaussian_train_only
    from robo.eval import fidelity_replacements
    monkeypatch.setattr(e3_gaussian_train_only,'require_published_full_source',lambda *a:None)
    monkeypatch.setattr(fidelity_replacements,'_tree_inventory',lambda *a,**kw:{'tree_sha256':'source-tree'})
    monkeypatch.setenv('SLURMD_NODENAME','sof1-test')
    monkeypatch.setenv('SLURM_JOB_ID','123')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    probes=[];calls=[]
    def probe(*a,**kw):
        probes.append(a);return '{}'
    def call(command,**kw):
        stage=command[command.index('--stage')+1];calls.append(stage)
        output=destination/'construction'
        if stage=='render':
            (output/'mesh_derive').mkdir()
            auto.write_new(output/'mesh_derive/training_views.json',{'frames':['a']})
            np.savez(output/'mesh_derive/view_0000.npz',fname='a')
        elif stage=='fuse': (output/'derived_mesh.ply').write_bytes(b'mesh')
        elif stage=='discover': np.savez(output/'auto_instances.npz',labels=['book'])
        elif stage=='prepare':
            (output/'objects').mkdir()
            auto.write_new(output/'objects/objects.json',[{'gt_object_id':1000,'index':0}])
        return 0
    monkeypatch.setattr(auto.subprocess,'check_output',probe)
    monkeypatch.setattr(auto.subprocess,'call',call)
    return config,tmp_path,destination,probes,calls


def run(staged,phase):
    config,root,*_=staged
    return auto.execute(config,root,'09c1414f1b',phase=phase)


def test_three_phases_use_original_actions_and_cpu_fusion_has_no_gpu_probe(staged):
    config,root,destination,probes,calls=staged
    run(staged,'run-render');assert len(probes)==1
    run(staged,'run-fuse');assert len(probes)==1
    assert json.loads((destination/'run-fuse_runtime.json').read_text())['gpu_reserved'] is False
    run(staged,'run-objects');assert len(probes)==2
    assert calls==['render','fuse','discover','prepare','refine']
    summary=json.loads((destination/'pilot_summary.json').read_text())
    assert [r['stage'] for r in summary['stages']]==calls
    assert summary['discovered_instances']==summary['prepared_instances']==1
    assert (destination/'all_jobs_manifest.json').is_file()
    assert summary['paper_ready'] is False


def test_fusion_refuses_missing_render_before_any_work(staged):
    with pytest.raises(FileNotFoundError):run(staged,'run-fuse')
    assert staged[-1]==[] and staged[-2]==[]


def test_completed_phase_refuses_overwrite(staged):
    run(staged,'run-render')
    with pytest.raises(FileExistsError):run(staged,'run-render')
    assert staged[-1]==['render']


@pytest.mark.parametrize('change',['source','config','input','failed','missing_views','missing_manifest','artifact','escape','predecessor','stage'])
def test_fusion_rejects_changed_or_incomplete_predecessor(staged,change):
    run(staged,'run-render');config,root,destination,*_=staged
    path=destination/'run-render_seal.json';seal=json.loads(path.read_text())
    if change=='source':seal['code_commit']='other'
    elif change=='config':seal['config_sha256']='other'
    elif change=='input':seal['input_manifest_sha256']='other'
    elif change=='failed':seal['stages'][0]['exit_code']=1
    elif change=='missing_views':seal['artifacts']={k:v for k,v in seal['artifacts'].items() if not k.endswith('.npz')}
    elif change=='missing_manifest':seal['artifacts'].pop('mesh_derive/training_views.json')
    elif change=='artifact':(destination/'construction/mesh_derive/view_0000.npz').write_bytes(b'tampered')
    elif change=='escape':seal['artifacts']['../outside']={}
    elif change=='predecessor':seal['predecessors']['unexpected']='digest'
    elif change=='stage':seal['stages'][0]['stage']='fuse'
    path.write_text(json.dumps(seal))
    with pytest.raises(ValueError):run(staged,'run-fuse')
    assert staged[-1]==['render']


def test_cpu_fusion_rejects_exposed_gpu(staged,monkeypatch):
    run(staged,'run-render');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError,match='must not reserve'):run(staged,'run-fuse')
    assert staged[-1]==['render']


def test_failed_fusion_is_not_retried_or_sealed(staged,monkeypatch):
    run(staged,'run-render');monkeypatch.setattr(auto.subprocess,'call',lambda *a,**kw:-9)
    with pytest.raises(ValueError,match='fuse failed'):run(staged,'run-fuse')
    assert not (staged[2]/'run-fuse_seal.json').exists()
    assert json.loads((staged[2]/'failure.json').read_text())['failed_stage']=='fuse'
    with pytest.raises(FileExistsError):run(staged,'run-fuse')


def test_added_render_file_rejected(staged):
    run(staged,'run-render')
    (staged[2]/'construction/mesh_derive/view_9999.npz').write_bytes(b'unplanned')
    with pytest.raises(ValueError,match='roster changed'):run(staged,'run-fuse')
    assert staged[-1]==['render']
