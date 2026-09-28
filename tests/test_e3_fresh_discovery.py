import copy
import json
import os
from pathlib import Path
import numpy as np
import pytest
import yaml
from run.icra2027 import e3_auto_discovery_pilot as auto
from run.icra2027 import e3_trellis_generation_pilot as generation


def real_inputs():
    config=yaml.safe_load((auto.CODE/'configs/experiments/icra2027/e3_fresh_discovery.yaml').read_text())
    source=Path(os.environ.get('SIMANY_EVIDENCE_ROOT','/nonexistent'))/'outputs/icra2027/20260904-07e8b05-v21/auto_discovery_pilot/input_manifest.json'
    if not source.exists(): pytest.skip('local source unavailable')
    manifest=json.loads(source.read_text())
    frozen_root = Path(config['gaussian_provenance']['freeze_root'])
    if not frozen_root.is_dir():
        pytest.skip('historical provenance integration requires its original frozen root')
    return config,manifest


def test_fresh_gaussian_exact_source_lineage_and_legacy_nuance():
    c,m=real_inputs()
    proof=auto.validate_gaussian_provenance(c,m['boundary'],m['input_images'],m['metadata'])
    assert proof['status']=='FRESH_OFFICIAL_TRAIN_ONLY' and proof['training_iterations']==15000
    assert len(proof['training_frames'])==48 and proof['independent_heldout_evaluation'] is False
    assert auto.validate_gaussian_provenance({},None,None,None)['status']=='UNKNOWN'


@pytest.mark.parametrize('mutation',['receipt','code','rgb','roster','pose','status'])
def test_gaussian_provenance_drift_fails_before_gpu(mutation):
    c,m=real_inputs();c=copy.deepcopy(c);m=copy.deepcopy(m)
    if mutation=='receipt':c['gaussian_provenance']['anchors']['gaussian_train_only/train-pilot_receipt.json']['sha256']='0'*64
    elif mutation=='code':c['gaussian_provenance']['producer_commit']='0'*40
    elif mutation=='rgb':m['input_images'][m['boundary']['training_frames'][0]]['sha256']='0'*64
    elif mutation=='roster':m['boundary']['training_frames'].pop()
    elif mutation=='pose':m['metadata']['colmap/images.txt']['sha256']='0'*64
    else:c['source_gaussian_training_provenance']='UNKNOWN'
    with pytest.raises((auto.PilotError,ValueError)):
        auto.validate_gaussian_provenance(c,m['boundary'],m['input_images'],m['metadata'])


def make_discovery(root,count):
    construction=root/'construction';objects=construction/'objects';objects.mkdir(parents=True)
    labels=['mug']*count
    metadata=[]
    if count:
        metadata=[{'gt_object_id':1000,'index':0,'frame':'train.jpg'}]
        crop=objects/'obj_00/rgba.png';crop.parent.mkdir();crop.write_bytes(b'crop')
    (objects/'objects.json').write_text(json.dumps(metadata))
    np.savez(construction/'auto_instances.npz',labels=np.asarray(labels,dtype='U3'))
    stages=['render','fuse','discover','prepare','refine']
    manifest={'boundary':{'training_frames':['train.jpg']},'planned_stages':stages,
              'source_gaussian_training_provenance':'FRESH_OFFICIAL_TRAIN_ONLY'}
    (root/'input_manifest.json').write_text(json.dumps(manifest))
    hashes={str(p.relative_to(construction)):auto.identity(p) for p in construction.rglob('*') if p.is_file()}
    (root/'output_hashes.json').write_text(json.dumps(hashes))
    rows=auto.summarize_instances(labels,metadata)
    summary={'paper_ready':False,'scene_id':'scene','freeze_id':'freeze','code_commit':'source',
      'discovered_instances':count,'prepared_instances':len(metadata),'rows':rows,
      'source_gaussian_training_provenance':'FRESH_OFFICIAL_TRAIN_ONLY',
      'stages':[{'stage':s,'exit_code':0} for s in stages]}
    (root/'pilot_summary.json').write_text(json.dumps(summary))
    all_jobs={'kind':'complete_automatic_discovery_jobs','scene_id':'scene','freeze_id':'freeze',
       'schema_version':1,'paper_ready':False,'source_gaussian_training_provenance':'FRESH_OFFICIAL_TRAIN_ONLY',
       'code_commit':'source','rows':rows,'planned_jobs':count,
       'input_manifest_sha256':auto.sha(root/'input_manifest.json'),
       'summary_sha256':auto.sha(root/'pilot_summary.json'),
       'output_hashes_sha256':auto.sha(root/'output_hashes.json')}
    (root/'all_jobs_manifest.json').write_text(json.dumps(all_jobs))
    audit={k:all_jobs[k] for k in ('summary_sha256','output_hashes_sha256')}
    audit['all_jobs_manifest_sha256']=auto.sha(root/'all_jobs_manifest.json')
    (root/'postrun_audit.json').write_text(json.dumps(audit))


@pytest.mark.parametrize('count',[0,3,7])
def test_dynamic_all_jobs_denominator_keeps_preparation_failures(tmp_path,count):
    make_discovery(tmp_path,count)
    rows=generation.source_jobs(tmp_path)
    assert len(rows)==count
    assert sum(r['prepared'] for r in rows)==min(1,count)
    assert [r['automatic_instance_id'] for r in rows]==list(range(1000,1000+count))


def test_dropped_job_resealed_summary_still_fails_against_actual_instances(tmp_path):
    make_discovery(tmp_path,7)
    path=tmp_path/'pilot_summary.json';r=json.loads(path.read_text());r['rows'].pop();r['discovered_instances']-=1;path.write_text(json.dumps(r))
    audit=tmp_path/'postrun_audit.json';a=json.loads(audit.read_text());a['summary_sha256']=auto.sha(path);audit.write_text(json.dumps(a))
    with pytest.raises(auto.PilotError,match='denominator'):
        generation.source_jobs(tmp_path)


def test_crop_comparison_matches_bytes_across_ids_and_preserves_missing(tmp_path,monkeypatch):
    previous=tmp_path/'previous';previous.mkdir()
    reference={'directory':str(previous),'anchors':{}}
    from agents.recon.colmap_poses import file_identity
    for name in ['pilot_summary.json','output_hashes.json','input_manifest.json','postrun_audit.json']:
        (previous/name).write_text('{}');reference['anchors'][name]=file_identity(previous/name)
    destination=tmp_path/'new/auto_discovery_pilot';destination.mkdir(parents=True)
    (destination/'pilot_summary.json').write_text(json.dumps({'code_commit':'source','freeze_id':'new'}))
    (destination/'all_jobs_manifest.json').write_text('{}')
    old=[{'job_id':'old:auto:1000','prepared':True,'input':{'sha256':'same'}}]
    new=[{'job_id':'new:auto:1009','prepared':True,'input':{'sha256':'same'}},
         {'job_id':'old:auto:1000','prepared':True,'input':{'sha256':'different'}},
         {'job_id':'new:auto:1011','prepared':False}]
    monkeypatch.setattr(auto,'load_context',lambda *_a:({'crop_comparison_reference':reference},'source'))
    monkeypatch.setattr(generation,'source_jobs',lambda path:copy.deepcopy(old if path==previous else new))
    auto.compare_crops(None,tmp_path/'new')
    report=json.loads((destination/'crop_hash_comparison.json').read_text())
    assert report['new_planned_jobs']==3 and report['exact_match_prepared_inputs']==1
    assert report['rows'][0]['exact_matching_prior_jobs']==['old:auto:1000']
    assert report['rows'][1]['exact_rgba_match'] is False and report['rows'][2]['exact_rgba_match'] is None
    assert report['generation_performed'] is False and report['artifact_reuse_performed'] is False
