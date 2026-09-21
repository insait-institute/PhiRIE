import copy
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def pilot():
    spec = importlib.util.spec_from_file_location('e3_auto_pilot', Path(__file__).resolve().parents[1] / 'run/icra2027/e3_auto_discovery_pilot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_first_scene_and_bounded_engineering_scope(pilot):
    config = {'paper_ready': False, 'scope': 'gt_isolated_engineering_discovery_crops',
              'scene_id': 'first', 'max_train_frames': 48, 'discovery_stride': 12,
              'render_stride': 3, 'generate_assets': False}
    roster = {'population': {'scene_ids': ['first', 'second']}}
    pilot.validate_config(config, roster)
    for key, value in [('scene_id','second'), ('max_train_frames',24),
                       ('paper_ready',True), ('generate_assets',True)]:
        changed = dict(config, **{key:value})
        with pytest.raises(pilot.PilotError):
            pilot.validate_config(changed, roster)


def test_all_discovered_instances_remain_in_denominator(pilot):
    rows = pilot.summarize_instances(['mug','bottle','book'], [
        {'gt_object_id':1001,'index':0,'mask_source':'sam3 iou=0.7'}])
    assert len(rows)==3 and sum(r['prepared'] for r in rows)==1
    assert rows[0]['failure_reason']=='filtered_by_existing_preparation_gates'
    assert rows[1]['automatic_instance_id']==1001
    assert rows[2]['mask_source'] is None
    with pytest.raises(pilot.PilotError, match='denominator'):
        pilot.summarize_instances(['mug'], [{'gt_object_id':2000,'index':0}])


def test_scan_gt_and_unselected_rgb_reads_fail_before_open(pilot, tmp_path):
    scene = tmp_path/'scene'
    images = scene/'dslr/resized_undistorted_images'
    options = dict(forbidden_scene=scene, image_root=images, allowed_images={'train.jpg'})
    pilot.enforce_read_boundary('open', (str(images/'train.jpg'),'r',0), **options)
    for path in [images/'heldout.jpg', scene/'scans/mesh_aligned_0.05.ply', scene/'scans/segments_anno.json']:
        with pytest.raises(pilot.PilotError):
            pilot.enforce_read_boundary('open', (str(path),'r',0), **options)


def test_output_records_never_overwrite(pilot,tmp_path):
    path=tmp_path/'record.json'
    pilot.write_new(path, {'first':True})
    with pytest.raises(FileExistsError):pilot.write_new(path, {'first':False})
    assert json.loads(path.read_text())=={'first':True}


def test_plan_stages_only_fixed_training_rgb_and_no_scan_gt(pilot, tmp_path, monkeypatch):
    from robo.eval.fidelity_replacements import _tree_inventory
    scene=tmp_path/'dataset/data/first'
    dslr=scene/'dslr'
    images=dslr/'resized_undistorted_images'
    images.mkdir(parents=True)
    train=[f'{i:03d}.jpg' for i in range(60)]
    test=['test.jpg']
    for name in train+test:(images/name).write_bytes(name.encode())
    (scene/'scans').mkdir()
    (scene/'scans/segments_anno.json').write_text('DO NOT READ')
    (dslr/'train_test_lists.json').write_text(json.dumps({'train':train,'test':test}))
    (dslr/'colmap').mkdir()
    (dslr/'colmap/images.txt').write_text(''.join(
        f'{i+1} 1 0 0 0 0 0 0 1 {name}\n0 0 -1\n' for i,name in enumerate(train+test)))
    (dslr/'nerfstudio').mkdir()
    (dslr/'nerfstudio/transforms_undistorted.json').write_text('{}')
    source=tmp_path/'sam3';source.mkdir();(source/'source.py').write_text('pass\n')
    ckpt=tmp_path/'checkpoint';ckpt.write_bytes(b'weights')
    gauss=tmp_path/'gaussian';gauss.write_bytes(b'gaussian')
    config={'scope':'gt_isolated_engineering_discovery_crops','dataset_root':str(tmp_path/'dataset'),
            'scene_id':'first','max_train_frames':48,'gaussian':str(gauss),'gaussian_sha256':pilot.sha(gauss),
            'sam3_checkpoint':{'path':str(ckpt),'sha256':pilot.sha(ckpt),'bytes':7},
            'sam3_source':{'path':str(source),'tree_sha256':_tree_inventory(source,label='test')['tree_sha256']}}
    cfg=tmp_path/'config.yaml';cfg.write_text('fixture')
    freeze=tmp_path/'freeze';freeze.mkdir()
    monkeypatch.setattr(pilot,'load_context',lambda *args:(config,'code'))
    pilot.plan(cfg,freeze)
    staged=freeze/'auto_discovery_pilot/inputs/data/first'
    assert not (staged/'scans').exists()
    assert sorted(p.name for p in (staged/'dslr/resized_undistorted_images').iterdir())==train[:48]
    manifest=json.loads((freeze/'auto_discovery_pilot/input_manifest.json').read_text())
    assert manifest['source_gaussian_training_provenance']=='UNKNOWN'
    assert not manifest['paper_ready'] and len(manifest['input_images'])==48
    destination = freeze/'auto_discovery_pilot'
    pilot.validate_staged_inputs(destination, config, manifest)
    for name in ['train_test_lists.json', 'colmap/images.txt', 'nerfstudio/transforms_undistorted.json']:
        target = staged/'dslr'/name
        original = target.read_bytes()
        target.write_bytes(b'tampered')
        with pytest.raises(pilot.PilotError, match='staged metadata differs'):
            pilot.validate_staged_inputs(destination, config, manifest)
        target.write_bytes(original)
    link = staged/'dslr/resized_undistorted_images'/train[0]
    link.unlink()
    link.symlink_to(images/test[0])
    with pytest.raises(pilot.PilotError, match='image target differs'):
        pilot.validate_staged_inputs(destination, config, manifest)
    with pytest.raises(FileExistsError):pilot.plan(cfg,freeze)


def test_a6000_usable_memory_and_free_budget(pilot):
    pilot.validate_gpu_memory({'total_bytes':46068*1024**2, 'free_bytes':30*1024**3})
    for total, free in [(46068*1024**2,30*1024**3-1), (44*1024**3-1,40*1024**3)]:
        with pytest.raises(pilot.PilotError, match='memory budget'):
            pilot.validate_gpu_memory({'total_bytes':total, 'free_bytes':free})
