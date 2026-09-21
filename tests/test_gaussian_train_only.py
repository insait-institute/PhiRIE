import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agents.recon import colmap_poses as producer


@pytest.fixture
def source(tmp_path):
    dslr=tmp_path/'source/dslr'
    (dslr/'colmap').mkdir(parents=True)
    (dslr/'nerfstudio').mkdir()
    (dslr/'resized_undistorted_images').mkdir()
    names=['a.png','b.png','c.png']
    (dslr/'train_test_lists.json').write_text(json.dumps({'train':names,'test':['test.png']}))
    meta={'fl_x':300.,'fl_y':300.,'cx':256.,'cy':192.,'w':512,'h':384}
    (dslr/'nerfstudio/transforms_undistorted.json').write_text(json.dumps(meta))
    lines=[]
    rng=np.random.default_rng(2)
    image=np.asarray(Image.fromarray(rng.integers(0,256,(384,512,3),dtype=np.uint8)).resize((256,192)).resize((512,384)))
    for i,name in enumerate(names):
        Image.fromarray(np.roll(image,-10*i,axis=1)).save(dslr/'resized_undistorted_images'/name)
        lines.extend([f'{i+1} 1 0 0 0 {-0.1*i} 0 0 1 {name}\n','INHERITED OBSERVATIONS MUST BE DISCARDED\n'])
    lines.extend(['4 1 0 0 0 0 0 0 1 test.png\n','FORBIDDEN LEGACY TRACKS\n'])
    (dslr/'colmap/images.txt').write_text(''.join(lines))
    # Deliberately unreadable/invalid geometry must not affect the producer.
    (dslr/'colmap/points3D.bin').write_bytes(b'NEVER READ')
    return tmp_path/'source'


def test_staging_reads_only_declared_rgb_and_drops_legacy_tracks(source,tmp_path):
    manifest=producer.prepare_training_scene(source,tmp_path/'stage',max_train_frames=3)
    assert [r['name'] for r in manifest['frames']]==['a.png','b.png','c.png']
    assert manifest['inherited_points_or_tracks_used'] is False
    assert not (tmp_path/'stage/dslr/colmap').exists()
    producer.validate_training_scene(tmp_path/'stage')
    with pytest.raises(FileExistsError):
        producer.prepare_training_scene(source,tmp_path/'stage',max_train_frames=3)


@pytest.mark.parametrize('mutation',['extra_rgb','changed_rgb','split','pose'])
def test_staged_boundary_drift_rejected(source,tmp_path,mutation):
    stage=tmp_path/'stage';producer.prepare_training_scene(source,stage,max_train_frames=3)
    if mutation=='extra_rgb': (stage/'dslr/resized_undistorted_images/test.png').write_bytes(b'heldout')
    if mutation=='changed_rgb': (stage/'dslr/resized_undistorted_images/a.png').write_bytes(b'changed')
    if mutation=='split': (source/'dslr/train_test_lists.json').write_text('{}')
    if mutation=='pose':
        p=stage/'training_inputs.json';r=json.loads(p.read_text());r['frames'][0]['w2c'][0][3]+=1;p.write_text(json.dumps(r))
    with pytest.raises(ValueError): producer.validate_training_scene(stage)


def test_unproven_init_rejected_before_training(tmp_path):
    p=tmp_path/'init_manifest.json';p.write_text(json.dumps({'kind':'inherited_gaussian'}))
    with pytest.raises(ValueError,match='fresh'):
        producer.validate_training_initialization(tmp_path,tmp_path/'unknown.ply',p)


def test_real_pycolmap_cpu_triangulation_and_track_lineage(source,tmp_path):
    pytest.importorskip('pycolmap')
    stage=tmp_path/'stage';producer.prepare_training_scene(source,stage,max_train_frames=3)
    init=tmp_path/'init'
    r=producer.triangulate_training_scene(stage,init,seed=42,num_threads=2)
    assert r['initial_point_count']==0 and r['n_points']>=4
    assert r['frame_names']==['a.png','b.png','c.png']
    producer.validate_training_initialization(stage,init/'init_points.ply',init/'init_manifest.json')
    with pytest.raises(FileExistsError): producer.triangulate_training_scene(stage,init)
    (init/'init_points.ply').write_bytes(b'changed')
    with pytest.raises(ValueError,match='PLY'):
        producer.validate_training_initialization(stage,init/'init_points.ply',init/'init_manifest.json')


def test_coordinated_pose_manifest_rewrite_rejected_by_prepare_receipt(source,tmp_path,monkeypatch):
    from run.icra2027 import e3_gaussian_train_only as launcher
    from robo.manifest.hash import canonical_hash
    destination=tmp_path/'freeze/gaussian_train_only';destination.mkdir(parents=True)
    producer.prepare_training_scene(source,destination/'scene',max_train_frames=3)
    config=tmp_path/'config.yaml';config.write_text('test: true\n')
    producer.write_new_json(destination/'prepare_receipt.json',{'status':'PASS','code':{},
        'config':producer.file_identity(config),
        'input_manifest':producer.file_identity(destination/'scene/training_inputs.json')})
    p=destination/'scene/training_inputs.json';r=json.loads(p.read_text())
    r['frames'][0]['w2c'][0][3]+=1
    r['pose_values_sha256']=canonical_hash({f['name']:f['w2c'] for f in r['frames']})
    p.write_text(json.dumps(r))
    # Self-consistency alone passes; the persisted preparation identity must not.
    producer.validate_training_scene(destination/'scene')
    monkeypatch.setattr(launcher,'context',lambda *_args:({'seed':42,'cpu_threads':2},{}))
    with pytest.raises(ValueError,match='manifest changed'):
        launcher.run(config,tmp_path/'freeze','triangulate')
    assert not (destination/'initialization').exists()
    assert json.loads((destination/'triangulate_receipt.json').read_text())['status']=='FAIL'


def test_training_environment_excludes_user_site_and_inherited_paths(monkeypatch):
    from run.icra2027.e3_gaussian_train_only import training_environment
    monkeypatch.setenv('PYTHONPATH','/untrusted/user/packages')
    monkeypatch.setenv('PYTHONHOME','/wrong/interpreter')
    env=training_environment({'training_dependency_root':'/scoped/deps','scene_id':'scene',
                              'torch_extensions_root':'/scoped/extensions'})
    assert '/untrusted' not in env['PYTHONPATH'] and env['PYTHONNOUSERSITE']=='1'
    assert 'PYTHONHOME' not in env and env['PYTHONHASHSEED']=='42'


def test_initialization_manifest_rewrite_rejected_by_triangulation_receipt(source,tmp_path,monkeypatch):
    from run.icra2027 import e3_gaussian_train_only as launcher
    destination=tmp_path/'freeze/gaussian_train_only';destination.mkdir(parents=True)
    producer.prepare_training_scene(source,destination/'scene',max_train_frames=3)
    config=tmp_path/'config.yaml';config.write_text('test: true\n')
    receipt={'status':'PASS','code':{},'config':producer.file_identity(config)}
    producer.write_new_json(destination/'prepare_receipt.json',{**receipt,
        'input_manifest':producer.file_identity(destination/'scene/training_inputs.json')})
    init=destination/'initialization';init.mkdir()
    producer.write_new_json(init/'init_manifest.json',{'kind':'before'})
    producer.write_new_json(destination/'triangulate_receipt.json',{**receipt,
        'init_manifest':producer.file_identity(init/'init_manifest.json')})
    (init/'init_manifest.json').write_text('{"kind":"coordinated rewritten manifest"}')
    monkeypatch.setattr(launcher,'context',lambda *_args:({},{}))
    with pytest.raises(ValueError,match='manifest differs from producer receipt'):
        launcher.run(config,tmp_path/'freeze','train-smoke')
    assert not (destination/'train-smoke').exists()
    assert json.loads((destination/'train-smoke_receipt.json').read_text())['status']=='FAIL'


def test_cohort_pins_all_fifty_ids_including_numeric_yaml_ids():
    import yaml
    from run.icra2027 import e3_gaussian_train_only as launcher
    config=yaml.safe_load((launcher.CODE/'configs/experiments/icra2027/e3_gaussian_cohort.yaml').read_text())
    scenes=launcher.cohort_scene_ids(config)
    assert len(scenes)==50 and scenes[0]=='09c1414f1b'
    assert '3864514494' in scenes and '5942004064' in scenes


@pytest.fixture
def cohort_context(tmp_path,monkeypatch):
    import yaml
    from run.icra2027 import e3_gaussian_train_only as launcher
    from robo.manifest.hash import canonical_hash
    config=yaml.safe_load((launcher.CODE/'configs/experiments/icra2027/e3_gaussian_cohort.yaml').read_text())
    for key in ('construction_roster','fidelity_roster'):
        path=tmp_path/config[key]['path'];path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((launcher.CODE/config[key]['path']).read_bytes())
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config))
    root=tmp_path/'freeze';(root/'contract').mkdir(parents=True)
    contract={'freeze_id':'freeze','code':{'commit':'a'*40,'dirty':False},
              'resource_inventory':[{'id':'e3_gaussian_config','sha256':producer.file_identity(path)['sha256']}]}
    contract['contract_sha256']=canonical_hash(contract)
    (root/'contract/freeze_manifest.json').write_text(json.dumps(contract))
    monkeypatch.setattr(launcher,'CODE',tmp_path)
    monkeypatch.setattr(launcher,'git_snapshot',lambda _:contract['code'])
    return launcher,path,root,config,contract


def test_cohort_context_selects_explicit_unit_without_changing_shared_config(cohort_context):
    launcher,path,root,config,contract=cohort_context
    before=path.read_bytes()
    a,_=launcher.context(path,root,'09c1414f1b')
    b,_=launcher.context(path,root,'0d2ee665be')
    assert path.read_bytes()==before
    assert launcher.gaussian_destination(root,a)!=launcher.gaussian_destination(root,b)
    assert launcher.gaussian_destination(root,a)==root/'gaussian_train_only/09c1414f1b'


@pytest.mark.parametrize('damage',['missing_scene','foreign_scene','changed_roster','lost_scene','different_rosters','duplicate_scene'])
def test_cohort_rejects_population_drift(cohort_context,damage):
    import yaml
    launcher,path,root,config,contract=cohort_context
    scene='09c1414f1b'
    if damage=='missing_scene':scene=None
    elif damage=='foreign_scene':scene='0000000000'
    else:
        roster=launcher.CODE/config['construction_roster']['path']
        data=yaml.safe_load(roster.read_text())
        if damage=='lost_scene':data['population']['scene_ids'].pop()
        elif damage=='different_rosters':data['population']['scene_ids'].reverse()
        elif damage=='duplicate_scene':data['population']['scene_ids'][1]=data['population']['scene_ids'][0]
        else:data['population']['split']='changed'
        roster.write_text(yaml.safe_dump(data))
        if damage!='changed_roster':
            config['construction_roster']['sha256']=producer.file_identity(roster)['sha256']
            path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):launcher.context(path,root,scene)


def test_full_source_must_be_on_published_main(tmp_path,monkeypatch):
    import subprocess
    from run.icra2027 import e3_gaussian_train_only as launcher
    def git(*args):return subprocess.check_output(['git','-C',str(tmp_path),*args],text=True).strip()
    git('init','-q');git('config','user.name','Fixture');git('config','user.email','fixture@example.invalid')
    (tmp_path/'file').write_text('before');git('add','.');git('commit','-qm','published')
    published=git('rev-parse','HEAD');git('update-ref','refs/remotes/origin/main',published)
    monkeypatch.setattr(launcher,'CODE',tmp_path)
    launcher.require_published_full_source({'commit':published})
    (tmp_path/'file').write_text('after');git('commit','-qam','unpublished')
    with pytest.raises(ValueError,match='published on main'):
        launcher.require_published_full_source({'commit':git('rev-parse','HEAD')})


def test_population_phase_keeps_scene_argument_through_final_validation(cohort_context,monkeypatch):
    launcher,path,root,config,contract=cohort_context
    def prepare(source,destination,**kw):
        destination.mkdir();producer.write_new_json(destination/'training_inputs.json',{'frames':[]})
        return {'frames':[]}
    monkeypatch.setattr(launcher,'prepare_training_scene',prepare)
    launcher.run(path,root,'prepare','09c1414f1b')
    receipt=json.loads((root/'gaussian_train_only/09c1414f1b/prepare_receipt.json').read_text())
    assert receipt['status']=='PASS' and receipt['scene_id']=='09c1414f1b'
    with pytest.raises(FileExistsError):launcher.run(path,root,'prepare','09c1414f1b')


def test_cohort_rejects_symlink_parent_before_output(cohort_context,tmp_path):
    launcher,path,root,config,contract=cohort_context
    outside=tmp_path/'outside';outside.mkdir()
    (root/'gaussian_train_only').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):launcher.run(path,root,'prepare','09c1414f1b')
    assert not list(outside.iterdir())


def test_cohort_cannot_bypass_full_main_guard_using_pilot(cohort_context):
    launcher,path,root,config,contract=cohort_context
    with pytest.raises(ValueError,match='published train-full'):
        launcher.run(path,root,'train-pilot','09c1414f1b')
    assert not (root/'gaussian_train_only').exists()


@pytest.mark.parametrize('field',['all','split','rgb'])
def test_valid_other_scene_staging_cannot_be_adopted(source,tmp_path,field):
    from run.icra2027 import e3_gaussian_train_only as launcher
    manifest=producer.prepare_training_scene(source,tmp_path/'stage-other',max_train_frames=None,initialization_max_frames=2)
    config={'dataset_root':str(tmp_path),'scene_id':'09c1414f1b','initialization_max_frames':2}
    dslr=tmp_path/'data/09c1414f1b/dslr'
    if field!='all':
        for name,value in manifest['source_metadata'].items():value['path']=str(dslr/name)
        if field=='rgb':manifest['boundary']['split']['path']=str(dslr/'train_test_lists.json')
    with pytest.raises(ValueError,match='another planned scene'):
        launcher.validate_cohort_staging(config,manifest)


def test_uniform_initialization_retains_full_training_population(source,tmp_path):
    pytest.importorskip('pycolmap')
    dslr=source/'dslr'
    split=json.loads((dslr/'train_test_lists.json').read_text())
    split['train'].append('d.png')
    (dslr/'train_test_lists.json').write_text(json.dumps(split))
    pixels=np.asarray(Image.open(dslr/'resized_undistorted_images/a.png'))
    Image.fromarray(np.roll(pixels,-30,axis=1)).save(dslr/'resized_undistorted_images/d.png')
    with (dslr/'colmap/images.txt').open('a') as stream:
        stream.write('5 1 0 0 0 -0.3 0 0 1 d.png\nDISCARD TRACKS\n')
    stage=tmp_path/'full-stage'
    manifest=producer.prepare_training_scene(source,stage,max_train_frames=None,initialization_max_frames=3)
    assert [r['name'] for r in manifest['frames']]==['a.png','b.png','c.png','d.png']
    assert producer.initialization_names(manifest)==['a.png','b.png','d.png']
    init=tmp_path/'subset-init'
    report=producer.triangulate_training_scene(stage,init,seed=42,num_threads=2)
    assert report['frame_names']==['a.png','b.png','d.png'] and report['n_points']>=4
    training,validated=producer.validate_training_initialization(stage,init/'init_points.ply',init/'init_manifest.json')
    assert len(training['frames'])==4 and len(validated['frame_names'])==3
    assert {o['frame'] for p in json.loads((init/'tracks.json').read_text()) for o in p['observations']}=={'a.png','b.png','d.png'}
    import pycolmap
    db=pycolmap.Database.open(init/'database.db')
    try: assert sorted(i.name for i in db.read_all_images())==['a.png','b.png','d.png']
    finally: db.close()
    manifest['initialization_selection']['frame_names']=['a.png','b.png','c.png']
    (stage/'training_inputs.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='selection changed'):producer.validate_training_scene(stage)


def test_uniform_initialization_rounding_is_predeclared():
    assert producer.uniform_initialization_names(list('abcdefghij'),4)==list('adgj')
    assert producer.uniform_initialization_names(list('abcdefghij'),5)==list('acegj')
    with pytest.raises(ValueError):producer.uniform_initialization_names(list('abc'),4)
    with pytest.raises(ValueError):producer.uniform_initialization_names(list('cba'),2)


@pytest.mark.parametrize('field,value',[('max_train_frames',48),('initialization_max_frames',47),('initialization_selection','first48')])
def test_full_cohort_rejects_narrowed_or_changed_initialization(cohort_context,field,value):
    import yaml
    launcher,path,root,config,contract=cohort_context
    config[field]=value
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='protocol differs|recipe or frame boundary'):
        launcher.context(path,root,'09c1414f1b')


def test_slurm_spool_routes_explicit_scene_and_full_recipe(tmp_path):
    import os
    import subprocess
    from run.icra2027 import e3_gaussian_train_only as launcher
    script=(launcher.CODE/'run/icra2027/e3_gaussian_train_only.sbatch').read_text()
    fake=tmp_path/'python';log=tmp_path/'calls'
    fake.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n')
    fake.chmod(0o755)
    spool=tmp_path/'slurm_script'
    spool.write_text(script.replace('/group/worldcept/PhiRIE/code/SimAny/.venv/bin/python',str(fake)))
    env=dict(os.environ,E3_GAUSSIAN_CODE=str(tmp_path),E3_GAUSSIAN_FREEZE=str(tmp_path/'freeze'),
             E3_GAUSSIAN_SCENE='09c1414f1b',E3_GAUSSIAN_CONFIG='cohort.yaml',
             E3_GAUSSIAN_PHASE='smoke-and-full',CALL_LOG=str(log))
    subprocess.run(['bash',str(spool)],env=env,check=True)
    calls=log.read_text().splitlines()
    assert len(calls)==2 and calls[0].endswith('--phase train-smoke') and calls[1].endswith('--phase train-full')
    assert all('--config cohort.yaml' in row and '--scene-id 09c1414f1b' in row for row in calls)
