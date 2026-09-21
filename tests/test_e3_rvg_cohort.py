"""Cohort RVG routing and TRAIN view-boundary regression checks."""
import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from run.icra2027 import e3_rvg_generation_pilot as rvg
from run.icra2027 import e3_fresh_rvg_config as builder
from run.icra2027.e3_auto_discovery_pilot import COHORT_SCOPE, PilotError
from run.icra2027.e3_fresh_generation_contract import DISCOVERY_FILES
from tests.test_e3_fresh_rvg import views_fixture


def test_scene_outputs_share_freeze_without_collision(tmp_path):
    a,b='38d58a7a31','5748ce6f01'
    assert rvg.output_directory({},tmp_path)==tmp_path/'rvg_initial'
    assert rvg.output_directory(dict(scene_id=a,output_scene_id=a),tmp_path)==tmp_path/'rvg_initial'/a
    assert rvg.output_directory(dict(scene_id=b,output_scene_id=b),tmp_path)==tmp_path/'rvg_initial'/b
    for scope in ['../escape',b,True]:
        with pytest.raises(PilotError,match='output scene'):
            rvg.output_directory(dict(scene_id=a,output_scene_id=scope),tmp_path)


@pytest.mark.parametrize('limit',[None,48])
def test_collector_uses_authenticated_training_limit(tmp_path,monkeypatch,limit):
    c,out,m,boundary,views,images=views_fixture(tmp_path,monkeypatch)
    boundary['boundary']['max_train_frames']=limit
    import agents.discover.training_views as training
    seen=[]
    def select(frames,path,stride,max_frames):
        seen.append((stride,max_frames))
        return frames,{'training_frames':boundary['boundary']['training_frames']}
    monkeypatch.setattr(training,'select_training_views',select)
    _,_,record=rvg.collect_frozen_views(c,out,boundary,write=True)
    assert seen==[(1,limit)] and record['n_views_max']==12


def test_collector_rejects_changed_training_population(tmp_path,monkeypatch):
    c,out,m,boundary,views,images=views_fixture(tmp_path,monkeypatch)
    boundary['boundary']['max_train_frames']=None
    import agents.discover.training_views as training
    monkeypatch.setattr(training,'select_training_views',lambda *args:([],{'training_frames':['test.png']}))
    with pytest.raises(PilotError,match='training-view population changed'):
        rvg.collect_frozen_views(c,out,boundary,write=True)


def test_cohort_builder_scopes_config_and_preserves_jobs(tmp_path,monkeypatch):
    scene='38d58a7a31';root=tmp_path/'source-freeze';source=root/'auto_discovery_pilot'/scene
    source.mkdir(parents=True);(root/'contract').mkdir()
    (root/'contract/freeze_manifest.json').write_text('{}')
    for name in DISCOVERY_FILES:(source/name).write_text('{}')
    source_config=tmp_path/'discovery.yaml';source_config.write_text(yaml.safe_dump({'scope':COHORT_SCOPE}))
    recipe=tmp_path/'recipe.yaml';recipe.write_text('seed: 42\n')
    import run.icra2027.e3_gaussian_train_only as gaussian
    monkeypatch.setattr(gaussian,'cohort_scene_ids',lambda _:[scene])
    calls=[]
    def binding(c):
        calls.append(c)
        return [{'prepared':True},{'prepared':False}],{},{'source_discovery_hashes':c['source_discovery_hashes']}
    monkeypatch.setattr(builder,'source_boundary',binding)
    result=builder.prepare(root,source_config,'1'*40,recipe,scene_id=scene)
    assert result['planned_jobs']==2 and result['prepared_inputs']==1
    c=calls[0]
    assert c['source_pilot']==str(source) and c['output_scene_id']==scene
    assert c['contract_resource_id']=='e3_rvg_config_'+scene
    assert set(c['source_discovery_hashes'])==DISCOVERY_FILES
    monkeypatch.setattr(gaussian,'cohort_scene_ids',lambda _:['5748ce6f01'])
    with pytest.raises(PilotError,match='outside frozen'):
        builder.prepare(root,source_config,'1'*40,recipe,scene_id=scene)


def test_launcher_rejects_array_before_any_work(tmp_path):
    script=Path(rvg.CODE)/'run/icra2027/e3_fresh_rvg.sbatch'
    env=dict(os.environ,E3_RVG_CODE=str(tmp_path),E3_RVG_FREEZE=str(tmp_path),SLURM_ARRAY_TASK_ID='0')
    result=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
    assert result.returncode==2 and 'independent jobs only' in result.stderr
    assert not list(tmp_path.iterdir())


def test_pool_uses_freeze_not_scene_as_proposal_identity(tmp_path,monkeypatch):
    freeze='20260905-123abcd-v1';scene='38d58a7a31'
    out=tmp_path/freeze/'rvg_initial'/scene;out.mkdir(parents=True)
    for name in ['view_manifest.json','views_receipt.json','input_manifest.json']:
        (out/name).write_text('{}')
    m={'jobs':[]}
    monkeypatch.setattr(rvg,'read_manifest',lambda *_:(m,{}))
    monkeypatch.setattr(rvg,'validate_view_receipt',lambda *_:None)
    seen=[]
    monkeypatch.setattr(rvg,'collect_records',lambda out,m,f,code:seen.append(f) or [])
    rvg.publish_pool({'freeze_id':freeze},'source',out,m,0,0.0,generation_performed=False)
    assert seen==[freeze]
    pool=json.loads((out/'proposal_pool.json').read_text())
    assert pool['freeze_id']==freeze and pool['planned_jobs']==0
