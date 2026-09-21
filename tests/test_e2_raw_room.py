import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from run.icra2027 import e2_raw_room as raw
from run.icra2027.e3_auto_discovery_pilot import identity


def fixture_plan(tmp_path):
    gaussian = tmp_path/'scene.ply'; gaussian.write_bytes(b'fake gaussian for CPU mock')
    images=[]
    for i in range(8):
        p=tmp_path/f'test{i}.jpg';Image.fromarray(np.full((8,10,3),32+i,dtype=np.uint8)).save(p)
        images.append(dict(frame=p.name,**identity(p),w2c=np.eye(4).tolist()))
    return dict(freeze_id='fixture',code_commit='a'*40,scene_id='scene',paper_ready=False,
                gaussian=identity(gaussian),evaluation_images=images,
                optimization_input_frames=['train.jpg'],
                calibration={'width':10,'height':8,'K':[[2,0,5],[0,2,4],[0,0,1]]})


def common_mock(plan, *, nan=False):
    calls=[]
    def render(gs,w2c,K,w,h):
        calls.append((w2c.tolist(),K.tolist(),w,h))
        return np.full((h,w,3),np.nan if nan else .5),None,None
    return SimpleNamespace(load_gaussians=lambda _:object(),render_view=render),calls


def test_cpu_export_exact_cameras_and_canonical_metric_pipeline(tmp_path,monkeypatch):
    from robo.eval import fidelity_metrics as metrics
    plan=fixture_plan(tmp_path);common,calls=common_mock(plan);bundle=tmp_path/'bundle'
    raw.export_raw(plan,bundle,common)
    assert len(calls)==8
    assert calls==[(r['w2c'],plan['calibration']['K'],10,8) for r in plan['evaluation_images']]
    manifest=raw.canonical_manifest(plan,bundle)
    path=tmp_path/'fidelity.json';path.write_text(json.dumps(manifest))
    class FakeLPIPS:
        error=None
        provenance={}
        def __init__(self,*args,**kwargs):pass
        def __call__(self,*args,**kwargs):return .25
    monkeypatch.setattr(metrics,'LPIPSEvaluator',FakeLPIPS)
    monkeypatch.setattr(metrics,'REPOSITORY_ROOT',tmp_path)
    result=metrics.evaluate_manifest(path,tmp_path/'table',bootstrap_samples=10)
    assert result['paper_ready'] is False
    assert len(result['rows'])==8
    row=next(r for r in result['rows'] if r['method']==raw.METHOD)
    assert row['n_images']==8 and row['n_scenes']==1
    assert row['metric_samples']=={'psnr':8,'ssim':8,'lpips':8}
    assert all(r['psnr'] is None for r in result['rows'] if r['method']!=raw.METHOD)


def test_changed_input_rgb_rejected(tmp_path):
    plan=fixture_plan(tmp_path);common,_=common_mock(plan)
    Path(plan['evaluation_images'][0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='anchor changed'):raw.export_raw(plan,tmp_path/'bundle',common)


def test_nonfinite_render_keeps_failed_staging(tmp_path):
    plan=fixture_plan(tmp_path);common,_=common_mock(plan,nan=True)
    with pytest.raises(ValueError,match='invalid rendered'):raw.export_raw(plan,tmp_path/'bundle',common)
    assert (tmp_path/'bundle.partial').exists()
    assert not (tmp_path/'bundle').exists()


@pytest.mark.parametrize('mutation',['drop','pixel','camera','view_mapping'])
def test_artifact_or_plan_drift_rejected(tmp_path,mutation):
    plan=fixture_plan(tmp_path);common,_=common_mock(plan);bundle=tmp_path/'bundle'
    raw.export_raw(plan,bundle,common)
    if mutation=='drop':
        m=json.loads((bundle/'manifest.json').read_text());m['artifacts'].pop()
        (bundle/'manifest.json').write_text(json.dumps(m))
    if mutation=='pixel':(bundle/'input_scene_gaussian/test0.png').write_bytes(b'bad')
    if mutation=='camera':plan=copy.deepcopy(plan);plan['evaluation_images'][0]['w2c'][0][3]=1
    if mutation=='view_mapping':
        m=json.loads((bundle/'manifest.json').read_text());m['views'][0]['view_id']='other.JPG'
        (bundle/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):raw.canonical_manifest(plan,bundle)


def test_no_overwrite_complete_or_failed_bundle(tmp_path):
    plan=fixture_plan(tmp_path);common,_=common_mock(plan);bundle=tmp_path/'bundle'
    raw.export_raw(plan,bundle,common)
    with pytest.raises(FileExistsError):raw.export_raw(plan,bundle,common)


def test_driver_uses_frozen_config_and_root_for_both_workers(tmp_path,monkeypatch):
    config=tmp_path/'config.yaml';config.write_text('frozen')
    root=tmp_path/'freeze';out=tmp_path/'local';out.mkdir()
    c={'python':{'render':'renderpython','metrics':'metricpython'},'freeze_id':'freeze','scope':'scope'}
    monkeypatch.setattr(raw,'context',lambda *args:(c,{'commit':'a'*40},out,{}))
    monkeypatch.setattr(raw,'checked_plan',lambda *args:{})
    monkeypatch.setattr(raw,'gpu_identity',lambda:{'node':'hala'})
    monkeypatch.setattr(raw,'environment',lambda *args:{})
    commands=[]
    def execute(command,**kwargs):
        commands.append(command)
        if command[command.index('--phase')+1]=='metrics':
            (out/'table').mkdir();(out/'table/fidelity_table.json').write_text('{}')
        else:(out/'render_receipt.json').write_text('{}')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(raw.subprocess,'run',execute)
    raw.run(config,root)
    assert [v[0] for v in commands]==['renderpython','metricpython']
    for v in commands:
        assert v[v.index('--config')+1]==str(config)
        assert v[v.index('--freeze-root')+1]==str(root)
    assert commands[1][-2:] == ['--upstream-receipt-sha256', raw.sha(out/'render_receipt.json')]
    with pytest.raises(FileExistsError):raw.run(config,root)


def test_driver_failure_preserves_claim_log_and_terminal_receipt(tmp_path,monkeypatch):
    out=tmp_path/'out';out.mkdir();root=tmp_path/'freeze'
    c={'python':{'render':'python'},'freeze_id':'freeze','scope':'scope'}
    monkeypatch.setattr(raw,'context',lambda *args:(c,{'commit':'a'*40},out,{}))
    monkeypatch.setattr(raw,'checked_plan',lambda *args:{})
    monkeypatch.setattr(raw,'gpu_identity',lambda:{'node':'hala'})
    monkeypatch.setattr(raw,'environment',lambda *args:{})
    monkeypatch.setattr(raw.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=1))
    with pytest.raises(ValueError,match='render failed'):raw.run(tmp_path/'config',root)
    assert (out/'execution_claim.json').exists() and (out/'render.log').exists()
    assert json.loads((root/'raw_room/execution_receipt.json').read_text())['status']=='FAIL'


@pytest.mark.parametrize('mutation',[None,'bundle','receipt','source','config','freeze','plan'])
def test_metrics_requires_original_render_receipt_and_bundle(tmp_path,mutation):
    plan=fixture_plan(tmp_path);common,_=common_mock(plan);bundle=tmp_path/'bundle'
    raw.export_raw(plan,bundle,common)
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    receipt=dict(status='PASS',code_commit=plan['code_commit'],freeze_id=plan['freeze_id'],
        config_sha256='c'*64,plan_sha256=raw.sha(tmp_path/'plan.json'),artifact=identity(bundle/'manifest.json'))
    if mutation=='source':receipt['code_commit']='d'*40
    if mutation=='config':receipt['config_sha256']='d'*64
    if mutation=='freeze':receipt['freeze_id']='other'
    if mutation=='plan':receipt['plan_sha256']='d'*64
    path=tmp_path/'render_receipt.json';path.write_text(json.dumps(receipt));anchor=raw.sha(path)
    if mutation=='bundle':
        # Even a newly hashed/otherwise valid manifest cannot replace the one
        # sealed by the actual rendering worker.
        with (bundle/'manifest.json').open('a') as f:f.write(' ')
    if mutation=='receipt':
        with path.open('a') as f:f.write(' ')
    if mutation is None:raw.check_render_receipt(tmp_path,plan,'c'*64,anchor)
    else:
        with pytest.raises(ValueError):raw.check_render_receipt(tmp_path,plan,'c'*64,anchor)
