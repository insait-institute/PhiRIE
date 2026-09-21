"""A malformed object must not suppress the remaining frozen initial jobs."""
import json
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image
import pytest
import torch

from run.icra2027 import e3_trellis_generation_pilot as trellis
from run.icra2027 import e3_rvg_generation_pilot as rvg


def setup_trellis(tmp_path,monkeypatch,failures,strict=True):
    import models.s4_trellis as model
    out=tmp_path/'construction';objects=out/'objects';objects.mkdir(parents=True)
    records=tmp_path/'producer_records';records.mkdir()
    metas=[{'index':i,'label':'synthetic'} for i in range(len(failures))]
    (objects/'objects.json').write_text(json.dumps(metas))
    for meta in metas:
        directory=objects/f"obj_{meta['index']:02d}";directory.mkdir()
        Image.fromarray(np.full((4,4,4),255,np.uint8)).save(directory/'rgba.png')
    monkeypatch.setattr(model.C,'OUT',out)
    if strict:monkeypatch.setenv('SIMANY_GENERATION_RECORDS',str(records))
    else:monkeypatch.delenv('SIMANY_GENERATION_RECORDS',raising=False)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    calls=[]
    class Gaussian:
        get_xyz=np.zeros((1,3))
        def save_ply(self,path,transform):
            assert transform is None
            Path(path).write_bytes(b'synthetic gaussian')
    class Pipeline:
        @classmethod
        def from_pretrained(cls,path):return cls()
        def cuda(self):return self
        def run(self,img,**kwargs):
            idx=len(calls);calls.append((idx,img.mode,kwargs))
            if failures[idx] is not None:raise failures[idx]
            mesh=types.SimpleNamespace(vertices=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),
                faces=torch.tensor([[0,1,2]]),vertex_attrs=torch.ones((3,3)))
            return {'mesh':[mesh],'gaussian':[Gaussian()]}
    package=types.ModuleType('trellis');package.__path__=[]
    pipelines=types.ModuleType('trellis.pipelines');pipelines.TrellisImageTo3DPipeline=Pipeline
    package.pipelines=pipelines
    monkeypatch.setitem(sys.modules,'trellis',package);monkeypatch.setitem(sys.modules,'trellis.pipelines',pipelines)
    return model,calls,records


def test_strict_trellis_records_both_failures_and_executes_later_job(tmp_path,monkeypatch):
    failures=[RuntimeError('empty sparse coords'),AttributeError("module 'rembg' has no attribute 'new_session'"),None]
    model,calls,records=setup_trellis(tmp_path,monkeypatch,failures)
    model.main()
    assert [c[0] for c in calls]==[0,1,2]
    assert all(c[1:] == ('RGBA',{'seed':42,'formats':['mesh','gaussian']}) for c in calls)
    actual=[json.loads((records/f'object_{i:02d}.json').read_text()) for i in range(3)]
    assert [r['status'] for r in actual]==['generation_failed','generation_failed','generated']
    assert [r['error_type'] for r in actual[:2]]==['RuntimeError','AttributeError']
    assert 'rembg' in actual[1]['reason'] and all(r['seed']==42 for r in actual)
    jobs=[dict(job_id=f'scene:auto:{i}',automatic_instance_id=i,proposal_id=f'proposal{i}',prepared=i<3,
               output_index=i,input={'sha256':'synthetic'},frame='train.png') for i in range(4)]
    rows=trellis.collect_records(tmp_path,{'jobs':jobs},0)
    assert len(rows)==4
    assert [r['status'] for r in rows]==['generation_failed','generation_failed','available','unavailable']
    assert rows[0]['reason']=='RuntimeError: empty sparse coords'
    assert rows[1]['reason']==actual[1]['reason']
    assert all(r['shared_initial_policy_rows']==['A1','A2','A3','A4'] for r in rows)


@pytest.mark.parametrize('failure',[RuntimeError('empty sparse coords'),AttributeError('rembg.new_session')])
def test_legacy_trellis_failure_still_raises(tmp_path,monkeypatch,failure):
    model,calls,records=setup_trellis(tmp_path,monkeypatch,[failure,None],strict=False)
    with pytest.raises(type(failure)):model.main()
    assert len(calls)==1 and not list(records.iterdir())


def test_prior_failed_record_cannot_trigger_same_object_retry(tmp_path,monkeypatch):
    model,calls,records=setup_trellis(tmp_path,monkeypatch,[None])
    record=records/'object_00.json';record.write_text('{"status":"generation_failed"}')
    before=record.read_bytes()
    with pytest.raises(FileExistsError):model.main()
    assert calls==[] and record.read_bytes()==before


@pytest.mark.parametrize('status',['generation_failed','unknown',None])
def test_trellis_complete_files_never_override_failed_or_missing_success_status(tmp_path,status):
    odir=tmp_path/'construction/objects/obj_00';odir.mkdir(parents=True)
    for name in trellis.OUTPUTS:(odir/name).write_bytes(b'partial export before failure')
    records=tmp_path/'producer_records';records.mkdir()
    record={'seed':42,'reason':'failure after output serialization'}
    if status is not None:record['status']=status
    (records/'object_00.json').write_text(json.dumps(record))
    job=dict(job_id='j',automatic_instance_id=1,proposal_id='p',prepared=True,output_index=0,input={'sha256':'crop'},frame='train.png')
    row=trellis.collect_records(tmp_path,{'jobs':[job]},0)[0]
    assert row['status']=='generation_failed' and row['reason']==record['reason']
    assert len(row['artifacts'])==4  # partial bytes retained for diagnosis


def test_rvg_records_failed_object_and_attempts_remaining_objects(tmp_path,monkeypatch):
    out=tmp_path/'rvg_initial';(out/'construction/objects').mkdir(parents=True);(out/'producer_records').mkdir()
    config=tmp_path/'config.yaml';config.write_text('synthetic')
    c={'python':'synthetic'};m={'code_commit':'source','config_sha256':rvg.sha(config),'source_discovery_hashes':{}}
    monkeypatch.setattr(rvg,'context',lambda *_:(c,'source'))
    monkeypatch.setattr(rvg,'read_manifest',lambda *_:(m,{}))
    env={key:'synthetic' for key in ['SIMANY_ROOT','SIMANY_OUT','SIMANY_SCENE','SIMANY_AUTO','SIMANY_MESH_SRC','SIMANY_SCANNETPP_ROOT','SIMANY_RVG_DIR','SIMANY_RVG_PINNED_CONFIG']}
    monkeypatch.setattr(rvg,'env_for',lambda *_:env)
    for k,v in env.items():monkeypatch.setenv(k,v)
    monkeypatch.setattr(rvg,'setup_read_guard',lambda *_:None)
    objects={i:({'index':i},{},[('a',),('b',)]) for i in range(3)}
    views={'rows':[{'object_index':i,'views':[]} for i in range(3)]}
    monkeypatch.setattr(rvg,'collect_frozen_views',lambda *a,**k:((),objects,views))
    monkeypatch.setattr(rvg,'validate_view_receipt',lambda *_:None)
    monkeypatch.setattr(rvg,'verify_consumed_views',lambda *_:[])
    (out/'view_manifest.json').write_text('{}')
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    calls=[];module=types.ModuleType('agents.assets.factory_hybrid')
    def run(holder,obj,odir,instance,ctx,n,precollected_views):
        calls.append(obj['index']);assert n==12
        if obj['index']==0:raise RuntimeError('empty sparse coords')
        if obj['index']==1:raise AttributeError('rembg.new_session')
        return 1.0
    module.run_rvg=run;monkeypatch.setitem(sys.modules,'agents.assets.factory_hybrid',module)
    rvg.views_worker(config,tmp_path,generate=True)
    assert calls==[0,1,2]
    rows=[json.loads((out/'producer_records'/f'object_{i:02d}.json').read_text()) for i in range(3)]
    assert [r['status'] for r in rows]==['generation_failed','generation_failed','generated']
    assert rows[0]['error_type']=='RuntimeError' and rows[1]['error_type']=='AttributeError'
