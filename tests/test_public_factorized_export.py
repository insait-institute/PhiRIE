import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agents.edit import inpaint_fill as fill
from agents.eval import fidelity_room_export as export
from robo.eval import agentic_ablation as e3
from run.icra2027 import e2_raw_room as raw
from tests.test_e2_raw_room import fixture_plan, common_mock


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


@pytest.fixture
def selected(tmp_path,monkeypatch):
    monkeypatch.setattr(e3,'REPOSITORY_ROOT',tmp_path)
    factory=tmp_path/'factory';obj=factory/'objects/obj_1000'
    records=[dict(index=1000,automatic_instance_id=1000,instance_namespace='automatic')]
    write(factory/'objects/objects.json',records)
    T=np.eye(4);T[0,3]=3
    aligned=dict(index=1000,terminal_action='accept',job_id='job',policy_id='A4',proposal_id='proposal',T=T.tolist())
    write(obj/'aligned.json',aligned)
    gs=obj/'trellis_gs.ply';gs.write_bytes(b'authenticated native Gaussian')
    chosen=dict(terminal_action='accept',object_slot=obj.name,job_id='job',policy_id='A4',selected_proposal_id='proposal',
        selected_asset=dict(artifact_hashes={'raw_gaussian':export._sha256(gs)},artifact_sizes={'raw_gaussian':gs.stat().st_size}))
    write(obj/'selected_asset.json',chosen)
    background=tmp_path/'fill/clean_background.ply';background.parent.mkdir();background.write_bytes(b'clean')
    seal=write(background.parent/'seal.json',{})
    public=dict(result={'cleaned_background_created':True},source=dict(factory=factory,blocked=[],accepted_objects=1,train_images={'TRAIN.jpg':{}}),
        seal_identity={'path':str(seal)},clean_background={'path':str(background)})
    monkeypatch.setattr(fill,'validate_public_fill',lambda _:public)
    transforms=[]
    def load(path):return {'sh_degree':0,'means':np.array([[0.,0.,0.]]) if Path(path)==background else np.array([[1.,0.,0.]])}
    def transform(gs,T):
        transforms.append(T.copy());return dict(gs,means=gs['means']+T[:3,3])
    common=SimpleNamespace(load_gaussians=load,pad_sh=lambda x,d:x,transform_gaussians=transform,
        cat_gaussians=lambda parts:dict(means=np.concatenate([p['means'] for p in parts])))
    return factory,obj,public,common,transforms


def test_selected_native_asset_uses_clean_background_and_single_transform(selected):
    factory,obj,public,common,transforms=selected
    gs,provenance=export._load_composite(common,None,factory,use_trellis_snapshot=False,public_fill='sealed')
    assert gs['means'].tolist()==[[0,0,0],[4,0,0]]
    assert len(transforms)==1 and provenance['accepted_object_ids']==[1000]
    assert provenance['optimization_input_frames']==['TRAIN.jpg']
    assert provenance['asset_variant']=='public_evidence_selected_native'
    assert not (obj/'meta.json').exists()


@pytest.mark.parametrize('kind',['GT_alias','wrong_auto_id','proposal','native_bytes','blocked','raw_background','legacy'])
def test_public_composite_rejects_wrong_identity_or_background(selected,kind):
    factory,obj,public,common,_=selected;kw={}
    if kind in {'GT_alias','wrong_auto_id'}:
        path=factory/'objects/objects.json';records=json.loads(path.read_text())
        records[0]['gt_object_id' if kind=='GT_alias' else 'automatic_instance_id']=999
        write(path,records)
    elif kind=='proposal':
        path=obj/'selected_asset.json';value=json.loads(path.read_text());value['selected_proposal_id']='different';write(path,value)
    elif kind=='native_bytes':(obj/'trellis_gs.ply').write_bytes(b'other')
    elif kind=='blocked':public['source']['blocked']=[{'reason':'NO_PLANE'}]
    elif kind=='raw_background':public['result']['cleaned_background_created']=False
    else:kw['replacements']={1000:{}}
    with pytest.raises(ValueError):export._load_composite(common,None,factory,use_trellis_snapshot=False,public_fill='sealed',**kw)


def test_preloaded_composite_reuses_exact_raw_renderer_camera_quantization(tmp_path):
    plan=fixture_plan(tmp_path);common,calls=common_mock(plan)
    common.load_gaussians=lambda _:pytest.fail('preloaded composite was replaced by raw room')
    views=raw.export_raw(plan,tmp_path/'bundle',common,gaussian=object(),method_slug='factorized_auto_discovery')
    assert len(views)==8 and len(calls)==8
    assert all('/factorized_auto_discovery/' in r['render_path'] for r in views)
    assert calls==[(r['w2c'],plan['calibration']['K'],10,8) for r in plan['evaluation_images']]
    from PIL import Image
    assert np.asarray(Image.open(views[0]['render_path'])).tolist()==np.full((8,10,3),127).tolist()
    with pytest.raises(FileExistsError):raw.export_raw(plan,tmp_path/'bundle',common,gaussian=object(),method_slug='factorized_auto_discovery')


@pytest.mark.parametrize('kwargs',[{'gaussian':object()},{'method_slug':'factorized_auto_discovery'},{'method_slug':'other'}])
def test_export_requires_explicit_method_and_asset_pair(tmp_path,kwargs):
    plan=fixture_plan(tmp_path);common,_=common_mock(plan)
    with pytest.raises(ValueError):raw.export_raw(plan,tmp_path/'bundle',common,**kwargs)
