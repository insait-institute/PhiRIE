import contextlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from agents.edit import inpaint_masks as masks
from agents.edit import inpaint_prepare as prep
from robo.eval import e3_factory_materializer  # initialize its evidence-root binding before fixture patches
from robo.eval import agentic_ablation as e3
from robo.manifest.hash import canonical_hash


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value));return path


def contract(path,context_path,freeze,commit):
    value={'freeze_id':freeze,'code':{'commit':commit,'dirty':False},'resource_inventory':[
        {'resolved_path':str(context_path),'hash_method':'content_sha256','sha256':masks._identity(context_path)['sha256']}]}
    value['contract_sha256']=canonical_hash(value)
    return write(path,value)


def seal(directory):
    return write(directory/'seal.json',{'schema_version':1,'members':{
        str(p.relative_to(directory)):masks._identity(p)['sha256'] for p in directory.rglob('*') if p.is_file() and p.name!='seal.json'}})


@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(e3,'REPOSITORY_ROOT',tmp_path)
    monkeypatch.setattr(e3,'_validate_cli_execution',lambda *a,**k:None)
    monkeypatch.setattr(prep,'_validate_source_factory',lambda *a:{'authenticated':'original-validator'})
    image=tmp_path/'data/scene/dslr/resized_undistorted_images/train.jpg'
    image.parent.mkdir(parents=True);Image.new('RGB',(16,16),(30,70,100)).save(image)
    split=write(image.parent.parent/'train_test_lists.json',{'train':['train.jpg'],'test':['test.jpg']})
    discovery=tmp_path/'discovery'
    boundary=write(discovery/'input_manifest.json',{'boundary':{'training_frames':['train.jpg']},
        'metadata':{'train_test_lists.json':masks._identity(split)},'input_images':{'train.jpg':masks._identity(image)}})
    generation=write(tmp_path/'generation.yaml',{'source_pilot':str(discovery),
        'source_discovery_hashes':{'input_manifest.json':masks._identity(boundary)['sha256']}})
    factory=tmp_path/'factory'
    manifest=write(factory/'materialization_manifest.json',{})
    objects=[{'index':1000,'automatic_instance_id':1000,'instance_namespace':'automatic','label':'bottle'},
             {'index':1001,'automatic_instance_id':1001,'instance_namespace':'automatic','label':'mug'}]
    write(factory/'objects/objects.json',objects)
    write(factory/'objects/obj_1000/aligned.json',{'terminal_action':'accept','rejected':None})
    write(factory/'objects/obj_1001/aligned.json',{'terminal_action':'abstain','rejected':'missing_evidence'})
    old_context=write(tmp_path/'old.yaml',dict(scope='automatic_train_only_removal_preparation',freeze_id='old',scene_id='0000000001',
        policy_id='A4',seed=0,algorithm=prep.PUBLIC_ALGORITHM,materialization_manifest=masks._identity(manifest),
        generation_config=masks._identity(generation)))
    old_contract=contract(tmp_path/'old_contract.json',old_context,'old','a'*40)
    directory=tmp_path/'outputs/icra2027/old/fidelity/removal/0000000001/inpaint'
    directory.mkdir(parents=True)
    projected=np.zeros((1,16,16),bool);projected[:,7:9,7:9]=True
    (directory/'obj_1000').mkdir()
    np.savez(directory/'obj_1000/proj_masks.npz',frames=np.array(['train.jpg']),masks=projected)
    write(directory/'obj_1000/views.json',[{'frame':'train.jpg','w2c':np.eye(4).tolist()}])
    statuses=[{'object_slot':'obj_1000','automatic_instance_id':1000,'terminal_action':'accept',
               'plane_status':'NO_PLANE','view_status':'VIEWS_SELECTED','selected_frames':['train.jpg'],'projected_mask_pixels':[4]},
              {'object_slot':'obj_1001','automatic_instance_id':1001,'terminal_action':'abstain','preparation_status':'NOT_APPLICABLE'}]
    result=dict(context_sha256=masks._identity(old_context)['sha256'],materialization_sha256=masks._identity(manifest)['sha256'],
        source_factory=str(factory),source_validation={'authenticated':'original-validator'},paper_ready=False,cleaned_background_created=False,
        official_test_images_read=0,seed=0,algorithm=prep.PUBLIC_ALGORITHM,training_frames=['train.jpg'],
        gaussian_provenance={'status':'FRESH_OFFICIAL_TRAIN_ONLY','training_frames':['train.jpg']},planned_objects=2,objects=statuses)
    write(directory/'public_prepare.json',result);seal_path=seal(directory)
    source=dict(directory=str(directory),seal=masks._identity(seal_path),context=masks._identity(old_context),
                contract=masks._identity(old_contract),producer_commit='a'*40)
    context=write(tmp_path/'new.yaml',dict(schema_version=1,scope='automatic_train_only_removal_masks',freeze_id='new',scene_id='0000000001',
        preparation=source,sam3_source={'path':str(tmp_path/'sam3'),'tree_sha256':'s'*64},sam3_checkpoint={},
        python=sys.executable,runtime={},seed=0,algorithm=masks.PUBLIC_ALGORITHM))
    current_contract=contract(tmp_path/'new_contract.json',context,'new','b'*40)
    return context,current_contract,directory,image


def test_union_iou_threshold_and_dilation_are_fixed():
    projected=np.zeros((30,30),bool);projected[10,10:30]=True
    candidate=np.zeros_like(projected);candidate[10,10:13]=True # IoU exactly .15: excluded
    final,selected=masks._merge_mask(projected,candidate[None])
    assert selected==0 and np.all(final[projected])
    assert final[6,10] and not final[5,10]
    candidate[10,13]=True
    assert masks._merge_mask(projected,candidate[None])[1]==1


@pytest.mark.parametrize('kind',['wrong_shape','nan','nonbinary'])
def test_malformed_sam_result_cannot_fall_back_to_projection(kind):
    projected=np.zeros((16,16),bool);candidate=np.zeros((1,16,16))
    if kind=='wrong_shape':candidate=np.zeros((1,8,8))
    elif kind=='nan':candidate[0,0,0]=np.nan
    else:candidate[0,0,0]=.5
    with pytest.raises(ValueError):masks._merge_mask(projected,candidate)


def test_preparation_keeps_complete_population_and_source_train_images(setup):
    context,_,directory,image=setup
    c=json.loads(context.read_text())
    result=masks._preparation(c['preparation'],'0000000001')
    assert len(result['objects'])==2 and len(result['jobs'])==1
    assert result['report']['objects'][1]['terminal_action']=='abstain'
    assert result['train_images']['train.jpg']['sha256']==masks._identity(image)['sha256']


@pytest.mark.parametrize('kind',['member','extra','empty_dir','symlink','rgb','test_view','code'])
def test_preparation_source_or_projection_drift_fails_closed(setup,kind):
    context,_,directory,image=setup;c=json.loads(context.read_text())
    if kind=='member':(directory/'obj_1000/views.json').write_text('[]')
    elif kind=='extra':(directory/'extra').write_text('bad')
    elif kind=='empty_dir':(directory/'extra').mkdir()
    elif kind=='symlink':(directory/'alias').symlink_to(image)
    elif kind=='rgb':Image.new('RGB',(16,16)).save(image)
    elif kind=='code':c['preparation']['producer_commit']='c'*40
    else:
        np.savez(directory/'obj_1000/proj_masks.npz',frames=np.array(['test.jpg']),masks=np.ones((1,16,16),bool))
        seal(directory);c['preparation']['seal']=masks._identity(directory/'seal.json')
    with pytest.raises((ValueError,FileNotFoundError)):
        masks._preparation(c['preparation'],'0000000001')


class Processor:
    def __init__(self,fail=False):self.images=0;self.prompts=0;self.fail=fail
    def set_image(self,image):self.images+=1;return {}
    def reset_all_prompts(self,state):pass
    def set_text_prompt(self,*,prompt,state):
        self.prompts+=1
        if self.fail:raise RuntimeError('SAM3 failed')
        value=torch.zeros((1,1,16,16),dtype=torch.bool);value[:,:,6:10,6:10]=True
        return {'masks':value}


def cpu_runtime(monkeypatch,processor):
    monkeypatch.setattr(masks,'_validate_model',lambda _:None)
    monkeypatch.setattr(masks,'_public_processor',lambda _:processor)
    monkeypatch.setattr(torch,'autocast',lambda *a,**k:contextlib.nullcontext())
    monkeypatch.setattr(sys,'addaudithook',lambda _:None)


def test_actual_mask_handoff_sealed_and_original_preparation_unchanged(setup,monkeypatch):
    context,contract_path,directory,_=setup
    before={str(p):p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    processor=Processor();cpu_runtime(monkeypatch,processor)
    result=masks.run_public(context,contract_path)
    output=e3.REPOSITORY_ROOT/'outputs/icra2027/new/fidelity/removal_masks/0000000001'
    handoff=masks.validate_public_masks(output)
    assert result['planned_objects']==2 and result['planned_views']==1
    assert processor.images==processor.prompts==result['model_loads']==1
    assert handoff['rows'][0]['status']=='MASK_READY'
    assert handoff['rows'][0]['final_pixels']>4
    assert before=={str(p):p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):masks.run_public(context,contract_path)
    mask=Path(handoff['rows'][0]['mask_identity']['path']);mask.write_bytes(b'changed')
    with pytest.raises(ValueError):masks.validate_public_masks(output)


def test_sam_failure_retains_partial_and_planned_rows_no_fallback(setup,monkeypatch):
    context,contract_path,_,_=setup;cpu_runtime(monkeypatch,Processor(fail=True))
    with pytest.raises(RuntimeError,match='SAM3 failed'):masks.run_public(context,contract_path)
    parent=e3.REPOSITORY_ROOT/'outputs/icra2027/new/fidelity/removal_masks'
    assert not (parent/'0000000001').exists()
    result=json.loads((parent/'0000000001_failure.json').read_text())
    assert result['status']=='FAILED' and len(result['planned_rows'])==1 and result['completed_rows']==[]
    assert (parent/'0000000001_failed_partial').is_dir()


def test_empty_projection_is_explicit_and_never_invokes_model(setup,monkeypatch):
    context,contract_path,directory,_=setup
    np.savez(directory/'obj_1000/proj_masks.npz',frames=np.array(['train.jpg']),masks=np.zeros((1,16,16),bool))
    report=json.loads((directory/'public_prepare.json').read_text());report['objects'][0]['projected_mask_pixels']=[0]
    write(directory/'public_prepare.json',report);seal(directory)
    c=json.loads(context.read_text());c['preparation']['seal']=masks._identity(directory/'seal.json');write(context,c)
    contract(contract_path,context,'new','b'*40)
    cpu_runtime(monkeypatch,Processor())
    monkeypatch.setattr(masks,'_public_processor',lambda _:pytest.fail('empty projection invoked model'))
    result=masks.run_public(context,contract_path)
    assert result['model_loads']==0 and result['status']=='NO_APPLICABLE_VIEWS'
    assert result['rows'][0]['status']=='EMPTY_PROJECTED_MASK' and result['rows'][0]['mask_identity'] is None
    handoff=masks.validate_public_masks(e3.REPOSITORY_ROOT/'outputs/icra2027/new/fidelity/removal_masks/0000000001')
    assert handoff['result']['planned_objects']==2


def refresh_preparation(context, directory):
    seal(directory)
    value=json.loads(context.read_text());value['preparation']['seal']=masks._identity(directory/'seal.json')
    write(context,value)
    return value


@pytest.mark.parametrize('kind',['dropped_row','renamed_frame','extra_product','promoted_status'])
def test_resealed_mask_semantic_tampering_rejected(setup,monkeypatch,kind):
    context,contract_path,_,_=setup;cpu_runtime(monkeypatch,Processor())
    masks.run_public(context,contract_path)
    output=e3.REPOSITORY_ROOT/'outputs/icra2027/new/fidelity/removal_masks/0000000001'
    result=json.loads((output/'public_masks.json').read_text())
    if kind=='dropped_row':result['rows']=[]
    elif kind=='renamed_frame':result['rows'][0]['frame']='test.jpg'
    elif kind=='extra_product':(output/'borrowed.png').write_bytes(b'extra')
    else:result['objects'][1]['terminal_action']='accept'
    write(output/'public_masks.json',result);seal(output)
    with pytest.raises(ValueError):masks.validate_public_masks(output)


@pytest.mark.parametrize('kind',['threshold','seed','source_commit','context_hash'])
def test_context_or_e0_tampering_rejected(setup,kind):
    context,contract_path,_,_=setup
    value=json.loads(context.read_text())
    if kind=='threshold':value['algorithm']['min_iou']=.2
    elif kind=='seed':value['seed']=1
    elif kind=='source_commit':value['preparation']['producer_commit']='z'*40
    else:value['scene_id']='0000000002'
    write(context,value)
    if kind!='context_hash':contract(contract_path,context,'new','b'*40)
    with pytest.raises(ValueError):masks._mask_context(context,contract_path,executing=True)


def test_empty_existing_destination_refused(setup,monkeypatch):
    context,contract_path,_,_=setup;cpu_runtime(monkeypatch,Processor())
    output=e3.REPOSITORY_ROOT/'outputs/icra2027/new/fidelity/removal_masks/0000000001'
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError):masks.run_public(context,contract_path)


def test_runtime_failure_is_terminal_and_never_loads_model(setup,monkeypatch):
    context,contract_path,_,_=setup;cpu_runtime(monkeypatch,Processor())
    def fail(_):raise ValueError('checkpoint drift')
    monkeypatch.setattr(masks,'_validate_model',fail)
    monkeypatch.setattr(masks,'_public_processor',lambda _:pytest.fail('unverified model loaded'))
    with pytest.raises(ValueError,match='checkpoint drift'):masks.run_public(context,contract_path)
    failure=e3.REPOSITORY_ROOT/'outputs/icra2027/new/fidelity/removal_masks/0000000001_failure.json'
    assert json.loads(failure.read_text())['error']=='checkpoint drift'
    with pytest.raises(FileExistsError):masks.run_public(context,contract_path)


def test_zero_accepted_objects_preserves_denominator_without_inference(setup,monkeypatch):
    import shutil
    context,contract_path,directory,_=setup
    report=json.loads((directory/'public_prepare.json').read_text())
    report['objects'][0]={'object_slot':'obj_1000','automatic_instance_id':1000,'terminal_action':'reject',
                          'preparation_status':'NOT_APPLICABLE'}
    factory=Path(report['source_factory'])
    write(factory/'objects/obj_1000/aligned.json',{'terminal_action':'reject','rejected':'invalid'})
    shutil.rmtree(directory/'obj_1000')
    write(directory/'public_prepare.json',report);refresh_preparation(context,directory)
    contract(contract_path,context,'new','b'*40);cpu_runtime(monkeypatch,Processor())
    monkeypatch.setattr(masks,'_public_processor',lambda _:pytest.fail('no accepted objects loaded model'))
    result=masks.run_public(context,contract_path)
    assert result['planned_objects']==2 and result['planned_views']==0 and result['model_loads']==0
    assert masks.validate_public_masks(e3.REPOSITORY_ROOT/'outputs/icra2027/new/fidelity/removal_masks/0000000001')['rows']==[]


@pytest.mark.parametrize('kind',['checkpoint','input_rgb'])
def test_mid_inference_drift_is_not_sealed(setup,monkeypatch,kind):
    context,contract_path,_,image=setup
    class DriftingProcessor(Processor):
        def set_text_prompt(self,**kwargs):
            result=super().set_text_prompt(**kwargs)
            if kind=='input_rgb':Image.new('RGB',(16,16)).save(image)
            return result
    cpu_runtime(monkeypatch,DriftingProcessor())
    calls=[]
    def verify_model(_):
        calls.append(1)
        if kind=='checkpoint' and len(calls)>1:raise ValueError('checkpoint changed during inference')
    monkeypatch.setattr(masks,'_validate_model',verify_model)
    with pytest.raises(ValueError):masks.run_public(context,contract_path)
    parent=e3.REPOSITORY_ROOT/'outputs/icra2027/new/fidelity/removal_masks'
    assert not (parent/'0000000001/seal.json').exists()
    assert (parent/'0000000001_failed_partial/obj_1000/mask_0.png').is_file()
    failure=json.loads((parent/'0000000001_failure.json').read_text())
    assert failure['status']=='FAILED' and len(failure['completed_rows'])==1
