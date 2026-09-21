"""CPU/numerical tests only. Fixtures are NOT simulator-performance evidence."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from robo.campaign import finalize as f
from robo.campaign.core import load, rows, save, receipt, digest
from robo.eval.final_release import validate_block, release
from robo.rendering.final_observation import Packet, Snapshot, StateBoundObserver

ROOT = Path(os.environ.get("SIMANY_TEST_REPO_ROOT", Path(__file__).resolve().parents[1]))


def put(path, value, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(('\n'.join(json.dumps(v) for v in value) if jsonl else json.dumps(value))+'\n')
    return receipt(path)


@pytest.fixture
def sealed(tmp_path):
    p=load(ROOT/'configs/experiments/final_submission/protocol.yaml')
    p.update(layouts=2, instances_per_task=2, resets=2)
    source=put(tmp_path/'source.json',{'official_source_index': True})
    pool=[]
    for layout in range(3):
        for task in f.TASKS:
            for n in range(2):
                cid=f'l{layout}-{task}-{n}'
                pool.append(dict(case_id=cid,canonical_instance_id='native-'+cid,layout_id=layout,
                    task_id=task,dataset='robocasa',split='test',prior_outcomes_seen=False,source_index=source))
    pp=tmp_path/'protocol.json';put(pp,p)
    poolp=tmp_path/'pool.jsonl';put(poolp,pool,True)
    selected=tmp_path/'selected';f.select(str(poolp),str(pp),str(selected))
    r=dict(frozen_before_outcomes=True,test_tuning=False,generator='trellis',segmentation='sam3',
           inpainting='telea',harmonizer='state',calibration='public_marker',policy='frozen',
           model_lock=source,dev_selection=source,policy_contract=source)
    rp=tmp_path/'recipe.json';put(rp,r)
    dest=tmp_path/'sealed';f.freeze(str(selected/'cases.jsonl'),str(pp),str(rp),str(dest))
    return dest,pp,poolp,rp


def block_fixture(root, block, *, status='RECORDED', engine='e1'):
    root.mkdir(parents=True,exist_ok=True)
    plan=[];ledger=[];bindings=[]
    control=put(root/'control.json',{'frozen':True})
    for i,u in enumerate(block['units']):
        uid='n'+str(i)
        p={k:u[k] for k in ('canonical_instance_id','task_id','policy_rng_seed','native_horizon',
              'controller_method','scope','sensor_regime','renderer','split','reset_id','layout_id')}
        p.update(unit_id=uid,cohort_id='native-final',policy_id='frozen-policy')
        plan.append(p)
        success=u['arm_id']=='REF_NATIVE'
        current='RECORDED' if u['arm_id']=='REF_NATIVE' and status in ('BUILD_FAILED','ABSTAINED') else status
        row=dict(p,terminal_status=current,executed=True,success=success)
        b=dict(final_unit_id=u['final_unit_id'],native_unit_id=uid)
        if current=='RECORDED':
            result={'executed':True,'success':success,'error':None,'execution_kind':'closed_loop_visual_policy',
                'policy_identity':{'policy_engine':{'engine_id':engine,'process_uuid':'uuid'},'checkpoint':'frozen'}}
            rr=put(root/f'{uid}/result.json',result)
            row.update(result_path=rr['path'],result_sha256=rr['sha256'])
            execution={'native_unit_id':uid,'result_sha256':rr['sha256'],'recipe_sha256':block['recipe_sha256']}
            execution.update({k:control for k in ('policy_contract','camera_contract','task_contract','reset_contract','physics')})
            execution['state_sync']=put(root/f'{uid}/sync.json',{'passed':True,'measurement_kind':'real_runtime',
                    'frames_checked':2,'max_state_lag_ticks':0,'silent_fallback_frames':0})
            if u['arm_id'] in ('SIMFOUNDRY_ADAPTED','POLARIS_ADAPTED'):
                execution['official_method']=put(root/f'{uid}/official.json',{
                    'method':'SimFoundry' if u['arm_id'].startswith('SIMFOUNDRY') else 'PolaRiS',
                    'adaptation':'common_importer','upstream_commit':'a'*40,'manual_minutes':0,'acquisition':'test_fixture_only'})
            b['execution_contract']=put(root/f'{uid}/execution.json',execution)
        elif current in ('BUILD_FAILED','ABSTAINED'):
            row.update(executed=False,success=None)
            b['failure_evidence']=put(root/f'{uid}/failure.json',{'native_unit_id':uid,'classification':current})
        else:row.update(executed=None if status=='NOT_SCHEDULED' else False,success=None)
        ledger.append(row);bindings.append(b)
    cp=put(root/'contract.json',block)
    index={'schema':f.SCHEMA,'block_id':block['block_id'],'contract':cp,
        'native_plan':put(root/'planned.jsonl',plan,True),'native_ledger':put(root/'ledger.jsonl',ledger,True),
        'unit_bindings':put(root/'bindings.jsonl',bindings,True)}
    return index


def first_block(sealed,suite='system'):
    root=sealed[0];p=load(root/'plan.json')
    ent=next(b for b in p['blocks'] if load(b['contract']['path'])['suite']['id']==suite)
    return ent,load(ent['contract']['path'])


def rebind(index, key, value, jsonl=False):
    index[key]=put(Path(index[key]['path']),value,jsonl)


def test_final_counts():
    p=load(ROOT/'configs/experiments/final_submission/protocol.yaml')
    f.validate_protocol(p)
    counts=[(24 if s['membership']=='all' else 12)*10*len(s['arms']) for s in p['suites']]
    assert counts==[960,720,360] and sum(counts)==2040


def test_deterministic_selection(sealed,tmp_path):
    root,pp,poolp,rp=sealed
    original=rows(root/'cases.jsonl')
    put(poolp,list(reversed(rows(poolp))),True)
    f.select(str(poolp),str(pp),str(tmp_path/'other'))
    assert rows(tmp_path/'other/cases.jsonl')==original


def test_freeze_counts_and_immutable(sealed):
    root,pp,poolp,rp=sealed;p=f.verify(root)
    assert p['planned_blocks']==16 and p['planned_units']==136
    with pytest.raises(FileExistsError):f.freeze(str(root/'cases.jsonl'),str(pp),str(rp),str(root))


def test_source_mutation(sealed):
    root=sealed[0];p=load(root/'plan.json')
    Path(p['recipe_dependencies']['model_lock']['path']).write_text('changed')
    with pytest.raises(ValueError,match='identity'):f.verify(root)


@pytest.mark.parametrize('key',['success','psnr','build_pass','quality'])
def test_no_outcome_selection(sealed,tmp_path,key):
    _,pp,poolp,_=sealed;rr=rows(poolp);rr[0][key]=True;put(poolp,rr,True)
    with pytest.raises(ValueError,match='outcomes'):f.select(str(poolp),str(pp),str(tmp_path/'bad'))


def test_insufficient_source_not_silent_shrink(sealed,tmp_path):
    _,pp,poolp,_=sealed;put(poolp,rows(poolp)[:1],True)
    with pytest.raises(ValueError,match='insufficient'):f.select(str(poolp),str(pp),str(tmp_path/'bad'))


def test_empty_release_preserves_roster(sealed,tmp_path):
    result=release(str(sealed[0]),[],str(tmp_path/'release'))
    assert result['status']=='PARTIAL' and result['planned_units']==136 and result['measured_units']==0
    t=list(__import__('csv').DictReader((tmp_path/'release/native_tables.csv').open()))
    assert len(t)==13 and all(r['success_per_planned']=='' for r in t)


@pytest.mark.parametrize('status',['RECORDED','BUILD_FAILED','ABSTAINED','NOT_SCHEDULED','RESOURCE_FAILED'])
def test_status_accounting(sealed,tmp_path,status):
    _,b=first_block(sealed);index=block_fixture(tmp_path/'b',b,status=status)
    values=validate_block(b,index)
    assert len(values)==len(b['units'])
    assert all(r['measured']==(status in ('RECORDED','BUILD_FAILED','ABSTAINED')) for r in values)
    if status in ('BUILD_FAILED','ABSTAINED'):assert all(r['service_success'] is False and r['success'] is None for r in values if r['arm_id']!='REF_NATIVE')


def test_missing_native_row_is_unknown(sealed,tmp_path):
    _,b=first_block(sealed);index=block_fixture(tmp_path/'b',b)
    rr=rows(index['native_ledger']['path']);rr=rr[1:];rebind(index,'native_ledger',rr,True)
    # Other executed arms may not borrow an absent reference.
    with pytest.raises(ValueError,match='reference'):validate_block(b,index)


def test_do_not_drop_build_failure_denominator(sealed,tmp_path):
    _,b=first_block(sealed);idx=block_fixture(tmp_path/'b',b)
    rr=rows(idx['unit_bindings']['path']);rebind(idx,'unit_bindings',rr[:-1],True)
    with pytest.raises(ValueError,match='ENTIRE'):validate_block(b,idx)


def test_result_bytes_changed(sealed,tmp_path):
    _,b=first_block(sealed);idx=block_fixture(tmp_path/'b',b)
    rr=rows(idx['native_ledger']['path']);Path(rr[0]['result_path']).write_text('{}')
    with pytest.raises(ValueError,match='identity'):validate_block(b,idx)


def test_cross_process_rejected(sealed,tmp_path):
    _,b=first_block(sealed);idx=block_fixture(tmp_path/'b',b)
    rr=rows(idx['native_ledger']['path']);r=load(rr[1]['result_path']);r['policy_identity']['policy_engine']['engine_id']='other'
    new=put(Path(rr[1]['result_path']),r);rr[1]['result_sha256']=new['sha256'];rebind(idx,'native_ledger',rr,True)
    bs=rows(idx['unit_bindings']['path']);x=load(bs[1]['execution_contract']['path']);x['result_sha256']=new['sha256']
    bs[1]['execution_contract']=put(Path(bs[1]['execution_contract']['path']),x);rebind(idx,'unit_bindings',bs,True)
    with pytest.raises(ValueError,match='processes'):validate_block(b,idx)


def test_observation_physics_unchanged(sealed,tmp_path):
    _,b=first_block(sealed,'observation');idx=block_fixture(tmp_path/'b',b)
    bs=rows(idx['unit_bindings']['path']);x=load(bs[2]['execution_contract']['path'])
    x['physics']=put(tmp_path/'different_physics.json',{'changed':True})
    bs[2]['execution_contract']=put(Path(bs[2]['execution_contract']['path']),x);rebind(idx,'unit_bindings',bs,True)
    with pytest.raises(ValueError,match='physical'):validate_block(b,idx)


def test_stale_frame_evidence_rejected(sealed,tmp_path):
    _,b=first_block(sealed,'observation');idx=block_fixture(tmp_path/'b',b)
    bs=rows(idx['unit_bindings']['path']);x=load(bs[2]['execution_contract']['path'])
    sync=load(x['state_sync']['path']);sync['max_state_lag_ticks']=1
    x['state_sync']=put(Path(x['state_sync']['path']),sync)
    bs[2]['execution_contract']=put(Path(bs[2]['execution_contract']['path']),x);rebind(idx,'unit_bindings',bs,True)
    with pytest.raises(ValueError,match='current-state'):validate_block(b,idx)


def test_complete_release_uses_existing_bootstrap(sealed,tmp_path):
    root=sealed[0];pp=load(root/'plan.json');paths=[]
    for i,entry in enumerate(pp['blocks']):
        b=load(entry['contract']['path']);idx=block_fixture(tmp_path/f'block{i}',b)
        idx['contract']=entry['contract'];path=tmp_path/f'index{i}.json';put(path,idx);paths.append(str(path))
    summary=release(str(root),paths,str(tmp_path/'final'))
    assert summary['native_data_complete'] and summary['executed_episodes']==136
    assert summary['paper_ready']=='REQUIRES_SCIENTIFIC_AND_LAYOUT_REVIEW'
    c=load(tmp_path/'final/comparisons.json')
    assert all(x['paired_complete'] for x in c) and all(x['positive_improvement_supported'] is False for x in c)


def test_unknown_or_duplicate_result_index(sealed,tmp_path):
    root=sealed[0];ent,b=first_block(sealed);idx=block_fixture(tmp_path/'b',b);idx['contract']=ent['contract']
    path=tmp_path/'i.json';put(path,idx)
    with pytest.raises(ValueError,match='repeated'):release(str(root),[str(path),str(path)],str(tmp_path/'r'))


def observer(mode='gaussian',enhance=None,constrain=None):
    state=[Snapshot('ep',0,'a')]
    def render(cam,s):
        return Packet(s,np.full((4,5,3),20,dtype=np.uint8),np.eye(4,5,dtype=bool),{})
    return StateBoundObserver(['left','wrist'],lambda:state[0],render,mode=mode,enhance=enhance,constrain=constrain),state


def test_observer_cached_reads_not_temporal_updates():
    o,s=observer();a=o.observe();a['left'][:]=0;b=o.observe()
    assert b['left'].min()==20 and len(o.events)==1


def test_observer_restores_exact_visible_robot_core():
    o,s=observer('robot_restore_h',lambda *args:np.full((4,5,3),100,dtype=np.uint8))
    image=o.observe()['left'];assert (image[np.eye(4,5,dtype=bool)]==20).all()
    assert (image[~np.eye(4,5,dtype=bool)]==100).all()


def test_observer_no_missing_enhancer_fallback():
    with pytest.raises(ValueError,match='enhancer'):observer('official_h')


def test_observer_temporal_reset():
    histories=[]
    def enhance(cam,raw,prev,buf):histories.append(prev);return raw
    o,s=observer('official_h',enhance);o.observe();s[0]=Snapshot('ep',1,'b');o.observe()
    assert histories[-1] is not None
    s[0]=Snapshot('new',0,'c');o.observe();assert histories[-1] is None


def test_observer_same_tick_change_rejected():
    o,s=observer();o.observe();s[0]=Snapshot('ep',0,'changed')
    with pytest.raises(ValueError,match='same/backward'):o.observe()


def test_observer_physics_advance_rollback():
    o,s=observer('official_h',lambda *args:np.full((4,5,3),100,dtype=np.uint8))
    def bad(cam,raw,prev,buf):s[0]=Snapshot('ep',1,'bad');return raw
    o.enhance=bad
    with pytest.raises(ValueError,match='advanced'):o.observe()
    assert not o.previous and not o.cache and o.summary()['failed_observation_requests']==1


def test_observer_stale_packet_rejected():
    o,s=observer();original=o.render
    def stale(cam,snap):p=original(cam,snap);p.snapshot=Snapshot('other',0,'bad');return p
    o.render=stale
    with pytest.raises(ValueError,match='stale'):o.observe()


def test_observer_second_camera_failure_rolls_back_all():
    def enhance(cam,raw,prev,buf):
        if cam=='wrist':raise RuntimeError('service failed')
        return raw
    o,s=observer('official_h',enhance)
    with pytest.raises(RuntimeError):o.observe()
    assert not o.previous and not o.cache


def test_observer_wrong_canvas():
    o,s=observer('official_h',lambda *a:np.zeros((1,1,3),dtype=np.uint8))
    with pytest.raises(ValueError,match='canvas'):o.observe()


def test_observer_state_mode_calls_corrector():
    o,s=observer('state_h',lambda cam,raw,*a:raw,lambda cam,raw,enh,*a:np.clip(enh+2,0,255).astype('uint8'))
    im=o.observe()['left'];assert im.max()==22 and im.min()==20


def test_dispatch_global_not_per_bundle(tmp_path,monkeypatch):
    from robo.campaign import runner
    calls=[]
    bundles=[]
    for i in range(2):
        b=tmp_path/f'b{i}';put(b/'plan.json',{})
        put(b/'jobs/a/intent.json',{});put(b/'jobs/a/submission.json',{'returncode':0,'job_id':str(i+10)})
        bundles.append(str(b))
    path=tmp_path/'registry.json';put(path,{'max_active_jobs':2,'bundles':bundles})
    monkeypatch.setattr(f.subprocess,'check_output',lambda *a,**k:'10\n11\n')
    monkeypatch.setattr(runner,'launch',lambda *a,**k:calls.append(k) or [])
    monkeypatch.setenv('USER','tester')
    assert f.dispatch(str(path),submit=True,max_jobs=4)==[] and calls==[]


def test_dispatch_ambiguous_intent_stops(tmp_path,monkeypatch):
    b=tmp_path/'bundle';put(b/'plan.json',{});put(b/'jobs/a/intent.json',{})
    path=tmp_path/'registry.json';put(path,{'max_active_jobs':2,'bundles':[str(b)]})
    monkeypatch.setattr(f.subprocess,'check_output',lambda *a,**k:'')
    monkeypatch.setenv('USER','tester')
    with pytest.raises(RuntimeError,match='uncertain'):f.dispatch(str(path),submit=True)


def test_policy_adapter_only_replaces_images():
    from robo.rendering.final_observation import PolicyObservationAdapter
    class Native:
        def get_policy_observation(self):
            return {'left_rgb':np.zeros((4,5,3),dtype=np.uint8),
                    'wrist_rgb':np.zeros((4,5,3),dtype=np.uint8),'joint_state':np.ones(7)}
        def native_success(self):return True
    o,s=observer();n=Native()
    wrapper=PolicyObservationAdapter(n,o,{'left':'left_rgb','wrist':'wrist_rgb'},lambda c,i:i)
    value=wrapper.get_policy_observation()
    assert value['left_rgb'].min()==20 and np.array_equal(value['joint_state'],np.ones(7))
    assert wrapper.native_success() is True and n.get_policy_observation()['left_rgb'].max()==0


def test_policy_adapter_preprocessing_contract():
    from robo.rendering.final_observation import PolicyObservationAdapter
    class Native:
        def get_policy_observation(self):
            return {k:np.zeros((4,5,3),dtype=np.uint8) for k in ('left','wrist')}
    o,s=observer();wrapper=PolicyObservationAdapter(Native(),o,{'left':'left','wrist':'wrist'},lambda c,i:i.astype(float))
    with pytest.raises(ValueError,match='preprocessing'):wrapper.get_policy_observation()


def test_native_admission_requires_actual_dev(tmp_path):
    gate=tmp_path/'gate.json';put(gate,{'passed':True})
    r={'split':'development','executed':True,'success':False,'error':None,
       'execution_kind':'closed_loop_visual_policy','policy_identity':{'policy_engine':{'process_uuid':'p'}}}
    rp=tmp_path/'ref.json';cp=tmp_path/'cmp.json';put(rp,r);put(cp,r)
    result=f.admit(str(gate),str(gate),str(rp),str(cp),str(tmp_path/'admitted'))
    assert load(result['same_engine_gate']['path'])['passed']
    r['split']='test';put(cp,r)
    with pytest.raises(ValueError,match='DEV'):f.admit(str(gate),str(gate),str(rp),str(cp),str(tmp_path/'bad'))
