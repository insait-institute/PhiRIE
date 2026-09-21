import copy
import json
from types import SimpleNamespace as NS

import pytest

from robo.manifest.hash import canonical_hash
from robo.roundtrip.paired import load_reference_bundle, prepare_paired_adapter, run_paired_episode


@pytest.fixture
def reference(tmp_path):
    config={'reset_seeds':[0,1], 'horizon':600}
    canonical=tmp_path/'canonical_seed0'; canonical.mkdir()
    episode=tmp_path/'episode_seed0'; episode.mkdir()
    state={'integration_state':[1,2], 'controller_state':[], 'observable_timing':{},
           'native_metadata':{'object':'canonical_object'}, 'object_states':{'obj':[0,0,0,1,0,0,0]}}
    policy={'checkpoint_receipt_sha256':'official-pinned', 'rng_protocol':'per-chunk'}
    result={'execution_kind':'closed_loop_visual_policy','treatment_id':'REF_NATIVE',
            'config_sha256':canonical_hash(config),'reset_seed':0,'policy_identity':policy,'success':False}
    for directory,name,value in [(canonical,'canonical_state.json',state),
        (episode,'initial_state.json',state),(episode,'result.json',result),(episode,'actions.json',[[0]*12])]:
        (directory/name).write_text(json.dumps(value))
    (canonical/'scene.xml').write_text('<canonical-native/>')
    return config,canonical,episode,policy


def bundle(ref):
    config,canonical,episode,_=ref
    return load_reference_bundle(episode,canonical,config=config,reset_seed=0)


def test_reference_failure_retained_and_canonical_hashes_recorded(reference):
    b=bundle(reference)
    assert b['result']['success'] is False
    assert len(b['provenance']['canonical_files']['scene.xml'])==64
    assert b['provenance']['reference_result_has_independent_xml_hash'] is False


@pytest.mark.parametrize('drift',['state','config','seed','sibling','execution_kind'])
def test_canonical_reference_drift_rejected(reference,drift):
    config,canonical,episode,_=reference
    if drift=='state':
        p=canonical/'canonical_state.json';s=json.loads(p.read_text());s['integration_state']=[3,4];p.write_text(json.dumps(s))
    elif drift=='config':config['horizon']=599
    elif drift=='seed':config['reset_seeds']=[1]
    elif drift=='sibling':canonical.rename(canonical.parent/'canonical_seed1');canonical=canonical.parent/'canonical_seed1'
    else:
        p=episode/'result.json';s=json.loads(p.read_text());s['execution_kind']='fixed_action_replay';p.write_text(json.dumps(s))
    with pytest.raises(ValueError):load_reference_bundle(episode,canonical,config=config,reset_seed=0)


def test_import_uses_canonical_xml_and_state_never_fresh_reset(monkeypatch,reference):
    import robo.roundtrip.importers.robocasa as importer
    b=bundle(reference);calls=[]
    class Adapter:
        native=NS(objects={'obj':NS(joints=['WRONG_FRESH_JOINT'],root_body='WRONG_FRESH_BODY')})
        def reset_from_spec(self,reset):calls.append(('reset',reset))
        def get_state(self):raise AssertionError('must not use fresh state')
        def source_xml(self):raise AssertionError('must not use fresh XML')
        def import_xml(self,xml,**kwargs):
            calls.append(('import',xml,copy.deepcopy(kwargs)))
            self.native.objects['obj']=NS(joints=['canonical_joint'],root_body='canonical_body')
    def reconstruct(xml,**kwargs):
        assert xml=='<canonical-native/>' and kwargs['body_name']=='canonical_body'
        return '<generated/>',{'position_m':[.2,.3,.4],'quaternion_wxyz':[1,0,0,0]}
    monkeypatch.setattr(importer,'import_reconstructed_object',reconstruct)
    monkeypatch.setattr(importer,'rebind_native_object',lambda *a,**k:{'native_scorer_binding':'BOUND'})
    receipt=prepare_paired_adapter(Adapter(),b,object_dir='frozen-build')
    assert calls[1][1]=='<canonical-native/>' and calls[2][1]=='<generated/>'
    assert calls[1][2]['canonical_state']==calls[2][2]['canonical_state']==b['state']
    assert calls[2][2]['replaced_joint']=='canonical_joint'
    assert calls[2][2]['estimated_pose']==[.2,.3,.4,1,0,0,0]
    assert receipt['binding']['native_scorer_binding']=='BOUND'


def test_checkpoint_drift_rejected_before_output_or_environment_mutation(reference,tmp_path):
    b=bundle(reference);policy=NS(is_visual_policy=True,metadata={'checkpoint':'wrong'})
    out=tmp_path/'paired'
    with pytest.raises(ValueError,match='policy/checkpoint/runtime'):
        run_paired_episode(object(),policy,bundle=b,config=reference[0],object_dir='build',out_dir=out)
    assert not out.exists()


def test_same_policy_and_full_native_config_reach_canonical_harness(monkeypatch,reference,tmp_path):
    import robo.roundtrip.paired as paired
    import robo.eval.harness_runner as harness
    b=bundle(reference);policy=NS(is_visual_policy=True,metadata=reference[3]);adapter=object()
    monkeypatch.setattr(paired,'prepare_paired_adapter',lambda *a,**k:{'scope':'target_only'})
    def run(a,p,**kwargs):
        assert a is adapter and p is policy
        assert kwargs['config']==reference[0] and kwargs['reset_seed']==0
        assert kwargs['execution_kind']=='closed_loop_visual_policy'
        result={'executed':True,'success':False,'error':None}
        kwargs['out_dir'].mkdir();(kwargs['out_dir']/'result.json').write_text(json.dumps(result))
        return result
    monkeypatch.setattr(harness,'run_native_episode',run)
    result=run_paired_episode(adapter,policy,bundle=b,config=reference[0],object_dir='build',out_dir=tmp_path/'paired')
    receipt=json.loads((tmp_path/'paired'/'pair_receipt.json').read_text())
    assert result['success'] is False and receipt['reference_success'] is False
    assert receipt['planned_comparison_episodes']==receipt['executed_comparison_episodes']==1


def test_build_import_failure_retains_planned_episode(monkeypatch,reference,tmp_path):
    import robo.roundtrip.paired as paired
    def fail(*a,**k):raise ValueError('invalid collision mesh')
    monkeypatch.setattr(paired,'prepare_paired_adapter',fail)
    with pytest.raises(ValueError,match='invalid collision'):
        run_paired_episode(object(),NS(is_visual_policy=True,metadata=reference[3]),bundle=bundle(reference),
                           config=reference[0],object_dir='build',out_dir=tmp_path/'paired')
    receipt=json.loads((tmp_path/'paired'/'pair_receipt.json').read_text())
    assert receipt['state']=='FAILED' and receipt['planned_comparison_episodes']==1
    assert receipt['executed_comparison_episodes']==0
