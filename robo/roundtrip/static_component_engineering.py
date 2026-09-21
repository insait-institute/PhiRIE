"""Generated static destination/native-target engineering control; zero policy episodes."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from robo.roundtrip.identity import file_hash
from robo.roundtrip.matrix import save_new


def run(binding_path,build_path,out,steps=10):
    from robo.roundtrip.adapters.robocasa import RoboCasaAdapter,observation_hashes
    from robo.roundtrip.fixture_scope import import_fixture_component,bind_fixture_component
    from robo.roundtrip.scope_contacts import bind_contact_inventory,sample_contacts
    from robo.roundtrip.scope_bundle import artifact_hashes
    from robo.roundtrip.scorer import predicate_components
    out=Path(out);out.mkdir(parents=True,exist_ok=False);b=json.loads(Path(binding_path).read_text());build=json.loads(Path(build_path).read_text())
    if build.get('status')!='BUILT' or build.get('canonical_instance_id')!=b['canonical_instance_id'] or build.get('capture_manifest_sha256')!=b['capture_manifest_sha256']:
        raise ValueError('static engineering requires matching sealed actual build/capture')
    p=Path(build['object_dir']);p=Path(build_path).parent/p.relative_to('/output') if p.is_relative_to('/output') else p
    for name,digest in artifact_hashes(p).items():
        if build['source_hashes'].get(str((p/name).relative_to(Path(build_path).parent)))!=digest:raise ValueError('generated static component changed')
    c=json.loads(Path(b['native_config']).read_text());task=c['instance']['task_id']
    if task not in ['PickPlaceCounterToCabinet','PickPlaceCounterToSink']:raise ValueError('unknown static fixture role')
    adapter=RoboCasaAdapter(c);started=time.monotonic();trace=[]
    try:
        adapter.reset_from_spec({'seed':c['reset_seeds'][0]})
        canonical=Path(b['bundle_dir']);state=json.loads((canonical/'canonical_state.json').read_text());source=(canonical/'scene.xml').read_text()
        adapter.import_xml(source,canonical_state=state)
        fixture=getattr(adapter.native,'cab' if task=='PickPlaceCounterToCabinet' else 'sink')
        source_observation=observation_hashes(adapter.get_policy_observation())
        xml,receipt=import_fixture_component(source,body_name=fixture.root_body,object_dir=p,object_id='destination',role=build['object_role'],component_kind=build['component_kind'])
        adapter.import_xml(xml,canonical_state=state,fixture_components=[receipt])
        fixture=getattr(adapter.native,'cab' if task=='PickPlaceCounterToCabinet' else 'sink')
        binding=bind_fixture_component(adapter.native,fixture_name=fixture.name,receipt=receipt)
        unchanged=np.array_equal(np.asarray(adapter.get_state()['qpos']),np.asarray(state['qpos']))
        if not unchanged:raise ValueError('generated static component changed initial native/robot qpos')
        inventory=bind_contact_inventory(adapter.native.sim.model._model,[receipt])
        rendered=adapter.get_policy_observation();initial_render_hashes=observation_hashes(rendered)
        camera_shapes={k:list(np.asarray(v).shape) for k,v in rendered.items() if k.startswith('video.')}
        if not camera_shapes or any(np.asarray(rendered[k]).dtype!=np.uint8 for k in camera_shapes):raise ValueError('native generated-role renderer invalid')
        for tick in range(steps):
            adapter.step_native_action(np.zeros(12))
            trace.append({'tick':tick,'native_predicates':predicate_components(adapter.native,task),
                'objects':adapter.tracked_objects(),'scope_contacts':sample_contacts(adapter.native.sim.model._model,adapter.native.sim.data._data,inventory)})
        save_new(out/'import_receipt.json',receipt);save_new(out/'binding_receipt.json',binding);save_new(out/'engineering_trace.json',trace)
        result={'status':'PASS','kind':'native_target_generated_static_component_engineering','native_policy_episodes':0,
            'scripted_control_steps':steps,'generated_target_available':False,'oracle_native_target_retained':True,
            'L1_reconstructed_target_and_destination_measured':False,'native_predicate_unchanged':True,
            'pre_step_qpos_byte_equal':unchanged,'renderer_camera_shapes':camera_shapes,'before_render_hashes':source_observation,
            'after_render_hashes':initial_render_hashes,'binding_sha256':file_hash(binding_path),'build_sha256':file_hash(build_path),
            'elapsed_s':time.monotonic()-started,'full_scope_or_native_success_claim':'NOT_RUN'}
        save_new(out/'result.json',result);return result
    except Exception as exc:
        save_new(out/'failure.json',{'type':type(exc).__name__,'reason':str(exc),'native_policy_episodes':0,'scripted_steps_completed':len(trace)});raise
    finally:adapter.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['binding','build','out']:p.add_argument('--'+n,required=True)
    a=p.parse_args();run(a.binding,a.build,a.out)
if __name__=='__main__':main()
