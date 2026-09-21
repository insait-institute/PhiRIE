"""Native RoboCasa reference/controls. Every execution phase creates a new output."""
import argparse
import json
from pathlib import Path
import numpy as np
from robo.manifest.hash import canonical_hash, git_snapshot
from robo.roundtrip.spec import load_spec


def save(path,value):
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value,indent=2,default=convert,allow_nan=False)+'\n')


def compare(a,b):
    keys=sorted(set(a)|set(b));diff={}
    for k in keys:
        if k not in a or k not in b:diff[k]={'missing':True};continue
        x,y=a[k],b[k]
        if isinstance(x,str):
            if x!=y:diff[k]={'different':True}
        elif not np.array_equal(x,y):
            diff[k]={'max_abs':float(np.max(np.abs(np.asarray(x,dtype=float)-np.asarray(y,dtype=float))))}
    return diff


def identity(c,out):
    from robo.roundtrip.adapters.robocasa import RoboCasaAdapter
    from robo.roundtrip.importers.robocasa import import_identity
    a=RoboCasaAdapter(c);b=RoboCasaAdapter(c);seed=c['reset_seeds'][0]
    try:
        a.reset_from_spec({'seed':seed});b.reset_from_spec({'seed':seed})
        # Official reference replay binding; U0 does not invoke the asset importer.
        b.import_xml(a.source_xml(),canonical_state=a.get_state())
        save(out/'u0_reset_audit.json', {name:{'state':adapter.get_state(),
            'instruction':adapter.get_policy_observation()['annotation.human.task_description'],
            'native_rng':adapter.native.rng.bit_generator.state,
            'source_xml_sha256':__import__('hashlib').sha256(adapter.source_xml().encode()).hexdigest()}
            for name,adapter in [('a',a),('b',b)]})
        initial_diff=compare(a.get_policy_observation(),b.get_policy_observation())
        initial_state_equal=compare(a.get_state()['object_states'],b.get_state()['object_states'])=={}
        actions=[]
        for i in range(10):
            action=np.zeros(12);action[6]=0.;action[11]=0.
            action[0]=.01*np.sin(i);actions.append(action)
        u0=[]
        for action in actions:
            # Control reaches the untouched official Gym directly on A; adapter on B.
            from robocasa.utils.env_utils import convert_action
            obs,_,_,_,_=a.env.step(convert_action(action));a.observation=obs
            b.step_native_action(action)
            u0.append({'observation_diff':compare(obs,b.get_policy_observation()),
                'state_max_abs':float(np.max(np.abs(a.native.sim.data.qpos-b.native.sim.data.qpos))),
                'predicate_equal':a.native_success()==b.native_success()})
        # Reset each to the same canonical initial state before lossless import.
        a.close();b.close();a=RoboCasaAdapter(c);b=RoboCasaAdapter(c)
        a.reset_from_spec({'seed':seed});b.reset_from_spec({'seed':seed})
        state=a.get_state();xml=a.source_xml();body=a.native.objects['obj'].root_body
        converted,receipt=import_identity(xml,body_name=body)
        b.import_xml(converted,canonical_state=state)
        u1_initial=compare(a.get_policy_observation(),b.get_policy_observation())
        u1=[]
        for action in actions:
            a.step_native_action(action);b.step_native_action(action)
            u1.append({'observation_diff':compare(a.get_policy_observation(),b.get_policy_observation()),
                'state_max_abs':float(np.max(np.abs(a.native.sim.data.qpos-b.native.sim.data.qpos))),
                'predicate_equal':a.native_success()==b.native_success()})
        report={'initialization':'official native metadata/XML/state replay; seed alone is not an instance ID',
            'initial_observation_diff':initial_diff,'initial_objects_equal':initial_state_equal,
            'u0':u0,'u1_import':receipt,'u1_initial_observation_diff':u1_initial,'u1':u1,
            'tolerance_policy':'initial gate exact; mismatch triggers adapter diagnosis, not threshold relaxation'}
        report['passed']=not initial_diff and initial_state_equal and not u1_initial and all(
            not r['observation_diff'] and r['state_max_abs']==0 and r['predicate_equal'] for r in u0+u1)
        save(out/'identity_report.json',report)
        return report
    finally:a.close();b.close()


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True)
    ap.add_argument('--phase',choices=['inventory','identity','pilot'],required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--host',default='localhost');ap.add_argument('--port',type=int,default=8765)
    args=ap.parse_args(argv);c=load_spec(args.config);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    save(out/'run_manifest.json',{'code':git_snapshot(),'config':c,'config_sha256':canonical_hash(c),'phase':args.phase})
    if args.phase=='identity':return 0 if identity(c,out)['passed'] else 2
    from robo.roundtrip.adapters.robocasa import RoboCasaAdapter
    a=RoboCasaAdapter(c)
    try:
        if args.phase=='inventory':
            a.reset_from_spec({'seed':c['reset_seeds'][0]})
            save(out/'environment_lock.json',a.environment_lock());save(out/'initial_state.json',a.get_state())
            import robocasa
            save(out/'available_tasks.json',sorted(robocasa.ALL_KITCHEN_ENVIRONMENTS))
            frame=a.render_capture({'native_name':'robot0_agentview_left'},width=256,height=256)
            from PIL import Image
            Image.fromarray(frame['rgb']).save(out/'native_initial.png')
            a.export_reference_for_evaluator(out/'reference_vault')
            return 0
        from robo.roundtrip.native_policy import NativePolicy
        from robo.eval.harness_runner import run_native_episode
        p=NativePolicy(args.host,args.port,replan_steps=5)
        # Native pilot outcomes all retained; no reconstruction measurement here.
        results=[]
        for seed in c['reset_seeds']:
            a.close();a=RoboCasaAdapter(c);a.reset_from_spec({'seed':seed})
            a.export_reference_for_evaluator(out/f'canonical_seed{seed}')
            results.append(run_native_episode(a,p,config=c,reset_seed=seed,
                out_dir=out/f'episode_seed{seed}',treatment_id='REF_NATIVE'))
        save(out/'pilot_results.json',results)
        return 0 if any(r['success'] is True for r in results) else 2
    finally:a.close()

if __name__=='__main__':raise SystemExit(main())
