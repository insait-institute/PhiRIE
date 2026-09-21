"""Source-bound accounting for completion of an existing native DEV pilot.

This is a development evidence receipt, consumed by the canonical table producer.
It does not turn the example pilot into an independent TEST cohort.
"""
import argparse
import hashlib
import json
from pathlib import Path
from robo.manifest.hash import canonical_hash,git_snapshot


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(pilot,pairs,build_failures):
    pilot=Path(pilot)
    manifest=json.loads((pilot/'run_manifest.json').read_text())
    seeds=manifest['config']['reset_seeds']
    if set(pairs)&set(build_failures) or set(pairs)|set(build_failures)!=set(seeds):
        raise ValueError('continuation must account for each original planned seed exactly once')
    rows=[]
    for seed in seeds:
        refpath=pilot/f'episode_seed{seed}/result.json';ref=json.loads(refpath.read_text())
        for arm,directory in [('REF_NATIVE',refpath.parent),('B0_FIXED_NATIVE',pairs.get(seed))]:
            if directory is not None:
                path=Path(directory)/'result.json';r=json.loads(path.read_text())
                if r.get('execution_kind')!='closed_loop_visual_policy' or r.get('reset_seed')!=seed:
                    raise ValueError('source is not the declared native policy reset')
                if r['config_sha256']!=ref['config_sha256'] or canonical_hash(r['policy_identity'])!=canonical_hash(ref['policy_identity']):
                    raise ValueError('continuation config/policy differs from paired reference')
                if arm!='REF_NATIVE':
                    pair_path=Path(directory).parent/'pair_receipt.json'
                    provenance=json.loads(pair_path.read_text())['reference']
                    if provenance['reference_files']['result.json']!=sha(refpath):
                        raise ValueError('paired source does not bind the declared native REF')
                    canonical=pilot/f'canonical_seed{seed}'
                    for name in ['scene.xml','canonical_state.json']:
                        if provenance['canonical_files'][name]!=sha(canonical/name):
                            raise ValueError('paired canonical instance differs')
                rows.append({'seed':seed,'arm':arm,'source_path':str(path.resolve()),'source_sha256':sha(path),
                    'executed':r['executed'],'native_success':r['success'],'service_success':r['success'] is True,
                    'ticks':r['ticks'],'horizon':r['horizon'],'error':r['error'],
                    'status':'completed' if r['error'] is None else 'execution_failure',
                    'config_sha256':r['config_sha256'],
                    'checkpoint_receipt_sha256':r['policy_identity']['checkpoint_receipt_sha256']})
            else:
                path=Path(build_failures[seed]);failure=json.loads(path.read_text())
                if failure.get('status')!='construction_unavailable' or failure.get('built_objects')!=0:
                    raise ValueError('unexecuted method unit requires an actual construction failure receipt')
                rows.append({'seed':seed,'arm':arm,'source_path':str(path.resolve()),'source_sha256':sha(path),
                    'executed':False,'native_success':None,'service_success':False,'ticks':None,
                    'horizon':ref['horizon'],'status':'method_build_failure','failure_phase':failure['phase']})
    summary={}
    for arm in ['REF_NATIVE','B0_FIXED_NATIVE']:
        group=[r for r in rows if r['arm']==arm];n=sum(r['executed'] for r in group);s=sum(r['service_success'] for r in group)
        summary[arm]={'planned':len(group),'executed':n,'successes':s,'success_per_planned':s/len(group),
            'success_per_executed':s/n if n else None,'execution_coverage':n/len(group),
            'method_build_failures':sum(r['status']=='method_build_failure' for r in group)}
    return {'schema_version':1,'scope':'original DEV native pilot continuation, one original reset per instance; legacy v1 RNG',
        'source_code':git_snapshot(),'pilot_manifest_sha256':sha(pilot/'run_manifest.json'),
        'rows':rows,'summary':summary,'full_accounting':len(rows)==2*len(seeds),
        'limitations':['Binding-specific raw native success; generated scorer correspondence not established.',
            'L0 target-only oracle room/destination context; ideal RGB-D; uniform-color reconstructed native meshes.',
            'DEV continuation, not held-out population retention.',
            'Reference and original seed0 used H200; continuation uses A6000 with same checkpoint/RNG protocol.']}


def mappings(values):
    result={}
    for text in values:
        seed,path=text.split('=',1);seed=int(seed)
        if seed in result:raise ValueError('duplicate seed binding')
        result[seed]=Path(path)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',required=True)
    p.add_argument('--pair',action='append',default=[],help='SEED=episode-directory')
    p.add_argument('--build-failure',action='append',default=[],help='SEED=build_failure.json')
    p.add_argument('--out',required=True);a=p.parse_args(argv)
    report=aggregate(a.pilot,mappings(a.pair),mappings(a.build_failure));path=Path(a.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report['summary'],indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
