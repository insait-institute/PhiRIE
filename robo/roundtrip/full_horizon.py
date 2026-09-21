"""Bounded legacy-DEV full-H continuation and identical robot-action replay.

Each native REF starts from the original canonical instance, keeps legacy RNG,
continues the same learned policy after first success, then supplies exactly H
recorded robot actions to native-import and reconstructed replay. Primary pilot
outcomes remain immutable and separate from this diagnostic protocol.
"""
import argparse
import copy
import json
from pathlib import Path
from robo.manifest.hash import canonical_hash,git_snapshot
from robo.roundtrip.identity import file_hash


def summarize_episode(directory):
    from robo.eval.episode_log import read_timeseries
    directory=Path(directory);r=json.loads((directory/'result.json').read_text())
    ticks=read_timeseries(directory/'trace.json.gz')
    success=[t['tick']+1 for t in ticks if t['native_predicates']['native_task_success']]
    return dict(result_sha256=file_hash(directory/'result.json'),actions_sha256=file_hash(directory/'actions.json'),
        directory=str(directory),planned_steps=r['horizon'],recorded_steps=len(ticks),
        first_success_step=min(success) if success else None,success_ever=bool(success),
        success_at_horizon=r['success'] if len(ticks)==r['horizon'] and r['error'] is None else None,
        full_horizon_completed=len(ticks)==r['horizon'] and r['error'] is None,error=r['error'],
        execution_kind=r['execution_kind'],native_predicates=r['native_predicates'])


def run_unit(config,pilot,seed,object_dir,public_points,out,host,port):
    from robo.roundtrip.adapters.robocasa import RoboCasaAdapter
    from robo.roundtrip.paired import load_reference_bundle,prepare_paired_adapter
    from robo.roundtrip.native_policy import NativePolicy
    from robo.roundtrip.replay import run_replay_episode,compare_episode_directories,write_public_marker_contract
    from robo.eval.harness_runner import run_native_episode
    from robo.roundtrip.spec import NATIVE_HORIZONS
    if config['schema_version']!=1 or seed not in config['reset_seeds']:
        raise ValueError('bounded continuation requires original v1 DEV reset roster')
    if config['horizon']!=NATIVE_HORIZONS.get(config['instance']['task_id']):
        raise ValueError('full-H must retain official native task horizon')
    pilot=Path(pilot);out=Path(out)
    bundle=load_reference_bundle(pilot/f'episode_seed{seed}',pilot/f'canonical_seed{seed}',config=config,reset_seed=seed)
    policy=NativePolicy(host,port)
    if canonical_hash(policy.metadata)!=canonical_hash(bundle['result']['policy_identity']):
        raise ValueError('learned policy checkpoint/runtime identity differs from original REF')
    diagnostic=copy.deepcopy(config);diagnostic['execution_protocol']='full_horizon_feedback_diagnostic'
    out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    marker=out/'public_marker.json';write_public_marker_contract(public_points,marker)
    receipt=dict(schema_version=1,source_code=git_snapshot(),reset_seed=seed,canonical_reference=bundle['provenance'],
        protocol='full_horizon_feedback_diagnostic',rng_protocol='legacy_v1_seed_unchanged',
        post_success_rule='continue_same_learned_policy_to_native_H',config_sha256=canonical_hash(diagnostic),
        planned_episodes=4,rows=[],claim_scope='binding-specific DEV dynamics diagnostic; absolute correspondence NULL')
    def save():
        p=out/'progress.json';temp=out/'progress.pending';temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(p)
    save()
    for method,generated,replay in [('REF_NATIVE',False,False),('REF_IMPORT_CONTROL',False,True),('B0_FIXED_NATIVE_REPLAY',True,True),('B0_FIXED_NATIVE',True,False)]:
        adapter=RoboCasaAdapter(config);stage=out/method;stage.mkdir()
        try:
            imported=prepare_paired_adapter(adapter,bundle,object_dir=object_dir if generated else None)
            (stage/'import_receipt.json').write_text(json.dumps(imported,indent=2)+'\n')
            if replay:
                reference=out/'REF_NATIVE/episode';ref=summarize_episode(reference)
                if not ref['full_horizon_completed']:raise ValueError('reference did not execute genuine full horizon')
                run_replay_episode(adapter,actions_path=reference/'actions.json',config=diagnostic,reset_seed=seed,
                    out_dir=stage/'episode',treatment_id=method,expected_actions_sha256=ref['actions_sha256'],
                    source_identity={'reference_result_sha256':ref['result_sha256'],'canonical_reference':bundle['provenance']})
                metrics=compare_episode_directories(reference,stage/'episode',frame_contract_path=marker)
                metrics['success_scope']='native predicate after genuine full-H identical robot-action replay'
                (stage/'replay_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
            else:
                run_native_episode(adapter,policy,config=diagnostic,reset_seed=seed,out_dir=stage/'episode',treatment_id=method)
            row=dict(method=method,**summarize_episode(stage/'episode'));receipt['rows'].append(row);save()
            if method=='REF_IMPORT_CONTROL':
                # This exact canonical importer control must preserve the whole
                # native response before generated replay is interpreted.
                from robo.eval.episode_log import read_timeseries
                import numpy as np
                a=read_timeseries(out/'REF_NATIVE/episode/trace.json.gz');b=read_timeseries(stage/'episode/trace.json.gz')
                equal=len(a)==len(b)==config['horizon'] and all(np.array_equal(x['qpos'],y['qpos']) and x['native_predicates']==y['native_predicates'] for x,y in zip(a,b))
                receipt['full_horizon_native_import_control_exact']=bool(equal);save()
                if not equal:raise ValueError('full-H native importer identity diverged; generated replay withheld')
        except Exception as exc:
            receipt['failure']=dict(method=method,error=f'{type(exc).__name__}: {exc}');save();raise
        finally:adapter.close()
    receipt['complete']=len(receipt['rows'])==4;save();return receipt


def aggregate_units(unit_paths):
    """Revalidate every canonical source before aggregating the predeclared DEV units."""
    import numpy as np
    paths=[Path(p) for p in unit_paths]
    if not paths or len(set(p.resolve() for p in paths))!=len(paths):
        raise ValueError('nonempty unique full-H unit roster required')
    rows=[];seeds=set();receipts=[]
    methods={'REF_NATIVE','REF_IMPORT_CONTROL','B0_FIXED_NATIVE_REPLAY','B0_FIXED_NATIVE'}
    for unit in paths:
        progress=unit/'progress.json';r=json.loads(progress.read_text())
        if r['reset_seed'] in seeds or not r.get('complete') or len(r['rows'])!=4 or {x['method'] for x in r['rows']}!=methods:
            raise ValueError('full-H unit incomplete or duplicate reset')
        seeds.add(r['reset_seed']);receipts.append(dict(path=str(progress),sha256=file_hash(progress)))
        reference=unit/'REF_NATIVE/episode';reference_actions=np.asarray(json.loads((reference/'actions.json').read_text()),dtype=np.float64)
        ref=json.loads((reference/'result.json').read_text())
        for row in r['rows']:
            directory=unit/row['method']/'episode';actual=summarize_episode(directory)
            if canonical_hash(dict(method=row['method'],**actual))!=canonical_hash(row):
                raise ValueError('full-H source result changed after completion')
            result=json.loads((directory/'result.json').read_text())
            record=dict(seed=r['reset_seed'],method=row['method'],**actual)
            if result['execution_kind']=='fixed_action_replay':
                identity=result['policy_identity']['source']
                if identity['reference_result_sha256']!=file_hash(reference/'result.json') or identity['actions_sha256']!=file_hash(reference/'actions.json'):
                    raise ValueError('replay source binding differs')
                actions=np.asarray(json.loads((directory/'actions.json').read_text()),dtype=np.float64)
                if actions.tobytes()!=reference_actions.tobytes():raise ValueError('full-H replay robot actions differ')
                metric_path=directory.parent/'replay_metrics.json';metric=json.loads(metric_path.read_text())
                record.update(replay_metrics_path=str(metric_path),replay_metrics_sha256=file_hash(metric_path),
                    relative_displacement_rmse_cm=metric['relative_displacement_rmse_cm'],
                    relative_rotation_increment_mean_deg=metric['relative_rotation_increment_mean_deg'],
                    matched_steps=metric['matched_steps'],position_rmse_cm=None,rotation_error_deg=None)
            elif result['config_sha256']!=ref['config_sha256'] or canonical_hash(result['policy_identity'])!=canonical_hash(ref['policy_identity']):
                raise ValueError('closed-loop treatment changes frozen config or policy')
            rows.append(record)
    counts={}
    for method in sorted(methods):
        group=[x for x in rows if x['method']==method]
        counts[method]=dict(planned=len(paths),executed=sum(x['recorded_steps']>0 for x in group),
            completed_full_horizon=sum(x['full_horizon_completed'] for x in group),
            success_ever=sum(x['success_ever'] for x in group),success_at_horizon=sum(x['success_at_horizon'] is True for x in group))
    return dict(schema_version=1,source_code=git_snapshot(),protocol='full_horizon_feedback_diagnostic',
        source_receipts=receipts,planned_instances=len(paths),planned_episodes=4*len(paths),rows=rows,counts=counts,
        claim_scope='two original DEV instances; binding-specific native success; relative dynamics diagnostics',
        absolute_correspondence=None,feedback_benefit_claim='NOT_RUN; no population inference from two DEV instances')


def main(argv=None):
    from robo.roundtrip.spec import load_spec
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--aggregate-bindings')
    for k in ['config','pilot','object-dir','public-points']:p.add_argument('--'+k)
    p.add_argument('--seed',type=int);p.add_argument('--host',default='hala');p.add_argument('--port',type=int,default=8017)
    a=p.parse_args(argv)
    if a.aggregate_bindings:
        bindings=json.loads(Path(a.aggregate_bindings).read_text());r=aggregate_units(bindings['units'])
        r['bindings_sha256']=file_hash(a.aggregate_bindings);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
    else:
        if any(getattr(a,k) is None for k in ['config','pilot','seed','object_dir','public_points']):p.error('episode execution requires config/pilot/seed/object-dir/public-points')
        r=run_unit(load_spec(a.config),a.pilot,a.seed,a.object_dir,a.public_points,a.out,a.host,a.port)
    print(json.dumps(r,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
