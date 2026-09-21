"""Static final-state native predicate and actual-asset origin audit (evaluator only)."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from robo.manifest.hash import git_snapshot
from robo.roundtrip.identity import file_hash
from robo.roundtrip.scorer import predicate_components,scorer_view,_View
from robo.roundtrip.scorer_bounds import audit_bounds
from robo.roundtrip.scorer_mesh_control import compare_compiled
from robo.roundtrip.fidelity_native import visual_surface


def diagnose(config,pilot,seed,episode,object_dir):
    from robo.roundtrip.paired import load_reference_bundle,prepare_paired_adapter
    from robo.roundtrip.adapters.robocasa import RoboCasaAdapter
    from robo.eval.episode_log import read_timeseries
    pilot=Path(pilot);episode=Path(episode)
    bundle=load_reference_bundle(pilot/f'episode_seed{seed}',pilot/f'canonical_seed{seed}',config=config,reset_seed=seed)
    result=json.loads((episode/'result.json').read_text())
    if result['reset_seed']!=seed or result['config_sha256']!=bundle['provenance']['config_sha256']:
        raise ValueError('diagnostic episode identity differs')
    adapter=RoboCasaAdapter(config)
    try:
        imported=prepare_paired_adapter(adapter,bundle,object_dir=object_dir)
        if object_dir:
            original=json.loads((episode.parent/'import_receipt.json').read_text())
            if imported['imported_xml_sha256']!=original['imported_xml_sha256']:
                raise ValueError('diagnostic reconstructed model differs from episode')
        native=adapter.native;trace=read_timeseries(episode/'trace.json.gz');tick=trace[-1]
        native.sim.data.qpos[:]=tick['qpos'];native.sim.data.qvel[:]=tick['qvel'];native.sim.forward()
        task=config['instance']['task_id'];raw=predicate_components(native,task)
        if raw['native_task_success']!=bool(tick['native_predicates']['native_task_success']):
            raise ValueError('static recomputation differs from recorded native predicate')
        model=native.sim.model._model;data=native.sim.data._data;body=native.obj_body_id['obj']
        mesh,geoms=visual_surface(model,data,body);bound=audit_bounds(native,task,mesh.vertices)
        control=None
        if object_dir:
            B=np.eye(4);B[:3,3]=[.1,-.07,.03];B[:3,:3]=Rotation.from_euler('xyz',[.1,.3,.7]).as_matrix()
            control,changed,inverse=compare_compiled(adapter.source_xml(),native.objects['obj'].root_body,data,B)
            view=_View(native,sim=_View(native.sim,data=_View(native.sim.data,body_xpos=changed.xpos,body_xquat=changed.xquat,site_xpos=changed.site_xpos)))
            fixed=predicate_components(view,task,body_from_scorer={'obj':inverse.tolist()})
            shifted=predicate_components(view,task)
            control.update(raw_reexpressed_predicates=shifted,fixed_scorer_predicates=fixed,
                original_native_success=raw['native_task_success'],
                fixed_native_success_equal=raw['native_task_success']==fixed['native_task_success'],
                fixed_gripper_distance_abs_error_m=abs(raw['gripper_distance_m']-fixed['gripper_distance_m']),
                original_raw_predicates=raw)
            control['passed'] &= control['fixed_native_success_equal'] and control['fixed_gripper_distance_abs_error_m']<1e-12
        return dict(seed=seed,episode=str(episode),source_trace_sha256=file_hash(episode/'trace.json.gz'),
            result_sha256=file_hash(episode/'result.json'),object_dir=str(object_dir) if object_dir else None,
            ticks=result['ticks'],raw_native_predicates=raw,origin_bounds=bound,known_generated_mesh_frame_control=control,
            native_visual_geoms=geoms,static_recomputed_predicate_matches_logged=True,
            evaluator_pose_source='final recorded qpos/qvel; forward only, no rollout',absolute_correspondence=None)
    finally:adapter.close()


def main(argv=None):
    from robo.roundtrip.spec import load_spec
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--bindings',required=True);p.add_argument('--out',required=True);a=p.parse_args(argv)
    bindings=json.loads(Path(a.bindings).read_text());rows=[]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    for row in bindings['episodes']:
        result=diagnose(load_spec(a.config),bindings['pilot'],row['seed'],row['episode'],row.get('object_dir'))
        rows.append(result)
        with (out/f"seed{row['seed']}_{'B0' if row.get('object_dir') else 'REF'}.json").open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    report=dict(schema_version=2,source=git_snapshot(),bindings_sha256=file_hash(a.bindings),rows=rows,
        scope='evaluator-only final-state binding-specific diagnosis; no policy episodes or construction feedback',
        native_thresholds_unchanged=True,absolute_pose_correspondence=None,
        all_known_frame_controls_pass=all(r['known_generated_mesh_frame_control']['passed'] for r in rows if r['known_generated_mesh_frame_control']),
        claim_scope='raw official native success binding-specific; physical task preservation NOT_RUN')
    with (out/'scorer_diagnosis.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report,allow_nan=False));return 0 if report['all_known_frame_controls_pass'] else 2


if __name__=='__main__':raise SystemExit(main())
