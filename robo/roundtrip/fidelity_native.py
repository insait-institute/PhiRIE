"""Evaluator-only native visual-surface export for canonical fidelity metrics.

No native asset or evaluator output is an input to the constructor. Frozen
estimated Sim(3) is applied once, without any evaluator registration or snapping.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from robo.roundtrip.identity import file_hash, validate_canonical_instance
from robo.roundtrip.build import _write


def fidelity_tier(build, canonical):
    """Bind evaluation split to the frozen build and native instance."""
    expected={'development':'DEV','test':'TEST'}.get(canonical.get('dataset_split'))
    if expected is None or build.get('tier')!=expected:
        raise ValueError('fidelity build/canonical split differs')
    return expected


def context_geometry_rows(path, binding, b0_manifest, candidate_pool):
    """Consume accepted frozen repair artifacts; never rerun selection here."""
    from robo.roundtrip.system_verification import asset_identity
    rows=[];sources={};seen=set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if row['canonical_instance_id']!=binding['canonical_instance_id']:continue
        method=row['controller_method']
        if method not in ('B4_ROOM_REPAIR_NATIVE','BM_BUDGET_MATCHED_NATIVE'):continue
        if method in seen:raise ValueError('duplicate context geometry method')
        seen.add(method)
        manifest=Path(row['build_manifest'])
        if file_hash(manifest)!=row['build_manifest_sha256']:raise ValueError('context build changed')
        m=json.loads(manifest.read_text())
        if row.get('terminal_status')=='BUILD_FAILED' and row.get('accepted') is False:
            sources[method]={'path':str(manifest.resolve()),'sha256':file_hash(manifest),'accepted':False,'terminal_status':'BUILD_FAILED'}
            continue
        if m.get('method')!={'B4_ROOM_REPAIR_NATIVE':'B4','BM_BUDGET_MATCHED_NATIVE':'BM'}[method]:
            raise ValueError('context treatment differs')
        if (m.get('capture_manifest_sha256')!=binding['capture_manifest_sha256'] or
                m.get('candidate_pool_sha256')!=file_hash(candidate_pool) or
                m.get('public_observation_provenance',{}).get('build_manifest_sha256')!=file_hash(b0_manifest)):
            raise ValueError('context geometry construction lineage differs')
        if m.get('accepted') is not row.get('accepted'):raise ValueError('context acceptance differs')
        sources[method]={'path':str(manifest.resolve()),'sha256':file_hash(manifest),'accepted':m['accepted']}
        if not m['accepted']:continue
        if row['terminal_status']!='READY' or Path(row['object_dir']).resolve()!=Path(m['selected_asset']).resolve():
            raise ValueError('context selected handle differs')
        if asset_identity(row['object_dir'])!=m['selected_asset_identity']:raise ValueError('context asset changed')
        for name in ('scene.xml','canonical_state.json'):
            p=Path(binding['bundle_dir'])/name
            if m['source_hashes'].get(str(p))!=file_hash(p):raise ValueError('context native identity differs')
        rows.append(dict(method=method,object_dir=row['object_dir']))
    if not seen:raise ValueError('no matching context geometry bindings')
    return rows,sources


def visual_surface(model, data, body_id):
    """Compiled mesh vertices include native mesh preprocessing; geom pose maps
    them to world. Include all visual mesh descendants, excluding collision and
    transparent bbox diagnostic geoms. Never approximate visual shape by hulls.
    """
    import mujoco
    import trimesh
    descendants={int(body_id)}
    for i in range(int(body_id)+1,model.nbody):
        if int(model.body_parentid[i]) in descendants:descendants.add(i)
    chunks=[];names=[]
    for g in range(model.ngeom):
        if (int(model.geom_bodyid[g]) not in descendants or model.geom_group[g]!=1
                or model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH or model.geom_rgba[g,3]<=0):continue
        m=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[m]);n=int(model.mesh_vertnum[m])
        f=int(model.mesh_faceadr[m]);nf=int(model.mesh_facenum[m])
        vertices=np.asarray(model.mesh_vert[a:a+n],dtype=float)@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
        faces=np.asarray(model.mesh_face[f:f+nf],dtype=int)
        chunks.append(trimesh.Trimesh(vertices=vertices,faces=faces,process=False))
        names.append(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g))
    if not chunks:raise ValueError('native target has no declared visual mesh surfaces')
    return trimesh.util.concatenate(chunks),names


def estimated_surface(directory):
    import trimesh
    from robo.roundtrip.importers.robocasa import decompose_alignment
    directory=Path(directory);aligned=json.loads((directory/'aligned.json').read_text())
    decompose_alignment(aligned)
    mesh=trimesh.load(directory/'mesh_sim.obj',force='mesh',process=False)
    if not np.isfinite(mesh.vertices).all() or len(mesh.faces)<4:raise ValueError('invalid estimated mesh')
    mesh.apply_transform(np.asarray(aligned['T']))
    return mesh


def evaluate(binding, b0_build, out, *, candidate_pool=None, reference_receipt=None, context_bindings=None, observed_control=None, sample_count=20000, seed=2027):
    import mujoco
    from robo.eval.agentic_ablation import _sample_mesh
    from robo.eval.fidelity_metrics import geometry_metrics
    from robo.roundtrip.shared_candidates import checked_b0
    started=time.monotonic();binding_path=Path(binding);binding=json.loads(binding_path.read_text())
    b0_build=Path(b0_build);out=Path(out)
    # Validate frozen construction before opening native evaluator-only data.
    built,_=checked_b0(b0_build,binding['capture_public'])
    if built.get('canonical_instance_id')!=binding['canonical_instance_id']:raise ValueError('build/canonical identity differs')
    if observed_control and (not reference_receipt or candidate_pool or context_bindings):
        raise ValueError('observed geometry requires cached reference and an independent control arm')
    methods=[dict(method='B0_FIXED_NATIVE',object_dir=str(b0_build/'construction/objects/obj_00'))]
    pool_hash=None
    if context_bindings and (not candidate_pool or not reference_receipt):
        raise ValueError('context geometry requires frozen pool and reference samples')
    if candidate_pool:
        cp=Path(candidate_pool);pool=json.loads(cp.read_text());pool_hash=file_hash(cp)
        if pool['b0_manifest_sha256']!=file_hash(b0_build/'build_manifest.json'):raise ValueError('candidate pool uses another B0')
        for row in ([] if context_bindings else pool['outcomes']):
            if row['native_method']=='B0_FIXED_NATIVE':continue
            path=cp.parent/Path(row['object_dir']).name
            for name,digest in row['artifact_hashes'].items():
                p=path/name
                if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or file_hash(p)!=digest:raise ValueError('selected artifact changed')
            methods.append(dict(method=row['native_method'],object_dir=str(path)))
    context_sources={}
    if context_bindings:
        methods,context_sources=context_geometry_rows(context_bindings,binding,b0_build/'build_manifest.json',candidate_pool)
    observed_source=None
    if observed_control:
        from robo.roundtrip.observed_native import validate_observed_control
        object_dir=Path(observed_control).resolve().parent
        observed_source=validate_observed_control(observed_control,object_dir,b0_build,binding['capture_public'],binding)
        methods=[dict(method='OBSERVED_SURFACE_TSDF',object_dir=str(object_dir))]
    frozen={r['method']:{name:file_hash(Path(r['object_dir'])/name) for name in ('aligned.json','mesh_sim.obj')} for r in methods}
    manifest=json.loads(Path(binding['canonical_manifest']).read_text());bundle=Path(binding['bundle_dir'])
    validate_canonical_instance(manifest,bundle)
    tier=fidelity_tier(built,manifest)
    if manifest['canonical_instance_id']!=binding['canonical_instance_id']:raise ValueError('binding canonical differs')
    reference_parent=None
    if reference_receipt:
        previous_path=Path(reference_receipt);previous=json.loads(previous_path.read_text())
        if (previous['canonical_manifest_sha256']!=file_hash(binding['canonical_manifest']) or
            previous['b0_manifest_sha256']!=file_hash(b0_build/'build_manifest.json') or
            previous['sample_count']!=sample_count or previous['seed']!=seed or previous['evaluator_alignment']!='NONE'):
            raise ValueError('cached evaluator surface identity differs')
        point_path=previous_path.parent/'reference_world_points.npy'
        if file_hash(point_path)!=previous['reference_points_sha256']:raise ValueError('cached native surface changed')
        reference=np.load(point_path,allow_pickle=False);names=previous['native_visual_geoms']
        reference_parent={'path':str(previous_path.resolve()),'sha256':file_hash(previous_path)}
        if candidate_pool:methods=[r for r in methods if r['method']!='B0_FIXED_NATIVE']
    else:
        model=mujoco.MjModel.from_xml_path(str(bundle/'scene.xml'));data=mujoco.MjData(model)
        state=json.loads((bundle/'canonical_state.json').read_text())
        mujoco.mj_setState(model,data,np.asarray(state['integration_state']),mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model,data)
        body=int(json.loads((bundle/'body_bindings.json').read_text())['obj'])
        if mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body)!='obj_main':raise ValueError('native target role binding differs')
        surface,names=visual_surface(model,data,body)
        reference=_sample_mesh(surface,sample_count,seed)
    out.mkdir(parents=True,exist_ok=False);np.save(out/'reference_world_points.npy',reference,allow_pickle=False)
    rows=[]
    for row in methods:
        points=_sample_mesh(estimated_surface(row['object_dir']),sample_count,seed)
        metric=geometry_metrics(points,reference,threshold_m=.02)
        np.save(out/(row['method']+'_world_points.npy'),points,allow_pickle=False)
        rows.append(dict(row,**metric,geometry_matched=True,predicted_samples=sample_count,reference_samples=sample_count))
    receipt=dict(schema_version=2,canonical_instance_id=binding['canonical_instance_id'],cohort_id=binding['cohort_id'],
        scope=binding['scope'],sensor_regime=built['sensor_regime'],tier=tier,geometry_producer='robo.eval.fidelity_metrics.geometry_metrics',
        sampling_producer='robo.eval.agentic_ablation._sample_mesh',sample_count=sample_count,seed=seed,
        reference_surface='native compiled visual triangle surface in frozen world frame',native_visual_geoms=names,
        evaluator_alignment='NONE',constructor_registration_target='TRAIN RGB-D only',heldout_native_asset_access='EVALUATOR_ONLY',
        binding_sha256=file_hash(binding_path),b0_manifest_sha256=file_hash(b0_build/'build_manifest.json'),candidate_pool_sha256=pool_hash,
        canonical_manifest_sha256=file_hash(binding['canonical_manifest']),frozen_estimated_artifacts=frozen,
        metric_implementation_sha256=file_hash(Path(__file__).resolve().parents[1]/'eval/fidelity_metrics.py'),
        reference_points_sha256=file_hash(out/'reference_world_points.npy'),reference_parent=reference_parent,rows=rows,wall_s=time.monotonic()-started,
        context_sources=context_sources,observed_source=observed_source,
        appearance_metrics='NOT_RUN; uniform native rendering is not Gaussian appearance',absolute_pose_error='NOT_RUN; no physical correspondence')
    _write(out/'geometry_metrics.json',receipt);return receipt


def checked_heldout(binding):
    """Use the predeclared private capture split, with no view reselection."""
    from robo.roundtrip.build import read_train
    public=Path(binding['capture_public']);manifest,train=read_train(public)
    if file_hash(public/'capture_manifest.json')!=binding['capture_manifest_sha256']:
        raise ValueError('appearance capture identity differs')
    private=Path(binding['bundle_dir']).parent/'capture'/public.name
    plan=json.loads((private/'capture_plan.json').read_text())
    rows=[json.loads(line) for line in (private/'test/cameras.jsonl').read_text().splitlines() if line.strip()]
    frames={r['frame_id']:r for r in plan['frames'] if r['split']=='test'}
    if len(rows)!=plan['spec']['counts']['test'] or {r['frame_id'] for r in rows}!=set(frames):
        raise ValueError('heldout frame roster differs')
    train_ids={r['frame_id'] for r in train}
    for row in rows:
        if row['frame_id'] in train_ids or any(np.allclose(row['T_world_from_camera'],r['T_world_from_camera'],rtol=0,atol=1e-10) for r in train):
            raise ValueError('appearance TRAIN/evaluation view overlap')
        rgb=private/row['rgb']
        if rgb.is_symlink() or file_hash(rgb)!=row['rgb_sha256']:raise ValueError('heldout RGB changed')
        row.update(reference_rgb=str(rgb),camera=frames[row['frame_id']]['camera'])
    return rows,plan['spec']


def static_render_state(model,before,after):
    """mj_forward refreshes solver warmstart without advancing physical state."""
    import mujoco
    before=np.asarray(before);after=np.asarray(after);changes={};offset=0
    for bit in [1<<i for i in range(int(mujoco.mjtState.mjNSTATE))]:
        if not int(mujoco.mjtState.mjSTATE_INTEGRATION)&bit:continue
        n=mujoco.mj_stateSize(model,bit);a=before[offset:offset+n];b=after[offset:offset+n];offset+=n
        if not np.array_equal(a,b):
            changes[str(mujoco.mjtState(bit))]=float(np.max(np.abs(a-b)))
            if bit!=int(mujoco.mjtState.mjSTATE_WARMSTART):raise ValueError('static rendering changed physical/control state '+str(bit))
    if offset!=len(before) or before.shape!=after.shape or not np.isfinite(after).all():raise ValueError('invalid render integration vector')
    return changes


def checked_train_render_prelude(binding):
    """Replay the declared capture prefix, without reading evaluation RGB."""
    from robo.roundtrip.build import read_train
    public=Path(binding['capture_public']);manifest,train=read_train(public)
    if file_hash(public/'capture_manifest.json')!=binding['capture_manifest_sha256']:
        raise ValueError('prelude capture identity differs')
    plan_path=Path(binding['bundle_dir']).parent/'capture'/public.name/'capture_plan.json'
    plan=json.loads(plan_path.read_text());first_test=next((i for i,r in enumerate(plan['frames']) if r['split']=='test'),None)
    if first_test is None:raise ValueError('prelude requires declared heldout suffix')
    prefix=plan['frames'][:first_test]
    if not prefix or any(r['split']!='train' for r in prefix) or [r['frame_id'] for r in prefix]!=[r['frame_id'] for r in train]:
        raise ValueError('prelude must preserve the complete original TRAIN prefix order')
    for declared,observed in zip(prefix,train):
        if not np.allclose(declared['camera']['T_world_from_camera'],observed['T_world_from_camera'],rtol=0,atol=1e-10):
            raise ValueError('prelude camera differs from frozen TRAIN calibration')
    return prefix,dict(protocol='capture_train_prelude_v1',capture_plan_path=str(plan_path),
        capture_plan_sha256=file_hash(plan_path),capture_manifest_sha256=binding['capture_manifest_sha256'],
        frame_ids=[r['frame_id'] for r in prefix],evaluation_rgb_accessed_for_prelude=False)


def metadata_render_methods(methods, frozen, correction_receipts):
    """Bind old evaluated surfaces to proven metadata-only import copies."""
    from robo.roundtrip.asset_metadata import validate_metadata_correction
    proofs=[(Path(p),validate_metadata_correction(p)) for p in correction_receipts]
    used=set();bound=[];overrides=[]
    for row in methods:
        hashes=frozen[row['method']]
        matches=[(p,d) for p,d in proofs if all(d['parent_hashes'].get(n)==h for n,h in hashes.items())]
        if len(matches)>1:raise ValueError('ambiguous metadata render correction')
        if not matches:bound.append(dict(row));continue
        p,d=matches[0];used.add(str(p))
        for n,h in hashes.items():
            if file_hash(Path(row['object_dir'])/n)!=h:raise ValueError('original evaluated asset changed')
        bound.append(dict(row,object_dir=d['corrected_asset']))
        overrides.append(dict(method=row['method'],original_object_dir=row['object_dir'],
            import_object_dir=d['corrected_asset'],correction_receipt=dict(path=str(p.resolve()),sha256=file_hash(p)),
            evaluated_artifact_hashes=hashes,import_artifact_hashes=d['child_hashes'],
            geometry_transform_collision_physics_unchanged=True,controller_action=False))
    if used!={str(p) for p,_ in proofs}:raise ValueError('unused metadata render correction')
    return bound,overrides


def render_appearance(binding_path, geometry_receipt, out, *, b0_build, candidate_pool=None, metadata_corrections=(), capture_render_prelude=False):
    """Frozen artifact render at static heldout poses; no physics settling/GT fit."""
    from PIL import Image
    from robo.roundtrip.shared_candidates import checked_b0
    from robo.roundtrip.adapters.robocasa import RoboCasaAdapter
    from robo.roundtrip.paired import prepare_paired_adapter
    from robo.manifest.hash import git_snapshot
    started=time.monotonic();code=git_snapshot()
    if code.get('dirty') is not False:raise ValueError('appearance requires clean committed source')
    binding=json.loads(Path(binding_path).read_text())
    built,_=checked_b0(b0_build,binding['capture_public'])
    geometry=json.loads(Path(geometry_receipt).read_text())
    if (geometry['binding_sha256']!=file_hash(binding_path) or geometry['evaluator_alignment']!='NONE'
            or geometry['b0_manifest_sha256']!=file_hash(Path(b0_build)/'build_manifest.json')):
        raise ValueError('geometry/render canonical binding differs')
    manifest=json.loads(Path(binding['canonical_manifest']).read_text());bundle=Path(binding['bundle_dir'])
    validate_canonical_instance(manifest,bundle)
    tier=fidelity_tier(built,manifest)
    if geometry.get('tier')!=tier:raise ValueError('geometry/render split differs')
    if geometry['canonical_manifest_sha256']!=file_hash(binding['canonical_manifest']):raise ValueError('canonical source changed')
    # Only consume already sealed geometry selection rows. Verify frozen meshes
    # and transforms before opening any private heldout image.
    methods=geometry['rows']
    if any(r['method']!='B0_FIXED_NATIVE' for r in methods):
        if candidate_pool is None or file_hash(candidate_pool)!=geometry['candidate_pool_sha256']:raise ValueError('appearance candidate pool differs')
        pool=json.loads(Path(candidate_pool).read_text())
        for row in pool['outcomes']:
            directory=Path(candidate_pool).parent/Path(row['object_dir']).name
            for name,digest in row['artifact_hashes'].items():
                if Path(name).is_absolute() or '..' in Path(name).parts or file_hash(directory/name)!=digest:raise ValueError('candidate appearance source changed')
    for row in methods:
        for name,digest in geometry['frozen_estimated_artifacts'][row['method']].items():
            if file_hash(Path(row['object_dir'])/name)!=digest:raise ValueError('frozen appearance asset changed')
    methods,metadata_overrides=metadata_render_methods(methods,geometry['frozen_estimated_artifacts'],metadata_corrections)
    frames,spec=checked_heldout(binding);config=json.loads(Path(binding['native_config']).read_text())
    source={'state':json.loads((bundle/'canonical_state.json').read_text()),'xml':(bundle/'scene.xml').read_text(),
        'provenance':{'reset_seed':config['reset_seeds'][0]}}
    prelude,prelude_source=checked_train_render_prelude(binding) if capture_render_prelude else ([],None)
    out=Path(out);out.mkdir(parents=True,exist_ok=False);records=[];identity=[];state_checks=[];prelude_checks=[];adapter=None
    try:
        adapter=RoboCasaAdapter(config)
        for method in [dict(method='REF_IMPORT_CONTROL',object_dir=None),*methods]:
            directory=out/method['method'];directory.mkdir()
            imported=prepare_paired_adapter(adapter,source,object_dir=method['object_dir'])
            _write(directory/'import_receipt.json',imported)
            state_before=np.asarray(adapter.get_state()['integration_state']).copy()
            for frame in prelude:
                adapter.render_capture(frame['camera'],width=spec['width'],height=spec['height'])
                changes=static_render_state(adapter.native.sim.model._model,state_before,adapter.get_state()['integration_state'])
                prelude_checks.append(dict(method=method['method'],frame_id=frame['frame_id'],physical_control_state_exact=True,solver_only_changes=changes))
            for frame in frames:
                rendered=adapter.render_capture(frame['camera'],width=spec['width'],height=spec['height'])
                pred=np.asarray(rendered['rgb']);target=np.asarray(Image.open(frame['reference_rgb']).convert('RGB'))
                if pred.shape!=target.shape or pred.dtype!=np.uint8:raise ValueError('heldout render RGB shape/type differs')
                if not np.allclose(rendered['K'],frame['K'],rtol=0,atol=1e-10) or not np.allclose(rendered['T_world_from_camera'],frame['T_world_from_camera'],rtol=0,atol=1e-10):
                    raise ValueError('heldout camera calibration differs')
                path=directory/(frame['frame_id']+'.png');Image.fromarray(pred).save(path)
                if method['method']=='REF_IMPORT_CONTROL':
                    exact=bool(np.array_equal(pred,target));identity.append(dict(frame_id=frame['frame_id'],byte_exact=exact,max_abs=int(np.abs(pred.astype(int)-target).max())))
                    if not exact:raise ValueError('unchanged-native heldout rendering differs')
                else:
                    records.append(dict(method=method['method'],frame_id=frame['frame_id'],pred_rgb=str(path.resolve()),pred_sha256=file_hash(path),
                        reference_rgb=frame['reference_rgb'],reference_sha256=frame['rgb_sha256'],width=spec['width'],height=spec['height']))
                changes=static_render_state(adapter.native.sim.model._model,state_before,adapter.get_state()['integration_state'])
                state_checks.append(dict(method=method['method'],frame_id=frame['frame_id'],solver_only_changes=changes,physical_control_state_exact=True))
        result=dict(schema_version=2,canonical_instance_id=binding['canonical_instance_id'],cohort_id=binding['cohort_id'],
            scope='L0_target_only',sensor_regime='ideal_rgbd_posed',renderer='native',tier=tier,
            source_code=code,binding={'path':str(Path(binding_path).resolve()),'sha256':file_hash(binding_path)},
            geometry_receipt={'path':str(Path(geometry_receipt).resolve()),'sha256':file_hash(geometry_receipt)},
            render_protocol='capture_train_prelude_v1' if capture_render_prelude else 'cold_import_v1',
            prelude_source=prelude_source,prelude_state_checks=prelude_checks,
            metadata_import_overrides=metadata_overrides,records=records,planned_views_per_method=len(frames),native_import_render_identity=identity,static_state_checks=state_checks,
            intervention='full reconstructed asset bundle; uniform reconstructed material; native context retained',
            support='full heldout frame; retained context dominates area; not object-only appearance',
            evaluator_alignment='NONE',exposure_matching='NONE',wall_s=time.monotonic()-started)
        if git_snapshot()!=code:raise ValueError('render source changed during execution')
        _write(out/'appearance_render.json',result);return result
    finally:
        if adapter is not None:adapter.close()


def serializable_appearance(values,pred,target):
    """Positive-infinite standard PSNR is an exact image match, never a floor."""
    values=dict(values);exact=bool(np.array_equal(pred,target))
    if np.isposinf(values['psnr']) and exact:
        values.update(psnr=None,psnr_status='POSITIVE_INFINITY',exact_rgb_match=True)
    elif np.isfinite(values['psnr']) and not exact:
        values.update(psnr_status='FINITE',exact_rgb_match=False)
    else:raise ValueError('PSNR/exact-image status is inconsistent')
    if not all(np.isfinite(values[k]) for k in ('ssim','lpips')):raise ValueError('appearance metric nonfinite')
    return values


def appearance_metrics(render_receipt,out):
    """Delegate PSNR/SSIM/LPIPS to existing canonical fidelity producers on CPU."""
    from robo.eval.fidelity_metrics import load_rgb,psnr,ssim,LPIPSEvaluator
    started=time.monotonic();path=Path(render_receipt);receipt=json.loads(path.read_text())
    if not all(r['byte_exact'] for r in receipt['native_import_render_identity']):raise ValueError('native render identity did not pass')
    lpips=LPIPSEvaluator('cpu')
    if lpips.model is None:raise RuntimeError('pinned LPIPS unavailable: '+str(lpips.error))
    out=Path(out);out.mkdir(parents=True,exist_ok=False);rows=[]
    for row in receipt['records']:
        for field,digest in [('pred_rgb','pred_sha256'),('reference_rgb','reference_sha256')]:
            if file_hash(row[field])!=row[digest]:raise ValueError('appearance image changed')
        pred=load_rgb(row['pred_rgb']);target=load_rgb(row['reference_rgb'])
        values=dict(psnr=psnr(pred,target),ssim=ssim(pred,target),lpips=lpips(pred,target))
        values=serializable_appearance(values,pred,target)
        rows.append(dict(row,**values))
    result={k:receipt[k] for k in ('schema_version','canonical_instance_id','cohort_id','scope','sensor_regime','renderer','tier','support','intervention','planned_views_per_method')}
    result.update(render_protocol=receipt.get('render_protocol','cold_import_v1'),rows=rows,source={'path':str(path.resolve()),'sha256':file_hash(path)},
        metric_implementation_sha256=file_hash(Path(__file__).resolve().parents[1]/'eval/fidelity_metrics.py'),
        lpips_provenance=lpips.provenance,wall_s=time.monotonic()-started)
    _write(out/'appearance_metrics.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--binding');p.add_argument('--b0-build');p.add_argument('--render-geometry-receipt');p.add_argument('--appearance-render-receipt')
    p.add_argument('--candidate-pool');p.add_argument('--reference-receipt');p.add_argument('--context-bindings');p.add_argument('--metadata-correction',action='append',default=[]);p.add_argument('--observed-control');p.add_argument('--capture-render-prelude',action='store_true');a=p.parse_args()
    if a.appearance_render_receipt:result=appearance_metrics(a.appearance_render_receipt,a.out)
    elif a.render_geometry_receipt:result=render_appearance(a.binding,a.render_geometry_receipt,a.out,b0_build=a.b0_build,candidate_pool=a.candidate_pool,metadata_corrections=a.metadata_correction,capture_render_prelude=a.capture_render_prelude)
    else:result=evaluate(a.binding,a.b0_build,a.out,candidate_pool=a.candidate_pool,reference_receipt=a.reference_receipt,context_bindings=a.context_bindings,observed_control=a.observed_control)
    print(json.dumps(result,allow_nan=False))

if __name__=='__main__':main()
