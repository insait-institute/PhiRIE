"""Bounded component batches: existing generation, registration and QA producers."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from .core import checked_path, load, receipt, save, source_identity
MODULE = 'robo.campaign.component_worker'


def validate_observation(obj):
    """Authenticate construction surfaces from their original TRAIN receipts."""
    manifest = load(checked_path(obj['input_manifest_receipt']))
    jobs = load(checked_path(obj['source_manifest_receipt']))
    meta = load(checked_path(obj['meta']))
    hashes = load(checked_path(obj['output_hashes']))
    if (manifest['scene_id'] != obj['scene_id'] or jobs['scene_id'] != obj['scene_id']
            or meta['gt_object_id'] != obj['automatic_instance_id']
            or jobs['output_hashes_sha256'] != obj['output_hashes']['sha256']
            or manifest['source_gaussian_training_provenance'] != 'FRESH_OFFICIAL_TRAIN_ONLY'
            or manifest['artifact_semantics']['gt_points.ply'] != 'derived construction-surface samples, not evaluation GT'
            or meta['frame'] not in manifest['boundary']['training_frames']):
        raise ValueError('construction input identity or TRAIN-only provenance differs')
    prefix = f"objects/obj_{meta['index']:02d}/"
    for key, name in [('rgba_receipt', 'rgba.png'), ('meta', 'meta.json'), ('observed_surface', 'gt_points.ply')]:
        checked_path(obj[key])
        if obj[key]['sha256'] != hashes[prefix + name]['sha256']:
            raise ValueError('original construction artifact changed: ' + name)
    return meta


def evaluate(config_path, out):
    # Keep CoACD/torch/Open3D in this CPU interpreter, outside the SAM runtime.
    from agents.assets.s5_align import align_object_with_signed_source_up, apply_T
    from agents.assets.s6_physics import coacd_parts, write_urdf
    from agents.eval.factory_report import drop_test
    from robo.eval.fidelity_metrics import load_surface, geometry_metrics
    import trimesh
    import open3d as o3d
    import pybullet as bullet
    from PIL import Image

    spec = load(config_path); obj = spec['object']; meta = validate_observation(obj)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(kind='generator_component_evaluation_v1', slot_id=spec['slot_id'],
                  status='ENGINEERING_FAILED', generation=spec['generation'],
                  config=receipt(config_path), evaluator_alignment='NONE',
                  registration_gt_read=False, appearance_metrics=None,
                  appearance_reason='No common measured appearance contract for these four backends',
                  source=source_identity(Path(__file__).resolve().parents[2]))
    try:
        gen = load(checked_path(spec['generation']))
        mesh = trimesh.load(checked_path(gen['artifacts']['mesh']), process=False, force='mesh')
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all() or mesh.area <= 0:
            raise ValueError('empty/nonfinite canonical mesh')
        points, _ = trimesh.sample.sample_surface(mesh, 20000, seed=42)
        t0 = time.monotonic()
        if spec.get('reuse_registration'):
            reg = load(checked_path(spec['reuse_registration']))
            train = load(checked_path(reg['config']))
            if (train['mesh']['sha256'] != gen['artifacts']['mesh']['sha256']
                    or train['observed_surface']['sha256'] != obj['observed_surface']['sha256']
                    or reg['evaluator_GT_read'] is not False
                    or reg['recipe'] != spec['protocol']['registration']):
                raise ValueError('existing registration is incompatible')
            T = np.asarray(reg['T_world_from_generator'])
            result['reused_registration'] = spec['reuse_registration']
            registration_seconds = reg['registration_seconds']
        else:
            target = load_surface(checked_path(obj['observed_surface']))
            target = target[::max(len(target) // 6000, 1)]
            T, residual, initial, tilt, up = align_object_with_signed_source_up(np.asarray(points, np.float64), target)
            registration_seconds = time.monotonic() - t0
            result.update(registration_residual_m=residual, initial_residual_m=initial, tilt_deg=tilt, source_up=up)
        if T.shape != (4, 4) or not np.isfinite(T).all() or np.linalg.det(T[:3, :3]) <= 0:
            raise ValueError('invalid world transform')
        world = apply_T(T, points)
        if spec.get('reuse_registration'):
            world = load_surface(checked_path(reg['world_surface']))
        np.save(out / 'world_surface.npy', world, allow_pickle=False)
        result.update(registered=True, T_world_from_generator=T.tolist(), registration_seconds=registration_seconds,
                      world_surface=receipt(out / 'world_surface.npy'))
        # Seal registration before opening the independent matching/evaluator data.
        save(out / 'registration.json', dict(result))
        refs = load(checked_path(spec['evaluation_matching']))
        scene = next(s for s in refs['scenes'] if s['scene_id'] == obj['scene_id'])
        original = scene['discovery_binding']['source_discovery_hashes']
        if (original['input_manifest.json'] != obj['input_manifest_receipt']['sha256']
                or original['all_jobs_manifest.json'] != obj['source_manifest_receipt']['sha256']):
            raise ValueError('independent matching belongs to a different discovery')
        matches = [r for r in refs['rows'] if r['scene_id'] == obj['scene_id'] and r['automatic_instance_id'] == obj['automatic_instance_id']]
        if len(matches) != 1:
            raise ValueError('independent matching is absent or ambiguous')
        surface = matches[0]['evaluation_surface']
        result['evaluation_surface'] = surface
        if surface:
            if surface['sha256'] == obj['observed_surface']['sha256']:
                raise ValueError('evaluator surface equals construction input')
            if spec.get('reuse_metric'):
                old = load(checked_path(spec['reuse_metric']))
                if (old['evaluation_surface'] != surface or old['evaluator_alignment'] != 'NONE'
                        or old['registration'] != spec.get('reuse_registration')):
                    raise ValueError('existing metric belongs to another registration')
                result['metrics'] = old['metrics']; result['reused_metric'] = spec['reuse_metric']
            else:
                result['metrics'] = geometry_metrics(world, load_surface(checked_path(surface)))
            result['metrics'] = dict(result['metrics'], f1_40=geometry_metrics(world, load_surface(checked_path(surface)), threshold_m=.04)['f1_20'])
            result['geometry_status'] = 'MEASURED'
        else:
            result.update(metrics=None, geometry_status='UNMATCHED_REFERENCE')
        t0 = time.monotonic()
        # Bake the observation-estimated rotation/scale once; placement translation
        # is omitted for the same canonical plane drop used by factory_report.
        mesh.vertices = np.asarray(mesh.vertices) @ T[:3, :3].T
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
        if len(om.triangles) > spec['protocol']['triangle_budget']:
            om = om.simplify_quadric_decimation(spec['protocol']['triangle_budget'])
        om.remove_degenerate_triangles()
        sim = trimesh.Trimesh(np.asarray(om.vertices), np.asarray(om.triangles), process=False)
        if not len(sim.faces) or not np.isfinite(sim.vertices).all() or sim.area <= 0:
            raise ValueError('invalid decimated mesh')
        sim.export(out / 'mesh_sim.ply'); sim.export(out / 'mesh_sim.obj')
        parts = coacd_parts(out, world_max_dim=float(sim.extents.max()))
        if not parts:
            raise ValueError('empty collision decomposition')
        phys = spec['protocol']['physics']
        write_urdf(out, spec['slot_id'], 1., sim.extents, phys, parts, sim.vertices.mean(axis=0))
        result.update(exported=True, collision_parts=len(parts), scale_baked_once=True,
                      registration_export_seconds=time.monotonic() - t0)
        bullet.connect(bullet.DIRECT)
        try:
            body = bullet.loadURDF(str(out / 'object.urdf'), flags=bullet.URDF_USE_INERTIA_FROM_FILE)
            if body < 0: raise ValueError('collision import failed')
            result['collision_import_valid'] = True
            result['stability'] = drop_test(bullet, out / 'object.urdf')
            # Real software-rendered diagnostic, independent from scientific metrics.
            extent = max(float(sim.extents.max()), .05)
            target = sim.centroid.tolist()
            view = bullet.computeViewMatrixFromYawPitchRoll(target, extent * 2.5, 35, -25, 0, 2)
            projection = bullet.computeProjectionMatrixFOV(45, 1., .001, extent * 20 + 10)
            image = bullet.getCameraImage(384, 384, view, projection, renderer=bullet.ER_TINY_RENDERER)
            Image.fromarray(np.asarray(image[2], dtype=np.uint8).reshape(384, 384, 4)).save(out / 'render.png')
        finally:
            bullet.disconnect()
        result.update(status='EVALUATED', physical_protocol='canonical_factory_report_drop_v1',
                      render=receipt(out / 'render.png'), urdf=receipt(out / 'object.urdf'))
    except Exception:
        result['error'] = traceback.format_exc()
    result['evaluation_seconds'] = time.monotonic() - started
    save(out / 'evaluation.json', result)
    return result


def generate(config_path, out):
    from models.s4_sam3d import load_component_pipeline, component_proposal
    config = load(config_path); out = Path(out); out.mkdir(parents=True, exist_ok=False)
    model = config['model']
    if source_identity(model['source']['root']) != model['source']:
        raise ValueError('SAM 3D source changed')
    for ref in load(checked_path(model['checkpoint_manifest']))['files']:
        checked_path(ref)
    checked_path(model['pipeline'])
    start = time.monotonic()
    moge = checked_path(model['moge_checkpoint']) if model.get('moge_checkpoint') else None
    pipe = load_component_pipeline(model['source']['root'], model['pipeline']['path'], moge_checkpoint=moge)
    initialization_seconds = time.monotonic() - start
    rows = []
    for obj in config['objects']:
        slot = obj['slot_id']; dest = out / slot; dest.mkdir()
        t0 = time.monotonic()
        row = dict(kind='sam3d_component_proposal_v1', backend='sam3d', slot_id=slot, seed=config['seed'],
                   input_images=[obj['rgba_receipt']], model=model, source=source_identity(Path(__file__).resolve().parents[2]),
                   status='GENERATION_FAILED', artifacts={}, initialization_seconds=initialization_seconds)
        try:
            validate_observation(obj)
            row.update(component_proposal(pipe, checked_path(obj['rgba_receipt']), dest, config['seed']))
            row['artifacts'] = {k: receipt(dest / f) for k, f in [('mesh','sam3d_mesh.ply'),('gaussians','sam3d_gs.ply'),('native_arrays','native_candidate.npz')]}
            row['status'] = 'GENERATION_COMPLETE'
        except Exception:
            row['error'] = traceback.format_exc()
        row['wall_s'] = time.monotonic() - t0
        save(dest / 'generation.json', row)
        if row['status'] == 'GENERATION_COMPLETE':
            evaluation = dict(slot_id=slot, object=obj, generation=receipt(dest / 'generation.json'),
                              protocol=config['protocol'], evaluation_matching=config['evaluation_matching'])
            save(dest / 'evaluation-config.json', evaluation)
            # No nested allocation: geometry is CPU work in this same allocation.
            child = subprocess.run([config['cpu_python'], '-m', MODULE, 'evaluate', '--config', str(dest / 'evaluation-config.json'), '--out', str(dest / 'evaluation')], timeout=1200)
            if child.returncode == 0 and (dest / 'evaluation/evaluation.json').exists():
                row['evaluation'] = receipt(dest / 'evaluation/evaluation.json')
        rows.append(row)
        print(json.dumps(dict(slot=slot, status=row['status'])), flush=True)
    good = all(r['status'] == 'GENERATION_COMPLETE' and r.get('evaluation')
               and load(r['evaluation']['path'])['status'] == 'EVALUATED' for r in rows)
    record = dict(status='VALIDATED' if good else 'PARTIAL', rows=rows, initialization_seconds=initialization_seconds,
                  wall_s=time.monotonic() - start, config=receipt(config_path), job_id=os.environ.get('SLURM_JOB_ID'))
    save(out / 'batch.json', record)
    save(out / 'outputs.json', {'artifacts': {'batch': receipt(out / 'batch.json')}})
    if not good: raise RuntimeError('component batch incomplete; inspect retained per-object records')
    return record


def evaluate_batch(config_path, out):
    config = load(config_path); out = Path(out); out.mkdir(parents=True, exist_ok=False)
    rows = []
    for spec in config['slots']:
        p = out / (spec['slot_id'] + '.json'); save(p, spec)
        run = subprocess.run([sys.executable, '-m', MODULE, 'evaluate', '--config', str(p), '--out', str(out / spec['slot_id'])], timeout=1200)
        q = out / spec['slot_id'] / 'evaluation.json'
        rows.append(dict(slot_id=spec['slot_id'], returncode=run.returncode, evaluation=receipt(q) if q.exists() else None))
    save(out / 'batch.json', {'rows': rows, 'config': receipt(config_path)})
    save(out / 'outputs.json', {'artifacts': {'batch': receipt(out / 'batch.json')}})
    if any(r['returncode'] != 0 or not r['evaluation'] for r in rows):
        raise RuntimeError('CPU evaluation has failed objects; inspect retained records')


def views(config_path, out):
    """Freeze existing occlusion-aware views using only the TRAIN-derived mesh."""
    from agents.core import common as C
    from models.s4_reconviagen import collect_views
    import open3d as o3d
    import trimesh
    from PIL import Image
    config = load(config_path); out = Path(out); out.mkdir(parents=True, exist_ok=False)
    rows = []; scenes = {}
    for obj in config['objects']:
        meta = validate_observation(obj); sid = obj['scene_id']
        if sid not in scenes:
            manifest = load(checked_path(obj['input_manifest_receipt']))
            hashes = load(checked_path(obj['output_hashes']))
            construction = Path(obj['meta']['path']).parents[2]
            for name in ['derived_mesh.ply', 'auto_instances.npz']:
                if receipt(construction / name)['sha256'] != hashes[name]['sha256']:
                    raise ValueError('original TRAIN construction mesh/membership changed')
            tm = trimesh.load(construction / 'derived_mesh.ply', process=False)
            ray = o3d.t.geometry.RaycastingScene()
            ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(tm.vertices), o3d.utility.Vector3iVector(tm.faces))))
            K, width, height, _ = C.load_intrinsics(checked_path(manifest['metadata']['nerfstudio/transforms_undistorted.json']))
            cameras = C.load_colmap_w2c(checked_path(manifest['metadata']['colmap/images.txt']))
            frames = [(name, cameras[name]) for name in sorted(manifest['boundary']['training_frames'])]
            instances = {r['object_id']: r for r in C.load_auto_instances(instances_path=construction / 'auto_instances.npz', mesh_path=construction / 'derived_mesh.ply')}
            scenes[sid] = (manifest, tm, ray, K, width, height, frames, instances)
        manifest, tm, ray, K, width, height, frames, instances = scenes[sid]
        dest = out / obj['object_key']; dest.mkdir()
        selected = collect_views(instances[obj['automatic_instance_id']], K, width, height, frames, ray, np.asarray(tm.vertices), np.asarray(tm.faces), 12)
        records = [dict(frame=meta['frame'], image=obj['rgba_receipt'], mask_source='unchanged shared anchor')]
        for frame, (x0,y0,x1,y1), mask in selected:
            if frame == meta['frame'] or len(records) >= 12: continue
            original = manifest['input_images'][frame]; image = np.asarray(Image.open(checked_path(original)).convert('RGB'))
            rgba = np.dstack([image[y0:y1, x0:x1], (mask * 255).astype(np.uint8)])
            path = dest / f'view_{len(records):02d}.png'; Image.fromarray(rgba).save(path)
            records.append(dict(frame=frame, image=receipt(path), source_image=original,
                                crop=[x0,y0,x1,y1], mask_source='occlusion-aware TRAIN-derived instance mesh projection'))
        row = dict(object_key=obj['object_key'], status='READY' if len(records) >= 2 else 'INSUFFICIENT_VIEWS',
                   views=records, input_manifest=obj['input_manifest_receipt'],
                   source_manifest=obj['source_manifest_receipt'], output_hashes=obj['output_hashes'],
                   selection='existing collect_views: visibility score, min frame gap4, maximum12 including unchanged anchor',
                   evaluation_geometry_read=False)
        save(dest / 'views.json', row); rows.append(row)
        print(json.dumps(dict(object=obj['object_key'], views=len(records))), flush=True)
    save(out / 'views.json', {'rows': rows, 'config': receipt(config_path)})
    save(out / 'outputs.json', {'artifacts': {'views': receipt(out / 'views.json')}})


def save_rvg_gaussian(gaussian, path):
    """Pinned RVG API saves its native canonical frame and accepts only path."""
    gaussian.save_ply(str(path))


def recover_rvg_smoke(config_path, out):
    """Recover retained mesh after a Gaussian-export error, without inference."""
    import trimesh
    config=load(config_path); out=Path(out);out.mkdir(parents=True,exist_ok=False)
    failed=load(checked_path(config['failed_generation']))
    if (failed['status']!='GENERATION_FAILED' or failed['backend']!='reconviagen'
            or "unexpected keyword argument 'transform'" not in failed.get('error','')):
        raise ValueError('recovery applies only to the diagnosed Gaussian export failure')
    source=load(checked_path(config['source_config']));obj=source['objects'][0]
    if len(source['objects'])!=1 or failed['slot_id']!=obj['slot_id'] or failed['seed']!=source['seed']:
        raise ValueError('failed output identity does not match the frozen smoke')
    mesh=trimesh.load(checked_path(config['mesh']),force='mesh',process=False)
    with np.load(checked_path(config['native_arrays']),allow_pickle=False) as native:
        if (not len(mesh.faces) or not np.isfinite(mesh.vertices).all()
                or not np.allclose(mesh.vertices,native['vertices'])
                or not np.array_equal(mesh.faces,native['faces'])
                or not np.isfinite(native['vertex_attrs']).all()):
            raise ValueError('retained native arrays and mesh differ or are invalid')
    start=time.monotonic();dest=out/obj['slot_id'];dest.mkdir()
    recovered=dict(failed,status='GENERATION_COMPLETE',recovered_from=config['failed_generation'],
                   recovery_config=receipt(config_path),no_regeneration=True,
                   artifact_status={'mesh':'RECOVERED','native_arrays':'RECOVERED','gaussians':'EXPORT_FAILED_STATE_NOT_RETAINED'},
                   artifacts={'mesh':config['mesh'],'native_arrays':config['native_arrays']})
    save(dest/'generation.json',recovered)
    spec=dict(slot_id=obj['slot_id'],object=obj,generation=receipt(dest/'generation.json'),
              protocol=source['protocol'],evaluation_matching=source['evaluation_matching'])
    save(dest/'evaluation-config.json',spec)
    evaluation=evaluate(dest/'evaluation-config.json',dest/'evaluation')
    recovered['evaluation']=receipt(dest/'evaluation/evaluation.json')
    prior=load(checked_path(config['failed_batch']))
    batch=dict(status='VALIDATED' if evaluation['status']=='EVALUATED' else 'PARTIAL',
               rows=[recovered],config=receipt(config_path),recovered_generation=True,
               gaussian_export='UNAVAILABLE_FROM_FAILED_SMOKE; production API corrected without regeneration',
               wall_s=prior['wall_s']+time.monotonic()-start,
               initialization_seconds=prior['initialization_seconds'])
    save(out/'batch.json',batch);save(out/'outputs.json',{'artifacts':{'batch':receipt(out/'batch.json')}})
    if batch['status']!='VALIDATED':raise RuntimeError('recovered geometry failed evaluation')


def rvg_generate(config_path, out):
    """Use the already admitted offline RVG loader and retain native outputs."""
    import torch
    import trimesh
    from PIL import Image
    config = load(config_path); out = Path(out); out.mkdir(parents=True, exist_ok=False)
    params = config['params']
    if source_identity(params['source_root'])['commit'] != params['source_commit']:
        raise ValueError('ReconViaGen source changed')
    from .core import local_model
    paths = {k: local_model(v) for k,v in params['local_dependencies'].items()}
    paths['rvg_snapshot'] = local_model(params)
    for name in ('mip_rasterizer','official_nvdiffrast','official_utils3d','rembg_import'):
        if name in paths: sys.path.insert(0, paths[name])
    # Same xformers compatibility shim as the existing factory RVG adapter.
    import xformers.ops.fmha as fmha
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
    if not hasattr(fmha, 'BlockDiagonalMask'): fmha.BlockDiagonalMask = BlockDiagonalMask
    sys.path[:0] = [params['source_root'], str(Path(params['source_root']) / 'wheels/vggt')]
    from models.rvg_pinned import load_pipeline
    loader = out / 'loader.json'; save(loader, {'models': {k:{'path':v} for k,v in paths.items()}})
    start = time.monotonic(); pipe = load_pipeline(loader)
    pipe._device = torch.device('cuda'); pipe.low_vram = params.get('low_vram', True)
    pipe.birefnet_model.cuda()
    if not pipe.low_vram:
        for module in pipe.models.values(): module.to(pipe._device)
        pipe.VGGT_model.to(pipe._device)
    init = time.monotonic() - start; rows = []
    for obj in config['objects']:
        validate_observation(obj)
        view = load(checked_path(obj['views']))
        if view['object_key'] != obj['object_key'] or view['status'] != 'READY' or len(view['views']) < 2:
            raise ValueError('true multiview input required')
        slot = obj['slot_id']; dest = out / slot; dest.mkdir(); t0 = time.monotonic()
        row = dict(kind='reconviagen_multiview_component_v1', backend='reconviagen', slot_id=slot,
                   seed=config['seed'], source=source_identity(params['source_root']), params=params,
                   input_images=[v['image'] for v in view['views']], views=obj['views'],
                   status='GENERATION_FAILED', artifacts={})
        try:
            torch.manual_seed(config['seed']); np.random.seed(config['seed']); torch.cuda.reset_peak_memory_stats()
            images = [pipe.preprocess_image(Image.open(checked_path(v['image'])).convert('RGBA')) for v in view['views']]
            tgen = time.monotonic()
            generated, _, _ = pipe.run(image=images, seed=config['seed'], formats=['mesh','gaussian'], preprocess_image=False, **params.get('inference', {}))
            row['inference_seconds'] = time.monotonic() - tgen
            native = generated['mesh'][0]; v = native.vertices.detach().cpu().numpy(); f = native.faces.detach().cpu().numpy()
            attrs = native.vertex_attrs.detach().cpu().numpy()
            if not len(f) or not np.isfinite(v).all() or not np.isfinite(attrs).all(): raise ValueError('invalid RVG mesh')
            np.savez_compressed(dest / 'native_candidate.npz', vertices=v, faces=f, vertex_attrs=attrs)
            trimesh.Trimesh(v, f, process=False, vertex_colors=(np.clip(attrs[:,:3],0,1)*255).astype(np.uint8)).export(dest / 'rvg_mesh.ply')
            save_rvg_gaussian(generated['gaussian'][0], dest / 'rvg_gs.ply')
            row.update(status='GENERATION_COMPLETE', peak_gpu_bytes=int(torch.cuda.max_memory_allocated()),
                       artifacts={k:receipt(dest / name) for k,name in [('mesh','rvg_mesh.ply'),('gaussians','rvg_gs.ply'),('native_arrays','native_candidate.npz')]})
            del generated; torch.cuda.empty_cache()
        except Exception:
            row['error'] = traceback.format_exc()
        row['wall_s'] = time.monotonic() - t0; save(dest / 'generation.json', row)
        if row['status'] == 'GENERATION_COMPLETE':
            spec = dict(slot_id=slot, object=obj, generation=receipt(dest / 'generation.json'), protocol=config['protocol'], evaluation_matching=config['evaluation_matching'])
            save(dest / 'evaluation-config.json', spec)
            child = subprocess.run([config['cpu_python'], '-m', MODULE, 'evaluate', '--config', str(dest / 'evaluation-config.json'), '--out', str(dest / 'evaluation')], timeout=1200)
            q = dest / 'evaluation/evaluation.json'
            if child.returncode == 0 and q.exists(): row['evaluation'] = receipt(q)
        rows.append(row); print(json.dumps(dict(slot=slot, status=row['status'])), flush=True)
    good = all(r['status'] == 'GENERATION_COMPLETE' and r.get('evaluation') and load(r['evaluation']['path'])['status'] == 'EVALUATED' for r in rows)
    save(out / 'batch.json', dict(status='VALIDATED' if good else 'PARTIAL', rows=rows, initialization_seconds=init, wall_s=time.monotonic()-start, config=receipt(config_path)))
    save(out / 'outputs.json', {'artifacts': {'batch': receipt(out / 'batch.json')}})
    if not good: raise RuntimeError('RVG batch incomplete; retained evidence needs diagnosis')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['generate','evaluate','evaluate-batch','views','rvg-generate','recover-rvg-smoke'])
    p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    if a.mode == 'generate': generate(a.config, a.out)
    elif a.mode == 'evaluate-batch': evaluate_batch(a.config, a.out)
    elif a.mode == 'views': views(a.config, a.out)
    elif a.mode == 'rvg-generate': rvg_generate(a.config, a.out)
    elif a.mode == 'recover-rvg-smoke': recover_rvg_smoke(a.config, a.out)
    else:
        if evaluate(a.config, a.out)['status'] != 'EVALUATED': raise SystemExit(1)


if __name__ == '__main__': main()
