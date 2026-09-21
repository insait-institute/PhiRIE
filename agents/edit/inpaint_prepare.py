"""Inpainting stage 1 (venv, CPU): per placed object, prepare everything the
removal+fill pipeline needs.

For each non-rejected object of a factory scene:
  - removal set: scene-splat gaussians within RADIUS of the GT instance
    vertices (these are the "original object" gaussians to delete)
  - support plane: least-squares plane through the scan-mesh ring around the
    object footprint (the desk the object stood on) + footprint hull in
    plane (u,v) coords
  - related views: top-K frames by the same visibility scoring used for the
    best-view crop, plus the PROJECTED object mask per view (occlusion-aware
    raycast) for the SAM3 refinement stage
Writes OUT/inpaint/{removal_mask.npz, obj_XX/{plane.json, views.json,
proj_masks.npz}}.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from agents.core import common as C

RADIUS = 0.03       # m, gaussian-removal distance from instance surface
TOP_K_VIEWS = 3
RING = 0.10         # m, xy ring around footprint for plane fitting


def surface_removal_indices(splat_tree, points, radius=RADIUS):
    """Canonical radius-based observed/asset surface removal primitive."""
    hit=splat_tree.query_ball_point(points,radius)
    parts=[np.asarray(h,dtype=np.int64) for h in hit if h]
    return np.unique(np.concatenate(parts or [np.array([],dtype=np.int64)]))


def fit_plane(pts):
    """SVD plane with 2 rounds of MAD trimming (nearby-object contamination
    sits 0-30mm above the desk and would bias/tilt a plain LSQ fit).
    Also returns trim_ok: False means outliers were detected on the first
    pass but trimming would leave <50 pts, so the result is the untrimmed
    one-shot fit (possibly biased by contamination)."""
    keep = pts
    trim_ok = False
    for _ in range(3):
        o = keep.mean(axis=0)
        _, _, vt = np.linalg.svd(keep - o, full_matrices=False)
        n = vt[2]
        res = (keep - o) @ n
        mad = np.median(np.abs(res - np.median(res))) + 1e-6
        sel = np.abs(res - np.median(res)) < 2.5 * mad
        if sel.all():       # fit is MAD-consistent: converged
            trim_ok = True
            break
        if sel.sum() < 50:  # can't trim without losing support: keep fit as-is
            break
        keep = keep[sel]    # trim BEFORE any break so the refit sees it
        trim_ok = True
    if n[2] < 0:
        n = -n
    u = np.cross(n, [1.0, 0, 0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(n, [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return o, n, u, v, trim_ok



PUBLIC_ALGORITHM = {"radius_m": RADIUS, "top_k_views": TOP_K_VIEWS,
                    "ring_m": RING, "asset_samples": 5000}



def _validate_source_factory(factory, manifest, scene_id):
    """Run the public validator at its recorded clean source, without relabeling."""
    import os
    import subprocess
    import sys
    from robo.eval import e3_factory_materializer as materializer
    provenance = manifest['provenance']
    source = Path(provenance['code_root']).resolve(strict=True)
    commit = provenance['validator_commit']
    if (provenance['materializer_commit'] != commit
            or provenance['materializer_dirty'] is not False
            or provenance['validator_dirty'] is not False):
        raise ValueError('original materializer source provenance differs')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=source, text=True)
    if head != commit or dirty.strip():
        raise ValueError('original materializer worktree is no longer exact and clean')
    if source == materializer.CODE_ROOT:
        return materializer.validate_materialized_factory(factory, expected_scene_id=scene_id, expected_policy_id='A4')
    script = ("import json,sys; from robo.eval.e3_factory_materializer import validate_materialized_factory; "
              "r=json.load(sys.stdin); print(json.dumps(validate_materialized_factory(r['factory'],"
              "expected_scene_id=r['scene'],expected_policy_id='A4')))")
    env = dict(os.environ, SIMANY_EVIDENCE_ROOT=str(materializer.REPOSITORY_ROOT), PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-c', script], cwd=source, env=env,
        input=json.dumps({'factory': str(factory), 'scene': scene_id}), text=True, capture_output=True, check=True)
    report = json.loads(result.stdout)
    if (report['scene_id'] != scene_id or report['policy_id'] != 'A4'
            or report['validator_commit'] != commit
            or report['manifest_sha256'] != materializer.sha256_file(factory/'materialization_manifest.json')):
        raise ValueError('original materializer validation receipt differs')
    return report

def _validate_public_context(context_path, contract_path):
    """Authenticate public automatic assets and their original TRAIN boundary."""
    import yaml
    from robo.eval import agentic_ablation as e3
    from robo.eval import e3_factory_materializer as materializer
    from run.icra2027.e3_fresh_generation_contract import checked_identity, discovery_binding, FRESH
    from run.icra2027.e3_trellis_generation_pilot import source_jobs

    context_path = Path(context_path).absolute()
    context = yaml.safe_load(context_path.read_text())
    expected = {'schema_version', 'scope', 'freeze_id', 'scene_id', 'policy_id',
                'materialization_manifest', 'generation_config', 'seed', 'algorithm'}
    if (set(context) != expected or context['schema_version'] != 1
            or context['scope'] != 'automatic_train_only_removal_preparation'
            or context['policy_id'] != 'A4' or type(context['seed']) is not int
            or context['seed'] != 0 or context['algorithm'] != PUBLIC_ALGORITHM):
        raise ValueError('public removal context or frozen algorithm differs')
    root = e3.REPOSITORY_ROOT/'outputs/icra2027'/context['freeze_id']
    e3._validate_cli_execution(contract_path, context['freeze_id'], root,
                               config_paths=[context_path])
    manifest_path = checked_identity(context['materialization_manifest'])
    factory = e3.checked_repo_path(manifest_path.parent, 'public factory', kind='dir')
    if manifest_path.name != 'materialization_manifest.json':
        raise ValueError('public materialization manifest name differs')
    e3._require_scene_id(context['scene_id'])
    preparation = e3.checked_repo_path(root/'fidelity/removal'/context['scene_id'], 'public preparation root', must_exist=False)
    if preparation == factory or factory in preparation.parents or preparation in factory.parents:
        raise ValueError('public preparation must be separate from immutable source factory')
    for name in ('inpaint', 'inpaint_failed_partial', 'inpaint_public_failure.json', '.inpaint-public.claim.json'):
        if (preparation/name).exists() or (preparation/name).is_symlink():
            raise FileExistsError(f'public inpaint output/claim already exists: {name}')
    manifest = json.loads(manifest_path.read_text())
    validation = _validate_source_factory(factory, manifest, context['scene_id'])
    if manifest['source_scene']['source_gaussian_training_provenance'] != FRESH:
        raise ValueError('public removal requires authenticated TRAIN Gaussian')
    construction = e3.checked_repo_path(manifest['e3_root'], 'source construction', kind='dir')
    jobs, _, audit, _ = materializer._verify_inventory(construction)
    scene_audit = materializer._automatic_scene_audit(jobs, audit, context['scene_id'])
    pool_path = e3.checked_repo_path(scene_audit['initial_pools']['trellis']['path'], 'source pool', kind='file')
    if e3.sha256_file(pool_path) != scene_audit['initial_pools']['trellis']['sha256']:
        raise ValueError('public initial source pool changed')
    pool = json.loads(pool_path.read_text())
    inputs_path = pool_path.parent/'input_manifest.json'
    if e3.sha256_file(inputs_path) != pool['input_manifest_sha256']:
        raise ValueError('public initial source input manifest changed')
    inputs = json.loads(inputs_path.read_text())
    generation_path = checked_identity(context['generation_config'])
    if e3.sha256_file(generation_path) != inputs['config_sha256']:
        raise ValueError('public generation config differs from sealed initial pool')
    generation = yaml.safe_load(generation_path.read_text())
    _, binding = discovery_binding(generation, source_jobs)
    discovery = Path(generation['source_pilot'])
    descriptor_path = e3.checked_repo_path(manifest['source_scene']['automatic_scene_descriptor']['path'],
        'public automatic scene descriptor', kind='file')
    descriptor = json.loads(descriptor_path.read_text())
    if (discovery != Path(descriptor['discovery_directory'])
            or binding['source_discovery_hashes'] != descriptor['discovery_hashes']):
        raise ValueError('public generation and materialization discovery differ')
    boundary = json.loads((discovery/'input_manifest.json').read_text())
    training = boundary['boundary']['training_frames']
    metadata = {key: checked_identity(value) for key, value in boundary['metadata'].items()}
    split = json.loads(metadata['train_test_lists.json'].read_text())
    if (boundary.get('scene_id') != context['scene_id']
            or binding['gaussian_provenance']['status'] != FRESH
            or binding['gaussian_provenance']['training_frames'] != training
            or len(training) != len(set(training)) or not training
            or not set(training) <= set(split['train']) or set(training) & set(split['test'])
            or set(boundary['input_images']) != set(training)):
        raise ValueError('public camera roster differs from authenticated official TRAIN')
    for name, record in boundary['input_images'].items():
        if Path(name).name != name or checked_identity(record).name != name:
            raise ValueError('public TRAIN image identity differs')
    scene = e3._scene_inventory(jobs, context['scene_id'])
    expected_sources = {'gaussian': scene['source_scene_gaussian'], **scene['camera_artifacts']}
    observed_sources = {'gaussian': boundary['gaussian'],
        'intrinsics': boundary['metadata']['nerfstudio/transforms_undistorted.json'],
        'poses': boundary['metadata']['colmap/images.txt']}
    for role, expected_identity in expected_sources.items():
        value = observed_sources[role]
        if (value['path'], value['bytes'], value['sha256']) != (
                expected_identity['path'], expected_identity['size_bytes'], expected_identity['sha256']):
            raise ValueError('public camera/background differs from construction')
        checked_identity(value)
    return dict(factory=factory, preparation_root=preparation, source_validation=validation, splat=Path(boundary['gaussian']['path']),
                intrinsics=metadata['nerfstudio/transforms_undistorted.json'],
                poses=metadata['colmap/images.txt'], training_frames=training,
                context=context, context_sha256=e3.sha256_file(context_path),
                materialization_sha256=e3.sha256_file(manifest_path),
                boundary=boundary, gaussian_provenance=binding['gaussian_provenance'])


def _public_objects(factory, instances):
    """All planned automatic slots stay explicit; GT aliases are never accepted."""
    objects = json.loads((factory/'objects/objects.json').read_text())
    by_id = {row['object_id']: row for row in instances}
    seen = set()
    for row in objects:
        index = row.get('automatic_instance_id')
        if ('gt_object_id' in row or row.get('instance_namespace') != 'automatic'
                or type(index) is not int or row.get('index') != index
                or index in seen or index not in by_id):
            raise ValueError('public automatic object identity differs or uses a GT alias')
        seen.add(index)
        alignment = json.loads((factory/'objects'/f'obj_{index:02d}'/'aligned.json').read_text())
        action = alignment.get('terminal_action')
        if (action not in {'accept', 'reject', 'abstain'}
                or alignment.get('construction_eligible') is not (action == 'accept')
                or bool(alignment.get('rejected')) != (action != 'accept')
                or 'gt_object_id' in alignment):
            raise ValueError('public object terminal state is malformed')
    if seen != set(by_id):
        raise ValueError('public automatic planned-object denominator differs')
    return objects


def _sample_asset(object_dir, alignment, *, public=False):
    import trimesh
    from agents.assets.s5_align import apply_T
    mesh = trimesh.load(object_dir/('trellis_mesh.ply' if public else 'mesh_sim.ply'), process=False)
    if public and (not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces)
                   or not np.isfinite(mesh.vertices).all() or not np.isfinite(mesh.area)
                   or mesh.area <= 0):
        raise ValueError('public accepted raw mesh is empty or non-finite')
    kwargs = {'seed': 0} if public else {}
    points, _ = trimesh.sample.sample_surface(mesh, 5000, **kwargs)
    transform = np.asarray(alignment['T'], dtype=float)
    if public and (transform.shape != (4, 4) or not np.isfinite(transform).all()
                   or not np.allclose(transform[3], [0, 0, 0, 1])):
        raise ValueError('public accepted transform is malformed')
    points = apply_T(transform, np.asarray(points))  # canonical raw mesh; exactly once
    if public and not np.isfinite(points).all():
        raise ValueError('public transformed mesh samples are non-finite')
    return points


def run_public(context_path, contract_path):
    """Atomic opt-in preparation only; no enhancement, fill or quality claim."""
    import os
    import sys
    from robo.eval import agentic_ablation as e3
    from robo.eval import e3_factory_materializer as materializer
    from run.icra2027.e3_auto_discovery_pilot import enforce_read_boundary
    public = _validate_public_context(context_path, contract_path)
    factory = public['factory']
    preparation = public['preparation_root']
    preparation.mkdir(parents=True, exist_ok=True)
    image_root = Path(next(iter(public['boundary']['input_images'].values()))['path']).parent
    forbidden_scene = image_root.parent.parent
    def guard(event, arguments):
        if event == 'socket.connect':
            raise ValueError('public removal preparation forbids network access')
        enforce_read_boundary(event, arguments, forbidden_scene=forbidden_scene,
            image_root=image_root, allowed_images=set(public['training_frames']))
    sys.addaudithook(guard)
    claim = preparation/'.inpaint-public.claim.json'
    materializer._write_inside(claim, materializer._json_bytes({
        'context_sha256': public['context_sha256'], 'pid': os.getpid()}))
    try:
        with e3._atomic_directory(preparation/'inpaint') as staging:
            public['inpaint'] = staging
            try:
                result = _prepare(public)
                result.update(schema_version=1, scope='automatic_train_only_removal_preparation',
                    context_sha256=public['context_sha256'], materialization_sha256=public['materialization_sha256'],
                    source_factory=str(factory), source_validation=public['source_validation'],
                    seed=0, algorithm=PUBLIC_ALGORITHM, training_frames=public['training_frames'],
                    gaussian_provenance=public['gaussian_provenance'], paper_ready=False,
                    cleaned_background_created=False, official_test_images_read=0)
                materializer._write_inside(staging/'public_prepare.json', materializer._json_bytes(result))
                members = {str(p.relative_to(staging)): e3.sha256_file(p) for p in staging.rglob('*') if p.is_file()}
                materializer._write_inside(staging/'seal.json', materializer._json_bytes({'schema_version': 1, 'members': members}))
            except Exception:
                staging.rename(preparation/'inpaint_failed_partial')
                raise
        return result
    except Exception as error:
        materializer._write_inside(preparation/'inpaint_public_failure.json', materializer._json_bytes({
            'status': 'FAILED', 'classification': 'construction_preparation_failure',
            'context_sha256': public['context_sha256'], 'error_type': type(error).__name__,
            'error': str(error), 'cause': str(error.__cause__) if error.__cause__ else None,
            'partial_output': str(preparation/'inpaint_failed_partial'), 'paper_ready': False}))
        raise

def _prepare(public=None):
    import open3d as o3d
    from plyfile import PlyData
    from agents.assets.s5_align import scene_mesh_arrays

    factory = public['factory'] if public else C.OUT
    if public:
        K, W, H, _ = C.load_intrinsics(public['intrinsics'])
        poses = C.load_colmap_w2c(public['poses'])
        if not set(public['training_frames']) <= set(poses):
            raise ValueError('public TRAIN camera is unregistered')
        w2c_all = [(name, poses[name]) for name in sorted(public['training_frames'])]
        mesh = o3d.io.read_triangle_mesh(str(factory/'derived_mesh.ply'))
        verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        if not len(verts) or not len(faces) or not np.isfinite(verts).all():
            raise ValueError('public derived mesh is empty or non-finite')
        scene_full = o3d.t.geometry.RaycastingScene()
        scene_full.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        instances = C.load_auto_instances(instances_path=factory/'auto_instances.npz', mesh_path=factory/'derived_mesh.ply')
        objects = _public_objects(factory, instances)
    else:
        K, W, H, _ = C.load_intrinsics()
        w2c_all = sorted(C.load_colmap_w2c().items())
        verts, faces = scene_mesh_arrays()
        scene_full = C.make_raycast_scene()
        instances = C.load_instances()
        objects = json.loads((factory/'objects/objects.json').read_text())
    gts = {g['object_id']: g for g in instances}

    splat_xyz = np.stack([np.asarray(PlyData.read(str(public['splat'] if public else C.SPLAT_PLY))["vertex"][a],
                                     dtype=np.float64) for a in "xyz"], axis=1)
    if public and (not len(splat_xyz) or not np.isfinite(splat_xyz).all()
                   or K.shape != (3, 3) or not np.isfinite(K).all()
                   or K[0, 0] <= 0 or K[1, 1] <= 0 or W <= 0 or H <= 0
                   or any(w.shape != (4, 4) or not np.isfinite(w).all() for _, w in w2c_all)):
        raise ValueError('public Gaussian coordinates or calibrated cameras are malformed')
    splat_tree = cKDTree(splat_xyz)

    inp = public["inpaint"] if public else factory / "inpaint"
    inp.mkdir(exist_ok=True)
    union = np.zeros(len(splat_xyz), bool)
    rng = np.random.RandomState(0)

    seen_gt = set()
    asset_rm_failed = []
    statuses = []
    for m in objects:
        name = f"obj_{m['index']:02d}"
        # stale per-object outputs poison downstream existence-guards on rerun
        import shutil
        if not public:
            shutil.rmtree(inp / name, ignore_errors=True)
        al = json.loads((factory / "objects" / name / "aligned.json").read_text())
        if al.get("rejected"):
            if public:
                statuses.append(dict(object_slot=name, automatic_instance_id=m['automatic_instance_id'],
                    terminal_action=al['terminal_action'], preparation_status='NOT_APPLICABLE',
                    reason_codes=al.get('reason_codes', [])))
            continue
        identity_key = 'automatic_instance_id' if public else 'gt_object_id'
        if m.get(identity_key) is None or m[identity_key] in seen_gt:
            print(f"[ip] {name}: no/duplicate GT id, skip")
            continue
        seen_gt.add(m[identity_key])
        odir = inp / name
        odir.mkdir(exist_ok=True)
        gt = gts[m[identity_key]]
        gv = verts[gt["vert_idx"]]

        # --- removal set ---------------------------------------------------
        # (a) near the GT instance surface; (b) near the ALIGNED generated
        # asset surface - scans miss transparent parts (bottle bodies), so
        # their splat gaussians sit far from any GT vertex, but the generated
        # asset covers the full extent
        idx_parts = [surface_removal_indices(splat_tree,gv,RADIUS)]
        try:
            apts = _sample_asset(factory/'objects'/name, al, public=bool(public))
            idx_parts.append(surface_removal_indices(splat_tree,apts,RADIUS * 0.8))
        except Exception as e:
            if public:
                raise ValueError(f"public accepted asset {name} failed removal preparation") from e
            print(f"[ip] {name}: asset-volume removal skipped ({e})")
            asset_rm_failed.append(name)
        idx = np.unique(np.concatenate(
            idx_parts or [np.array([], dtype=np.int64)]))
        mask = np.zeros(len(splat_xyz), bool)
        mask[idx] = True
        union |= mask
        np.save(odir / "removal_idx.npy", idx)

        # --- support plane + footprint ------------------------------------
        zmin = gv[:, 2].min()
        cen = gv.mean(axis=0)
        d_xy = np.linalg.norm(verts[:, :2] - cen[:2], axis=1)
        r_obj = np.linalg.norm(gv[:, :2] - cen[:2], axis=1).max()
        ring = ((d_xy > r_obj) & (d_xy < r_obj + RING) &
                (np.abs(verts[:, 2] - zmin) < 0.03))
        inset = np.zeros(len(verts), bool)
        inset[gt["vert_idx"]] = True
        ring &= ~inset
        if ring.sum() < 100:
            print(f"[ip] {name}: no support ring ({ring.sum()} pts) - "
                  "floor-standing or shelved object; skip plane fill")
            plane = None
        else:
            o, n, u, v, trim_ok = fit_plane(verts[ring])
            if not trim_ok:
                print(f"[ip] {name}: WARNING plane MAD-trim fell back to "
                      f"one-shot fit ({int(ring.sum())} ring pts, "
                      "contamination kept)")
            fp = gv - o
            uv = np.stack([fp @ u, fp @ v], axis=1)  # hull computed in fill
            plane = {"origin": o, "normal": n, "u": u, "v": v,
                     "trim_ok": trim_ok,
                     "footprint_uv": uv[rng.choice(len(uv),
                                                   min(len(uv), 800),
                                                   replace=False)]}
            C.save_json(odir / "plane.json", plane)

        # --- related views + projected masks -------------------------------
        sample = gv[rng.choice(len(gv), min(120, len(gv)), replace=False)]
        cand = []
        for fname, w2c in w2c_all:
            pc = sample @ w2c[:3, :3].T + w2c[:3, 3]
            z = pc[:, 2]
            if (z < 0.25).mean() > 0.05:
                continue
            upx = pc[:, 0] / z * K[0, 0] + K[0, 2]
            vpx = pc[:, 1] / z * K[1, 1] + K[1, 2]
            inb = (z > 0.25) & (upx >= 0) & (upx < W) & (vpx >= 0) & (vpx < H)
            if inb.mean() < 0.7:
                continue
            cam = np.linalg.inv(w2c)[:3, 3]
            dirs = sample[inb] - cam
            dist = np.linalg.norm(dirs, axis=1)
            rays = o3d.core.Tensor(np.concatenate(
                [np.broadcast_to(cam, dirs.shape), dirs / dist[:, None]],
                axis=1).astype(np.float32))
            t_hit = scene_full.cast_rays(rays)["t_hit"].numpy()
            vis = float((np.abs(t_hit - dist) < 0.02).mean()) * inb.mean()
            if vis < 0.5:
                continue
            cand.append((vis * np.sqrt(np.ptp(upx[inb]) * np.ptp(vpx[inb])),
                         fname, w2c))
        cand.sort(key=lambda c: -c[0])
        views = cand[:TOP_K_VIEWS]

        # occlusion-aware projected mask per view (instance submesh raycast)
        fmask = inset[faces].all(axis=1)
        sub = o3d.t.geometry.RaycastingScene()
        sub.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts),
                o3d.utility.Vector3iVector(faces[fmask]))))
        pm, vnames = [], []
        for _, fname, w2c in views:
            c2w = np.linalg.inv(w2c)
            uu, vv = np.meshgrid(np.arange(W, dtype=np.float64) + 0.5,
                                 np.arange(H, dtype=np.float64) + 0.5)
            d_cam = np.stack([(uu - K[0, 2]) / K[0, 0],
                              (vv - K[1, 2]) / K[1, 1],
                              np.ones_like(uu)], axis=-1)
            d_world = (d_cam @ c2w[:3, :3].T).reshape(-1, 3).astype(np.float32)
            o_world = np.broadcast_to(c2w[:3, 3], d_world.shape).astype(np.float32)
            rays = o3d.core.Tensor(np.concatenate([o_world, d_world], axis=1))
            t_i = sub.cast_rays(rays)["t_hit"].numpy()
            t_f = scene_full.cast_rays(rays)["t_hit"].numpy()
            pm.append((np.isfinite(t_i) & (t_i < t_f + 0.005))
                      .reshape(H, W))
            vnames.append(fname)
        np.savez_compressed(odir / "proj_masks.npz",
                            masks=np.stack(pm) if pm else np.zeros((0, H, W), bool),
                            frames=np.array(vnames))
        C.save_json(odir / "views.json",
                    [{"frame": f, "w2c": w.tolist()} for _, f, w in views])
        if public:
            statuses.append(dict(object_slot=name, automatic_instance_id=m['automatic_instance_id'],
                terminal_action='accept', preparation_status='PREPARED', removal_gaussians=len(idx),
                plane_status='NO_PLANE' if plane is None else 'FIT' if plane['trim_ok'] else 'TRIM_NOT_CONVERGED',
                view_status='VIEWS_SELECTED' if views else 'NO_VIEW', selected_frames=vnames,
                projected_mask_pixels=[int(mask.sum()) for mask in pm]))
        print(f"[ip] {name} {m['label']}: remove {len(idx)} gaussians, "
              f"plane={'ok' if plane else 'none'}, views={vnames}")

    np.save(inp / "removal_union_idx.npy", np.nonzero(union)[0])
    C.save_json(inp / "prepare_meta.json",
                {"asset_removal_failures": len(asset_rm_failed),
                 "asset_removal_failed": asset_rm_failed})
    print(f"[ip] union removal: {int(union.sum())} / {len(splat_xyz)} gaussians")
    print(f"[ip] asset-volume removal failures: {len(asset_rm_failed)}"
          + (f" -> {asset_rm_failed}" if asset_rm_failed else ""))

    if public:
        accepted = sum(row['terminal_action'] == 'accept' for row in statuses)
        return dict(status='PREPARED' if accepted else 'NO_ACCEPTED_OBJECTS', planned_objects=len(objects),
            accepted_objects=accepted, rejected_objects=sum(r['terminal_action']=='reject' for r in statuses),
            abstained_objects=sum(r['terminal_action']=='abstain' for r in statuses),
            no_plane_objects=sum(r.get('plane_status')=='NO_PLANE' for r in statuses),
            no_view_objects=sum(r.get('view_status')=='NO_VIEW' for r in statuses),
            empty_projected_mask_views=sum(n == 0 for r in statuses for n in r.get('projected_mask_pixels', [])),
            plane_trim_not_converged_objects=sum(r.get('plane_status')=='TRIM_NOT_CONVERGED' for r in statuses),
            removed_gaussians=int(union.sum()), source_gaussians=len(splat_xyz), objects=statuses)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-context')
    parser.add_argument('--contract-manifest')
    args = parser.parse_args(argv)
    if bool(args.public_context) != bool(args.contract_manifest):
        parser.error('public context and E0 contract are required together')
    if args.public_context:
        return run_public(args.public_context, args.contract_manifest)
    return _prepare()


if __name__ == "__main__":
    main()
