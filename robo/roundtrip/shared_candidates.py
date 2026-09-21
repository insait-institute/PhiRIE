"""Shared native TRAIN candidate pool using existing generators/controller.

Run inside the constructor allowlist. The B0 build is read-only; native assets,
states, task outcomes and held-out observations are not arguments to this CLI.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import hashlib
import os
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image

from robo.roundtrip.build import _sha, _write, read_train


def checked_b0(b0, capture):
    b0, capture = Path(b0), Path(capture)
    manifest = json.loads((b0 / "build_manifest.json").read_text())
    discovery = json.loads((b0 / "discovery/discovery_manifest.json").read_text())
    read_train(capture)
    if manifest["status"] != "BUILT" or manifest["capture_manifest_sha256"] != _sha(capture / "capture_manifest.json"):
        raise ValueError("B0/capture identity differs")
    for name, digest in manifest["source_hashes"].items():
        path = b0 / name
        if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink() or _sha(path) != digest:
            raise ValueError("B0 source closure differs")
    if discovery["capture_manifest_sha256"] != manifest["capture_manifest_sha256"]:
        raise ValueError("discovery capture differs")
    return manifest, discovery


def proposal_prefix(manifest, discovery):
    capture=discovery['capture_manifest_sha256']
    role=manifest.get('object_role','target')
    if role=='target':return capture[:20]  # preserve original frozen target IDs
    if role not in ('receptacle','support'):raise ValueError('unsupported candidate-pool dependency role')
    return hashlib.sha256((capture+'|'+role).encode()).hexdigest()[:20]


def rvg_views(b0, capture):
    """Use already selected automatic TRAIN masks, never oracle view selection."""
    _, discovery = checked_b0(b0, capture)
    _, cameras = read_train(capture)
    cameras = {r["frame_id"]: r for r in cameras}
    candidates = {r["candidate_id"]: r for r in json.loads((Path(b0) / "discovery/candidates.json").read_text())}
    views, receipts = [], []
    for cid in discovery["accepted_candidates"]:
        row = candidates[cid]; camera = cameras[row["frame_id"]]
        path = Path(b0) / "discovery" / (cid + ".png")
        if _sha(path) != row["mask_sha256"]:
            raise ValueError("automatic source mask changed")
        mask = np.asarray(Image.open(path)) > 127
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            raise ValueError("confirmed source mask is empty")
        height, width = mask.shape
        pad = int(.15 * max(int(xs.max()-xs.min()), int(ys.max()-ys.min())) + 8)
        box = (max(0, int(xs.min())-pad), max(0, int(ys.min())-pad),
               min(width, int(xs.max())+pad+1), min(height, int(ys.max())+pad+1))
        u0,v0,u1,v1 = box
        views.append((camera["rgb"], box, mask[v0:v1,u0:u1]))
        receipts.append(dict(candidate_id=cid, frame_id=row["frame_id"], bbox=list(box),
                             rgb_sha256=_sha(Path(capture)/camera["rgb"]), mask_sha256=row["mask_sha256"]))
    if len(views) < 2:
        raise ValueError("RVG requires at least two confirmed TRAIN views")
    return views, receipts


def generate_rvg(b0, capture, out, model_config):
    """Delegate inference and mesh export to the canonical RVG factory wrapper."""
    from run.icra2027.e3_trellis_generation_pilot import validate_models
    validate_models(model_config)
    models = model_config["models"]
    os.environ.update(SIMANY_RVG_DIR=models["rvg_source"]["path"],
                      TORCH_HOME=str(Path(models["dinov2_source"]["path"]).parent.parent),
                      SIMANY_NO_GT="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from agents.assets.factory_hybrid import run_rvg, build_sim_mesh
    from agents.core import common as C
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    _write(out / "pinned_models.json", model_config)
    os.environ["SIMANY_RVG_PINNED_CONFIG"] = str(out / "pinned_models.json")
    views, view_receipts = rvg_views(b0, capture)
    _, discovery = checked_b0(b0, capture)
    _write(out / "view_receipt.json", dict(read_splits=["train"], rows=view_receipts,
        b0_manifest_sha256=_sha(Path(b0)/"build_manifest.json"),
        capture_manifest_sha256=discovery["capture_manifest_sha256"]))
    previous = C.IMAGES_DIR
    started = time.monotonic()
    try:
        C.IMAGES_DIR = Path(capture)
        # None geometry/GT is safe ONLY because the existing wrapper receives
        # explicit frozen observations and never calls its legacy view collector.
        producer_s = run_rvg({"pipe": None}, {"index":0,"label":discovery["target_prompt"]},
                            out, None, (None,)*7, len(views), precollected_views=views)
        build_sim_mesh(out, out/"rvg/rvg_mesh.ply")
        result = dict(status="available", tool="reconviagen", producer_wall_s=producer_s,
                      artifacts={str(p.relative_to(out)):_sha(p) for p in out.rglob("*") if p.is_file()})
    except Exception as exc:
        result = dict(status="tool_failure", tool="reconviagen", error_type=type(exc).__name__, reason=str(exc))
    finally:
        C.IMAGES_DIR = previous
    result.update(wall_s=time.monotonic()-started, seed=42,
                  b0_manifest_sha256=_sha(Path(b0)/"build_manifest.json"),
                  model_config_sha256=_sha(out/"pinned_models.json"))
    _write(out/"rvg_receipt.json", result)
    return result


def select_pool(b0, capture, rvg, out, policy_config, source_commit, *, aligner=None, collision=None):
    """B0/B1/B2/B3 reuse the exact initial pool; A3 never applies A4 abstention."""
    from agents.orchestrator.controller import run_policies
    from agents.orchestrator.runtime import align_and_probe
    from agents.assets.s6_physics import coacd_parts
    aligner = align_and_probe if aligner is None else aligner
    collision = coacd_parts if collision is None else collision
    started=time.monotonic()
    manifest, discovery = checked_b0(b0, capture)
    b0,rvg,out = Path(b0),Path(rvg),Path(out)
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("exact source commit required")
    rr = json.loads((rvg/"rvg_receipt.json").read_text())
    if rr["b0_manifest_sha256"] != _sha(b0/"build_manifest.json"):
        raise ValueError("RVG pool belongs to a different canonical B0 build")
    out.mkdir(parents=True,exist_ok=False)
    observation = np.load(b0/"discovery/observation_points.npy", allow_pickle=False)
    initial, sources, alignments, failures = [], {}, {}, []
    prefix = proposal_prefix(manifest,discovery)
    trellis_dir = b0/"construction/objects/obj_00"
    ev = json.loads((b0/"registration/evidence.json").read_text())
    reg = json.loads((b0/"registration/registration.json").read_text())
    pid = prefix+":trellis:initial:42"
    initial.append(dict(proposal_id=pid,tool="trellis",evidence=ev["raw_values"],parent_proposal_ids=[]))
    sources[pid]=trellis_dir;alignments[pid]=reg
    if rr["status"] == "available":
        for name,digest in rr["artifacts"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts or (rvg/name).is_symlink() or _sha(rvg/name)!=digest:
                raise ValueError("RVG candidate closure differs")
        pid=prefix+":reconviagen:initial:42"
        result=aligner(rvg/"mesh_sim.ply",observation,label=discovery["target_prompt"],
            visible_fraction=discovery["anchor_mask_pixel_fraction"],out_dir=out/"rvg_registration",
            signed_source_up=False,producer_commit=source_commit,
            input_hashes={"mesh":_sha(rvg/"mesh_sim.ply"),"observation":discovery["observation_sha256"]})
        initial.append(dict(proposal_id=pid,tool="reconviagen",evidence=result["evidence"],parent_proposal_ids=[]))
        sources[pid]=rvg;alignments[pid]=result["alignment"]
    elif rr["status"] == "tool_failure":
        failures.append(rr)
    else:
        raise ValueError("RVG must have a terminal available/tool_failure receipt")
    before=run_policies(initial,policy_config)
    retry=None
    if before.retry_required and before.retry_parent_proposal_id is not None:
        parent=before.retry_parent_proposal_id
        result=aligner(sources[parent]/"mesh_sim.ply",observation,label=discovery["target_prompt"],
            visible_fraction=discovery["anchor_mask_pixel_fraction"],out_dir=out/"registration_retry",
            signed_source_up=True,producer_commit=source_commit,
            input_hashes={"mesh":_sha(sources[parent]/"mesh_sim.ply"),"observation":discovery["observation_sha256"]})
        if np.array_equal(np.asarray(result["alignment"]["T"]), np.asarray(alignments[parent]["T"])):
            failures.append(dict(tool="registration_retry",status="no_new_transform",parent_proposal_id=parent,
                                 actual_action="alternative_signed_source_up",wall_s=result["wall_s"]))
        else:
            pid=prefix+":registration_retry:1"
            retry=dict(proposal_id=pid,tool="registration_retry",evidence=result["evidence"],parent_proposal_ids=[parent])
            sources[pid]=sources[parent];alignments[pid]=result["alignment"]
    final=run_policies(initial,policy_config,retry_candidate=retry)
    rows=[];materialized={}
    for outcome in final.outcomes[:4]:
        row=asdict(outcome); row["native_method"]={"A0":"B0_FIXED_NATIVE","A1":"B1_FIXED_PRIORITY","A2":"B2_EVIDENCE","A3":"B3_AGENT_NATIVE"}[outcome.policy_id]
        pid=outcome.selected_proposal_id
        if pid is not None:
            if pid not in materialized:
                dst=out/f"selected_{len(materialized):02d}";dst.mkdir()
                for name in ("mesh_sim.ply","mesh_sim.obj"):
                    shutil.copyfile(sources[pid]/name,dst/name)
                r=alignments[pid]
                if pid == initial[0]["proposal_id"]:
                    # Reuse exact B0 export: do not regenerate a random collision
                    # decomposition and turn reuse into a hidden intervention.
                    shutil.copyfile(trellis_dir/"aligned.json",dst/"aligned.json")
                    shutil.copytree(trellis_dir/"collision",dst/"collision")
                else:
                    from robo.roundtrip.asset_metadata import exported_dimensions
                    _write(dst/"aligned.json",dict(T=r["T"],scale=r["scale"],world_dims=exported_dimensions(dst,r["scale"])))
                    parts=collision(dst,world_max_dim=max(r["world_dims_m"]))
                    if not parts:raise ValueError("selected candidate produced no native collision parts")
                # Uniform appearance and category-independent physics for all arms.
                shutil.copyfile(trellis_dir/"physics.json",dst/"physics.json")
                materialized[pid]=dst
            dst=materialized[pid]
            row.update(object_dir=str(dst),artifact_hashes={str(p.relative_to(dst)):_sha(p) for p in dst.rglob("*") if p.is_file()})
        rows.append(row)
    receipt=dict(schema_version=2,initial_candidates=initial,retry_candidate=retry,outcomes=rows,
        failures=failures,source_commit=source_commit,planned_objects=1,
        capture_manifest_sha256=discovery["capture_manifest_sha256"],b0_manifest_sha256=_sha(b0/"build_manifest.json"),
        rvg_receipt_sha256=_sha(rvg/"rvg_receipt.json"),controller="agents.orchestrator.controller.run_policies",
        isolated_probe_is_native_context_evidence=False,policy_outcomes="NOT_RUN",
        object_role=manifest.get("object_role","target"),canonical_instance_id=manifest.get("canonical_instance_id"),
        cohort_id=manifest.get("cohort_id"),selection_wall_s=time.monotonic()-started)
    _write(out/"candidate_pool.json",receipt)
    with (out/"selection_ledger.jsonl").open("x") as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
    return receipt


def fuse_masked_observations(capture,views,center,physics_path,out,config,*,workspace_bounds_world_m=None):
    """Shared existing masked TSDF/CoACD core; observation center, never GT fit."""
    import open3d as o3d
    import trimesh
    from agents.discover.derive_mesh_from_splat import new_tsdf_volume,integrate_rgbd
    from agents.assets.s6_physics import coacd_parts
    from robo.roundtrip.capture import backproject_camera_z
    if config!={'voxel_m':.002,'sdf_trunc_m':.006}:raise ValueError('frozen DEV observed-surface config required')
    capture=Path(capture);out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    center=np.asarray(center,dtype=float)
    if center.shape!=(3,) or not np.isfinite(center).all():raise ValueError('finite observed center required')
    bounds=None if workspace_bounds_world_m is None else np.asarray(workspace_bounds_world_m,dtype=float)
    if bounds is not None and (bounds.shape!=(2,3) or not np.isfinite(bounds).all() or np.any(bounds[1]<=bounds[0])):raise ValueError('invalid observed workspace bound')
    volume=new_tsdf_volume(config['voxel_m'],config['sdf_trunc_m']);source_views=[]
    for view in views:
        camera=view['camera'];maskpath=Path(view['mask_path'])
        if _sha(maskpath)!=view['mask_sha256']:raise ValueError('observed control mask changed')
        mask=np.asarray(Image.open(maskpath))>127
        rgb=np.asarray(Image.open(capture/camera['rgb']).convert('RGB'))
        depth=np.load(capture/camera['depth_m'],allow_pickle=False).astype(np.float32)
        if mask.shape!=depth.shape:raise ValueError('mask/depth grid differs')
        if bounds is not None:
            points,valid=backproject_camera_z(depth,camera['K'],camera['T_world_from_camera'])
            mask=mask&valid&np.all(points>=bounds[0],axis=-1)&np.all(points<=bounds[1],axis=-1)
        depth[~mask|~np.isfinite(depth)|(depth<.05)]=0
        K=np.asarray(camera['K']);h,w=depth.shape
        intr=o3d.camera.PinholeCameraIntrinsic(w,h,K[0,0],K[1,1],K[0,2],K[1,2])
        integrate_rgbd(volume,rgb,depth,intr,np.linalg.inv(camera['T_world_from_camera']))
        source_views.append(dict(candidate_id=view['candidate_id'],frame_id=camera['frame_id'],mask_sha256=view['mask_sha256']))
    fused=volume.extract_triangle_mesh()
    mesh=trimesh.Trimesh(vertices=np.asarray(fused.vertices),faces=np.asarray(fused.triangles),process=False)
    if len(mesh.faces)<4 or not np.isfinite(mesh.vertices).all():raise ValueError('observed TSDF has no usable surface')
    mesh.vertices-=center;mesh.export(out/'mesh_sim.obj');mesh.export(out/'mesh_sim.ply')
    T=np.eye(4);T[:3,3]=center;dimensions=np.ptp(mesh.vertices,axis=0)
    _write(out/'aligned.json',dict(T=T.tolist(),scale=1.,world_dims=dimensions.tolist()))
    shutil.copyfile(physics_path,out/'physics.json')
    fusion_wall=time.monotonic()-started;collision_started=time.monotonic()
    parts=coacd_parts(out,world_max_dim=float(max(dimensions)))
    if not parts:raise ValueError('observed control has no collision export')
    return dict(source_views=source_views,visual_mesh_watertight=bool(mesh.is_watertight),visual_mesh_faces=len(mesh.faces),
        collision_parts=len(parts),fusion_wall_s=fusion_wall,collision_wall_s=time.monotonic()-collision_started,
        total_wall_s=time.monotonic()-started,artifact_hashes={str(p.relative_to(out)):_sha(p) for p in out.rglob('*') if p.is_file()})


def observed_surface(b0, capture, out, config):
    """Original TRAIN-only TSDF control, same masks and B0 physical-prior bytes."""
    manifest,discovery=checked_b0(b0,capture);b0=Path(b0);capture=Path(capture);out=Path(out)
    _,cameras=read_train(capture);cameras={r['frame_id']:r for r in cameras}
    candidates={r['candidate_id']:r for r in json.loads((b0/'discovery/candidates.json').read_text())}
    views=[dict(camera=cameras[candidates[cid]['frame_id']],mask_path=b0/'discovery'/(cid+'.png'),
        mask_sha256=candidates[cid]['mask_sha256'],candidate_id=cid) for cid in discovery['accepted_candidates']]
    summary=fuse_masked_observations(capture,views,np.load(b0/'discovery/observation_points.npy',allow_pickle=False).mean(0),
        b0/'construction/objects/obj_00/physics.json',out,config)
    result=dict(schema_version=2,method='OBSERVED_SURFACE_TSDF',tier='DEV',status='BUILT',
        canonical_instance_id=manifest.get('canonical_instance_id'),cohort_id=manifest.get('cohort_id'),
        scope='L0_target_only',object_role='target',config=config,
        b0_manifest_sha256=_sha(b0/'build_manifest.json'),capture_manifest_sha256=discovery['capture_manifest_sha256'],
        completion='CoACD convex collision approximation; partial TSDF visual surface unchanged',
        physical_prior='same category-independent B0 mass/friction',heldout_access=False,
        native_import='NOT_RUN',policy_outcomes='NOT_RUN',**summary)
    _write(out/'observed_surface_receipt.json',result);return result


def main():
    import yaml
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("b0-build","capture","out","config"):parser.add_argument("--"+key,required=True)
    parser.add_argument("--phase",choices=("rvg","select","observed"),required=True)
    parser.add_argument("--rvg");parser.add_argument("--source-commit")
    args=parser.parse_args();config=yaml.safe_load(Path(args.config).read_text())
    if args.phase=="rvg":result=generate_rvg(args.b0_build,args.capture,args.out,config)
    elif args.phase=="observed":result=observed_surface(args.b0_build,args.capture,args.out,config)
    else:result=select_pool(args.b0_build,args.capture,args.rvg,args.out,config,args.source_commit)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=="__main__":main()
