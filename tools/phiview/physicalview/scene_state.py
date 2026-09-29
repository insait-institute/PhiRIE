"""Scene discovery and loading for Studio (no GPU; CPU-only, cacheable).

CONTRACT (implemented by the scene/render agent):

    discover_result_sets(config) -> list[ResultSet]
        Every directory under config.outputs_root that has objects/objects.json OR a
        recon/ + derived_mesh.ply (DROID/phone builds), plus ScanNet++ scenes that have a
        splat but no build yet (kind="unbuilt"). ResultSet fields: name (dir name, e.g.
        "c50d2d1d42_factory"), scene_id, kind in {"factory","auto","full","droid","video",
        "behavior","unbuilt","other"}, out_dir, scene_dir (ScanNet++ scene dir or
        data/recon_scenes/data/<scene>), splat_ply, mesh_ply, has_clean_bg, has_sim_export,
        has_tasks, n_objects, n_accepted, mtime.

    load_scene(config, result_set, device="cpu") -> SceneState
        SceneState: result_set, K (3x3), W, H, cameras {frame_name: w2c 4x4} (from colmap
        images.txt), splat_gs (gaussian dict as agents.core.common.load_gaussians returns,
        CPU tensors), clean_bg_gs | None, mesh (trimesh or None), objects: dict[str,
        ObjectRecord] keyed "obj_XX", tasks (parsed pi05_tasks.json or None), scene_xml
        path or None, sim_export dir, timings (dict from timings.txt), report (report.json
        dict or None).

    ObjectRecord: name, index, label, meta (meta.json), aligned (aligned.json or None),
        accepted: bool (aligned present and not rejected), rejected_reason, tier,
        T_world (4x4 from aligned["T"]) or None, world_dims, proposals: dict[str,
        Proposal] for keys in {"trellis","rvg","sam3d"} present on disk (Proposal: dir,
        mesh_ply, gs_ply, aligned json dict|None, evidence dict with sym_chamfer_m,
        chamfer_med_m, scale, size_ratio, f1_20 (evaluation-only, flagged
        eval_only=True), gen_seconds), chosen_source (aligned["vmesh_source"]),
        hybrid (hybrid.json or None), physics (physics.json or None), collision_parts
        (count), drop_test (from report.json / drop_v2.json), artifacts: dict[str,bool]
        (rgba, trellis_mesh, rvg_mesh, sam3d_mesh, aligned, urdf, collision, physics,
        inpainted), rgba_png path.

    reload_objects(state) -> list[str]   re-reads objects/*; returns ids whose artifacts
                                          changed (used after jobs finish)

    object_canonical_gs(state, obj_id, source=None) -> gaussian dict (CPU) of the chosen
        (or given source's) canonical gaussians (trellis_gs.ply / rvg/rvg_gs.ply /
        sam3d/...); cached.

Scene-independent: never read agents.core.common module constants for paths (they are
bound to SIMANY_SCENE at import time); compute paths from result_set instead, but reuse
C.load_gaussians / C.load_intrinsics(path) / C.load_colmap_w2c(path) helpers, which
accept explicit paths.

Implementation notes (not part of the contract):
  * ``load_scene(..., load_mesh=False)``: the scan mesh (tens of MB) is only read when
    asked for; ``SceneState.mesh`` stays None otherwise and the path is on
    ``result_set.mesh_ply``.
  * ``SceneState.edited_poses`` holds user-edited world transforms (obj_id -> 4x4) set by
    the scene panel's gizmos so other panels (robot, inpaint) can read them.
  * DROID/phone/BEHAVIOR scene dirs live under ``data/recon_scenes``; that tree is looked
    up under ``config.repo_root`` first and, when absent (git worktrees share ``outputs/``
    via a symlink but not ``data/``), next to the resolved ``outputs_root``.
  * Every reader degrades to None/0/{} on missing or unparsable files; nothing here raises
    on a partially built scene.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from physicalview.config import StudioConfig

log = logging.getLogger("studio.scene")

PROPOSAL_SOURCES: tuple[str, ...] = ("trellis", "rvg", "sam3d")
_SUFFIX_KINDS: tuple[tuple[str, str], ...] = (
    ("_factory", "factory"), ("_auto", "auto"), ("_full", "full"),
    ("_rowC2", "other"), ("_rowC", "other"),
)
_OBJ_RE = re.compile(r"^obj_(\d{2,})$")


@dataclass
class ResultSet:
    name: str
    scene_id: str
    kind: str
    out_dir: Path
    scene_dir: Path | None
    splat_ply: Path | None
    mesh_ply: Path | None
    has_clean_bg: bool = False
    has_sim_export: bool = False
    has_tasks: bool = False
    n_objects: int = 0
    n_accepted: int = 0
    mtime: float = 0.0


@dataclass
class Proposal:
    source: str
    dir: Path
    mesh_ply: Path | None
    gs_ply: Path | None
    aligned: dict | None
    evidence: dict = field(default_factory=dict)
    eval_only_fields: tuple[str, ...] = ("f1_20", "f1_40")


@dataclass
class ObjectRecord:
    name: str
    index: int
    label: str
    meta: dict
    aligned: dict | None
    accepted: bool
    rejected_reason: str | None
    T_world: np.ndarray | None
    world_dims: list[float] | None
    proposals: dict[str, Proposal]
    chosen_source: str | None
    hybrid: dict | None
    physics: dict | None
    collision_parts: int
    drop_test: dict | None
    artifacts: dict[str, bool]
    rgba_png: Path | None
    dir: Path

    @property
    def tier(self) -> str | None:
        return None if self.aligned is None else self.aligned.get("tier")

    @property
    def f1_20(self) -> float | None:
        """Evaluation-only F1@20mm from aligned["eval"] (None when no GT eval ran)."""
        if not self.aligned:
            return None
        ev = self.aligned.get("eval") or {}
        f = (ev.get("f1@20mm") or {}).get("f1")
        return None if f is None else float(f)

    @property
    def stable(self) -> bool | None:
        if not self.drop_test:
            return None
        v = self.drop_test.get("stable")
        return None if v is None else bool(v)


@dataclass
class SceneState:
    result_set: ResultSet
    K: np.ndarray
    W: int
    H: int
    cameras: dict[str, np.ndarray]
    splat_gs: dict[str, Any] | None
    clean_bg_gs: dict[str, Any] | None
    mesh: Any | None
    objects: dict[str, ObjectRecord]
    tasks: dict | None
    scene_xml: Path | None
    sim_export: Path
    timings: dict[str, float]
    report: dict | None
    edited_poses: dict[str, np.ndarray] = field(default_factory=dict)


# ----------------------------------------------------------------- helpers --

def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        if Path(path).exists():
            log.warning("unreadable json %s: %s", path, exc)
        return None


def _exists(p: Path | None) -> bool:
    return p is not None and p.exists()


def _recon_roots(config: StudioConfig) -> list[Path]:
    roots = [config.repo_root / "data" / "recon_scenes"]
    try:
        alt = config.outputs_root.resolve().parent / "data" / "recon_scenes"
    except OSError:
        alt = None
    if alt is not None and alt not in roots:
        roots.append(alt)
    return roots


def classify_result_dir(name: str) -> tuple[str, str]:
    """Directory name -> (kind, scene_id). Pure string logic, unit-tested."""
    base, kind = name, "other"
    for suffix, k in _SUFFIX_KINDS:
        if name.endswith(suffix):
            base, kind = name[: -len(suffix)], k
            break
    if base.startswith("droid_"):
        return "droid", base
    if base.startswith("video_"):
        return "video", base[len("video_"):]
    if base.startswith("behavior_task"):
        # run/run_behavior_recon.sh writes outputs/behavior_task-0020 but builds the scene
        # as behavior_task0020 (SCENE_NAME=behavior_${TASK//-/}); the scene id is the
        # dash-free form, so scene_dir/splat lookup and SIMANY_SCENE both resolve.
        return "behavior", base.replace("-", "")
    return kind, base


def resolve_scene_paths(config: StudioConfig, scene_id: str, out_dir: Path
                        ) -> tuple[Path | None, Path | None, Path | None]:
    """(scene_dir, splat_ply, mesh_ply) for a scene id; None where nothing exists."""
    scene_dir: Path | None = None
    cand = config.scannetpp_root / "data" / scene_id
    if cand.is_dir():
        scene_dir = cand
    else:
        for root in _recon_roots(config):
            cand = root / "data" / scene_id
            if cand.is_dir():
                scene_dir = cand
                break
    splat: Path | None = None
    cand = config.splats_root / f"{scene_id}.ply"
    if cand.exists():
        splat = cand
    else:
        for root in _recon_roots(config):
            cand = root / "splats" / f"{scene_id}.ply"
            if cand.exists():
                splat = cand
                break
    mesh: Path | None = None
    if scene_dir is not None and (scene_dir / "scans" / "mesh_aligned_0.05.ply").exists():
        mesh = scene_dir / "scans" / "mesh_aligned_0.05.ply"
    elif (out_dir / "derived_mesh.ply").exists():
        mesh = out_dir / "derived_mesh.ply"
    return scene_dir, splat, mesh


def _count_accepted(aligned_all: Any) -> int:
    if not isinstance(aligned_all, list):
        return 0
    return sum(1 for a in aligned_all
               if isinstance(a, dict) and a.get("T") is not None and not a.get("rejected"))


# ---------------------------------------------------------------- discovery --

def discover_result_sets(config: StudioConfig) -> list[ResultSet]:
    root = Path(config.outputs_root)
    found: list[ResultSet] = []
    built_scene_ids: set[str] = set()
    if root.is_dir():
        with os.scandir(root) as it:
            entries = [e for e in it if e.is_dir(follow_symlinks=True) and not e.name.startswith(".")]
        for e in entries:
            out_dir = Path(e.path)
            objects_json = out_dir / "objects" / "objects.json"
            has_objects = objects_json.exists()
            is_recon = (not has_objects and (out_dir / "recon").is_dir()
                        and (out_dir / "derived_mesh.ply").exists())
            if not (has_objects or is_recon):
                continue
            kind, scene_id = classify_result_dir(e.name)
            scene_dir, splat, mesh = resolve_scene_paths(config, scene_id, out_dir)
            n_obj = n_acc = 0
            mtime = 0.0
            try:
                mtime = e.stat().st_mtime
            except OSError:
                pass
            if has_objects:
                objs = _read_json(objects_json)
                n_obj = len(objs) if isinstance(objs, list) else 0
                n_acc = _count_accepted(_read_json(out_dir / "objects" / "aligned_all.json"))
                try:
                    mtime = max(mtime, objects_json.stat().st_mtime)
                except OSError:
                    pass
            sim_export = out_dir / "sim_export"
            found.append(ResultSet(
                name=e.name, scene_id=scene_id, kind=kind, out_dir=out_dir,
                scene_dir=scene_dir, splat_ply=splat, mesh_ply=mesh,
                has_clean_bg=(out_dir / "inpaint" / "clean_background.ply").exists(),
                has_sim_export=(sim_export / "scene.xml").exists(),
                has_tasks=(sim_export / "pi05_tasks.json").exists(),
                n_objects=n_obj, n_accepted=n_acc, mtime=mtime))
            built_scene_ids.add(scene_id)

    # ScanNet++ validation scenes with a splat but no build yet.
    split_txt = config.scannetpp_root / "splits" / "nvs_sem_val.txt"
    if split_txt.exists():
        try:
            val_ids = [ln.strip() for ln in split_txt.read_text().splitlines() if ln.strip()]
        except OSError:
            val_ids = []
        for sid in val_ids:
            if sid in built_scene_ids:
                continue
            splat = config.splats_root / f"{sid}.ply"
            if not splat.exists():
                continue
            out_dir = root / f"{sid}_factory"
            scene_dir, _, mesh = resolve_scene_paths(config, sid, out_dir)
            found.append(ResultSet(name=sid, scene_id=sid, kind="unbuilt", out_dir=out_dir,
                                   scene_dir=scene_dir, splat_ply=splat, mesh_ply=mesh))

    def sort_key(rs: ResultSet):
        if rs.kind in ("factory", "auto"):
            group = 0
        elif rs.kind == "unbuilt":
            group = 2
        else:
            group = 1
        return (group, -rs.mtime, rs.name)

    found.sort(key=sort_key)
    return found


# ------------------------------------------------------------------ objects --

def _proposal_evidence(source: str, hybrid: dict | None, p_aligned: dict | None,
                       pdir: Path) -> dict:
    ev: dict[str, Any] = {}
    h = (hybrid or {}).get(source) if isinstance(hybrid, dict) else None
    if isinstance(h, dict):
        for k in ("chamfer_med_m", "f1_20", "f1_40", "tier", "scale", "rejected"):
            if k in h:
                ev[k] = h[k]
    if isinstance(hybrid, dict):
        sc = hybrid.get(f"sym_chamfer_{source}_m")
        if sc is not None:
            ev["sym_chamfer_m"] = sc
        if source == "rvg" and hybrid.get("gen_seconds") is not None:
            ev["gen_seconds"] = hybrid["gen_seconds"]
    if isinstance(p_aligned, dict):
        for k in ("chamfer_med_m", "scale", "tier", "icp_tilt_deg"):
            if k in p_aligned and k not in ev:
                ev[k] = p_aligned[k]
        if "size_ratio_vs_obs" in p_aligned:
            ev["size_ratio"] = p_aligned["size_ratio_vs_obs"]
        evald = p_aligned.get("eval") or {}
        for key, name in (("f1@20mm", "f1_20"), ("f1@40mm", "f1_40")):
            f = (evald.get(key) or {}).get("f1") if isinstance(evald, dict) else None
            if f is not None and name not in ev:
                ev[name] = f
        if p_aligned.get("rejected") and "rejected" not in ev:
            ev["rejected"] = p_aligned["rejected"]
    if source == "sam3d":
        meta = _read_json(pdir / "sam3d_meta.json")
        if isinstance(meta, dict) and meta.get("runtime_s") is not None:
            ev["gen_seconds"] = meta["runtime_s"]
    return ev


def _load_proposals(odir: Path, hybrid: dict | None) -> dict[str, Proposal]:
    props: dict[str, Proposal] = {}
    for src in PROPOSAL_SOURCES:
        pdir = odir / src
        if pdir.is_dir():
            mesh = pdir / f"{src}_mesh.ply"
            gs = pdir / f"{src}_gs.ply"
        elif src == "trellis" and ((odir / "trellis_gs.ply").exists()
                                   or (odir / "trellis_mesh.ply").exists()):
            pdir = odir  # legacy flat layout: trellis assets live in the object dir
            mesh = odir / "trellis_mesh.ply"
            gs = odir / "trellis_gs.ply"
        else:
            continue
        p_aligned = _read_json(pdir / "aligned.json") if pdir != odir else None
        if pdir == odir:
            # flat layout: the object's aligned.json is the trellis registration unless a
            # different source was chosen
            top = _read_json(odir / "aligned.json")
            if isinstance(top, dict) and top.get("vmesh_source", "trellis") == "trellis":
                p_aligned = top
        props[src] = Proposal(
            source=src, dir=pdir,
            mesh_ply=mesh if mesh.exists() else None,
            gs_ply=gs if gs.exists() else None,
            aligned=p_aligned if isinstance(p_aligned, dict) else None,
            evidence=_proposal_evidence(src, hybrid, p_aligned if isinstance(p_aligned, dict) else None, pdir))
    return props


def _drop_test_for(name: str, index: int, drop_v2: Any, report: Any) -> dict | None:
    if isinstance(drop_v2, dict):
        for o in drop_v2.get("objects") or []:
            if isinstance(o, dict) and o.get("name") == name:
                return {k: v for k, v in o.items() if k not in ("name", "label")}
    if isinstance(report, dict):
        for o in report.get("objects") or []:
            if isinstance(o, dict) and o.get("index") == index and isinstance(o.get("drop_test"), dict):
                return dict(o["drop_test"])
    return None


def _load_object(out_dir: Path, meta: dict, drop_v2: Any, report: Any,
                 inpaint_meta: Any, clean_bg: bool) -> ObjectRecord:
    index = int(meta.get("index", 0))
    name = f"obj_{index:02d}"
    odir = out_dir / "objects" / name
    disk_meta = _read_json(odir / "meta.json")
    if isinstance(disk_meta, dict):
        meta = {**meta, **disk_meta}
    aligned = _read_json(odir / "aligned.json")
    aligned = aligned if isinstance(aligned, dict) else None
    hybrid = _read_json(odir / "hybrid.json")
    hybrid = hybrid if isinstance(hybrid, dict) else None
    physics = _read_json(odir / "physics.json")
    physics = physics if isinstance(physics, dict) else None

    T_world = None
    if aligned is not None and aligned.get("T") is not None:
        try:
            T = np.asarray(aligned["T"], dtype=np.float64)
            if T.shape == (4, 4):
                T_world = T
        except (TypeError, ValueError):
            T_world = None
    rejected = aligned.get("rejected") if aligned else None
    rejected_reason = str(rejected) if rejected else None
    accepted = aligned is not None and T_world is not None and not rejected

    proposals = _load_proposals(odir, hybrid)
    chosen = None
    if aligned is not None and aligned.get("vmesh_source"):
        chosen = str(aligned["vmesh_source"])
    elif hybrid is not None and (hybrid.get("canonical_source") or hybrid.get("winner")):
        chosen = str(hybrid.get("canonical_source") or hybrid.get("winner"))
    elif "trellis" in proposals:
        chosen = "trellis"

    coll_dir = odir / "collision"
    collision_parts = 0
    if coll_dir.is_dir():
        collision_parts = sum(1 for p in os.listdir(coll_dir)
                              if p.startswith("part_") and p.endswith(".obj"))
    inpainted = False
    if isinstance(inpaint_meta, dict):
        inpainted = any(str(k).startswith(name) for k in inpaint_meta)
    elif clean_bg and accepted:
        inpainted = True  # clean background exists but no per-object record: best effort
    rgba = odir / "rgba.png"
    artifacts = {
        "rgba": rgba.exists(),
        "trellis_mesh": _exists(proposals.get("trellis").mesh_ply) if "trellis" in proposals else False,
        "rvg_mesh": _exists(proposals.get("rvg").mesh_ply) if "rvg" in proposals else False,
        "sam3d_mesh": _exists(proposals.get("sam3d").mesh_ply) if "sam3d" in proposals else False,
        "aligned": aligned is not None,
        "urdf": (odir / "object.urdf").exists(),
        "collision": collision_parts > 0,
        "physics": physics is not None,
        "inpainted": inpainted,
    }
    return ObjectRecord(
        name=name, index=index, label=str(meta.get("label", name)), meta=meta,
        aligned=aligned, accepted=bool(accepted), rejected_reason=rejected_reason,
        T_world=T_world, world_dims=aligned.get("world_dims") if aligned else None,
        proposals=proposals, chosen_source=chosen, hybrid=hybrid, physics=physics,
        collision_parts=collision_parts,
        drop_test=_drop_test_for(name, index, drop_v2, report),
        artifacts=artifacts, rgba_png=rgba if rgba.exists() else None, dir=odir)


def load_objects(out_dir: Path, report: dict | None = None) -> dict[str, ObjectRecord]:
    """objects/objects.json + per-object artifacts -> {obj_XX: ObjectRecord}."""
    out_dir = Path(out_dir)
    objs = _read_json(out_dir / "objects" / "objects.json")
    if not isinstance(objs, list):
        return {}
    drop_v2 = _read_json(out_dir / "drop_v2.json")
    if report is None:
        report = _read_json(out_dir / "report.json")
    inpaint_meta = _read_json(out_dir / "inpaint" / "inpaint_meta.json")
    clean_bg = (out_dir / "inpaint" / "clean_background.ply").exists()
    out: dict[str, ObjectRecord] = {}
    for m in objs:
        if not isinstance(m, dict) or "index" not in m:
            continue
        try:
            rec = _load_object(out_dir, m, drop_v2, report, inpaint_meta, clean_bg)
        except Exception as exc:  # noqa: BLE001 - one broken object must not hide the rest
            log.warning("object %s in %s unreadable: %s", m.get("index"), out_dir, exc)
            continue
        out[rec.name] = rec
    return out


# ------------------------------------------------------------------- timings --

def parse_timings(path: Path) -> dict[str, float]:
    """timings.txt lines "<stage description> <seconds>"; when a pipeline was re-run the
    file holds several blocks and the LAST occurrence of each stage wins."""
    out: dict[str, float] = {}
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return out
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        head, _, tail = ln.rpartition(" ")
        try:
            secs = float(tail)
        except ValueError:
            continue
        if head:
            out[head] = secs  # dict insertion order keeps first-seen position, last value wins
    return out


# ------------------------------------------------------------------- loading --

def load_scene(config: StudioConfig, result_set: ResultSet, device: str = "cpu",
               load_mesh: bool = False, load_splats: bool = True) -> SceneState:
    from agents.core import common as C

    rs = result_set
    out_dir = Path(rs.out_dir)
    K: np.ndarray | None = None
    W = H = 0
    cameras: dict[str, np.ndarray] = {}
    if rs.scene_dir is not None:
        tj = rs.scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json"
        it = rs.scene_dir / "dslr" / "colmap" / "images.txt"
        if tj.exists():
            try:
                K, W, H, _ = C.load_intrinsics(tj)
            except Exception as exc:  # noqa: BLE001
                log.warning("intrinsics unreadable %s: %s", tj, exc)
        if it.exists():
            try:
                cameras = C.load_colmap_w2c(it)
            except Exception as exc:  # noqa: BLE001
                log.warning("colmap images.txt unreadable %s: %s", it, exc)

    splat_gs = clean_bg_gs = None
    if load_splats:
        if rs.splat_ply is not None and rs.splat_ply.exists():
            try:
                splat_gs = C.load_gaussians(rs.splat_ply, device=device)
            except Exception as exc:  # noqa: BLE001
                log.warning("splat unreadable %s: %s", rs.splat_ply, exc)
        clean = out_dir / "inpaint" / "clean_background.ply"
        if clean.exists():
            try:
                clean_bg_gs = C.load_gaussians(clean, device=device)
            except Exception as exc:  # noqa: BLE001
                log.warning("clean background unreadable %s: %s", clean, exc)

    mesh = None
    if load_mesh and rs.mesh_ply is not None and rs.mesh_ply.exists():
        try:
            import trimesh
            mesh = trimesh.load(str(rs.mesh_ply), process=False, force="mesh")
        except Exception as exc:  # noqa: BLE001
            log.warning("mesh unreadable %s: %s", rs.mesh_ply, exc)

    report = _read_json(out_dir / "report.json")
    report = report if isinstance(report, dict) else None
    objects = load_objects(out_dir, report)
    sim_export = out_dir / "sim_export"
    tasks = _read_json(sim_export / "pi05_tasks.json")
    tasks = tasks if isinstance(tasks, dict) else None
    scene_xml = sim_export / "scene.xml"
    return SceneState(
        result_set=rs, K=K, W=W, H=H, cameras=cameras, splat_gs=splat_gs,
        clean_bg_gs=clean_bg_gs, mesh=mesh, objects=objects, tasks=tasks,
        scene_xml=scene_xml if scene_xml.exists() else None, sim_export=sim_export,
        timings=parse_timings(out_dir / "timings.txt"), report=report)


def _object_signature(rec: ObjectRecord) -> tuple:
    def _fp(p: Path | None) -> tuple | None:
        if p is None:
            return None
        try:
            st = p.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    return (
        tuple(sorted(rec.artifacts.items())),
        rec.accepted, rec.rejected_reason, rec.chosen_source, rec.collision_parts,
        None if rec.T_world is None else tuple(np.round(rec.T_world, 9).ravel().tolist()),
        json.dumps(rec.hybrid, sort_keys=True) if rec.hybrid else None,
        json.dumps(rec.physics, sort_keys=True) if rec.physics else None,
        json.dumps(rec.drop_test, sort_keys=True) if rec.drop_test else None,
        tuple((s, _fp(p.gs_ply), _fp(p.mesh_ply)) for s, p in sorted(rec.proposals.items())),
    )


def reload_objects(state: SceneState) -> list[str]:
    out_dir = Path(state.result_set.out_dir)
    report = _read_json(out_dir / "report.json")
    state.report = report if isinstance(report, dict) else state.report
    new = load_objects(out_dir, state.report)
    old = state.objects
    changed: list[str] = []
    for name, rec in new.items():
        prev = old.get(name)
        if prev is None or _object_signature(prev) != _object_signature(rec):
            changed.append(name)
    changed += [name for name in old if name not in new]
    state.objects = new
    for name in changed:
        for key in list(_canon_cache):
            if key[0] == str(out_dir) and key[1] == name:
                _canon_cache.pop(key, None)
    return sorted(changed)


# ---------------------------------------------------------- canonical splats --

_canon_cache: "OrderedDict[tuple[str, str, str], dict]" = OrderedDict()
_CANON_CACHE_MAX = 64


def canonical_gs_path(rec: ObjectRecord, source: str | None = None) -> tuple[str, Path] | None:
    """(source, gs_ply) of the requested/chosen proposal, else the first with gaussians."""
    order: list[str] = []
    if source:
        order.append(source)
    elif rec.chosen_source:
        order.append(rec.chosen_source)
    order += [s for s in PROPOSAL_SOURCES if s not in order]
    for s in order:
        p = rec.proposals.get(s)
        if p is not None and p.gs_ply is not None and p.gs_ply.exists():
            return s, p.gs_ply
        if source and s == source:
            return None
    return None


def object_canonical_gs(state: SceneState, obj_id: str, source: str | None = None) -> dict:
    from agents.core import common as C

    rec = state.objects.get(obj_id)
    if rec is None:
        raise KeyError(f"unknown object {obj_id!r}")
    found = canonical_gs_path(rec, source)
    if found is None:
        raise FileNotFoundError(
            f"{obj_id}: no canonical gaussians for source {source or rec.chosen_source!r} "
            f"(available: {[s for s, p in rec.proposals.items() if p.gs_ply]})")
    src, ply = found
    key = (str(state.result_set.out_dir), obj_id, src)
    try:
        st = ply.stat()
        key = key + ((st.st_mtime_ns, st.st_size),)  # type: ignore[assignment]
    except OSError:
        pass
    hit = _canon_cache.get(key)
    if hit is not None:
        _canon_cache.move_to_end(key)
        return hit
    gs = C.load_gaussians(ply, device="cpu")
    _canon_cache[key] = gs
    while len(_canon_cache) > _CANON_CACHE_MAX:
        _canon_cache.popitem(last=False)
    return gs
