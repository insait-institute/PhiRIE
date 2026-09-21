"""Task 05: ground-truth-free build audit -- one consolidated JSON + a
human-readable report per build, assembled from the outputs of the existing,
disparate QA components rather than re-running any of their computation:

  - agents/eval/factory_report.py       report.json            (tiers, F1,
                                                                  chamfer, drop_test)
  - robo/sim/redrop.py                  drop_v2.json            (link-frame-
                                                                  corrected drop test)
  - agents/eval/factory_eval_render.py  render_metrics_v2.json  (held-out
                                                                  PSNR/SSIM/LPIPS)
  - agents/edit/inpaint_prepare.py      inpaint/obj_XX/plane.json (support-
                                                                  plane fit)
  - agents/eval/verify_removal_render.py inpaint/verify/verify_render.json
                                                                (alpha/depth)
  - agents/eval/verify_removal_check.py  inpaint/verify/verify_report.json
                                                                (SAM3 residual
                                                                 re-detection)
  - agents/assets/s5_align.py / factory_align.py / factory_hybrid.py
                                         objects/obj_XX/aligned.json,
                                         objects/obj_XX/hybrid.json
  - agents/eval/eval_vs_gt.py            eval_vs_gt.json         (GT instance
                                                                  match, optional)

and, defensively (read with .get(), degrade to "not_applicable" when absent),
two artifacts from tasks that have not landed yet at the time this module was
written:

  - robo/sim/room_collision.py (Task 06)   room_collision_report.json
      EXPECTED fields: coverage_fraction, penetration_max_mm,
      disconnected_components, settle_drift_mm
  - agents/recon/robot_align.py / alignment_report.py (Task 04)
                                            alignment_report.json
      EXPECTED fields (plan/04_METRIC_SCALE_ROBOT_ALIGNMENT.md "Outputs"):
      scale_factor, scale_ci, rotation_residual_deg, translation_residual_mm,
      reprojection_residual_px, floor_normal_error_deg, table_height_error_mm,
      held_out_error_m

Full field-by-field documentation: docs/BUILD_AUDIT_SCHEMA.md

THE ONE INVARIANT THAT MATTERS: every check result is exactly one of
pass / warning / fail / not_applicable. Missing evidence is ALWAYS
not_applicable, never pass -- see _mk()/VALID_STATUSES and every check_*()
function below, all of which return not_applicable as soon as a required
input is absent, before ever touching a numeric threshold.

Usage:
    python -m agents.eval.build_audit outputs/<scene>_factory
    python -m agents.eval.build_audit outputs/<scene>_factory --out /tmp/a.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0.0"

STATUS_PASS = "pass"
STATUS_WARNING = "warning"
STATUS_FAIL = "fail"
STATUS_NA = "not_applicable"
# Severity order for worst-of aggregation. not_applicable is the *weakest*
# signal (no evidence, not "no problem"): it must never outrank a real pass,
# and a real fail/warning must never be outranked by any number of passes or
# not_applicable entries sitting next to it (the "cannot be averaged away"
# requirement -- see _aggregate_over_objects()/_overall_status()).
_SEVERITY = {STATUS_NA: 0, STATUS_PASS: 1, STATUS_WARNING: 2, STATUS_FAIL: 3}
VALID_STATUSES = frozenset(_SEVERITY)
# Fixed iteration order (frozenset order is hash-seed dependent across
# process runs; a plain tuple keeps counts{} key order -- and hence the
# on-disk JSON -- byte-identical across reruns regardless of PYTHONHASHSEED).
_STATUS_ORDER = (STATUS_PASS, STATUS_WARNING, STATUS_FAIL, STATUS_NA)

# --------------------------------------------------------------------------
# Thresholds. Two provenance classes, both cited inline; full rationale for
# the "new" ones lives in docs/BUILD_AUDIT_SCHEMA.md's Thresholds section
# per the task instruction to never leave an unexplained magic number.
# --------------------------------------------------------------------------

# -- reused verbatim from already-frozen pipeline thresholds (not new) ----
DROP_STABLE_DRIFT_M = 0.03        # agents/eval/factory_report.py drop_test()
DROP_SUNK_Z_M = -0.05             # agents/eval/factory_report.py drop_test()
TIER_A_F1_20 = 0.40               # agents/assets/factory_align.py
TIER_B_F1_40 = 0.20               # agents/assets/factory_align.py
SIZE_RATIO_RANGE = (0.4, 2.5)     # agents/assets/factory_align.py SIZE_RATIO_RANGE
REDETECTION_FAIL_SCORE = 0.5      # agents/eval/verify_removal_check.py removed_ok
BG_PSNR_INCLUSION_DB = 26.0       # docs/GAP_STUDY.md zeroshot-100 scene-inclusion bar

# -- new thresholds authored for this audit; see docs/BUILD_AUDIT_SCHEMA.md --
REG_RESIDUAL_WARN_MM = 20.0
REG_RESIDUAL_FAIL_MM = 40.0
CANDIDATE_DISAGREEMENT_WARN_MM = 10.0
CANDIDATE_DISAGREEMENT_FAIL_MM = 30.0
VIS_COLLISION_WARN_MM = 5.0
VIS_COLLISION_FAIL_MM = 15.0
ALPHA_COV_PASS = 0.98
ALPHA_COV_WARN = 0.90
DEPTH_PLANE_WARN_MM = 15.0
DEPTH_PLANE_FAIL_MM = 40.0
REDETECTION_WARN_SCORE = 0.3      # inner band before the existing 0.5 cliff
SETTLE_DRIFT_WARN_MM = 10.0       # FAIL boundary reuses DROP_STABLE_DRIFT_M*1000
COVERAGE_WARN_FRAC = 0.85
COVERAGE_FAIL_FRAC = 0.50
MISSING_INSTANCE_WARN_FRAC = 0.30
DUPLICATE_HEURISTIC_DIST_M = 0.05
TWIN_PSNR_WARN_DB = BG_PSNR_INCLUSION_DB
TWIN_PSNR_FAIL_DB = 20.0
PSNR_DELTA_WARN_DB = 4.0
PSNR_DELTA_FAIL_DB = 8.0
SCALE_SANITY_INNER_MARGIN_FRAC = 0.10   # warn within 10% of the SIZE_RATIO_RANGE edge
# room_collision (Task 06) forward-declared thresholds, inactive until that
# component lands -- see docs/BUILD_AUDIT_SCHEMA.md
ROOM_COVERAGE_WARN_FRAC = 0.85
ROOM_COVERAGE_FAIL_FRAC = 0.50
ROOM_DISCONNECTED_WARN_N = 1
ROOM_DISCONNECTED_FAIL_N = 3
ROOM_PENETRATION_WARN_MM = 2.0
ROOM_PENETRATION_FAIL_MM = 10.0
ROOM_SETTLE_DRIFT_WARN_MM = 5.0
ROOM_SETTLE_DRIFT_FAIL_MM = 20.0


def _mk(status: str, reason: str, **evidence: Any) -> dict:
    assert status in VALID_STATUSES, f"invalid status {status!r}"
    out = {"status": status, "reason": reason}
    out.update(evidence)
    return out


def _band(value: float, warn: float, fail: float, higher_is_worse: bool = True):
    """Classic pass/warning/fail banding. higher_is_worse=True means value
    below `warn` is pass, [warn, fail) is warning, >= fail is fail."""
    if higher_is_worse:
        if value >= fail:
            return STATUS_FAIL
        if value >= warn:
            return STATUS_WARNING
        return STATUS_PASS
    if value <= fail:
        return STATUS_FAIL
    if value <= warn:
        return STATUS_WARNING
    return STATUS_PASS


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _index_by(rows: list[dict] | None, key: str) -> dict[Any, dict]:
    if not rows:
        return {}
    return {r[key]: r for r in rows if key in r}


def _oid(index: int) -> str:
    return f"obj_{index:02d}"


# --------------------------------------------------------------------------
# Loading raw evidence
# --------------------------------------------------------------------------

def load_build(build_dir: Path) -> dict:
    """Read every source file defensively (missing files -> None). Never
    raises on a missing/malformed file; that is exactly the condition every
    check_*() below must turn into not_applicable rather than pass."""
    objects_json = _load_json(build_dir / "objects" / "objects.json") or []
    objects_meta = {_oid(m["index"]): m for m in objects_json if "index" in m}

    aligned = {}
    hybrid = {}
    plane = {}
    for oid in objects_meta:
        aligned[oid] = _load_json(build_dir / "objects" / oid / "aligned.json")
        hybrid[oid] = _load_json(build_dir / "objects" / oid / "hybrid.json")
        plane[oid] = _load_json(build_dir / "inpaint" / oid / "plane.json")

    report = _load_json(build_dir / "report.json")
    report_by_oid = {
        _oid(r["index"]): r for r in (report or {}).get("objects", []) if "index" in r
    }
    drop_v2 = _load_json(build_dir / "drop_v2.json")
    drop_v2_by_oid = _index_by((drop_v2 or {}).get("objects"), "name")

    render_metrics = _load_json(build_dir / "render_metrics_v2.json")
    eval_vs_gt = _load_json(build_dir / "eval_vs_gt.json")

    verify_render = _load_json(build_dir / "inpaint" / "verify" / "verify_render.json")
    verify_report = _load_json(build_dir / "inpaint" / "verify" / "verify_report.json")
    verify_summary_by_oid = _index_by((verify_report or {}).get("summary"), "obj")

    room_collision = _load_json(build_dir / "room_collision_report.json")
    alignment_report = _load_json(build_dir / "alignment_report.json")

    return {
        "objects_meta": objects_meta,
        "aligned": aligned,
        "hybrid": hybrid,
        "plane": plane,
        "report": report,
        "report_by_oid": report_by_oid,
        "drop_v2": drop_v2,
        "drop_v2_by_oid": drop_v2_by_oid,
        "render_metrics": render_metrics,
        "eval_vs_gt": eval_vs_gt,
        "verify_render": verify_render,
        "verify_report": verify_report,
        "verify_summary_by_oid": verify_summary_by_oid,
        "room_collision": room_collision,
        "alignment_report": alignment_report,
        "sources_present": {
            "objects.json": bool(objects_json) or (build_dir / "objects" / "objects.json").exists(),
            "report.json": report is not None,
            "drop_v2.json": drop_v2 is not None,
            "render_metrics_v2.json": render_metrics is not None,
            "eval_vs_gt.json": eval_vs_gt is not None,
            "inpaint/verify/verify_render.json": verify_render is not None,
            "inpaint/verify/verify_report.json": verify_report is not None,
            "room_collision_report.json": room_collision is not None,
            "alignment_report.json": alignment_report is not None,
        },
    }


def _drop_evidence(oid: str, raw: dict) -> dict | None:
    """drop_v2.json (link-frame corrected) supersedes report.json's embedded
    drop_test when both exist -- same policy as redrop.py's own docstring."""
    if oid in raw["drop_v2_by_oid"]:
        return raw["drop_v2_by_oid"][oid]
    row = raw["report_by_oid"].get(oid)
    if row and "drop_test" in row:
        return row["drop_test"]
    return None


def _removal_evidence(oid: str, raw: dict) -> dict | None:
    """Prefer verify_report.json's stage-2 summary row (has det_before/after);
    fall back to averaging verify_render.json's stage-1 per-view fields when
    stage 2 was never run."""
    row = raw["verify_summary_by_oid"].get(oid)
    if row is not None:
        return row
    vr = (raw["verify_render"] or {}).get(oid)
    if vr is None:
        return None
    views = vr.get("views", [])
    covs = [v["alpha_cov"] for v in views if v.get("alpha_cov") is not None]
    deps = [v["depth_vs_plane_med_mm"] for v in views
            if v.get("depth_vs_plane_med_mm") is not None]
    return {
        "alpha_cov": float(np.mean(covs)) if covs else None,
        "depth_plane_mm": float(np.median(deps)) if deps else None,
        "det_before": None, "det_after": None, "removed_ok": None,
    }


# --------------------------------------------------------------------------
# Object-scoped checks
# --------------------------------------------------------------------------

def check_ghost_object(aligned_rec, drop_evidence, physics_stage_ran: bool) -> dict:
    if aligned_rec is None:
        return _mk(STATUS_NA, "no aligned.json for this object; cannot tell "
                    "whether it was ever accepted into the render/physics build")
    if aligned_rec.get("rejected"):
        return _mk(STATUS_NA, "object was rejected at registration; it is "
                    "expected to have no physics record")
    if not physics_stage_ran:
        return _mk(STATUS_NA, "physics stage (report.json / drop_v2.json) has "
                    "not been run for this build yet")
    if drop_evidence is not None:
        return _mk(STATUS_PASS, "accepted object has a recorded drop test")
    return _mk(STATUS_FAIL, "accepted object (tier A/B) has no drop-test "
                "record in report.json or drop_v2.json -- present in "
                "reconstruction/render but never entered physics")


def check_registration_residual(aligned_rec) -> dict:
    if aligned_rec is None:
        return _mk(STATUS_NA, "no aligned.json for this object")
    chamfer_m = aligned_rec.get("chamfer_med_m")
    if chamfer_m is None:
        return _mk(STATUS_NA, "aligned.json missing chamfer_med_m")
    chamfer_mm = chamfer_m * 1000.0
    status = _band(chamfer_mm, REG_RESIDUAL_WARN_MM, REG_RESIDUAL_FAIL_MM)
    return _mk(status, f"registration chamfer residual {chamfer_mm:.1f}mm "
                f"(icp_tilt={aligned_rec.get('icp_tilt_deg')})",
                chamfer_med_mm=chamfer_mm, icp_tilt_deg=aligned_rec.get("icp_tilt_deg"))


def check_scale_sanity(aligned_rec) -> dict:
    if aligned_rec is None or "size_ratio_vs_obs" not in aligned_rec:
        return _mk(STATUS_NA, "no size_ratio_vs_obs recorded for this object")
    ratio = aligned_rec["size_ratio_vs_obs"]
    lo, hi = SIZE_RATIO_RANGE
    if ratio < lo or ratio > hi:
        return _mk(STATUS_FAIL, f"size ratio {ratio:.2f}x vs observation is "
                    f"outside the accepted [{lo}, {hi}] range", size_ratio_vs_obs=ratio)
    margin = SCALE_SANITY_INNER_MARGIN_FRAC * (hi - lo)
    if ratio <= lo + margin or ratio >= hi - margin:
        return _mk(STATUS_WARNING, f"size ratio {ratio:.2f}x is within "
                    f"{SCALE_SANITY_INNER_MARGIN_FRAC:.0%} of the size-sanity "
                    "boundary", size_ratio_vs_obs=ratio)
    return _mk(STATUS_PASS, f"size ratio {ratio:.2f}x within accepted range",
                size_ratio_vs_obs=ratio)


def check_candidate_disagreement(hybrid_rec) -> dict:
    if hybrid_rec is None:
        return _mk(STATUS_NA, "no hybrid.json (single-candidate pipeline run; "
                    "SIMANY_HYBRID_CANDIDATES not used for this object)")
    vals = {c: hybrid_rec.get(f"sym_chamfer_{c}_m") for c in ("trellis", "rvg", "sam3d")}
    scored = {c: v for c, v in vals.items() if v is not None}
    if len(scored) < 2:
        return _mk(STATUS_NA, "fewer than 2 scored registration candidates")
    spread_mm = (max(scored.values()) - min(scored.values())) * 1000.0
    status = _band(spread_mm, CANDIDATE_DISAGREEMENT_WARN_MM, CANDIDATE_DISAGREEMENT_FAIL_MM)
    return _mk(status, f"registration candidates disagree by {spread_mm:.1f}mm "
                f"(winner={hybrid_rec.get('winner')})",
                spread_mm=spread_mm, candidates=scored, winner=hybrid_rec.get("winner"))


def check_visual_collision_distance(build_dir: Path, oid: str) -> dict:
    odir = build_dir / "objects" / oid
    visual_path = None
    for name in ("trellis_mesh.ply", "mesh_sim.ply"):
        if (odir / name).exists():
            visual_path = odir / name
            break
    cdir = odir / "collision"
    parts = sorted(cdir.glob("part_*.obj")) if cdir.exists() else []
    if visual_path is None or not parts:
        return _mk(STATUS_NA, "no visual+collision mesh pair found for this "
                    "object (need <visual mesh> and collision/part_*.obj)")
    try:
        import trimesh
        from scipy.spatial import cKDTree
    except ImportError as e:
        return _mk(STATUS_NA, f"geometry libraries unavailable ({e})")
    try:
        vmesh = trimesh.load(visual_path, process=False)
        vpts, _ = trimesh.sample.sample_surface(vmesh, 4000)
        cpts = np.concatenate([
            np.asarray(trimesh.sample.sample_surface(trimesh.load(p, process=False), 1000)[0])
            for p in parts
        ], axis=0)
        d, _ = cKDTree(cpts).query(np.asarray(vpts), k=1)
        dist_mm = float(np.percentile(d, 95)) * 1000.0
    except Exception as e:  # pragma: no cover -- defensive, not exercised by fixtures
        return _mk(STATUS_NA, f"surface-distance computation failed "
                    f"({type(e).__name__}: {e})")
    status = _band(dist_mm, VIS_COLLISION_WARN_MM, VIS_COLLISION_FAIL_MM)
    return _mk(status, f"95th-pct visual-to-collision surface distance "
                f"{dist_mm:.1f}mm", p95_distance_mm=dist_mm)


def check_removal_alpha(removal_ev) -> dict:
    if removal_ev is None or removal_ev.get("alpha_cov") is None:
        return _mk(STATUS_NA, "no alpha-coverage evidence (object was not "
                    "part of a removal-verification run)")
    a = removal_ev["alpha_cov"]
    if a >= ALPHA_COV_PASS:
        status = STATUS_PASS
    elif a >= ALPHA_COV_WARN:
        status = STATUS_WARNING
    else:
        status = STATUS_FAIL
    return _mk(status, f"removal-mask alpha coverage {a:.3f}", alpha_cov=a)


def check_removal_depth(removal_ev) -> dict:
    if removal_ev is None or removal_ev.get("depth_plane_mm") is None:
        return _mk(STATUS_NA, "no depth-vs-support-plane evidence for the "
                    "removed region")
    d = removal_ev["depth_plane_mm"]
    status = _band(d, DEPTH_PLANE_WARN_MM, DEPTH_PLANE_FAIL_MM)
    return _mk(status, f"filled-region depth-vs-plane residual {d:.1f}mm",
                depth_plane_mm=d)


def check_removal_redetection(removal_ev) -> dict:
    if removal_ev is None or removal_ev.get("det_after") is None:
        return _mk(STATUS_NA, "no residual re-detection score (SAM3 stage 2 "
                    "of verify_removal_check.py has not been run for this object)")
    d = removal_ev["det_after"]
    status = _band(d, REDETECTION_WARN_SCORE, REDETECTION_FAIL_SCORE)
    return _mk(status, f"post-removal residual detection score {d:.3f} "
                f"(removed_ok={removal_ev.get('removed_ok')})",
                det_after=d, det_before=removal_ev.get("det_before"),
                removed_ok=removal_ev.get("removed_ok"))


def check_penetration(drop_ev) -> dict:
    if drop_ev is None or drop_ev.get("sunk") is None:
        return _mk(STATUS_NA, "no drop-test record with a 'sunk' field for this object")
    if drop_ev["sunk"]:
        return _mk(STATUS_FAIL, f"object sank below the support plane during "
                    f"settle (z < {DROP_SUNK_Z_M}m)", sunk=True)
    return _mk(STATUS_PASS, "object did not sink through its support during "
                "the drop test", sunk=False)


def check_drop_response(drop_ev) -> dict:
    if drop_ev is None or drop_ev.get("stable") is None:
        return _mk(STATUS_NA, "no drop-test record with a 'stable' field for this object")
    if drop_ev["stable"]:
        return _mk(STATUS_PASS, "drop test settled stably")
    return _mk(STATUS_FAIL, "drop test reports an unstable settle (excess "
                "drift and/or sinking)", stable=False)


def check_settle_drift(drop_ev) -> dict:
    if drop_ev is None or drop_ev.get("drift_m") is None:
        return _mk(STATUS_NA, "no drop-test record with a 'drift_m' field for this object")
    drift_mm = drop_ev["drift_m"] * 1000.0
    status = _band(drift_mm, SETTLE_DRIFT_WARN_MM, DROP_STABLE_DRIFT_M * 1000.0)
    return _mk(status, f"post-settle drift {drift_mm:.1f}mm", drift_mm=drift_mm)


def check_support_overlap(plane_rec) -> dict:
    if plane_rec is None:
        return _mk(STATUS_NA, "no support-plane evidence for this object "
                    "(removal verification not run, or object is legitimately "
                    "floor-standing with no support ring to fit)")
    trim_ok = plane_rec.get("trim_ok")
    if trim_ok is None:
        return _mk(STATUS_NA, "plane.json is missing the 'trim_ok' field")
    if trim_ok:
        return _mk(STATUS_PASS, "support plane fit cleanly from the ring "
                    "around the object's footprint")
    return _mk(STATUS_FAIL, "support-plane fit fell back to a contaminated "
                "one-shot fit -- the object's support surface cannot be "
                "confidently modeled")


OBJECT_CHECK_NAMES = (
    "ghost_object", "registration_residual", "scale_sanity",
    "candidate_disagreement", "visual_collision_surface_distance",
    "removal_alpha_coverage", "removal_depth_residual", "removal_redetection",
    "penetration", "support_overlap", "settle_drift", "drop_response",
)


def compute_object_checks(build_dir: Path, raw: dict) -> dict[str, dict]:
    """oid -> {check_name: check_dict}"""
    physics_stage_ran = raw["sources_present"]["report.json"] or raw["sources_present"]["drop_v2.json"]
    out: dict[str, dict] = {}
    for oid in raw["objects_meta"]:
        aligned_rec = raw["aligned"].get(oid)
        drop_ev = _drop_evidence(oid, raw)
        removal_ev = _removal_evidence(oid, raw)
        out[oid] = {
            "ghost_object": check_ghost_object(aligned_rec, drop_ev, physics_stage_ran),
            "registration_residual": check_registration_residual(aligned_rec),
            "scale_sanity": check_scale_sanity(aligned_rec),
            "candidate_disagreement": check_candidate_disagreement(raw["hybrid"].get(oid)),
            "visual_collision_surface_distance": check_visual_collision_distance(build_dir, oid),
            "removal_alpha_coverage": check_removal_alpha(removal_ev),
            "removal_depth_residual": check_removal_depth(removal_ev),
            "removal_redetection": check_removal_redetection(removal_ev),
            "penetration": check_penetration(drop_ev),
            "support_overlap": check_support_overlap(raw["plane"].get(oid)),
            "settle_drift": check_settle_drift(drop_ev),
            "drop_response": check_drop_response(drop_ev),
        }
    return out


# --------------------------------------------------------------------------
# Scene-only checks
# --------------------------------------------------------------------------

def check_reconstruction_coverage(raw: dict) -> dict:
    objects_meta = raw["objects_meta"]
    n_total = len(objects_meta)
    if n_total == 0:
        return _mk(STATUS_NA, "objects.json lists no discovered objects for this build")
    n_accepted = sum(
        1 for oid in objects_meta
        if raw["aligned"].get(oid) and not raw["aligned"][oid].get("rejected")
    )
    frac = n_accepted / n_total
    status = _band(frac, COVERAGE_WARN_FRAC, COVERAGE_FAIL_FRAC, higher_is_worse=False)
    return _mk(status, f"{n_accepted}/{n_total} discovered objects reached an "
                f"accepted registration ({frac:.0%})",
                n_total=n_total, n_accepted=n_accepted, coverage_fraction=frac)


def check_held_out_rendering(render_metrics: dict | None) -> dict:
    if render_metrics is None:
        return _mk(STATUS_NA, "no render_metrics_v2.json (factory_eval_render.py not run)")
    mean = render_metrics.get("mean", {})
    twin = mean.get("twin")
    if twin is None or twin.get("psnr") is None:
        return _mk(STATUS_NA, "render_metrics_v2.json missing the 'twin' variant's psnr")
    psnr = twin["psnr"]
    bg = mean.get("bg") or {}
    delta = (bg["psnr"] - psnr) if bg.get("psnr") is not None else None
    fail = psnr < TWIN_PSNR_FAIL_DB or (delta is not None and delta > PSNR_DELTA_FAIL_DB)
    warn = psnr < TWIN_PSNR_WARN_DB or (delta is not None and delta > PSNR_DELTA_WARN_DB)
    status = STATUS_FAIL if fail else STATUS_WARNING if warn else STATUS_PASS
    return _mk(status, f"held-out twin render PSNR {psnr:.1f}dB"
                + (f" (bg-twin delta {delta:.1f}dB)" if delta is not None else ""),
                psnr=psnr, ssim=twin.get("ssim"), lpips=twin.get("lpips"),
                bg_psnr=bg.get("psnr"), delta_psnr=delta,
                split=render_metrics.get("split"),
                n_objects_composited=render_metrics.get("n_objects_composited"),
                source="render_metrics_v2.json")


def check_missing_instances(eval_vs_gt: dict | None) -> dict:
    if eval_vs_gt is None:
        return _mk(STATUS_NA, "no eval_vs_gt.json (ScanNet++ GT-based instance "
                    "recall not evaluated for this build)")
    n_missed, n_wl = eval_vs_gt.get("n_gt_missed"), eval_vs_gt.get("n_gt_whitelist")
    if n_missed is None or n_wl is None:
        return _mk(STATUS_NA, "eval_vs_gt.json missing n_gt_missed/n_gt_whitelist")
    if n_wl == 0:
        return _mk(STATUS_NA, "no GT whitelist instances recorded for this scene")
    frac = n_missed / n_wl
    if n_missed == 0:
        status = STATUS_PASS
    elif frac <= MISSING_INSTANCE_WARN_FRAC:
        status = STATUS_WARNING
    else:
        status = STATUS_FAIL
    return _mk(status, f"{n_missed}/{n_wl} GT instances missed ({frac:.0%})",
                n_gt_missed=n_missed, n_gt_whitelist=n_wl, missed_fraction=frac)


def check_duplicate_instances(raw: dict) -> dict:
    eval_vs_gt = raw["eval_vs_gt"]
    if eval_vs_gt is not None:
        groups: dict[int, list[str]] = defaultdict(list)
        for row in eval_vs_gt.get("objects", []):
            gid = row.get("matched_gt_id")
            if gid is not None and row.get("tier") in ("A", "B"):
                groups[gid].append(row["name"])
        dups = {gid: names for gid, names in groups.items() if len(names) > 1}
        if dups:
            involved = sorted({n for names in dups.values() for n in names})
            return _mk(STATUS_FAIL, f"{len(dups)} GT instance(s) matched by "
                        f"multiple accepted objects: {dups}",
                        duplicate_groups=dups, involved_objects=involved,
                        source="eval_vs_gt.json")
        return _mk(STATUS_PASS, "no GT instance matched by more than one "
                    "accepted object", source="eval_vs_gt.json")

    # GT-free fallback: same label + near-identical world position among
    # accepted objects is a heuristic duplicate signal, not authoritative,
    # hence capped at "warning" (never "fail") -- see docs/BUILD_AUDIT_SCHEMA.md.
    accepted = [
        (oid, meta, raw["aligned"][oid]) for oid, meta in raw["objects_meta"].items()
        if raw["aligned"].get(oid) and not raw["aligned"][oid].get("rejected")
        and raw["aligned"][oid].get("T") is not None
    ]
    if not accepted:
        return _mk(STATUS_NA, "no accepted objects with pose data to compare "
                    "(and no eval_vs_gt.json)")
    pairs = []
    for i in range(len(accepted)):
        oid1, m1, a1 = accepted[i]
        t1 = np.asarray(a1["T"])[:3, 3]
        for j in range(i + 1, len(accepted)):
            oid2, m2, a2 = accepted[j]
            if str(m1.get("label", "")).strip().lower() != str(m2.get("label", "")).strip().lower():
                continue
            t2 = np.asarray(a2["T"])[:3, 3]
            if float(np.linalg.norm(t1 - t2)) < DUPLICATE_HEURISTIC_DIST_M:
                pairs.append([oid1, oid2])
    if pairs:
        involved = sorted({oid for pair in pairs for oid in pair})
        return _mk(STATUS_WARNING, f"{len(pairs)} same-label object pair(s) "
                    f"within {DUPLICATE_HEURISTIC_DIST_M * 100:.0f}cm of each "
                    "other (heuristic; no GT available to confirm)",
                    heuristic_pairs=pairs, involved_objects=involved,
                    source="objects.json+aligned.json (GT-free heuristic)")
    return _mk(STATUS_PASS, "no duplicate signal found (GT-free heuristic; "
                "no eval_vs_gt.json available)", source="objects.json+aligned.json")


def _room_field_check(room: dict | None, field: str, warn: float, fail: float,
                       higher_is_worse: bool, label: str) -> dict:
    if room is None:
        return _mk(STATUS_NA, "room_collision_report.json not found -- Task 06 "
                    "(robo/sim/room_collision.py) has not landed yet; expected "
                    "fields: coverage_fraction, penetration_max_mm, "
                    "disconnected_components, settle_drift_mm")
    val = room.get(field)
    if val is None:
        return _mk(STATUS_NA, f"room_collision_report.json present but missing '{field}'")
    status = _band(val, warn, fail, higher_is_worse=higher_is_worse)
    return _mk(status, f"{label} = {val}", **{field: val})


def check_room_collision_coverage(room: dict | None) -> dict:
    return _room_field_check(room, "coverage_fraction", ROOM_COVERAGE_WARN_FRAC,
                              ROOM_COVERAGE_FAIL_FRAC, higher_is_worse=False,
                              label="room-collision coverage fraction")


def check_room_disconnected_components(room: dict | None) -> dict:
    return _room_field_check(room, "disconnected_components", ROOM_DISCONNECTED_WARN_N,
                              ROOM_DISCONNECTED_FAIL_N, higher_is_worse=True,
                              label="disconnected floating components")


def check_room_penetration_max(room: dict | None) -> dict:
    return _room_field_check(room, "penetration_max_mm", ROOM_PENETRATION_WARN_MM,
                              ROOM_PENETRATION_FAIL_MM, higher_is_worse=True,
                              label="max room-scale penetration (mm)")


def check_room_settle_drift(room: dict | None) -> dict:
    return _room_field_check(room, "settle_drift_mm", ROOM_SETTLE_DRIFT_WARN_MM,
                              ROOM_SETTLE_DRIFT_FAIL_MM, higher_is_worse=True,
                              label="room-scale settle drift (mm)")


def check_metric_scale(alignment_report: dict | None) -> dict:
    if alignment_report is None:
        return _mk(STATUS_NA, "alignment_report.json not found -- Task 04 "
                    "(agents/recon/robot_align.py / alignment_report.py) has "
                    "not landed yet; expected fields: scale_factor, scale_ci, "
                    "translation_residual_mm, rotation_residual_deg, "
                    "reprojection_residual_px, floor_normal_error_deg, "
                    "table_height_error_mm, held_out_error_m")
    status = alignment_report.get("metric_scale_status")
    if status in VALID_STATUSES:
        return _mk(status, "status reported directly by alignment_report.json",
                    scale_factor=alignment_report.get("scale_factor"),
                    scale_ci=alignment_report.get("scale_ci"))
    return _mk(STATUS_NA, "alignment_report.json present but does not report "
                "a recognized metric_scale_status")


def check_robot_alignment(alignment_report: dict | None) -> dict:
    if alignment_report is None:
        return _mk(STATUS_NA, "alignment_report.json not found -- Task 04 "
                    "(agents/recon/robot_align.py / alignment_report.py) has "
                    "not landed yet; expected fields: held_out_error_m, "
                    "reprojection_residual_px, robot_alignment_status")
    status = alignment_report.get("robot_alignment_status")
    if status in VALID_STATUSES:
        return _mk(status, "status reported directly by alignment_report.json",
                    held_out_error_m=alignment_report.get("held_out_error_m"),
                    reprojection_residual_px=alignment_report.get("reprojection_residual_px"))
    return _mk(STATUS_NA, "alignment_report.json present but does not report "
                "a recognized robot_alignment_status")


SCENE_ONLY_CHECK_NAMES = (
    "reconstruction_coverage", "held_out_rendering", "duplicate_instances",
    "missing_instances", "room_collision_coverage",
    "room_disconnected_components", "room_penetration_max",
    "room_settle_drift", "metric_scale", "robot_alignment",
)


def compute_scene_only_checks(raw: dict) -> dict[str, dict]:
    room = raw["room_collision"]
    align = raw["alignment_report"]
    return {
        "reconstruction_coverage": check_reconstruction_coverage(raw),
        "held_out_rendering": check_held_out_rendering(raw["render_metrics"]),
        "duplicate_instances": check_duplicate_instances(raw),
        "missing_instances": check_missing_instances(raw["eval_vs_gt"]),
        "room_collision_coverage": check_room_collision_coverage(room),
        "room_disconnected_components": check_room_disconnected_components(room),
        "room_penetration_max": check_room_penetration_max(room),
        "room_settle_drift": check_room_settle_drift(room),
        "metric_scale": check_metric_scale(align),
        "robot_alignment": check_robot_alignment(align),
    }


# --------------------------------------------------------------------------
# Aggregation (worst-of; never averaged)
# --------------------------------------------------------------------------

def _aggregate_over_objects(check_name: str, per_object: dict[str, dict]) -> dict:
    entries = {oid: obj_checks[check_name] for oid, obj_checks in per_object.items()}
    if not entries:
        return _mk(STATUS_NA, "no objects were evaluated for this check")
    worst_oid, worst_sev = None, -1
    counts = dict.fromkeys(_STATUS_ORDER, 0)
    for oid, res in entries.items():
        counts[res["status"]] += 1
        sev = _SEVERITY[res["status"]]
        if sev > worst_sev:
            worst_sev, worst_oid = sev, oid
    status = next(s for s, sev in _SEVERITY.items() if sev == worst_sev)
    if status == STATUS_NA:
        reason = "no object had usable evidence for this check"
    else:
        reason = f"worst case {worst_oid}: {entries[worst_oid]['reason']}"
    return _mk(status, reason, counts=counts,
               worst_object=worst_oid if status != STATUS_NA else None)


def _overall_status(checks: dict[str, dict]) -> str:
    sev, status = -1, STATUS_NA
    for c in checks.values():
        s = _SEVERITY[c["status"]]
        if s > sev:
            sev, status = s, c["status"]
    return status


# --------------------------------------------------------------------------
# Top-level assembly
# --------------------------------------------------------------------------

def _object_evidence_paths(build_dir: Path, oid: str, raw: dict) -> dict[str, str]:
    rel = {}
    candidates = {
        "aligned_json": Path("objects") / oid / "aligned.json",
        "hybrid_json": Path("objects") / oid / "hybrid.json",
        "plane_json": Path("inpaint") / oid / "plane.json",
        "rgba_png": Path("objects") / oid / "rgba.png",
    }
    for key, relpath in candidates.items():
        if (build_dir / relpath).exists():
            rel[key] = str(relpath)
    return rel


def run_audit(build_dir: Path, now: str | None = None) -> dict:
    build_dir = Path(build_dir)
    raw = load_build(build_dir)
    per_object_raw = compute_object_checks(build_dir, raw)
    scene_only = compute_scene_only_checks(raw)

    checks: dict[str, dict] = {}
    for name in OBJECT_CHECK_NAMES:
        checks[name] = _aggregate_over_objects(name, per_object_raw)
    checks.update(scene_only)

    objects_summary = []
    for oid, meta in sorted(raw["objects_meta"].items(), key=lambda kv: kv[1]["index"]):
        obj_checks = per_object_raw[oid]
        applicable = [c["status"] for c in obj_checks.values()]
        obj_status = next(
            (s for s, sev in sorted(_SEVERITY.items(), key=lambda kv: -kv[1])
             if s in applicable),
            STATUS_NA,
        )
        objects_summary.append({
            "object_id": oid,
            "label": meta.get("label"),
            "gt_object_id": meta.get("gt_object_id"),
            "tier": (raw["aligned"].get(oid) or {}).get("tier"),
            "rejected": (raw["aligned"].get(oid) or {}).get("rejected"),
            "object_status": obj_status,
            "checks": obj_checks,
            "evidence": _object_evidence_paths(build_dir, oid, raw),
        })

    repair_queue = []
    for obj in objects_summary:
        for check_name, res in obj["checks"].items():
            if res["status"] == STATUS_FAIL:
                repair_queue.append({"object_id": obj["object_id"],
                                      "stage": check_name, "reason": res["reason"]})
    for check_name in SCENE_ONLY_CHECK_NAMES:
        res = checks[check_name]
        if res["status"] != STATUS_FAIL:
            continue
        involved = res.get("involved_objects")
        if involved:
            for oid in involved:
                repair_queue.append({"object_id": oid, "stage": check_name,
                                      "reason": res["reason"]})
        else:
            repair_queue.append({"object_id": "scene", "stage": check_name,
                                  "reason": res["reason"]})

    audit = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now,
        "build_dir": str(build_dir),
        "scene_id": build_dir.name,
        "overall_status": _overall_status(checks),
        "checks": checks,
        "objects": objects_summary,
        "repair_queue": repair_queue,
        "sources_present": raw["sources_present"],
    }
    return audit


# --------------------------------------------------------------------------
# Human-readable report
# --------------------------------------------------------------------------

_STATUS_GLYPH = {STATUS_PASS: "PASS", STATUS_WARNING: "WARN",
                 STATUS_FAIL: "FAIL", STATUS_NA: "N/A"}


def render_human_report(audit: dict) -> str:
    lines = [
        f"# Build audit: {audit['scene_id']}",
        "",
        f"schema_version: {audit['schema_version']}  "
        f"generated_at: {audit.get('generated_at')}",
        f"build_dir: {audit['build_dir']}",
        "",
        f"## Overall: {_STATUS_GLYPH[audit['overall_status']]}",
        "",
        "## Scene-level checks",
        "",
        "| check | status | reason |",
        "|---|---|---|",
    ]
    for name, res in audit["checks"].items():
        lines.append(f"| {name} | {_STATUS_GLYPH[res['status']]} | {res['reason']} |")
    lines += ["", "## Objects", "", "| object | label | tier | status |", "|---|---|---|---|"]
    for obj in audit["objects"]:
        lines.append(f"| {obj['object_id']} | {obj['label']} | {obj['tier']} | "
                      f"{_STATUS_GLYPH[obj['object_status']]} |")
    if audit["repair_queue"]:
        lines += ["", "## Repair queue (Task 13)", "", "| object | stage | reason |", "|---|---|---|"]
        for item in audit["repair_queue"]:
            lines.append(f"| {item['object_id']} | {item['stage']} | {item['reason']} |")
    else:
        lines += ["", "## Repair queue (Task 13)", "", "(empty)"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    import sys
    from datetime import datetime, timezone

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("build_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                     help="audit JSON output path (default: <build_dir>/build_audit.json)")
    ap.add_argument("--report-out", type=Path, default=None,
                     help="human-readable report path "
                          "(default: <build_dir>/build_audit_report.md)")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    now = datetime.now(timezone.utc).isoformat()
    audit = run_audit(args.build_dir, now=now)

    out = args.out or (args.build_dir / "build_audit.json")
    out.write_text(json.dumps(audit, indent=1))
    report_out = args.report_out or (args.build_dir / "build_audit_report.md")
    report_out.write_text(render_human_report(audit))
    print(f"[build_audit] {args.build_dir}: overall={audit['overall_status']} "
          f"-> {out}")
    return 0 if audit["overall_status"] != STATUS_FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
