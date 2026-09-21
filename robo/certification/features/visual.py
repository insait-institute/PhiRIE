"""Task 12 -- Visual feature group for the task-conditioned twin certificate.

Plan (`plan/12_TASK_CONDITIONED_CERTIFICATE.md`) group 1: "task-local render
feature residual, alpha/depth consistency, residual detection, ghosting,
camera reprojection."

Every feature here is read, verbatim, from evidence Task 05
(`agents/eval/build_audit.py`) or Task 04 (`agents/recon/alignment_report.py`)
already computed -- this module does not touch a mesh, a render, or a raw
sensor file. It reads exactly two kinds of input:

  - `build_audit`: the dict returned by `agents.eval.build_audit.run_audit()`
    (or loaded straight from `build_audit.json`) -- see docs/BUILD_AUDIT_SCHEMA.md.
  - `alignment_report`: the dict returned by
    `agents.recon.alignment_report.build_report()` (optional -- falls back to
    `build_audit["checks"]["robot_alignment"]` when omitted, since build_audit
    already folds Task 04's own status/fields in when its file was present at
    audit time).

`node_ids`, when given, is the set of graph-local object ids from a
`robo.certification.task_graph.build_graph()` JSON (its `nodes` where
`kind == "object"`) -- restricting the alpha/depth/ghosting/registration-style
per-object aggregation to objects the task graph says are actually relevant to
*this* task, per the plan's "restrict feature extraction to graph-local node
IDs only" instruction. `node_ids=None` falls back to every object in the
build (scene-global), which is honest but no longer task-local -- callers
should pass the real set whenever a task graph is available.

Missingness contract (shared by every feature module in this package): every
feature name `X` is always paired with `X_missing` (1.0/0.0). A feature is
recorded as `None` (never a numeric stand-in, and never the value a "pass"
verdict would produce) whenever the underlying check's status is
`not_applicable` for every object in scope, or the evidence field itself is
absent. See `test_certificate.py::test_missingness_never_imputed_as_good`.

Camera reprojection residual is a genuinely scene-level (not per-object)
number in both Task 04's and Task 05's schemas -- there is no per-object
reprojection check anywhere upstream -- so `visual_camera_reprojection_px`
is honestly scene-level, not task-local, and this docstring says so rather
than silently implying otherwise.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

FEATURE_NAMES: tuple[str, ...] = (
    "visual_removal_alpha_coverage",
    "visual_removal_depth_residual_mm",
    "visual_removal_redetection_score",
    "visual_ghost_object_any_fail",
    "visual_held_out_psnr_db",
    "visual_held_out_psnr_delta_db",
    "visual_camera_reprojection_px",
)


def _set(out: dict, name: str, value) -> None:
    out[name] = None if value is None else float(value)
    out[f"{name}_missing"] = 1.0 if value is None else 0.0


def _objects(build_audit: Mapping[str, Any] | None, node_ids: Sequence[str] | None):
    objs = (build_audit or {}).get("objects") or []
    if node_ids is None:
        return objs
    node_ids = set(node_ids)
    return [o for o in objs if o.get("object_id") in node_ids]


def _obj_check(obj: Mapping[str, Any], name: str) -> dict:
    return ((obj.get("checks") or {}).get(name)) or {}


def _worst_numeric(objs, check_name: str, field: str, higher_is_worse: bool = True):
    vals = []
    for o in objs:
        c = _obj_check(o, check_name)
        if c.get("status") == "not_applicable":
            continue
        v = c.get(field)
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None
    return max(vals) if higher_is_worse else min(vals)


def _any_status(objs, check_name: str, status: str):
    """1.0 if any in-scope object hit `status` for `check_name`, 0.0 if every
    in-scope object had evidence and none hit it, None if no object had any
    evidence for this check at all (honestly missing, not a favorable 0.0)."""
    saw_evidence = False
    for o in objs:
        c = _obj_check(o, check_name)
        if c.get("status") == "not_applicable":
            continue
        saw_evidence = True
        if c.get("status") == status:
            return 1.0
    return 0.0 if saw_evidence else None


def extract_visual_features(build_audit: Mapping[str, Any] | None,
                             node_ids: Sequence[str] | None = None,
                             alignment_report: Mapping[str, Any] | None = None
                             ) -> dict[str, float | None]:
    build_audit = build_audit or {}
    objs = _objects(build_audit, node_ids)
    checks = build_audit.get("checks") or {}
    out: dict[str, float | None] = {}

    # -- alpha/depth consistency + residual re-detection (removal QA) -------
    _set(out, "visual_removal_alpha_coverage",
         _worst_numeric(objs, "removal_alpha_coverage", "alpha_cov", higher_is_worse=False))
    _set(out, "visual_removal_depth_residual_mm",
         _worst_numeric(objs, "removal_depth_residual", "depth_plane_mm", higher_is_worse=True))
    _set(out, "visual_removal_redetection_score",
         _worst_numeric(objs, "removal_redetection", "det_after", higher_is_worse=True))

    # -- ghosting -------------------------------------------------------------
    _set(out, "visual_ghost_object_any_fail", _any_status(objs, "ghost_object", "fail"))

    # -- held-out render residual (scene-composited; see module docstring) ---
    hor = checks.get("held_out_rendering") or {}
    if hor.get("status") == "not_applicable":
        _set(out, "visual_held_out_psnr_db", None)
        _set(out, "visual_held_out_psnr_delta_db", None)
    else:
        _set(out, "visual_held_out_psnr_db", hor.get("psnr"))
        _set(out, "visual_held_out_psnr_delta_db", hor.get("delta_psnr"))

    # -- camera reprojection residual (scene-level; Task 04 evidence) --------
    robot_align = checks.get("robot_alignment") or {}
    reproj_px = None
    if robot_align.get("status") != "not_applicable":
        reproj_px = robot_align.get("reprojection_residual_px")
    if reproj_px is None and alignment_report is not None:
        hard = (alignment_report.get("hard_thresholds") or {}).get("reprojection_residual_px") or {}
        reproj_px = hard.get("value")
        if reproj_px is None:
            reproj_px = (alignment_report.get("soft_metrics") or {}).get("reproj_median_px")
    _set(out, "visual_camera_reprojection_px", reproj_px)

    assert set(out) == set(FEATURE_NAMES) | {f"{n}_missing" for n in FEATURE_NAMES}
    return out
