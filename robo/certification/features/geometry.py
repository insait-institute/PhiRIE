"""Task 12 -- Pose/geometry feature group.

Plan group 2: "registration residual, candidate disagreement, visible
fraction, scale uncertainty, visual-collision distance."

Sources:
  - `build_audit` (Task 05): registration_residual, candidate_disagreement,
    scale_sanity, visual_collision_surface_distance (all per-object checks),
    plus the scene-level metric_scale check for scale uncertainty.
  - `alignment_report` (Task 04, `agents.recon.alignment_report.build_report()`
    output): the *authoritative* source for scale uncertainty
    (`hard_thresholds.scale_ci95_halfwidth_pct`) -- preferred over
    reconstructing it from build_audit's `metric_scale.scale_ci`, since Task 04
    owns that number and already computes the halfwidth-vs-scale percentage
    itself (see `agents/recon/alignment_report.py::build_report`).
  - `task_graph` (Task 11, `robo.certification.task_graph.build_graph()`
    output): the *only* source for "visible fraction" -- build_audit's camera
    reasoning is scene-composited PSNR, not per-object visibility; task_graph
    nodes carry a real per-object `camera_visible` list computed from the
    scene's actual camera frustums (`grounding.py::camera_visible_ids`).
    Restricting this to `node_ids` (the eligible/task-relevant object set) is
    what makes "visible fraction" a task-local number instead of a
    whole-scene one.

Missingness contract: identical to `visual.py` -- see that module's docstring.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

FEATURE_NAMES: tuple[str, ...] = (
    "geom_registration_residual_mm",
    "geom_icp_tilt_deg",
    "geom_candidate_disagreement_mm",
    "geom_scale_sanity_margin",
    "geom_visual_collision_distance_mm",
    "geom_visible_fraction",
    "geom_scale_ci95_halfwidth_pct",
)

# Reused verbatim from agents/eval/build_audit.py's SIZE_RATIO_RANGE (not a
# new number -- see docs/BUILD_AUDIT_SCHEMA.md section 3).
SIZE_RATIO_RANGE = (0.4, 2.5)


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


def _scale_sanity_margin(objs) -> float | None:
    """Worst (smallest) distance from `size_ratio_vs_obs` to the nearer edge
    of SIZE_RATIO_RANGE, normalized by the range width -- 0.0 means an object
    sits exactly on the accepted/rejected boundary, larger is safer. `None`
    when no in-scope object has a recorded size ratio."""
    lo, hi = SIZE_RATIO_RANGE
    width = hi - lo
    margins = []
    for o in objs:
        c = _obj_check(o, "scale_sanity")
        if c.get("status") == "not_applicable":
            continue
        ratio = c.get("size_ratio_vs_obs")
        if ratio is None:
            continue
        margins.append(min(ratio - lo, hi - ratio) / width)
    if not margins:
        return None
    return min(margins)


def _visible_fraction(task_graph: Mapping[str, Any] | None, node_ids: Sequence[str] | None):
    """Fraction of task-graph object nodes (restricted to `node_ids` when
    given) that are visible from at least one camera, per task_graph's own
    `camera_visible` field. `None` when no task graph was supplied or it has
    no object nodes in scope -- never defaulted to 1.0 (favorable)."""
    if task_graph is None or "policy_cameras" in task_graph.get("missing_evidence", []):
        return None
    nodes = [n for n in (task_graph.get("nodes") or []) if n.get("kind") == "object"]
    if node_ids is not None:
        node_id_set = set(node_ids)
        nodes = [n for n in nodes if n.get("node_id") in node_id_set]
    if not nodes:
        return None
    n_visible = sum(1 for n in nodes if n.get("camera_visible"))
    return n_visible / len(nodes)


def _scale_ci95_halfwidth_pct(build_audit: Mapping[str, Any],
                               alignment_report: Mapping[str, Any] | None):
    if alignment_report is not None:
        hard = (alignment_report.get("hard_thresholds") or {}).get("scale_ci95_halfwidth_pct") or {}
        if hard.get("value") is not None:
            return hard["value"]
    checks = build_audit.get("checks") or {}
    ms = checks.get("metric_scale") or {}
    if ms.get("status") == "not_applicable":
        return None
    scale, ci = ms.get("scale_factor"), ms.get("scale_ci")
    if scale is None or not ci or len(ci) != 2:
        return None
    return (ci[1] - ci[0]) / 2.0 / max(abs(scale), 1e-9) * 100.0


def extract_geometry_features(build_audit: Mapping[str, Any] | None,
                               node_ids: Sequence[str] | None = None,
                               task_graph: Mapping[str, Any] | None = None,
                               alignment_report: Mapping[str, Any] | None = None
                               ) -> dict[str, float | None]:
    build_audit = build_audit or {}
    objs = _objects(build_audit, node_ids)
    out: dict[str, float | None] = {}

    _set(out, "geom_registration_residual_mm",
         _worst_numeric(objs, "registration_residual", "chamfer_med_mm", higher_is_worse=True))
    _set(out, "geom_icp_tilt_deg",
         _worst_numeric(objs, "registration_residual", "icp_tilt_deg", higher_is_worse=True))
    _set(out, "geom_candidate_disagreement_mm",
         _worst_numeric(objs, "candidate_disagreement", "spread_mm", higher_is_worse=True))
    _set(out, "geom_scale_sanity_margin", _scale_sanity_margin(objs))
    _set(out, "geom_visual_collision_distance_mm",
         _worst_numeric(objs, "visual_collision_surface_distance", "p95_distance_mm",
                         higher_is_worse=True))
    _set(out, "geom_visible_fraction", _visible_fraction(task_graph, node_ids))
    _set(out, "geom_scale_ci95_halfwidth_pct",
         _scale_ci95_halfwidth_pct(build_audit, alignment_report))

    assert set(out) == set(FEATURE_NAMES) | {f"{n}_missing" for n in FEATURE_NAMES}
    return out
