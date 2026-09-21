"""Task 12 -- Support/contact feature group.

Plan group 3: "penetration, support area/COM margin, settle drift, bounded
perturbation sensitivity."

Sources:
  - `build_audit` (Task 05): penetration, drop_response, settle_drift,
    support_overlap -- all per-object drop-test-derived checks. These four
    checks read *different fields of the same upstream drop-test record* by
    design (see docs/BUILD_AUDIT_SCHEMA.md section 6): `penetration` and
    `settle_drift` are surfaced here as genuinely distinct signals rather
    than one blob, matching build_audit's own "a repair agent needs to know
    *which* problem it is looking at" rationale.
  - `task_graph` (Task 11): the *only* source for a real "support area / COM
    margin" proxy. `build_graph()`'s `compute_support_edges` already computes,
    per support edge, the xy-footprint overlap fraction between an object and
    whatever it `rests_on` (`grounding.py::compute_support_edges`,
    `min_xy_overlap_frac` gate). `support_com_margin_risk` below is
    `1 - overlap_fraction` for the manipulated object's resolved support edge
    -- a coarser but real, purely-geometric stand-in for a true center-of-mass
    stability margin (no COM/mass model exists anywhere upstream in this
    repo, so this is honestly a footprint-overlap proxy, not a physical COM
    computation).

"Bounded perturbation sensitivity" (plan group 3's last bullet) has no
dedicated upstream probe either -- the closest real evidence is the drop
test's own stability verdict (an object released from a small height and
settled is, in effect, a single bounded perturbation trial). `drop_response`
already captures that outcome; this module does not invent a second reading
of the same evidence under a different name, it is simply named for what it
is (`support_drop_unstable_any`) rather than for the more general probe the
plan envisions. See `dynamics_probes.py`'s docstring for a fuller discussion
of what real probe evidence does and does not exist yet.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

FEATURE_NAMES: tuple[str, ...] = (
    "support_penetration_any",
    "support_settle_drift_mm",
    "support_overlap_fail_any",
    "support_drop_unstable_any",
    "support_com_margin_risk",
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
    saw_evidence = False
    for o in objs:
        c = _obj_check(o, check_name)
        if c.get("status") == "not_applicable":
            continue
        saw_evidence = True
        if c.get("status") == status:
            return 1.0
    return 0.0 if saw_evidence else None


def _support_com_margin_risk(task_graph: Mapping[str, Any] | None):
    """1 - (xy footprint overlap fraction) for the manipulated object's
    resolved `support` edge (rests_on relation). `None` when no task graph,
    no resolved support edge, or no manipulated_object hypothesis exists --
    never defaulted to 0.0 (which would read as "perfectly centered")."""
    if task_graph is None:
        return None
    role_res = task_graph.get("role_resolutions") or {}
    if (role_res.get("support") or {}).get("missing_evidence"):
        return None
    manip = role_res.get("manipulated_object") or {}
    manip_hyps = manip.get("hypotheses") or []
    if not manip_hyps:
        return None
    manip_id = manip_hyps[0]["object_id"]
    for e in task_graph.get("edges") or []:
        if e.get("kind") == "support" and e.get("src") == manip_id:
            overlap_frac = e.get("confidence")
            if overlap_frac is None:
                return None
            return max(0.0, 1.0 - float(overlap_frac))
    return None


def extract_support_contact_features(build_audit: Mapping[str, Any] | None,
                                      node_ids: Sequence[str] | None = None,
                                      task_graph: Mapping[str, Any] | None = None
                                      ) -> dict[str, float | None]:
    build_audit = build_audit or {}
    objs = _objects(build_audit, node_ids)
    out: dict[str, float | None] = {}

    _set(out, "support_penetration_any", _any_status(objs, "penetration", "fail"))
    _set(out, "support_settle_drift_mm",
         _worst_numeric(objs, "settle_drift", "drift_mm", higher_is_worse=True))
    _set(out, "support_overlap_fail_any", _any_status(objs, "support_overlap", "fail"))
    _set(out, "support_drop_unstable_any", _any_status(objs, "drop_response", "fail"))
    _set(out, "support_com_margin_risk", _support_com_margin_risk(task_graph))

    assert set(out) == set(FEATURE_NAMES) | {f"{n}_missing" for n in FEATURE_NAMES}
    return out
