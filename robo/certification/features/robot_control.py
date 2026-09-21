"""Task 12 -- Robot/control feature group.

Plan group 5: "IK and collision margins, target visibility, open-loop replay
error."

Honest status: no real inverse-kinematics solver or collision-margin
computation exists anywhere in this repo yet (checked by grep before writing
this module). Two real, purely-geometric proxies stand in, both computed
directly from `task_graph` (Task 11) evidence -- never from an eval-result
field, since task_graph.py's own docstring guarantees its output contains
none:

  - `robot_reach_margin_m`: reach-envelope slack for the resolved
    manipulated_object/target/receptacle anchor, i.e. how much room is left
    between the robot base and its own `reach_envelope_m` before the target
    falls outside the eligibility gate `task_graph.py::eligible_object_ids`
    already used to decide graph membership. This is a coarse reachability
    proxy, *not* a real IK feasibility check -- named distinctly so a reader
    (or a future repair module) never mistakes it for one.
  - `robot_obstacle_count_in_sweep`: number of objects the task graph's own
    swept-workspace capsule test (`grounding.py::resolve_obstacle_role`)
    resolved as `obstacle` -- a real, task-local collision-risk proxy (more
    objects in the sweep path is more collision risk), again not a true
    collision-margin distance.

`robot_target_visible` reads task_graph's per-object `camera_visible` list
for whichever role (manipulated_object / target / receptacle) actually
resolved -- distinct from `geometry.py`'s `geom_visible_fraction` (which
averages over *every* task-local object) in that this one asks specifically
"can the policy see the thing it needs to act on."

`robot_open_loop_replay_error_m` is the one number here with a real upstream
source: Task 04's held-out alignment residual
(`agents/recon/alignment_report.py::build_report`'s `held_out.rms_m`, or
`agents/eval/build_audit.py`'s `robot_alignment` check's `held_out_error_m`)
is exactly an "open-loop replay error" -- markers/trajectory points predicted
by the fitted transform vs. their independently-measured held-out positions.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

FEATURE_NAMES: tuple[str, ...] = (
    "robot_ik_collision_margin",
    "robot_reach_margin_m",
    "robot_obstacle_count_in_sweep",
    "robot_target_visible",
    "robot_open_loop_replay_error_m",
)

_ANCHOR_ROLES = ("manipulated_object", "target", "receptacle")


def _set(out: dict, name: str, value) -> None:
    out[name] = None if value is None else float(value)
    out[f"{name}_missing"] = 1.0 if value is None else 0.0


def _resolved_anchor_id(task_graph: Mapping[str, Any]) -> str | None:
    role_res = task_graph.get("role_resolutions") or {}
    for role in _ANCHOR_ROLES:
        hyps = (role_res.get(role) or {}).get("hypotheses") or []
        if hyps:
            return hyps[0]["object_id"]
    return None


def _node(task_graph: Mapping[str, Any], node_id: str) -> dict | None:
    for n in task_graph.get("nodes") or []:
        if n.get("node_id") == node_id:
            return n
    return None


def _robot_node(task_graph: Mapping[str, Any]) -> dict | None:
    return _node(task_graph, "robot")


def _reach_margin_m(task_graph: Mapping[str, Any]) -> float | None:
    import math

    robot = _robot_node(task_graph)
    anchor_id = _resolved_anchor_id(task_graph)
    if robot is None or robot.get("transform") is None or anchor_id is None:
        return None
    anchor = _node(task_graph, anchor_id)
    if anchor is None or "aabb" not in anchor:
        return None
    lo, hi = anchor["aabb"][0], anchor["aabb"][1]
    center_xy = [(lo[i] + hi[i]) / 2.0 for i in range(2)]
    base_xy = robot["transform"]["pos"][:2]
    dist = math.hypot(center_xy[0] - base_xy[0], center_xy[1] - base_xy[1])
    reach_m = task_graph.get("_reach_envelope_m")  # optional caller-supplied override
    if reach_m is None:
        reach_m = 0.80  # robo.certification.task_graph.DEFAULT_REACH_M, kept in sync manually
    return reach_m - dist


def _obstacle_count(task_graph: Mapping[str, Any]) -> float | None:
    if "robot_frame" in task_graph.get("missing_evidence", []):
        return None
    role_res = task_graph.get("role_resolutions") or {}
    obstacle = role_res.get("obstacle")
    if obstacle is None or obstacle.get("missing_evidence") or _resolved_anchor_id(task_graph) is None:
        return None
    return float(len(obstacle.get("hypotheses") or []))


def _target_visible(task_graph: Mapping[str, Any]) -> float | None:
    if "policy_cameras" in task_graph.get("missing_evidence", []):
        return None
    anchor_id = _resolved_anchor_id(task_graph)
    if anchor_id is None:
        return None
    anchor = _node(task_graph, anchor_id)
    if anchor is None:
        return None
    return 1.0 if anchor.get("camera_visible") else 0.0


def _open_loop_replay_error_m(build_audit: Mapping[str, Any] | None,
                               alignment_report: Mapping[str, Any] | None) -> float | None:
    if alignment_report is not None:
        held = alignment_report.get("held_out") or {}
        if held.get("rms_m") is not None:
            return held["rms_m"]
    if build_audit is not None:
        robot_align = (build_audit.get("checks") or {}).get("robot_alignment") or {}
        if robot_align.get("status") != "not_applicable":
            v = robot_align.get("held_out_error_m")
            if v is not None:
                return v
    return None


def extract_robot_control_features(build_audit: Mapping[str, Any] | None = None,
                                    task_graph: Mapping[str, Any] | None = None,
                                    alignment_report: Mapping[str, Any] | None = None
                                    ) -> dict[str, float | None]:
    out: dict[str, float | None] = {}

    # No real IK/collision-margin solver exists upstream -- forward-declared.
    _set(out, "robot_ik_collision_margin", None)

    if task_graph is not None:
        _set(out, "robot_reach_margin_m", _reach_margin_m(task_graph))
        _set(out, "robot_obstacle_count_in_sweep", _obstacle_count(task_graph))
        _set(out, "robot_target_visible", _target_visible(task_graph))
    else:
        _set(out, "robot_reach_margin_m", None)
        _set(out, "robot_obstacle_count_in_sweep", None)
        _set(out, "robot_target_visible", None)

    _set(out, "robot_open_loop_replay_error_m",
         _open_loop_replay_error_m(build_audit, alignment_report))

    assert set(out) == set(FEATURE_NAMES) | {f"{n}_missing" for n in FEATURE_NAMES}
    return out
