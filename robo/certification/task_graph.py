"""Task-conditioned interaction graph (Task 11).

`build_graph(scene, task)` grounds a task into the smallest scene subgraph
that can causally affect its outcome: the robot, the cameras the policy
actually sees through, the manipulated object, its target/receptacle,
whatever it visibly rests on, and anything geometrically in the way of the
reach path. Role resolution (language/rubric -> scene ids) lives in
grounding.py; this module owns:

  - the eligibility gate (which objects even get to be candidates -- plan/11
    step 4, "select camera-visible and collision-relevant objects"),
  - assembling nodes/edges/role-resolutions into one serializable graph
    (step 5),
  - the CLI and the optional top-down overlay (step 6).

Everything here is a pure function of the scene manifest and the task
manifest read from disk -- no eval-result field is read anywhere in this
module or in grounding.py, so certificate features (Task 12) computed from
this graph's JSON are graph-local and leakage-free by construction (plan/11
acceptance criterion, shared with Task 12: "certificate features can be
computed only on graph-local evidence").

CLI:
    python -m robo.certification.task_graph \
        --task tests/data/tasks/food_bussing.yaml \
        --scene tests/data/scenes/task_graph_fixture
"""
import argparse
import json
from pathlib import Path

import yaml

from robo.certification import grounding as G

DEFAULT_REACH_M = 0.80
DEFAULT_REACH_MIN_M = 0.15
# Wider than robo/tasks/pi05_tasks.py's 60 deg front cone: this is the
# *candidate-eligibility* gate (should this object be a node at all?), not
# the tighter per-task manipulability check the task suite makes downstream,
# so it is deliberately permissive.
DEFAULT_FRONT_CONE_DEG = 75.0
DEFAULT_CAPSULE_RADIUS_M = 0.12
DEFAULT_LIFT_M = 0.12


# ---------------------------------------------------------------------- I/O
def load_scene(path, robot_defaults=None):
    """Load a scene manifest. `path` may be the YAML file itself or a
    directory containing scene.yaml (the CLI/fast-validation contract
    passes the fixture directory: --scene tests/data/scenes/<name>).

    `robot_defaults` (reach_envelope_m/reach_min_m/front_cone_deg) fills in
    only fields the scene itself doesn't set; it comes from a loaded
    configs/task_graph/*.yaml (see load_config) or, absent one, this
    module's hardcoded DEFAULT_* constants -- a scene manifest can always
    override either source explicitly."""
    path = Path(path)
    scene_file = path / "scene.yaml" if path.is_dir() else path
    scene = yaml.safe_load(scene_file.read_text())
    scene.setdefault("cameras", [])
    scene.setdefault("objects", [])
    robot = scene.setdefault("robot", {})
    robot.setdefault("base_yaw", 0.0)
    defaults = {"reach_envelope_m": DEFAULT_REACH_M, "reach_min_m": DEFAULT_REACH_MIN_M,
                "front_cone_deg": DEFAULT_FRONT_CONE_DEG}
    defaults.update(robot_defaults or {})
    for k, v in defaults.items():
        robot.setdefault(k, v)
    for o in scene["objects"]:
        o["aabb"] = [[float(x) for x in o["aabb"][0]], [float(x) for x in o["aabb"][1]]]
    return scene


def load_task(path, capsule_defaults=None):
    """`capsule_defaults` (capsule_radius_m/lift_m) works the same way as
    load_scene's robot_defaults: config-file fallback, task-file override."""
    task = yaml.safe_load(Path(path).read_text())
    task.setdefault("rubric", {})
    task.setdefault("language", {})
    task.setdefault("roles", ["manipulated_object", "receptacle"])
    defaults = {"capsule_radius_m": DEFAULT_CAPSULE_RADIUS_M, "lift_m": DEFAULT_LIFT_M}
    defaults.update(capsule_defaults or {})
    for k, v in defaults.items():
        task.setdefault(k, v)
    return task


def load_config(path):
    """Load a configs/task_graph/*.yaml system config (grounding evidence
    weights + robot/capsule geometry defaults). Returns {} when `path` is
    None -- the CLI's --config flag is optional, defaults come from this
    module's and grounding.py's hardcoded constants otherwise."""
    cfg = (yaml.safe_load(Path(path).read_text()) or {}) if path is not None else {}
    cfg.setdefault("grounding_weights", {})
    cfg.setdefault("robot_defaults", {})
    cfg.setdefault("capsule_defaults", {})
    return cfg


# --------------------------------------------------------------- eligibility
def eligible_object_ids(scene, task):
    """Node-membership gate, applied once, independent of role outcome: an
    object earns a place in the graph only if the policy could plausibly see
    it, the arm could plausibly reach it, or the rubric's explicit
    benchmark_mapping pointed at it by id. Everything else -- same-category
    or not -- never becomes a node or a role hypothesis.

    This is plan/11's negative test in code: "a distant/irrelevant room
    object must be excluded unless it intersects camera view or swept
    workspace." (We use the generic reach envelope rather than the precise
    target-conditioned capsule here, to avoid a chicken-and-egg dependency
    on role resolution; the precise capsule is applied afterwards, within
    this already-eligible set, only to decide the 'obstacle' role.)

    Returns (eligible_ids: set[str], camera_visible: {id: [camera_id,...]}).
    """
    cams = scene.get("cameras") or []
    visible = G.camera_visible_ids(scene["objects"], cams)
    robot = scene.get("robot") or {}
    frame = G.robot_frame(scene)
    base_pos, base_yaw = frame if frame is not None else (None, None)
    reach_m = robot.get("reach_envelope_m", DEFAULT_REACH_M)
    reach_min_m = robot.get("reach_min_m", DEFAULT_REACH_MIN_M)
    cone = robot.get("front_cone_deg", DEFAULT_FRONT_CONE_DEG)
    explicit_refs = set((task.get("rubric") or {}).get("role_refs", {}).values())
    explicit_refs.update(b["object_id"] for b in G.construction_bindings(scene, task).values())

    eligible = set()
    for o in scene["objects"]:
        oid = o["id"]
        reachable = frame is not None and G.in_reach_envelope(
            base_pos, base_yaw, G.aabb_center(o["aabb"]), reach_m, reach_min_m, cone)
        if visible.get(oid) or reachable or oid in explicit_refs:
            eligible.add(oid)
    return eligible, visible


def _available_scene(scene):
    """Keep missing camera/frame declarations explicit without fabricated geometry."""
    import numpy as np
    available = dict(scene)
    missing = []
    if G.robot_frame(scene) is None:
        missing.append("robot_frame")
    cameras = []
    absent_cameras = []
    for camera in scene.get("cameras") or []:
        if not camera.get("id"):
            raise ValueError("camera declaration requires an identity")
        if any(camera.get(key) is None for key in ("pos", "look_at")) or camera.get("fovy_deg", 60.0) is None:
            absent_cameras.append(camera["id"])
            missing.append("camera_frame:" + camera["id"])
            continue
        for key in ("pos", "look_at"):
            value = np.asarray(camera[key], dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError("malformed/nonfinite camera frame")
        if np.linalg.norm(np.asarray(camera["look_at"]) - camera["pos"]) < 1e-9:
            raise ValueError("degenerate camera orientation")
        fovy = float(camera.get("fovy_deg", 60.0))
        distance = float(camera.get("max_range_m", 10.0))
        if not np.isfinite([fovy, distance]).all() or not 0 < fovy <= 180 or distance <= 0:
            raise ValueError("malformed/nonfinite camera projection")
        cameras.append(camera)
    if not cameras or absent_cameras:
        missing.append("policy_cameras")
    available["cameras"] = cameras
    if scene.get("objects") is None:
        missing.append("scene_geometry")
    available["objects"] = scene.get("objects") or []
    return available, missing, absent_cameras


# ------------------------------------------------------------------- graph
def build_graph(scene, task):
    """Ground `task` into `scene` and return one deterministic, JSON-ready
    graph dict. Deterministic under a frozen scene+task manifest: all
    iteration below is over sorted ids, and no wall-clock/random value is
    embedded in the output (plan/11 acceptance criterion)."""
    scene, missing, absent_cameras = _available_scene(scene)
    frame = G.robot_frame(scene)
    eligible, cam_map = eligible_object_ids(scene, task)
    objects_by_id = {o["id"]: o for o in scene["objects"]}
    eligible_objects = [objects_by_id[i] for i in sorted(eligible)]

    lang_res = G.resolve_language_roles(scene, task, eligible)
    edges = G.compute_support_edges(eligible_objects)

    manip_res = lang_res.get("manipulated_object")
    dest_res = lang_res.get("receptacle") or lang_res.get("target")
    anchor_res = manip_res if (manip_res and manip_res["hypotheses"]) else dest_res
    target_point = None
    if anchor_res and anchor_res["hypotheses"]:
        target_point = G.aabb_center(objects_by_id[anchor_res["hypotheses"][0]["object_id"]]["aabb"])

    support_res = G.resolve_support_role(manip_res, edges)
    declared_missing = task.get("unresolved_roles") or {}
    if "support" in declared_missing:
        support_res = G.finalize_hypotheses("support", {})
        support_res["missing_evidence"] = [declared_missing["support"]]
        edges = [edge for edge in edges if edge["kind"] != "support"]

    exclude_ids = set()
    for res in (manip_res, dest_res, support_res):
        if res and res["hypotheses"]:
            exclude_ids.add(res["hypotheses"][0]["object_id"])

    obstacle_res = G.resolve_obstacle_role(
        eligible_objects, frame[0] if frame is not None else None, target_point, exclude_ids,
        capsule_radius_m=task.get("capsule_radius_m", DEFAULT_CAPSULE_RADIUS_M),
        lift_m=task.get("lift_m", DEFAULT_LIFT_M),
    )

    if "obstacle" in declared_missing:
        obstacle_res = G.finalize_hypotheses("obstacle", {}, multi_select=True)
        obstacle_res["missing_evidence"] = [declared_missing["obstacle"]]
        edges = [edge for edge in edges if edge["kind"] != "obstacle"]

    role_resolutions = dict(lang_res)
    role_resolutions["support"] = support_res
    role_resolutions["obstacle"] = obstacle_res
    for role, reason in (task.get("unresolved_roles") or {}).items():
        if role not in G.ROLES:
            raise ValueError("unknown unresolved task role")
        missing.append("role:" + role)

    nodes = _build_nodes(scene, eligible_objects, cam_map, role_resolutions)
    for camera_id in absent_cameras:
        nodes.append({"node_id": "camera:" + camera_id, "kind": "camera",
                      "transform": None, "construction_status": "unresolved"})
    nodes.sort(key=lambda node: node["node_id"])
    edges = sorted(edges, key=lambda e: (e["src"], e["dst"], e["kind"]))

    graph = {
        "task_id": task.get("task_id"),
        "scene_id": scene.get("scene_id"),
        "roles_requested": sorted(set(task.get("roles") or [])),
        "nodes": nodes,
        "edges": edges,
        "role_resolutions": {r: role_resolutions[r] for r in sorted(role_resolutions)},
        "ambiguous_roles": sorted(r for r, res in role_resolutions.items()
                                   if res["status"] != "resolved"),
    }
    if missing:
        graph["missing_evidence"] = sorted(set(missing))
        graph["unresolved_references"] = sorted(set(missing) | {
            role for role in graph["roles_requested"]
            if not role_resolutions.get(role, {}).get("hypotheses")})
    return graph


def _build_nodes(scene, eligible_objects, cam_map, role_resolutions):
    frame = G.robot_frame(scene)
    nodes = [{"node_id": "robot", "kind": "robot", "transform": None,
              "construction_status": "unresolved"}] if frame is None else [{
        "node_id": "robot", "kind": "robot",
        "transform": {"pos": [float(x) for x in frame[0]], "yaw": float(frame[1])},
    }]
    for cam in scene.get("cameras", []):
        nodes.append({
            "node_id": f"camera:{cam['id']}", "kind": "camera",
            "transform": {"pos": [float(x) for x in cam["pos"]],
                          "look_at": [float(x) for x in cam["look_at"]]},
            "fovy_deg": float(cam.get("fovy_deg", 60.0)),
        })

    role_hits = {}
    for role, res in role_resolutions.items():
        for i, h in enumerate(res["hypotheses"]):
            selected = True if res["multi_select"] else (i == 0)
            role_hits.setdefault(h["object_id"], []).append({
                "role": role, "confidence": h["confidence"],
                "selected": bool(selected), "status": res["status"],
            })

    for o in eligible_objects:
        oid = o["id"]
        nodes.append({
            "node_id": oid, "kind": "object", "label": o["label"],
            "aabb": [[float(x) for x in o["aabb"][0]], [float(x) for x in o["aabb"][1]]],
            "transform": o.get("pose", {}),
            "camera_visible": sorted(cam_map.get(oid, [])),
            "roles": sorted(role_hits.get(oid, []), key=lambda r: (r["role"], -r["confidence"])),
        })
    return sorted(nodes, key=lambda n: n["node_id"])


# --------------------------------------------------------------- serialize
def to_json(graph, indent=1):
    return json.dumps(graph, indent=indent, sort_keys=True)


def save_graph(graph, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "task_graph.json"
    json_path.write_text(to_json(graph))
    overlay_path = _write_overlay(graph, out_dir)
    return json_path, overlay_path


# ---------------------------------------------------------------- visualize
_ROLE_COLORS = {"manipulated_object": "tab:orange", "target": "tab:blue",
                 "receptacle": "tab:blue", "support": "tab:green",
                 "obstacle": "tab:red", "tool": "tab:purple"}


def _write_overlay(graph, out_dir):
    """Top-down role overlay. Uses matplotlib when importable (true in this
    repo's .venv); otherwise falls back to a text/JSON overlay description,
    which plan/11 explicitly allows as a substitute -- noted here as a
    trade-off rather than blocking the CLI on a plotting backend."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except Exception:
        return _write_text_overlay(graph, out_dir)

    fig, ax = plt.subplots(figsize=(6, 6))
    for node in graph["nodes"]:
        if node["kind"] in {"robot", "camera"} and node.get("transform") is None:
            continue
        if node["kind"] == "robot":
            pos = node["transform"]["pos"]
            ax.plot(pos[0], pos[1], "ks", markersize=10, zorder=5)
            ax.annotate("robot", (pos[0], pos[1]), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
        elif node["kind"] == "camera":
            pos = node["transform"]["pos"]
            ax.plot(pos[0], pos[1], "k^", markersize=8, zorder=5)
            ax.annotate(node["node_id"], (pos[0], pos[1]), fontsize=6, xytext=(3, 3),
                        textcoords="offset points")
        else:
            lo, hi = node["aabb"][0], node["aabb"][1]
            selected_roles = [r["role"] for r in node.get("roles", []) if r["selected"]]
            color = _ROLE_COLORS.get(selected_roles[0], "lightgray") if selected_roles else "lightgray"
            ax.add_patch(patches.Rectangle(
                (lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
                fill=True, alpha=0.55 if selected_roles else 0.15,
                edgecolor="black", facecolor=color, linewidth=1.0))
            ax.annotate(node["node_id"], (lo[0], hi[1]), fontsize=6, xytext=(2, 2),
                        textcoords="offset points")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"task_graph: {graph.get('task_id')} / {graph.get('scene_id')}")
    handles = [patches.Patch(color=c, label=r) for r, c in _ROLE_COLORS.items()]
    ax.legend(handles=handles, fontsize=6, loc="best")
    path = out_dir / "task_graph_overlay.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_text_overlay(graph, out_dir):
    lines = [f"task_graph overlay (text fallback, no matplotlib backend): "
             f"{graph.get('task_id')} / {graph.get('scene_id')}"]
    for role, res in sorted(graph["role_resolutions"].items()):
        lines.append(f"- {role} [{res['status']}]:")
        for i, h in enumerate(res["hypotheses"]):
            mark = "*" if (i == 0 or res["multi_select"]) else " "
            lines.append(f"    {mark} {h['object_id']} conf={h['confidence']:.2f} "
                         f"evidence={h['evidence']}")
        if not res["hypotheses"]:
            lines.append("    (no eligible candidate)")
    path = out_dir / "task_graph_overlay.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="task manifest YAML")
    ap.add_argument("--scene", required=True,
                     help="scene manifest YAML, or a directory containing scene.yaml")
    ap.add_argument("--out-dir", default=None,
                     help="where to write task_graph.json + overlay "
                          "(default: outputs/task_graph/<scene_id>__<task_id>/)")
    ap.add_argument("--config", default=None,
                     help="optional configs/task_graph/*.yaml overriding "
                          "grounding evidence weights and robot/capsule "
                          "geometry defaults (see configs/task_graph/default_thresholds.yaml)")
    ap.add_argument("--quiet", action="store_true", help="suppress the JSON dump to stdout")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    G.configure(cfg["grounding_weights"])
    scene = load_scene(args.scene, robot_defaults=cfg["robot_defaults"])
    task = load_task(args.task, capsule_defaults=cfg["capsule_defaults"])
    graph = build_graph(scene, task)

    out_dir = args.out_dir or f"outputs/task_graph/{scene.get('scene_id', 'scene')}__{task.get('task_id', 'task')}"
    json_path, overlay_path = save_graph(graph, out_dir)
    if not args.quiet:
        print(to_json(graph))
    print(f"[task_graph] {len(graph['nodes'])} nodes, {len(graph['edges'])} edges, "
          f"ambiguous_roles={graph['ambiguous_roles']}")
    print(f"[task_graph] wrote {json_path}")
    print(f"[task_graph] wrote {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
