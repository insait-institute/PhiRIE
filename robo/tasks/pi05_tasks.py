"""Pick-and-place task suite over SimAny scenes (RoboLab-style).

generate: reads a scene's objects/ tree + mujoco_settle.json, picks the main
tabletop cluster, places the robot at the table edge, and emits
sim_export/pi05_tasks.json with one task per (graspable, receptacle) pair
(or a move-to-region task when the scene has no receptacle).

TaskScorer: staged predicate scoring, 0.25 credit per stage
(grasp -> lift -> hover -> place); success = place held for 1 s.

Usage: pi05_tasks.py --out-dir outputs/<scene>_factory [--max-tasks 10]
"""
import argparse
import json
from pathlib import Path

import numpy as np

GRASP_LABELS = {
    "bottle", "plastic bottle", "mug", "cup", "mouse", "box",
    "cardboard box", "book", "pen", "stapler", "jar", "can", "remote",
    "scissors", "headphone", "headphones", "clock",
}
RECEPTACLE_LABELS = {
    "box", "cardboard box", "bowl", "plate", "tray", "basket",
    "plant pot", "pen holder", "pot",
}
REACH_M = 0.80          # usable Franka reach for tabletop targets
REACH_MIN_M = 0.25      # inside this the arm folds back on itself / self-collides
FRONT_CONE_DEG = 60.0   # admissible bearing off the base heading
BASE_INSET = 0.10       # robot base inset from table edge
BASE_CLEAR = 0.18       # base footprint clearance radius vs objects


def _bearing_off(center_xy, base_xy, yaw):
    """Signed bearing of a point off the base heading, in degrees [-180,180]."""
    v = np.asarray(center_xy[:2]) - np.asarray(base_xy[:2])
    return (np.degrees(np.arctan2(v[1], v[0]) - yaw) + 180.0) % 360.0 - 180.0


def _in_workspace(center_xy, base_xy, yaw):
    """Radius AND frontal cone. A Franka on a desk cannot work behind itself;
    the pre-2026-07-28 test was radius-only, which admitted targets up to 143
    deg off the heading and made them unreachable in practice."""
    d = float(np.linalg.norm(np.asarray(center_xy[:2]) - np.asarray(base_xy[:2])))
    return (REACH_MIN_M <= d <= REACH_M
            and abs(_bearing_off(center_xy, base_xy, yaw)) <= FRONT_CONE_DEG)


# Comfortable annulus: inside this the arm is neither folded back on itself nor
# fully extended (a target at 0.78 m of a 0.80 m reach is technically admissible
# but has almost no manipulability left for the grasp approach).
SWEET_LO, SWEET_HI = 0.35, 0.65


def _workspace_quality(center_xy, base_xy, yaw):
    """0 outside the hard workspace, else 0..1 preferring the comfortable
    annulus and small bearings. Used to SCORE base candidates, so the optimiser
    prefers a pose where targets are actually manipulable, not merely inside
    the radius+cone envelope."""
    if not _in_workspace(center_xy, base_xy, yaw):
        return 0.0
    d = float(np.linalg.norm(np.asarray(center_xy[:2]) - np.asarray(base_xy[:2])))
    if d < SWEET_LO:
        f_d = (d - REACH_MIN_M) / max(SWEET_LO - REACH_MIN_M, 1e-6)
    elif d > SWEET_HI:
        f_d = (REACH_M - d) / max(REACH_M - SWEET_HI, 1e-6)
    else:
        f_d = 1.0
    off = abs(_bearing_off(center_xy, base_xy, yaw))
    f_a = 1.0 - 0.8 * (off / FRONT_CONE_DEG)
    return float(np.clip(f_d, 0.0, 1.0) * f_a)


def _load_objects(out_dir):
    rows = []
    objects = json.loads((out_dir / "objects" / "objects.json").read_text())
    settle = {}
    sj = out_dir / "sim_export" / "mujoco_settle.json"
    if sj.exists():
        settle = json.loads(sj.read_text())["drift_m"]
    for m in objects:
        odir = out_dir / "objects" / f"obj_{m['index']:02d}"
        af = odir / "aligned.json"
        if not af.exists():
            continue
        al = json.loads(af.read_text())
        if al.get("rejected") or not (odir / "object.urdf").exists():
            continue
        ph = json.loads((odir / "physics.json").read_text())
        aabb = np.array(m["aabb"], dtype=float)
        dims = np.array(al["world_dims"], dtype=float)
        name = f"obj_{m['index']:02d}"
        rows.append({
            "name": name, "label": m["label"].lower(),
            "aabb": aabb, "dims": dims, "mass": ph["mass_kg"],
            "tier": al.get("tier", "A") if al.get("schema_version") != 2 else al.get("tier"),
            **({"construction_eligible": al.get("construction_eligible")}
               if al.get("schema_version") == 2 else {}),
            "bottom_z": float(aabb[0][2]),
            "center": aabb.mean(axis=0),
            "drift": float(settle.get(name, 0.0)),
        })
    return rows


def _is_graspable(o):
    return (o["label"] in GRASP_LABELS and (o.get("construction_eligible") is True or o["tier"] in ("A", "B"))
            and o["dims"].max() <= 0.28 and o["dims"].min() >= 0.005
            and o["mass"] <= 1.0 and o["drift"] < 0.10)


def _is_receptacle(o):
    return (o["label"] in RECEPTACLE_LABELS and (o.get("construction_eligible") is True or o["tier"] in ("A", "B"))
            and 0.10 <= o["dims"].max() <= 0.95 and o["dims"].min() >= 0.02
            and o["drift"] < 0.10)


def _cluster(zs, tol=0.06):
    groups = []
    for z in sorted(zs):
        if groups and z - groups[-1][-1] < tol:
            groups[-1].append(z)
        else:
            groups.append([z])
    return [float(np.mean(g)) for g in groups]


def _pick_table(rows, *, planning_predicate=None):
    predicate = planning_predicate or _is_graspable
    grasp = [o for o in rows if predicate(o)]
    best, best_n = None, 0
    for cz in _cluster([o["bottom_z"] for o in rows]):
        if not 0.20 <= cz <= 1.40:
            continue
        n = sum(1 for o in grasp if abs(o["bottom_z"] - cz) < 0.05)
        if n > best_n:
            best, best_n = cz, n
    if best is None:
        raise SystemExit("no tabletop cluster with graspable objects")
    members = [o for o in rows if abs(o["bottom_z"] - best) < 0.05]
    return best, members


def _place_robot(members, *, planning_predicate=None):
    lo = np.min([o["aabb"][0][:2] for o in members], axis=0) - 0.30
    hi = np.max([o["aabb"][1][:2] for o in members], axis=0) + 0.30
    cen = np.mean([o["center"][:2] for o in members], axis=0)
    # candidate base points: grid over the table surface (DROID arms are
    # desk-mounted, so anywhere on the table is fair game); pick the spot
    # that brings the most graspables into reach
    il, ih = lo + BASE_INSET, hi - BASE_INSET
    xs = np.arange(il[0], ih[0] + 1e-6, 0.15)
    ys = np.arange(il[1], ih[1] + 1e-6, 0.15)
    cands = [np.array([x, y]) for x in xs for y in ys]

    def clear(p):
        for o in members:
            l, h = o["aabb"][0][:2] - 0.03, o["aabb"][1][:2] + 0.03
            near = np.maximum(l, np.minimum(p, h))
            if np.linalg.norm(near - p) < BASE_CLEAR:
                return False
        return True

    predicate = planning_predicate or _is_graspable
    graspables = [o for o in members if predicate(o)]
    gcen = (np.mean([o["center"][:2] for o in graspables], axis=0)
            if graspables else cen)

    def best_yaw(p):
        """Pick the heading that puts the most graspables in the frontal cone.
        Candidate headings: straight at each graspable, plus at their centroid.
        Optimising yaw per base (instead of always aiming at the centroid of
        ALL members) is what stops targets landing behind the arm."""
        opts = [np.arctan2(o["center"][1] - p[1], o["center"][0] - p[0])
                for o in graspables]
        opts.append(np.arctan2(gcen[1] - p[1], gcen[0] - p[0]))
        scored = []
        for y in opts:
            q = sum(_workspace_quality(o["center"], p, y) for o in graspables)
            n = sum(1 for o in graspables if _in_workspace(o["center"], p, y))
            scored.append((q, n, float(y)))
        scored.sort(reverse=True)
        return scored[0][2], scored[0][0]

    # Score every candidate base by how many graspables it can actually work
    # on (radius AND cone). This also naturally seats the base at the EDGE of
    # the clutter: from the middle, a +-60 deg cone sees fewer objects.
    scored_cands = []
    for p in cands:
        y, q = best_yaw(p)
        scored_cands.append((-q, float(np.linalg.norm(p - gcen)), p, y))
    scored_cands.sort(key=lambda t: (t[0], t[1]))
    pick = next((t for t in scored_cands if clear(t[2])), scored_cands[0])
    base_xy, yaw = pick[2], pick[3]
    return {
        "table": {"cx": float((lo[0] + hi[0]) / 2), "cy": float((lo[1] + hi[1]) / 2),
                  "hx": float((hi[0] - lo[0]) / 2), "hy": float((hi[1] - lo[1]) / 2),
                  "top_z": float(min(o["bottom_z"] for o in members) - 0.02)},
        "base_pos": [float(base_xy[0]), float(base_xy[1]),
                     float(min(o["bottom_z"] for o in members) - 0.02)],
        "base_yaw": yaw,
    }


def _qualifier(target, same, base_pos, base_yaw):
    """Disambiguate duplicates by position in the robot base frame."""
    if len(same) <= 1:
        return ""
    c, s = np.cos(base_yaw), np.sin(base_yaw)
    def bf(o):  # (forward, left)
        d = o["center"][:2] - np.array(base_pos[:2])
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    t = bf(target)
    others = [bf(o) for o in same if o["name"] != target["name"]]
    keys = {"leftmost": lambda v: v[1], "rightmost": lambda v: -v[1],
            "nearest": lambda v: -v[0], "farthest": lambda v: v[0]}
    for tag, key in keys.items():
        if key(t) > max(key(o) for o in others) + 0.03:
            return tag
    return ""


def ext_cam_from_frame(frame):
    """Build an ext_cam dict for a NAMED scan frame.

    Split out of _pick_scan_camera so a demo render can pin a viewpoint that
    the scoring heuristic would never choose: that score only rewards
    workspace alignment/distance/height and is blind to what else fills the
    frame, so on c50d2d1d42 it picks DSC01593, where the office chair (baked
    into clean_background.ply as furniture, hence unremovable) eats ~25% of
    the image. Returns None if the frame is not in the scan trajectory.
    """
    from agents.core import common as C
    from robo.rigs import pi05_rig as rig
    fn = frame if frame.endswith(".JPG") else frame + ".JPG"
    w2c_all = C.load_colmap_w2c()
    if fn not in w2c_all:
        return None
    K, _W, H, _ = C.load_intrinsics()
    w2c = np.array(w2c_all[fn])
    R, t = w2c[:3, :3], w2c[:3, 3]
    pos = -R.T @ t
    c2w = R.T @ np.diag([1.0, -1.0, -1.0])
    return {"mode": "world", "pos": pos.tolist(),
            "quat_wxyz": rig.rot_to_quat_wxyz(c2w).tolist(),
            "fovy": float(np.degrees(2 * np.arctan(H / (2 * K[1, 1])))),
            "frame": fn}


def _pick_scan_camera(out_dir, target, top_z):
    """Pick a real scan-trajectory camera pose as the external DROID camera:
    at scan viewpoints the splat render is photoreal (28+ dB), while novel
    far-out viewpoints degrade into blurry gaussians. Returns a world-pose
    ext_cam dict, or None if the scan poses are unavailable."""
    import os
    os.environ.setdefault("SIMANY_SCENE",
                          Path(out_dir).name.replace("_factory", ""))
    os.environ.setdefault("SIMANY_OUT", str(out_dir))
    try:
        from agents.core import common as C
        w2c_all = C.load_colmap_w2c()
        K, W, H, _ = C.load_intrinsics()
    except Exception as e:
        print(f"[tasks] scan-camera pick unavailable ({e}); "
              "falling back to synthetic camera")
        return None
    best, best_score = None, -1e9
    for name, w2c in sorted(w2c_all.items()):
        R, t = np.array(w2c)[:3, :3], np.array(w2c)[:3, 3]
        pos = -R.T @ t
        fwd = R.T @ np.array([0.0, 0.0, 1.0])
        to_t = target - pos
        d = float(np.linalg.norm(to_t))
        if not 0.5 <= d <= 1.8:
            continue
        align = float(fwd @ (to_t / d))          # look toward workspace
        if align < 0.85:
            continue
        h = pos[2] - top_z                        # DROID cams sit above table
        if not 0.15 <= h <= 1.0:
            continue
        score = align * 2 - abs(d - 1.0) - abs(h - 0.55)
        if score > best_score:
            best_score, best = score, (name, pos, R)
    if best is None:
        return None
    name, pos, R = best
    # OpenCV cam (+z fwd, +y down) -> MuJoCo cam (-z fwd, +y up):
    # c2w_gl = c2w_cv @ diag(1,-1,-1)
    c2w = R.T @ np.diag([1.0, -1.0, -1.0])
    fovy = float(np.degrees(2 * np.arctan(H / (2 * K[1, 1]))))
    from robo.rigs import pi05_rig as rig
    print(f"[tasks] ext cam = scan frame {name} (d={np.linalg.norm(target-pos):.2f} m)")
    return {"mode": "world", "pos": pos.tolist(),
            "quat_wxyz": rig.rot_to_quat_wxyz(c2w).tolist(),
            "fovy": fovy, "frame": name}


def generate(out_dir, max_tasks=10, cam_side=1.0):
    out_dir = Path(out_dir)
    rows = _load_objects(out_dir)
    top_z, members = _pick_table(rows)
    placement = _place_robot(members)
    base_pos, base_yaw = placement["base_pos"], placement["base_yaw"]

    grasp = [o for o in members if _is_graspable(o)]
    recep = [o for o in members if _is_receptacle(o)]
    exclude = [o["name"] for o in rows if o["drift"] > 0.25]

    def reachable(o):
        return _in_workspace(o["center"], base_pos[:2], base_yaw)

    tasks = []
    for g in sorted(grasp, key=lambda o: -float(o.get("tier") == "A")):
        if not reachable(g):
            continue
        same = [o for o in grasp if o["label"] == g["label"]]
        qual = _qualifier(g, same, base_pos, base_yaw)
        gname = f"the {qual} {g['label']}" if qual else f"the {g['label']}"
        if len(same) > 1 and not qual:
            gname = f"the {g['label']}"  # ambiguous; scorer accepts any twin
        cands = [r for r in recep if r["name"] != g["name"] and reachable(r)
                 and np.linalg.norm(r["center"][:2] - g["center"][:2]) < 0.70
                 and np.linalg.norm(r["center"][:2] - g["center"][:2]) > 0.10]
        cands.sort(key=lambda r: np.linalg.norm(
            r["center"][:2] - g["center"][:2]))
        if cands:
            r = cands[0]
            tasks.append({
                "task_id": f"{out_dir.name}__{g['name']}_into_{r['name']}",
                "target": g["name"], "target_label": g["label"],
                "any_instance": len(same) > 1 and not qual,
                "receptacle": r["name"],
                "receptacle_dims": r["dims"].tolist(),
                "instructions": {
                    "default": f"put {gname} in the {r['label']}",
                    "vague": f"put {gname} away",
                    "specific": f"pick up {gname} and place it inside "
                                f"the {r['label']}",
                },
            })
        else:
            # move-to-region task: opposite half of the table
            t = placement["table"]
            side = -1.0 if g["center"][1] > t["cy"] else 1.0
            word = "left" if side * np.cos(base_yaw) >= 0 else "right"
            tasks.append({
                "task_id": f"{out_dir.name}__{g['name']}_to_region",
                "target": g["name"], "target_label": g["label"],
                "any_instance": len(same) > 1 and not qual,
                "receptacle": None,
                "region": {"cx": t["cx"], "cy": t["cy"] + side * t["hy"] / 2,
                           "hx": min(t["hx"], 0.30), "hy": t["hy"] / 3,
                           "zlo": t["top_z"] - 0.02, "zhi": t["top_z"] + 0.30},
                "instructions": {
                    "default": f"move {gname} to the {word} side of the table",
                    "vague": f"move {gname}",
                    "specific": f"pick up {gname} and set it down on the "
                                f"{word} side of the table",
                },
            })
        if len(tasks) >= max_tasks:
            break

    aim = [o for o in grasp if reachable(o)] or grasp
    cam_target = [float(np.mean([o["center"][0] for o in aim])),
                  float(np.mean([o["center"][1] for o in aim])),
                  placement["table"]["top_z"] + 0.12]
    scan_cam = _pick_scan_camera(out_dir, np.array(cam_target),
                                 placement["table"]["top_z"])
    suite = {
        "scene": out_dir.name,
        "scene_xml": str(out_dir / "sim_export" / "scene.xml"),
        "robot": {"base_pos": base_pos, "base_yaw": base_yaw},
        "table": placement["table"],
        "ext_cam": scan_cam or {"pos": [0.05, 0.70 * cam_side, 0.75],
                                "target": cam_target, "fovy": 68.0},
        "exclude_objects": exclude,
        "time_limit_s": 16.0,
        "tasks": tasks,
    }
    dst = out_dir / "sim_export" / "pi05_tasks.json"
    dst.write_text(json.dumps(suite, indent=1))
    print(f"[tasks] {dst}: {len(tasks)} tasks, table top "
          f"{placement['table']['top_z']:.3f} m, base {base_pos}, "
          f"yaw {np.degrees(base_yaw):.0f} deg, exclude {exclude}")
    return suite


# --------------------------------------------------------------------- score
class TaskScorer:
    """Stages: grasp -> lift -> hover -> place. Success = place held 1 s."""

    def __init__(self, env, task, hold_ticks=15):
        self.env, self.task = env, task
        self.hold_ticks = hold_ticks
        self.targets = [task["target"]]
        if task.get("any_instance"):
            # accept any same-label twin that exists in the sim
            label = task["target_label"]
            suite_objs = getattr(env, "_task_rows", None)
            if suite_objs:
                self.targets = [o["name"] for o in suite_objs
                                if o["label"] == label and
                                o["name"] in env.free_bodies]
        self.stages = {t: {"grasp": False, "lift": False, "hover": False,
                           "place": False} for t in self.targets}
        self._held = {t: 0 for t in self.targets}
        self.region = self._resolve_region(task)
        self.wrong_grasps = set()

    def _resolve_region(self, task):
        if task.get("receptacle"):
            pos, _ = self.env.body_pose(task["receptacle"])
            d = np.asarray(task["receptacle_dims"])
            return {"cx": float(pos[0]), "cy": float(pos[1]),
                    "hx": float(max(d[0] / 2, 0.07)),
                    "hy": float(max(d[1] / 2, 0.07)),
                    "zlo": float(pos[2] - d[2] / 2 - 0.02),
                    "zhi": float(pos[2] + d[2] / 2 + 0.25)}
        return dict(task["region"])

    def update(self):
        for t in self.targets:
            st = self.stages[t]
            if self.env.grasped(t):
                st["grasp"] = True
                if self.env.lifted(t):
                    st["lift"] = True
            p = self.env.data.xpos[self.env.model.body(t).id]
            if st["lift"] and (abs(p[0] - self.region["cx"]) <= self.region["hx"] + 0.05
                               and abs(p[1] - self.region["cy"]) <= self.region["hy"] + 0.05):
                st["hover"] = True
            in_r = (abs(p[0] - self.region["cx"]) <= self.region["hx"] and
                    abs(p[1] - self.region["cy"]) <= self.region["hy"] and
                    self.region["zlo"] <= p[2] <= self.region["zhi"])
            settled = (st["hover"] and in_r and not self.env.grasped(t)
                       and self.env.at_rest(t))
            self._held[t] = self._held[t] + 1 if settled else 0
            if self._held[t] >= self.hold_ticks:
                st["place"] = True
        # diagnostics: grasping a non-target object
        for b in self.env.free_bodies:
            if b not in self.targets and self.env.grasped(b):
                self.wrong_grasps.add(b)

    @property
    def success(self):
        return any(all(st.values()) for st in self.stages.values())

    def score(self):
        best = max(self.stages.values(),
                   key=lambda st: sum(st.values()))
        return sum(best.values()) / 4.0

    def summary(self):
        best_t = max(self.stages, key=lambda t: sum(self.stages[t].values()))
        return {"success": bool(self.success), "score": self.score(),
                "stages": self.stages[best_t], "scored_target": best_t,
                "wrong_grasps": sorted(self.wrong_grasps)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-tasks", type=int, default=10)
    ap.add_argument("--cam-side", type=float, default=1.0,
                    help="+1 external cam on robot's left, -1 right")
    args = ap.parse_args()
    generate(args.out_dir, args.max_tasks, args.cam_side)
