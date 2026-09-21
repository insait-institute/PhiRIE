"""Role grounding for the task-conditioned interaction graph (Task 11).

Resolves each task role (manipulated_object, target, receptacle, tool,
support, obstacle) to zero, one, or several ranked scene-instance
hypotheses, each carrying an explicit evidence trail and a confidence.

Evidence sources, from strongest to weakest:
  1. benchmark_mapping  -- an explicit rubric-provided {role: object_id}
                           lookup table. This is a human/benchmark
                           annotation, not a policy-evaluation result, so
                           it is permitted (plan/11 step 2: "permit explicit
                           benchmark mappings ... but no evaluation-result
                           leakage").
  2. rubric_label_match -- the rubric names a label for the role.
  3. language_match     -- a very small pattern-matcher over the task's
                           natural-language instruction (not a real parser;
                           robustness comes from combining it with the
                           other evidence sources, not from language alone).
  4. category_prior     -- the object's label is in a generic role/label
                           prior table (e.g. "mug" is graspable-shaped).
  5. geometry bonus      -- a tie-break among same-scoring candidates using
                           either an explicit rubric spatial hint
                           (leftmost/rightmost/nearest_to_base/...) or
                           proximity to an already-resolved anchor role.

Nothing in this module reads a success/score/reward field. There is no such
field in its inputs (scene manifest, task manifest) by construction --
callers must not pass rollout output into `scene` or `task`.
"""
import math
import re

import numpy as np

def robot_frame(scene):
    """Return an observed/declared frame, or None; never invent a base pose."""
    robot = scene.get("robot") or {}
    base = robot.get("base_pos")
    yaw = robot.get("base_yaw", 0.0)  # Preserve the existing declared convention.
    if base is None or yaw is None:
        return None
    p = np.asarray(base, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or not math.isfinite(float(yaw)):
        raise ValueError("malformed/nonfinite robot frame")
    return base, yaw


# --------------------------------------------------------------------- roles
ROLES = ("manipulated_object", "target", "receptacle", "support", "tool", "obstacle")
# Grounded via language/rubric/label evidence (grounding pipeline below).
LANGUAGE_ROLES = ("manipulated_object", "target", "receptacle", "tool")
# Derived purely from geometry once at least one LANGUAGE_ROLE is resolved.
DERIVED_ROLES = ("support", "obstacle")
# Roles where *every* hypothesis clearing the floor is jointly selected
# (there is no single "the" support object or "the" obstacle -- a task can
# have several of each), vs. single-select roles where hypotheses compete
# for one slot and uncertainty is expressed via `status`/ranking instead.
MULTI_SELECT_ROLES = ("support", "obstacle")

STOPWORDS = {"the", "a", "an", "it", "its", "to", "in", "into", "on", "onto",
             "of", "and", "then", "please", "up", "down"}

# Generic label priors per role -- deliberately small and reused across
# fixtures/configs; a real deployment would load these per-benchmark rather
# than hardcoding, but a small built-in prior keeps grounding usable even
# when a rubric under-specifies a role.
ROLE_LABEL_PRIORS = {
    "manipulated_object": {"mug", "cup", "bottle", "bowl", "box", "book",
                            "plate", "can", "jar", "remote", "stapler"},
    "target": {"tray", "bowl", "box", "basket", "plate", "bin", "pot"},
    "receptacle": {"tray", "bowl", "box", "basket", "plate", "bin", "pot"},
    "tool": {"tongs", "scoop", "spatula", "brush"},
}

# ---- evidence tags (also used verbatim as strings in serialized output) ----
EV_BENCHMARK_MAPPING = "benchmark_mapping"
EV_RUBRIC_LABEL = "rubric_label_match"
EV_LANGUAGE = "language_match"
EV_CATEGORY_PRIOR = "category_prior"
EV_GEOMETRY_NEAREST = "geometry_nearest_anchor"
EV_SPATIAL_QUALIFIER = "spatial_qualifier"
EV_CONTACT_SUPPORT = "contact_support"
EV_SWEPT_WORKSPACE = "swept_workspace_intersection"

# ---- confidence weights. Deliberately conservative: language alone should
# never outweigh rubric label + any geometric corroboration, and a lone
# category-prior hit should never look "resolved" on its own. ----
W_BENCHMARK_MAPPING = 0.97
W_RUBRIC_LABEL = 0.55
W_LANGUAGE = 0.42
W_CATEGORY_PRIOR = 0.15
W_GEOMETRY_BONUS = 0.30
W_SUPPORT_EDGE = 0.90
W_OBSTACLE_BASE = 0.55
W_OBSTACLE_MARGIN = 0.35

# A role is "ambiguous" (never an arbitrary single pick) when the top
# hypothesis isn't decisively ahead of the runner-up, or is confident-looking
# only because it is the sole candidate despite thin evidence.
AMBIGUITY_MARGIN = 0.12
CONFIDENT_FLOOR = 0.35

# Names configure() is allowed to override from a configs/task_graph/*.yaml
# file. Deliberately an allowlist (not "any module attribute") so a typo'd
# config key fails loudly instead of silently doing nothing.
_CONFIGURABLE = ("AMBIGUITY_MARGIN", "CONFIDENT_FLOOR", "W_BENCHMARK_MAPPING",
                 "W_RUBRIC_LABEL", "W_LANGUAGE", "W_CATEGORY_PRIOR",
                 "W_GEOMETRY_BONUS", "W_SUPPORT_EDGE", "W_OBSTACLE_BASE",
                 "W_OBSTACLE_MARGIN")


def configure(overrides):
    """Override selected evidence weights / ambiguity thresholds from a
    loaded configs/task_graph/*.yaml file (see task_graph.main --config).
    Every resolve_*/finalize_hypotheses call reads these names as module
    globals, so a configure() call before build_graph() affects every
    subsequent resolution in the process -- callers that need per-call
    isolation (e.g. tests exercising two different weight sets) should
    call configure() with the previous snapshot afterwards to restore it.
    """
    if not overrides:
        return {}
    unknown = set(overrides) - set(_CONFIGURABLE)
    if unknown:
        raise ValueError(f"unknown task_graph config keys: {sorted(unknown)}")
    previous = {k: globals()[k] for k in overrides}
    globals().update(overrides)
    return previous


# ------------------------------------------------------------------ geometry
def aabb_center(aabb):
    lo, hi = aabb[0], aabb[1]
    return [(float(lo[i]) + float(hi[i])) / 2.0 for i in range(3)]


def aabb_dims(aabb):
    lo, hi = aabb[0], aabb[1]
    return [float(hi[i]) - float(lo[i]) for i in range(3)]


def aabb_bounding_radius(aabb):
    """Radius of the sphere circumscribing the aabb -- the conservative,
    approximate object shape used everywhere in this module (support
    footprint overlap is the only place true aabb extents are used)."""
    d = aabb_dims(aabb)
    return 0.5 * math.sqrt(sum(x * x for x in d))


def _bearing_off(center_xy, base_xy, yaw):
    v = np.asarray(center_xy[:2], dtype=float) - np.asarray(base_xy[:2], dtype=float)
    return (math.degrees(math.atan2(v[1], v[0]) - yaw) + 180.0) % 360.0 - 180.0


def in_reach_envelope(base_pos, base_yaw, point, reach_m, reach_min_m, front_cone_deg):
    """Generic, target-independent reach test (radius + frontal cone), the
    same shape as robo/tasks/pi05_tasks.py's `_in_workspace` but exposed here
    as the *eligibility* gate: a wide, permissive envelope deciding whether
    an object is a graph node candidate at all, not the tighter per-task
    manipulability check the task suite makes downstream."""
    bp = np.asarray(base_pos[:2], dtype=float)
    p = np.asarray(point[:2], dtype=float)
    d = float(np.linalg.norm(p - bp))
    if not (reach_min_m <= d <= reach_m):
        return False
    return abs(_bearing_off(point, base_pos, base_yaw)) <= front_cone_deg


def _bearing_and_distance(cam, point):
    pos = np.asarray(cam["pos"], dtype=float)
    look = np.asarray(cam["look_at"], dtype=float)
    fwd = look - pos
    n = float(np.linalg.norm(fwd))
    if n < 1e-9:
        return None
    fwd_unit = fwd / n
    to_p = np.asarray(point, dtype=float) - pos
    d = float(np.linalg.norm(to_p))
    if d < 1e-9:
        return 0.0, 0.0
    cos_ang = float(np.clip(np.dot(fwd_unit, to_p / d), -1.0, 1.0))
    return math.degrees(math.acos(cos_ang)), d


def camera_visible_ids(objects, cameras):
    """Approximate line-of-sight: is the object's aabb center within the
    camera's fovy/2 cone and max_range_m? This is deliberately coarse (a
    single fovy used as a round cone rather than separate fovx/fovy, and no
    occlusion reasoning at all -- occlusion/ghosting belongs to the render
    audit, Task 05). Good enough for the graph-membership gate; certificate
    features (Task 12) that need real visibility should re-render.

    Returns {object_id: [camera_id, ...]} (empty list, never a missing key).
    """
    visible = {}
    for o in objects:
        center = aabb_center(o["aabb"])
        seeing = []
        for cam in cameras:
            res = _bearing_and_distance(cam, center)
            if res is None:
                continue
            ang, dist = res
            if ang <= cam.get("fovy_deg", 60.0) / 2.0 and dist <= cam.get("max_range_m", 3.0):
                seeing.append(cam["id"])
        visible[o["id"]] = seeing
    return visible


def point_segment_distance(p, a, b):
    p, a, b = np.asarray(p, dtype=float), np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ab = b - a
    l2 = float(np.dot(ab, ab))
    t = 0.0 if l2 < 1e-12 else float(np.clip(np.dot(p - a, ab) / l2, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def swept_workspace_segment(base_pos, target_point, lift_m):
    """Approximate the pick-and-place sweep as a single 3D line segment from
    (roughly) the base, raised a little toward the shoulder, to a point
    `lift_m` above the target -- the straight-line stand-in for the lift
    arc the plan asks to approximate as a capsule/box (plan/11 step 3)."""
    a = np.array([base_pos[0], base_pos[1], base_pos[2] + 0.5 * lift_m], dtype=float)
    b = np.array([target_point[0], target_point[1], target_point[2] + lift_m], dtype=float)
    return a, b


def capsule_margin(aabb, seg_a, seg_b, radius):
    """Positive => the object (approximated as its circumscribing sphere)
    intersects the capsule; the value is the penetration depth, used only to
    rank obstacle confidence, not as a physical clearance number."""
    center = np.array(aabb_center(aabb))
    r_obj = aabb_bounding_radius(aabb)
    dist = point_segment_distance(center, seg_a, seg_b)
    return (radius + r_obj) - dist


def swept_workspace_hits(objects, base_pos, target_point, radius, lift_m, exclude_ids=()):
    a, b = swept_workspace_segment(base_pos, target_point, lift_m)
    hits = []
    for o in objects:
        if o["id"] in exclude_ids:
            continue
        margin = capsule_margin(o["aabb"], a, b, radius)
        if margin > 0:
            hits.append((o["id"], margin))
    return hits


def compute_support_edges(objects, z_tol=0.015, min_xy_overlap_frac=0.15):
    """object A `rests_on` object B when A's aabb bottom sits within z_tol of
    B's aabb top and their xy footprints overlap by at least
    min_xy_overlap_frac of A's own footprint area. Full O(n^2) scan over the
    *eligible* object set only -- graphs are task-local by construction so n
    stays small; this is not meant to scale to a whole-room object count."""
    edges = []
    for a in objects:
        a_bot = a["aabb"][0][2]
        a_lo, a_hi = a["aabb"][0][:2], a["aabb"][1][:2]
        a_area = max((a_hi[0] - a_lo[0]) * (a_hi[1] - a_lo[1]), 1e-9)
        best = None
        for b in objects:
            if a["id"] == b["id"]:
                continue
            b_top = b["aabb"][1][2]
            if abs(a_bot - b_top) > z_tol:
                continue
            b_lo, b_hi = b["aabb"][0][:2], b["aabb"][1][:2]
            ox = max(0.0, min(a_hi[0], b_hi[0]) - max(a_lo[0], b_lo[0]))
            oy = max(0.0, min(a_hi[1], b_hi[1]) - max(a_lo[1], b_lo[1]))
            frac = (ox * oy) / a_area
            if frac >= min_xy_overlap_frac and (best is None or frac > best[1]):
                best = (b["id"], frac)
        if best is not None:
            edges.append({"src": a["id"], "dst": best[0], "kind": "support",
                          "relation": "rests_on", "confidence": float(min(1.0, best[1])),
                          "evidence": [EV_CONTACT_SUPPORT]})
    return edges


# ------------------------------------------------------------------ language
_MANIP_PAT = re.compile(
    r"(?:pick up|grasp|take|lift|move|grab)\s+(?:the\s+)?([a-z][a-z\s]*?)"
    r"(?=\s+(?:and|into|in|onto|on|to|off|from)\b|[.,]|$)"
)
_DEST_PAT = re.compile(
    r"\b(?:into|in|onto|on|to)\b\s+(?:the\s+)?([a-z][a-z\s]*?)(?=[.,]|$)"
)


def parse_instruction(instruction):
    """A deliberately small pattern-matcher over "pick up/move/grab the X
    (and put it) in/on/into the Y" style instructions -- not a real parser.
    Returns {"manipulated_object": phrase|None, "receptacle": phrase|None}.
    Callers should treat a match as one more piece of evidence, never as
    ground truth on its own (see module docstring)."""
    text = (instruction or "").strip().lower()
    manip = _MANIP_PAT.search(text)
    dest = _DEST_PAT.search(text)
    return {
        "manipulated_object": manip.group(1).strip() if manip else None,
        "receptacle": dest.group(1).strip() if dest else None,
    }


def _label_tokens(text):
    return {w for w in re.findall(r"[a-z]+", (text or "").lower()) if w not in STOPWORDS}


def label_match_score(object_label, wanted):
    """Token-overlap score in [0, 1]: 1.0 for an exact label match, a partial
    score for shared tokens (so a multi-word rubric/label like "cardboard
    box" still matches a scene label of plain "box"), 0.0 when disjoint."""
    if not wanted:
        return 0.0
    ol, wl = (object_label or "").strip().lower(), wanted.strip().lower()
    if ol == wl:
        return 1.0
    ot, wt = _label_tokens(ol), _label_tokens(wl)
    if not ot or not wt:
        return 0.0
    inter = ot & wt
    return len(inter) / len(wt) if inter else 0.0


# -------------------------------------------------------------- resolution
def finalize_hypotheses(role, hyps, multi_select=False):
    """Turn a {object_id: {"confidence": raw, "evidence": [...]}} accumulator
    into a serializable, ranked RoleResolution dict.

    `confidence` is an unbounded, additive evidence score (sum of the
    W_* weights of every corroborating source), not a calibrated
    probability -- deliberately so: clipping it to [0, 1] would make a
    benchmark_mapping+geometry-corroborated pick (e.g. 1.4) display
    identically to a label-only distractor (e.g. 1.1) even though the
    `status`/ranking logic below (which uses these same raw sums) is quite
    decisive about which one actually won. Leaving it unclipped keeps the
    serialized JSON itself informative, not just the status flag.
    """
    items = sorted(hyps.items(), key=lambda kv: (-kv[1]["confidence"], kv[0]))
    out_hyps = [{
        "object_id": oid,
        "confidence": round(h["confidence"], 4),
        "evidence": sorted(set(h["evidence"])),
    } for oid, h in items]

    if not out_hyps:
        status = "unresolved"
    elif multi_select:
        # Every hypothesis already cleared the margin>0 selection test
        # (e.g. swept_workspace_hits only returns positive-margin hits);
        # there is no single winner to be decisive about.
        status = "resolved"
    else:
        top = items[0][1]["confidence"]
        second = items[1][1]["confidence"] if len(items) > 1 else float("-inf")
        if top < CONFIDENT_FLOOR or (len(items) > 1 and (top - second) < AMBIGUITY_MARGIN):
            status = "ambiguous"
        else:
            status = "resolved"

    return {"role": role, "status": status, "multi_select": bool(multi_select),
            "hypotheses": out_hyps}


def _apply_geometry_bonus(hyps, objects, anchor):
    dists = {oid: float(np.linalg.norm(np.array(aabb_center(objects[oid]["aabb"])) - np.array(anchor)))
             for oid in hyps}
    dmin, dmax = min(dists.values()), max(dists.values())
    span = max(dmax - dmin, 1e-6)
    for oid, h in hyps.items():
        closeness = 1.0 - (dists[oid] - dmin) / span  # 1.0 nearest, 0.0 farthest among candidates
        bonus = W_GEOMETRY_BONUS * closeness
        if bonus > 1e-9:  # don't claim geometry evidence for a exactly-zero contribution
            h["confidence"] += bonus
            h["evidence"].append(EV_GEOMETRY_NEAREST)


_SPATIAL_KEYS = {
    "nearest_to_base": lambda v: -math.hypot(v[0], v[1]),
    "farthest_from_base": lambda v: math.hypot(v[0], v[1]),
    "leftmost": lambda v: v[1],
    "rightmost": lambda v: -v[1],
}


def _apply_spatial_hint(hyps, objects, hint, base_pos, base_yaw):
    key = _SPATIAL_KEYS.get(hint)
    if key is None:
        return False
    c, s = math.cos(base_yaw), math.sin(base_yaw)

    def base_frame(oid):
        d = np.array(aabb_center(objects[oid]["aabb"])[:2]) - np.array(base_pos[:2])
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])  # (forward, left)

    scored = {oid: key(base_frame(oid)) for oid in hyps}
    lo, hi = min(scored.values()), max(scored.values())
    span = max(hi - lo, 1e-6)
    for oid, h in hyps.items():
        closeness = (scored[oid] - lo) / span
        bonus = W_GEOMETRY_BONUS * closeness
        if bonus > 1e-9:
            h["confidence"] += bonus
            h["evidence"].append(f"{EV_SPATIAL_QUALIFIER}:{hint}")
    return True


def construction_bindings(scene, task):
    """Typed public selections; source authentication belongs to the feature bridge."""
    bindings = task.get("construction_role_bindings") or {}
    objects = {o["id"] for o in scene.get("objects") or []}
    if not isinstance(bindings, dict) or set(bindings) - set(LANGUAGE_ROLES):
        raise ValueError("unsupported public construction role")
    if bindings and (task.get("rubric") or {}).get("role_refs"):
        raise ValueError("public construction cannot use benchmark role aliases")
    for role, binding in bindings.items():
        if (role not in (task.get("roles") or []) or role in (task.get("unresolved_roles") or {})
            or set(binding) != {"object_id", "query_sha256", "source_gate", "overlap_iou"}
            or binding["object_id"] not in objects
            or binding["query_sha256"] != task.get("query_sha256")
            or len(binding["query_sha256"]) != 64
            or set(binding["source_gate"]) != {"path", "sha256", "size_bytes"}
            or len(binding["source_gate"]["sha256"]) != 64
            or not math.isfinite(float(binding["overlap_iou"]))
            or not 0 <= float(binding["overlap_iou"]) <= 1):
            raise ValueError("malformed or contradictory public construction binding")
    return bindings


def resolve_language_roles(scene, task, eligible_ids):
    """Resolve every LANGUAGE_ROLES entry in task["roles"] to ranked
    hypotheses, restricted to `eligible_ids` (the graph-membership gate
    computed once by task_graph.eligible_object_ids -- this function never
    widens that set, it only ranks within it).

    Roles are resolved in a fixed priority order (destination roles before
    manipulated_object before tool) so that, absent an explicit spatial
    hint, "which same-label object is the real target" can fall back on
    proximity to an already-resolved anchor role (e.g. the mug nearest the
    already-resolved receptacle), which is exactly the distractor-rejection
    path plan/11's negative test exercises.
    """
    bindings = construction_bindings(scene, task)
    objects = {o["id"]: o for o in scene["objects"] if o["id"] in eligible_ids}
    rubric = task.get("rubric") or {}
    role_refs = rubric.get("role_refs") or {}
    role_labels = rubric.get("role_labels") or {}
    spatial_hints = rubric.get("spatial_hints") or {}
    parsed = parse_instruction((task.get("language") or {}).get("instruction"))
    frame = robot_frame(scene)
    base_pos, base_yaw = frame if frame is not None else (None, None)

    requested = [r for r in task.get("roles") or [] if r in LANGUAGE_ROLES]
    order = {"receptacle": 0, "target": 0, "manipulated_object": 1, "tool": 2}
    requested = sorted(requested, key=lambda r: order.get(r, 1))

    missing = task.get("unresolved_roles") or {}
    if not isinstance(missing, dict) or set(missing) - set(task.get("roles") or []):
        raise ValueError("unresolved roles must reference requested roles")
    if any(not isinstance(reason, str) or not reason.strip() for reason in missing.values()):
        raise ValueError("unresolved role requires a concrete reason")
    resolutions = {}
    anchor_points = {}
    for role in requested:
        if role in missing:
            resolution = finalize_hypotheses(role, {})
            resolution["missing_evidence"] = [missing[role]]
            resolutions[role] = resolution
            continue
        if role in bindings:
            binding = bindings[role]
            if binding["object_id"] not in objects:
                raise ValueError("public selected object absent from graph eligibility")
            resolutions[role] = {"role": role, "status": "resolved", "multi_select": False,
                "hypotheses": [{"object_id": binding["object_id"],
                    "confidence": float(binding["overlap_iou"]),
                    "evidence": ["authenticated_public_region_selection"]}],
                "confidence_kind": "observed_public_region_overlap_iou",
                "construction_binding": binding}
            continue
        hyps = {}

        def add(oid, w, tag):
            h = hyps.setdefault(oid, {"confidence": 0.0, "evidence": []})
            h["confidence"] += w
            h["evidence"].append(tag)

        ref = role_refs.get(role)
        if ref is not None and ref in objects:
            add(ref, W_BENCHMARK_MAPPING, EV_BENCHMARK_MAPPING)

        wanted_label = role_labels.get(role)
        phrase = parsed.get(role)
        priors = ROLE_LABEL_PRIORS.get(role, ())
        for oid, o in objects.items():
            if wanted_label:
                s = label_match_score(o["label"], wanted_label)
                if s > 0:
                    add(oid, W_RUBRIC_LABEL * s, EV_RUBRIC_LABEL)
            if phrase:
                s = label_match_score(o["label"], phrase)
                if s > 0:
                    add(oid, W_LANGUAGE * s, EV_LANGUAGE)
            if o["label"] in priors:
                add(oid, W_CATEGORY_PRIOR, EV_CATEGORY_PRIOR)

        hint = spatial_hints.get(role)
        if hint and len(hyps) > 1:
            if frame is not None:
                _apply_spatial_hint(hyps, objects, hint, base_pos, base_yaw)
        elif len(hyps) > 1:
            for anchor_role in ("receptacle", "target", "manipulated_object"):
                if anchor_role != role and anchor_role in anchor_points:
                    _apply_geometry_bonus(hyps, objects, anchor_points[anchor_role])
                    break

        resolution = finalize_hypotheses(role, hyps, multi_select=False)
        resolutions[role] = resolution
        if resolution["hypotheses"]:
            anchor_points[role] = aabb_center(objects[resolution["hypotheses"][0]["object_id"]]["aabb"])
    return resolutions


def resolve_support_role(manipulated_res, support_edges):
    """The 'support' role is derived, not language-grounded: whatever
    object the resolved manipulated_object's top hypothesis rests on."""
    hyps = {}
    if manipulated_res and manipulated_res["hypotheses"]:
        top = manipulated_res["hypotheses"][0]["object_id"]
        for e in support_edges:
            if e["src"] == top:
                hyps[e["dst"]] = {"confidence": W_SUPPORT_EDGE * e["confidence"],
                                   "evidence": [EV_CONTACT_SUPPORT]}
    return finalize_hypotheses("support", hyps, multi_select=False)


def resolve_obstacle_role(eligible_objects, base_pos, target_point, exclude_ids,
                           capsule_radius_m, lift_m):
    """The 'obstacle' role is set-valued: every eligible object whose aabb
    intersects the base->target swept-workspace capsule is jointly selected
    (there is no single "the" obstacle)."""
    if target_point is None or base_pos is None:
        return finalize_hypotheses("obstacle", {}, multi_select=True)
    hits = swept_workspace_hits(eligible_objects, base_pos, target_point,
                                 capsule_radius_m, lift_m, exclude_ids=exclude_ids)
    hyps = {}
    if hits:
        max_margin = max(m for _, m in hits)
        for oid, margin in hits:
            norm = margin / max(max_margin, 1e-6)
            hyps[oid] = {"confidence": W_OBSTACLE_BASE + W_OBSTACLE_MARGIN * norm,
                         "evidence": [EV_SWEPT_WORKSPACE]}
    return finalize_hypotheses("obstacle", hyps, multi_select=True)


# ------------------------------------------------------------------ reports
def role_accuracy_report(cases):
    """Aggregate role-resolution accuracy / ambiguity rate over hand-labeled
    test cases (plan/11 acceptance criterion: "role accuracy and ambiguity
    rate are reported").

    cases: iterable of {"role_resolutions": {role: RoleResolution, ...},
                         "expected": {role: expected_object_id_or_None, ...}}
    Only roles present in a case's "expected" dict are scored for that case
    (a case that only asserts on "manipulated_object" doesn't get penalized
    for "obstacle"). expected_object_id may be None to assert "correctly
    reports no confident single answer" (checked via status != "resolved"
    rather than object-id equality).
    """
    n_total = n_correct = n_ambiguous = 0
    per_role = {}
    for case in cases:
        for role, expected_id in case["expected"].items():
            res = case["role_resolutions"].get(role, {"status": "unresolved", "hypotheses": []})
            best = res["hypotheses"][0]["object_id"] if res["hypotheses"] else None
            is_ambiguous = res["status"] != "resolved"
            if expected_id is None:
                correct = is_ambiguous  # "correctly flagged as uncertain"
            else:
                correct = (best == expected_id) and not is_ambiguous
            n_total += 1
            n_correct += int(correct)
            n_ambiguous += int(is_ambiguous)
            bucket = per_role.setdefault(role, {"n": 0, "correct": 0, "ambiguous": 0})
            bucket["n"] += 1
            bucket["correct"] += int(correct)
            bucket["ambiguous"] += int(is_ambiguous)

    per_role_out = {r: {"accuracy": b["correct"] / b["n"], "ambiguity_rate": b["ambiguous"] / b["n"],
                         "n": b["n"]} for r, b in per_role.items()}
    return {"n_cases": n_total,
            "role_accuracy": (n_correct / n_total) if n_total else None,
            "ambiguity_rate": (n_ambiguous / n_total) if n_total else None,
            "per_role": per_role_out}
