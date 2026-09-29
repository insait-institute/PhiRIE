"""Tier-1 bridge: package objects already made sim-ready by s6_physics.py
(CoACD collision parts + object.urdf + physics.json mass/friction/restitution)
for import into BEHAVIOR-1K/OmniGibson, and map our free-text labels to BDDL
WordNet synsets.

No simulator, no GPU here. This only copies existing files and writes a
manifest; the actual URDF->USD conversion needs a live Isaac Sim process
(see omnigibson_bridge/import_and_run.py, run inside the `behavior1k` env).

Physics is NOT recomputed: the collision meshes are byte-identical copies of
s6's collision/part_*.obj, and mass comes straight from physics.json. Only
friction/restitution ride along in the manifest, since plain URDF has no
slot for them - import_and_run.py applies them as a UsdPhysics material
after conversion.

Usage: python export_omnigibson.py [--task NAME ...]  (default: all 7 tasks
in TASK_PLAN). Reads across ALL outputs/*_factory scenes (not just SIMANY_SCENE)
since objects are pooled/deduped by label, then staged per task.
"""
import argparse
import csv
import json
import re
import shutil

from agents.core import common as C

BDDL_DATA = (C.ROOT / "third_party" / "BEHAVIOR-1K" / "bddl3" / "bddl"
             / "generated_data")
EXPORT_DIR = C.ROOT / "outputs" / "omnigibson_export"

# task -> (movable synset -> (count, our-vocab label hint), fixture synset)
# Counts/labels reality-checked by hand against the actual problem*.bddl
# files (parse_bddl() in behavior1k_coverage.py only looks at rigid_needed
# counts per-synset, not per-object multiplicities used in :goal, which is
# what actually matters here).
# placement: (predicate, target) used to actually place each movable object
# via OmniGibson's own object_states rejection-sampling setter (the same
# primitive BehaviorTask's init-sampler uses) rather than a hand-rolled drop
# position - "Inside" a closed articulated container (dishwasher/fridge)
# needs the container's own fillable-volume/door-state handling, which only
# the real sampler knows how to do; "OnTop" the floor has no such access
# problem, so a plain drop is fine there. None = no real-checker placement
# needed at all (goal is 100% scripted).
TASK_PLAN = {
    "loading_the_dishwasher": {
        "movable": {"plate.n.04": (1, "plate"), "mug.n.04": (2, "mug"),
                    "bowl.n.01": (1, "bowl")},
        "fixture_synset": "dishwasher.n.01",
        "goal_kind": "geometric",  # :goal is pure Inside, no substance dep
        "placement": ("Inside", "fixture"),
    },
    "putting_dirty_dishes_in_sink": {
        "movable": {"bowl.n.01": (3, "bowl"), "plate.n.04": (2, "plate")},
        "fixture_synset": "sink.n.01",
        "goal_kind": "geometric",
        "placement": ("Inside", "fixture"),
    },
    "store_beer": {
        "movable": {"beer_bottle.n.01": (4, "bottle")},
        "fixture_synset": "electric_refrigerator.n.01",
        "goal_kind": "geometric",
        "placement": ("Inside", "fixture"),
    },
    "clearing_table_after_coffee": {
        "movable": {"mug.n.04": (2, "mug")},
        "fixture_synset": "dishwasher.n.01",
        "goal_kind": "hybrid",  # Inside (real) AND not-filled-with-coffee (scripted)
        "placement": ("Inside", "fixture"),
    },
    "changing_dogs_water": {
        "movable": {"bowl.n.01": (1, "bowl")},
        "fixture_synset": "sink.n.01",
        "goal_kind": "hybrid",  # ontop floor (real) AND filled-with-water (scripted)
        "placement": ("OnTop", "floor"),
    },
    "clean_clear_plastic": {
        "movable": {"liquid_soap__bottle.n.01": (1, "bottle"),
                    "sodium_carbonate__jar.n.01": (1, "jar")},
        "fixture_synset": "sink.n.01",
        "goal_kind": "scripted",  # goal is entirely "not covered/not filled"
        "placement": None,
    },
    "fold_a_plastic_bag": {
        "movable": {"plastic_bag.n.01": (1, "bag")},
        "fixture_synset": "clothes_dryer.n.01",
        "goal_kind": "scripted",  # folded() is cloth-only, no geometry to place
        "placement": None,
    },
}

FIXTURE_CATEGORY = {
    # official BEHAVIOR-1K dataset category to use as a stand-in container,
    # since none of our ScanNet++ office/desk scans contain a real one -
    # see category_mapping.csv (synset column).
    "dishwasher.n.01": "dishwasher",
    # pedestal_sink's basin was too small/oddly-shaped for the Inside
    # sampler to ever fit a bowl/plate in it (empirically: 0/5 objects
    # placeable across every attempt) - commercial_kitchen_sink is a bigger,
    # also more semantically apt basin for a kitchen dish-washing task.
    "sink.n.01": "commercial_kitchen_sink",
    "electric_refrigerator.n.01": "fridge",
    "clothes_dryer.n.01": "clothes_dryer",
}


def norm(word):
    return re.sub(r"[_.]", " ", word).strip().lower()


def load_synset_type():
    d = {}
    with open(BDDL_DATA / "synsets.csv") as f:
        for row in csv.DictReader(f):
            d[row["synset"]] = row["objectType"]
    return d


def load_category_to_synset():
    """category_mapping.csv: official (free-text category -> synset) table
    the BEHAVIOR-1K authors curated - more authoritative than re-deriving
    from synset name patterns, so we use it as the primary lookup and only
    fall back to a synset-name-word match for labels it doesn't cover.
    """
    cat2syn = {}
    with open(BDDL_DATA / "category_mapping.csv") as f:
        for row in csv.DictReader(f):
            cat2syn[norm(row["category"])] = row["synset"]
    return cat2syn


def gather_pool(label_words):
    """Scan every outputs/*_factory scene for non-rejected, URDF-ready
    objects whose label matches one of label_words (loose containment, same
    rule as behavior1k_coverage.py's matches()). Returns
    {label_word: [(scene, index, tier), ...]} sorted best-tier first.
    """
    pool = {w: [] for w in label_words}
    for p in sorted((C.ROOT / "outputs").glob("*_factory/objects/objects.json")):
        scene_factory = p.parent.parent.name
        try:
            objs = json.loads(p.read_text())
        except Exception:
            continue
        for o in objs:
            lbl = o["label"].strip().lower()
            odir = p.parent / f"obj_{o['index']:02d}"
            al_p = odir / "aligned.json"
            if not al_p.exists() or not (odir / "object.urdf").exists():
                continue
            try:
                al = json.loads(al_p.read_text())
            except Exception:
                continue
            if al.get("rejected"):
                continue
            for w in label_words:
                if w == lbl or lbl.endswith(" " + w) or w.endswith(" " + lbl):
                    pool[w].append((scene_factory, o["index"], al.get("tier", "C")))
    for w in pool:
        pool[w].sort(key=lambda x: x[2])  # tier A first
    return pool


def stage_object(scene_factory, index, model_id, out_root):
    """Copy the s6-produced object dir as-is (URDF's mesh/collision paths
    are relative, so a straight directory copy keeps them valid).

    s6_physics.py names its single link "base" (fine for MuJoCo/export_mjcf.py,
    which don't care) - OmniGibson's own asset pipeline hardcodes the
    convention that a URDF's root link is named "base_link"
    (generate_urdf_for_mesh in asset_conversion_utils.py does this too), and
    fails with a bare KeyError if it isn't. Rename only in the staged copy;
    s6's own output (and its other consumers) stay untouched.
    """
    src = C.ROOT / "outputs" / scene_factory / "objects" / f"obj_{index:02d}"
    dst = out_root / "objects" / model_id
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    urdf_path = dst / "object.urdf"
    urdf_path.write_text(urdf_path.read_text().replace(
        '<link name="base">', '<link name="base_link">'))
    return dst


def write_baseline_urdf(obj_dir, phys, scale):
    """Naive-digital-twin ablation arm: single whole-mesh convex hull instead
    of s6's per-part CoACD decomposition, everything else (mass, friction,
    restitution, visual mesh, canonical->world scale) held identical -
    isolates collision-geometry fidelity as the one variable under test,
    same spirit as the project's existing MuJoCo drop-stable ablations.
    """
    import numpy as np
    import trimesh

    tm = trimesh.load(obj_dir / "mesh_sim.ply", process=False)
    hull = tm.convex_hull
    (obj_dir / "collision_baseline").mkdir(exist_ok=True)
    hull_path = obj_dir / "collision_baseline" / "hull.obj"
    hull.export(hull_path)

    m = phys["mass_kg"]
    dims_world = (tm.vertices.max(axis=0) - tm.vertices.min(axis=0)) * scale
    dx, dy, dz = np.maximum(dims_world, 1e-3)
    ixx, iyy, izz = (m / 12 * (dy**2 + dz**2), m / 12 * (dx**2 + dz**2),
                     m / 12 * (dx**2 + dy**2))
    com = tm.vertices.mean(axis=0) * scale
    s = f"{scale:.6f} {scale:.6f} {scale:.6f}"
    urdf = f"""<?xml version="1.0"?>
<robot name="{obj_dir.name}_baseline">
  <link name="base_link">
    <inertial>
      <origin xyz="{com[0]:.5f} {com[1]:.5f} {com[2]:.5f}"/>
      <mass value="{m:.4f}"/>
      <inertia ixx="{ixx:.6e}" iyy="{iyy:.6e}" izz="{izz:.6e}"
               ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual><geometry><mesh filename="mesh_sim.obj" scale="{s}"/></geometry></visual>
    <collision><geometry>
      <mesh filename="collision_baseline/hull.obj" scale="{s}"/></geometry></collision>
  </link>
</robot>
"""
    (obj_dir / "object_baseline.urdf").write_text(urdf)
    return obj_dir / "object_baseline.urdf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", nargs="*", default=None,
                     help="subset of TASK_PLAN keys (default: all)")
    args = ap.parse_args()

    tasks = args.task or list(TASK_PLAN)
    unknown = set(tasks) - set(TASK_PLAN)
    assert not unknown, f"unknown task(s): {unknown}"

    synset_type = load_synset_type()
    load_category_to_synset()  # validated below; kept for audit/logging only

    # sanity: every hardcoded synset in TASK_PLAN must actually exist in this
    # bddl release, and every fixture must have a curated dataset category.
    for task, plan in TASK_PLAN.items():
        for syn in plan["movable"]:
            assert syn in synset_type, f"{task}: unknown synset {syn}"
        assert plan["fixture_synset"] in FIXTURE_CATEGORY, task

    all_label_words = {lbl for plan in TASK_PLAN.values()
                        for _, lbl in plan["movable"].values()}
    pool = gather_pool(all_label_words)
    for w, items in pool.items():
        print(f"[export_og] pool '{w}': {len(items)} candidates")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    (EXPORT_DIR / "objects").mkdir(exist_ok=True)

    manifest = {"tasks": {}, "fixtures": FIXTURE_CATEGORY,
                "label_to_synset_used": {}}
    used_cursor = {w: 0 for w in all_label_words}  # round-robins through the
    # pool so repeated tasks reuse variety before duplicating any one object

    for task in tasks:
        plan = TASK_PLAN[task]
        objects = []
        for synset, (count, label) in plan["movable"].items():
            candidates = pool.get(label, [])
            assert candidates, f"{task}: no reconstructed '{label}' objects found at all"
            manifest["label_to_synset_used"][label] = synset
            for i in range(count):
                cursor = used_cursor[label]
                scene_factory, index, tier = candidates[cursor % len(candidates)]
                duplicated = cursor >= len(candidates)
                used_cursor[label] += 1
                model_id = f"{synset.split('.')[0]}_{scene_factory[:10]}_{index:02d}_{i}"
                model_id = re.sub(r"[^a-zA-Z0-9_]", "", model_id)
                obj_dir = stage_object(scene_factory, index, model_id, EXPORT_DIR)
                phys = json.loads((obj_dir / "physics.json").read_text())
                aligned = json.loads((obj_dir / "aligned.json").read_text())
                baseline_urdf = write_baseline_urdf(obj_dir, phys, aligned["scale"])
                objects.append({
                    "synset": synset,
                    "instance_name": f"{synset}_{i + 1}",
                    "category": synset.split(".")[0],
                    "model_id": model_id,
                    "source_scene": scene_factory,
                    "source_index": index,
                    "source_tier": tier,
                    "urdf": str((obj_dir / "object.urdf").relative_to(C.ROOT)),
                    "urdf_baseline": str(baseline_urdf.relative_to(C.ROOT)),
                    "duplicated": duplicated,
                    "physics": phys,
                })
        manifest["tasks"][task] = {
            "goal_kind": plan["goal_kind"],
            "fixture_synset": plan["fixture_synset"],
            "fixture_category": FIXTURE_CATEGORY[plan["fixture_synset"]],
            "placement": list(plan["placement"]) if plan["placement"] else None,
            "objects": objects,
        }
        n_dup = sum(1 for o in objects if o["duplicated"])
        print(f"[export_og] {task}: {len(objects)} objects staged "
              f"({n_dup} are duplicated instances of an already-used asset)")

    (EXPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[export_og] wrote {EXPORT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
