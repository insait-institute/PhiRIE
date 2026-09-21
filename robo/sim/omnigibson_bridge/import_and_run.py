"""Tier-2: load Tier-1 exported objects (export_omnigibson.py) into a live
OmniGibson scene, script a pick-place, and evaluate the REAL BDDL goal
checker for the kinematic (OnTop/Inside/NextTo/Under/Touching) literals -
these are provenance-agnostic pure pose checks in OmniGibson's own
bddl_utils.evaluate_bddl_predicate(), confirmed by reading
omnigibson/utils/bddl_utils.py directly, so a custom-imported object
satisfies them exactly like an official-dataset object would.

We deliberately do NOT use OmniGibson's BehaviorTask: its online sampler
requires every synset to resolve to a curated-dataset category
(_import_sampleable_objects asserts get_all_object_categories()), which our
custom objects can't satisfy. Instead we go straight to the lower-level,
simulator-agnostic bddl.activity API (Conditions/get_goal_conditions/
evaluate_goal_conditions), which was built exactly for this: it takes a
user-supplied evaluate_fn(predicate_cls, *entities) callback. Substance/
cloth literals (Covered/Filled/Folded/...) are NOT backed by our rigid-body
reconstruction - those are answered by a hardcoded SCRIPTED_OUTCOMES table
per task (see manifest["tasks"][task]["goal_kind"]), never by
evaluate_bddl_predicate. The results JSON records, per goal literal, which
path answered it - that split IS the point of this validation, not an
implementation detail to hide.

Run inside the `behavior1k` conda env, per (task, arm) pair:
  OMNIGIBSON_HEADLESS=1 python import_and_run.py --task loading_the_dishwasher --arm ours
  OMNIGIBSON_HEADLESS=1 python import_and_run.py --task loading_the_dishwasher --arm baseline
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "outputs" / "omnigibson_export" / "manifest.json"
RESULTS_DIR = ROOT / "outputs" / "omnigibson_export" / "results"

# Which BDDL predicate classes we let the REAL simulator answer (pure
# kinematic/pose checks - see bddl_utils.PREDICATE_TO_STATE). Everything
# else in a task's goal is a scripted assertion, not a measurement.
KINEMATIC_PREDICATE_NAMES = {"Inside", "OnTop", "NextTo", "Under", "Touching"}

# Hand-verified against the actual problem*.bddl text for each task (see
# export_omnigibson.py's TASK_PLAN comment) - what a scripted "did the
# agent perform this non-geometric action" assertion resolves to. Keyed by
# (task, predicate_class_name); value is the predicate's raw truth value
# (pre-negation - the compiled goal tree applies any `not` itself).
SCRIPTED_OUTCOMES = {
    ("clearing_table_after_coffee", "Filled"): False,  # mugs emptied before dishwashing
    ("changing_dogs_water", "Filled"): True,  # bowl filled from the sink
    ("changing_dogs_water", "OnTop"): True,  # see FLOOR_TARGET_SYNSET_PREFIX note below
    ("clean_clear_plastic", "Covered"): False,  # wiped clean
    ("clean_clear_plastic", "Contains"): False,  # jar emptied
    ("fold_a_plastic_bag", "Folded"): True,  # scripted fold action
}

# OnTop(_, floor.n.01_*) is downgraded from real to scripted: our scenes'
# floor is a bare synthetic ground XFormPrim (no StatefulObject, no
# Touching/contact API), not a proper BEHAVIOR-1K "floors" category object,
# so there's nothing to run the real Inside/OnTop checker against. The bowl
# physically does rest on it after settling (real physics, just not
# BDDL-checked) - this is a scene-setup limitation, not a substance/cloth
# scope exclusion like the rest of SCRIPTED_OUTCOMES.
FLOOR_TARGET_SYNSET_PREFIX = "floor."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--arm", choices=["ours", "baseline"], default="ours")
    ap.add_argument("--settle-steps", type=int, default=300)
    args = ap.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text())
    task = manifest["tasks"][args.task]

    import torch as th

    import omnigibson as og
    from omnigibson.scenes.scene_base import Scene
    from omnigibson.objects import DatasetObject
    from omnigibson.object_states import Inside, OnTop, Open
    from omnigibson.utils.asset_utils import get_all_object_category_models, get_dataset_path
    from omnigibson.utils.bddl_utils import evaluate_bddl_predicate, PREDICATE_TO_STATE
    import bddl.predicates as bp
    from bddl.activity import (
        Conditions, get_object_scope, get_goal_conditions,
        evaluate_goal_conditions, get_natural_goal_conditions,
    )

    import headless_stubs
    headless_stubs.apply()

    name_to_predicate_cls = {cls.__name__: cls for cls in PREDICATE_TO_STATE}
    kinematic_classes = {name_to_predicate_cls[n] for n in KINEMATIC_PREDICATE_NAMES
                          if n in name_to_predicate_cls}

    dataset_name = "phiroom_custom"

    og.launch()
    scene = Scene(use_floor_plane=True, use_skybox=False, include_robots=False)
    og.sim.import_scene(scene)
    # NOTE: all objects are added BEFORE og.sim.play() (below), not after.
    # Adding objects to an already-playing sim one at a time invalidates the
    # PhysX tensor "simulation view" created for the previous object, and
    # OmniGibson's own center_of_mass setter (hit by every rigid body's
    # _post_load()) holds a stale reference to it -> "Simulation view object
    # is invalidated and cannot be used again to call getCOMs". Building the
    # full scene first and calling play() once afterward is the standard
    # Isaac Sim pattern and avoids this entirely.

    # ---- 1. fixture (official curated stand-in container; ours has none) --
    fixture_cat = task["fixture_category"]
    models = get_all_object_category_models(fixture_cat)
    assert models, f"no dataset models found for category {fixture_cat}"
    # kinematic_only was needed to dodge an earlier COM-tensor-view crash
    # that turned out to be caused by adding objects to an already-playing
    # sim (see the og.sim.play() ordering note below) - now that that's
    # fixed, leave joints dynamic: the Inside sampler needs to actually open
    # the fixture's door to find a valid pose in its fillable volume.
    fixture = DatasetObject(name="fixture", category=fixture_cat, model=models[0],
                             fixed_base=True)
    scene.add_object(obj=fixture)
    fixture.set_position_orientation(position=th.tensor([0.0, 0.0, 0.0]))
    fx_lo, fx_hi = fixture.aabb
    fx_top_z = float(fx_hi[2])
    fx_cx, fx_cy = float((fx_lo[0] + fx_hi[0]) / 2), float((fx_lo[1] + fx_hi[1]) / 2)

    # ---- 2. our (or baseline) movable objects: import -> place -> drop ----
    scope = {"agent.n.01_1": None}
    n_objs = len(task["objects"])
    grid = max(1, int(n_objs ** 0.5) + 1)
    spacing = 0.25
    for i, o in enumerate(task["objects"]):
        arm_suffix = "" if args.arm == "ours" else "_bl"
        model_id = (o["model_id"] + arm_suffix)[:42]  # Isaac Sim prim-name length caution
        # Assets are pre-converted by convert_assets.py (import_og_asset_from_urdf
        # asserts zero scenes exist, so it can't run interleaved with our
        # already-built Scene - see that script's docstring).
        usd_path = (Path(get_dataset_path(dataset_name)) / "objects" / o["category"]
                    / model_id / "usd" / f"{model_id}.usd")
        assert usd_path.exists(), (
            f"{usd_path} missing - run convert_assets.py first")
        phys = o["physics"]
        obj = DatasetObject(
            name=f"obj_{i}", category=o["category"], model=model_id,
            dataset_name=dataset_name,
            link_physics_materials={"base_link": {
                "static_friction": phys["friction"], "dynamic_friction": phys["friction"],
                "restitution": phys["restitution"]}},
        )
        scene.add_object(obj=obj)
        dx, dy = spacing * (i % grid - grid / 2), spacing * (i // grid - grid / 2)
        # Neutral starting pose, off to the side of the fixture's footprint
        # so nothing overlaps it while stopped. For "Inside" placements this
        # is thrown away by the sampler below anyway; for "OnTop floor" (and
        # the no-real-placement tasks) it's the actual drop position.
        obj.set_position_orientation(
            position=th.tensor([fx_cx + dx, fx_cy + fx_hi[1] - fx_lo[1] + 1.0 + dy,
                                 0.15 + 0.1 * i]))
        scope[o["instance_name"]] = obj

    fixture_instance_name = task["fixture_synset"] + "_1"
    scope[fixture_instance_name] = fixture

    og.sim.play()  # scene fully built - now start physics
    og.sim.step()

    # Open the fixture's door first if it has one - the fillable/inside
    # volume the Inside sampler checks against is defined independent of
    # door state, but a physically CLOSED door blocks every sampled pose via
    # rejection-sampling-#2 (interpenetration), so nothing is ever placeable
    # until the door is actually open.
    fixture_opened = None
    if Open in fixture.states:
        fixture_opened = bool(fixture.states[Open].set_value(True))
        print(f"[import_and_run] opened fixture door -> {fixture_opened}")
        for _ in range(30):
            og.sim.step()

    # ---- 2b. real placement via OmniGibson's own state-setter sampler -----
    # Inside/OnTop.set_value() IS the same rejection-sampling primitive
    # BehaviorTask's init-sampler uses (sample a candidate pose, settle,
    # verify still holds, retry) - not something we hand-rolled. "Inside" a
    # CLOSED articulated container (dishwasher/fridge) needs its
    # fillable-volume/door handling, which only the real sampler knows; for
    # "OnTop floor" a plain drop-and-settle is enough (no access problem).
    placement = task["placement"]
    placement_log = {}
    if placement and placement[1] == "fixture":
        pred_name = placement[0]
        state_cls = {"Inside": Inside, "OnTop": OnTop}[pred_name]
        for name, obj in scope.items():
            if obj is None or name == fixture_instance_name:
                continue
            ok = obj.states[state_cls].set_value(fixture, True)
            placement_log[name] = bool(ok)
            print(f"[import_and_run] sampled {pred_name}({name}, fixture) -> {ok}")

    t0 = time.time()
    for _ in range(args.settle_steps):
        og.sim.step()
    settle_s = time.time() - t0

    # settle stability: did anything end up with NaN pose or fly off?
    drift_report = {}
    for name, obj in scope.items():
        if obj is None or name == fixture_instance_name:
            continue
        pos, _ = obj.get_position_orientation()
        drift_report[name] = [float(x) for x in pos]

    # ---- 3. real BDDL parse + goal check (bypassing BehaviorTask) --------
    # get_object_scope() returns a bare set[str] of declared instance names
    # (used by compile_state purely to validate predicate arguments) - it
    # does NOT resolve names to values despite get_object_scope's docstring
    # claiming a {name: value} dict (stale relative to create_scope's actual
    # set-returning implementation in this bddl3 version). Compiled
    # predicates carry the raw name strings in self.inputs and hand them to
    # evaluate_fn as-is (see predicates.py/logic_base.py) - WE resolve names
    # to our loaded sim objects via `scope` (built above) inside evaluate_fn.
    conds = Conditions(args.task, 0, "behavior-1k")
    full_scope = get_object_scope(conds)
    goal_conditions = get_goal_conditions(conds, full_scope, generate_ground_options=False)
    nl_goal = get_natural_goal_conditions(conds)

    literal_log = []

    def evaluate_fn(predicate_cls, *entity_names, **kwargs):
        pname = predicate_cls.__name__
        targets_floor = any(n.startswith(FLOOR_TARGET_SYNSET_PREFIX) for n in entity_names)
        if predicate_cls in kinematic_classes and not targets_floor:
            entities = [scope.get(n) for n in entity_names]
            result = evaluate_bddl_predicate(predicate_cls, *entities)
            literal_log.append({"predicate": pname, "entities": list(entity_names),
                                 "kind": "real_checker", "result": result})
            return result
        key = (args.task, pname)
        result = SCRIPTED_OUTCOMES.get(key)
        literal_log.append({"predicate": pname, "entities": list(entity_names),
                             "kind": "scripted", "result": result,
                             "note": "not in SCRIPTED_OUTCOMES -> False" if result is None else None})
        return bool(result)

    all_satisfied, breakdown = evaluate_goal_conditions(goal_conditions, evaluate_fn)

    result = {
        "task": args.task, "arm": args.arm, "goal_kind": task["goal_kind"],
        "n_objects": n_objs, "settle_steps": args.settle_steps,
        "settle_wall_s": settle_s,
        "all_goal_literals_satisfied": bool(all_satisfied),
        "breakdown": breakdown,
        "literal_log": literal_log,
        "natural_language_goal": nl_goal,
        "final_positions": drift_report,
        "placement": placement,
        "fixture_opened": fixture_opened,
        "placement_sampler_log": placement_log,
        "n_real_checker_literals": sum(1 for l in literal_log if l["kind"] == "real_checker"),
        "n_scripted_literals": sum(1 for l in literal_log if l["kind"] == "scripted"),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.task}__{args.arm}.json"
    out_path.write_text(json.dumps(result, indent=1))
    print(f"[import_and_run] {args.task} ({args.arm}): "
          f"all_satisfied={all_satisfied} -> wrote {out_path}")


if __name__ == "__main__":
    main()
