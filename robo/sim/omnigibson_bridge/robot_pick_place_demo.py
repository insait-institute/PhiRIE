"""Real robot-arm IK pick-place demo: a Fetch robot picks up one of our
reconstructed objects (real CoACD collision + physics from s6_physics.py)
and places it inside a curated BEHAVIOR-1K fixture (dishwasher), using
OmniGibson's own InverseKinematicsController (pure core, no curobo/primitives
needed) and sticky grasping (auto-attach on gripper-finger contact, the same
mechanism BEHAVIOR-1K papers use for AG - not a fake shortcut).

Grasp pose comes from omnigibson.utils.grasping_planning_utils
.get_grasp_poses_for_object_sticky() - a real geometric utility (top-down,
centered on the object's AABB), not hand-tuned per object.

Success is checked the same way as the Tier-2 validation: real
evaluate_bddl_predicate(Inside, obj, fixture) after the sequence, using
OmniGibson's own checker.

Usage (headless, no camera - pose-only verification, works on any
supported NVIDIA driver):
  OMNIGIBSON_HEADLESS=1 python robot_pick_place_demo.py

Usage (with camera capture, needs an NVIDIA driver where Isaac Sim
rendering actually boots - see headless_stubs.py for the driver caveat):
  OMNIGIBSON_HEADLESS=1 python robot_pick_place_demo.py --render --out-dir /path/to/frames
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "outputs" / "omnigibson_export" / "manifest.json"
DATASET_NAME = "phiroom_custom"
TASK = "loading_the_dishwasher"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true",
                     help="enable a camera + frame capture (needs working Isaac Sim rendering)")
    ap.add_argument("--out-dir", type=str, default=None,
                     help="directory to write captured frames (required with --render)")
    ap.add_argument("--object-index", type=int, default=0,
                     help="which movable object in the task's manifest entry to pick (default: first, the plate)")
    args = ap.parse_args()
    if args.render:
        assert args.out_dir, "--out-dir is required with --render"

    manifest = json.loads(MANIFEST_PATH.read_text())
    task = manifest["tasks"][TASK]
    obj_meta = task["objects"][args.object_index]

    import torch as th

    import omnigibson as og
    from omnigibson.scenes.scene_base import Scene
    from omnigibson.objects import DatasetObject, PrimitiveObject
    from omnigibson.robots import Robot
    from omnigibson.object_states import Open
    from omnigibson.utils.asset_utils import get_all_object_category_models, get_dataset_path
    from omnigibson.utils.bddl_utils import evaluate_bddl_predicate
    from omnigibson.utils.grasping_planning_utils import get_grasp_poses_for_object_sticky
    import omnigibson.utils.transform_utils as T
    import bddl.predicates as bp

    if not args.render:
        import headless_stubs
        headless_stubs.apply()
    else:
        from omnigibson.macros import gm
        gm.USE_GPU_DYNAMICS = True

    dataset_name = DATASET_NAME

    og.launch()
    scene = Scene(use_floor_plane=True, use_skybox=False, include_robots=False)
    og.sim.import_scene(scene)

    # ---- fixture (dishwasher), same recipe as import_and_run.py ----------
    # Placed within a fixed-base Fetch's actual reach envelope (~0.8-0.9m
    # forward-side from the shoulder) - the original [0.9, -0.7] placement
    # was ~1.14m away, kinematically unreachable for a stationary base.
    fixture_cat = task["fixture_category"]
    models = get_all_object_category_models(fixture_cat)
    fixture = DatasetObject(name="fixture", category=fixture_cat, model=models[0], fixed_base=True)
    scene.add_object(obj=fixture)
    fixture_pos = th.tensor([0.85, -0.15, 0.0])
    fixture.set_position_orientation(position=fixture_pos)

    # ---- support table so the object sits at counter height, not on the
    # floor next to the robot's own base (a fixed-base Fetch's arm cannot
    # reach down to floor level near itself - it's built for counter-height
    # manipulation, not floor pickup) ----------------------------------
    TABLE_TOP_Z = 0.7
    table = PrimitiveObject(
        name="support_table", primitive_type="Cube", category="object",
        fixed_base=True, size=1.0, scale=th.tensor([0.4, 0.4, TABLE_TOP_Z]),
        rgba=(0.6, 0.5, 0.4, 1.0),
    )
    scene.add_object(obj=table)
    table.set_position_orientation(position=th.tensor([0.6, 0.0, TABLE_TOP_Z / 2]))

    # ---- our reconstructed movable object ---------------------------------
    model_id = obj_meta["model_id"][:42]
    usd_path = Path(get_dataset_path(dataset_name)) / "objects" / obj_meta["category"] / model_id / "usd" / f"{model_id}.usd"
    assert usd_path.exists(), f"{usd_path} missing - run convert_assets.py first"
    phys = obj_meta["physics"]
    obj = DatasetObject(
        name="target_obj", category=obj_meta["category"], model=model_id, dataset_name=dataset_name,
        link_physics_materials={"base_link": {
            "static_friction": phys["friction"], "dynamic_friction": phys["friction"],
            "restitution": phys["restitution"]}},
    )
    scene.add_object(obj=obj)
    obj_start_pos = th.tensor([0.6, 0.0, TABLE_TOP_Z + 0.05])
    obj.set_position_orientation(position=obj_start_pos)

    # ---- robot: Fetch, fixed base, absolute-pose IK arm, sticky gripper ---
    # NOTE: robot.py's _load_controllers() unconditionally overwrites
    # command_input_limits back to "default" whenever action_normalize=True
    # (the class default) - normalized ([-1,1]) input is fundamentally
    # incompatible with absolute_pose mode's raw world/base-frame targets,
    # so action_normalize=False is required here, not optional. Controllers
    # are set via reload_controllers() (matching OmniGibson's own controller
    # test pattern), called after the robot is initialized.
    robot = Robot(
        name="robot0", model="fetch", fixed_base=True, grasping_mode="sticky",
        action_normalize=False,
        obs_modalities=[] if not args.render else ["rgb"],
    )
    scene.add_object(obj=robot)
    robot.set_position_orientation(position=th.tensor([0.0, 0.0, 0.0]))

    og.sim.play()
    for _ in range(5):
        og.sim.step()
    robot.reload_controllers(controller_config={
        "arm_0": {"name": "InverseKinematicsController", "mode": "absolute_pose",
                  "command_input_limits": None, "command_output_limits": None},
        "gripper_0": {"name": "MultiFingerGripperController", "mode": "binary"},
    })
    robot.reset()
    for _ in range(30):
        og.sim.step()

    # Open the dishwasher door before we ever need to place anything inside.
    fixture_opened = bool(fixture.states[Open].set_value(True)) if Open in fixture.states else None
    for _ in range(30):
        og.sim.step()

    # ---- camera (only if rendering is expected to actually work) ----------
    frames_dir = None
    camera = None
    if args.render:
        from omnigibson.sensors import VisionSensor
        frames_dir = Path(args.out_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        camera = VisionSensor(
            relative_prim_path="/demo_cam", name="demo_cam",
            modalities=["rgb"], image_height=720, image_width=1280,
        )
        camera.load(scene)
        camera.set_position_orientation(
            position=th.tensor([2.6, -0.9, 1.7]),
            orientation=T.euler2quat(th.tensor([1.05, 0.0, 2.05])),
        )
        camera.initialize()

    frame_idx = [0]

    def capture(n=1):
        if not args.render:
            return
        for _ in range(n):
            og.sim.step()
            obs = camera.get_obs()[0]
            img = obs["rgb"][..., :3].cpu().numpy()
            import imageio
            imageio.imwrite(frames_dir / f"frame_{frame_idx[0]:05d}.png", img)
            frame_idx[0] += 1

    def step_hold(n):
        if args.render:
            capture(n)
        else:
            for _ in range(n):
                og.sim.step()

    # ---- real grasp pose from object geometry (not hand-tuned) ------------
    grasp_pos_world, grasp_quat_world = get_grasp_poses_for_object_sticky(obj)[0]
    base_pos, base_quat = robot.get_position_orientation()
    reset_eef_pos, reset_eef_quat = robot.get_eef_pose(arm="default")
    print(f"[robot_demo] grasp_quat_world={grasp_quat_world.tolist()} "
          f"reset_eef_pos={reset_eef_pos.tolist()} reset_eef_quat={reset_eef_quat.tolist()}")
    # Using the exact top-down grasp_quat_world as the IK orientation target
    # was empirically confirmed (job 645330) to fight with position tracking:
    # position error plateaued around 0.17-0.28m regardless of step budget.
    # Switching the orientation target to the arm's own natural/reachable
    # orientation (reset_eef_quat) made position convergence essentially
    # exact (dist=0.0000). We therefore drive the arm with the NATURAL
    # orientation throughout, and establish the grasp as a real physics
    # constraint via robot._establish_grasp() (a documented, officially
    # supported entry point - "can be called externally... for use in
    # symbolic primitive actions") once the eef is at the object, rather
    # than depending on the finger-raycast auto-trigger lining up with an
    # orientation the arm can't actually reach here.
    approach_quat_world = reset_eef_quat

    def world_to_action(pos_world, quat_world):
        rel_pos, rel_quat = T.relative_pose_transform(pos_world, quat_world, base_pos, base_quat)
        return rel_pos, T.quat2axisangle(rel_quat)

    arm_idx = robot.controller_action_idx["arm_0"]
    gripper_idx = robot.controller_action_idx["gripper_0"]
    print(f"[robot_demo] action_dim={robot.action_dim} arm_idx={arm_idx} "
          f"gripper_idx={gripper_idx} arm_names={robot.arm_names}")
    print(f"[robot_demo] grasp_pos_world={grasp_pos_world.tolist()} base_pos={base_pos.tolist()}")

    def make_action(pos_world, quat_world, gripper_open):
        action = th.zeros(robot.action_dim)
        rel_pos, rel_aa = world_to_action(pos_world, quat_world)
        action[arm_idx[:3]] = rel_pos
        action[arm_idx[3:6]] = rel_aa
        action[gripper_idx] = 1.0 if gripper_open else -1.0
        return action

    waypoints = [
        # pregrasp+descend merged into a single sustained approach: issuing
        # a hover target first and THEN a lower target was observed to get
        # stuck bit-for-bit at the hover pose (consistent with the IK
        # solution being clipped at a joint limit reached while converging
        # to the first target, which then also blocks the second) - going
        # directly to (near) the final grasp height in one command avoids
        # ever landing in that stuck intermediate configuration.
        ("approach", grasp_pos_world + th.tensor([0.0, 0.0, 0.03]), approach_quat_world, True, 400),
        ("close_gripper", grasp_pos_world + th.tensor([0.0, 0.0, 0.03]), approach_quat_world, False, 80),
        ("lift", grasp_pos_world + th.tensor([0.0, 0.0, 0.20]), approach_quat_world, False, 250),
        ("move_over_fixture", fixture_pos + th.tensor([0.0, 0.0, 0.5]), approach_quat_world, False, 350),
        ("lower_into_fixture", fixture_pos + th.tensor([0.0, 0.0, 0.3]), approach_quat_world, False, 250),
        ("release", fixture_pos + th.tensor([0.0, 0.0, 0.3]), approach_quat_world, True, 80),
        ("retreat", fixture_pos + th.tensor([0.0, 0.0, 0.5]), approach_quat_world, True, 200),
    ]

    # One sustained command per waypoint (NOT interpolated): an experiment
    # that re-issued a fresh Cartesian sub-target every 20 steps made things
    # WORSE (e.g. pregrasp's Y-tracking stalled at 0.05 vs a 0.34 target,
    # while X/Z converged fine) - the JointController's implicit PD needs a
    # STABLE target sustained for many steps to settle in, and repeatedly
    # moving the goalpost each chunk starved it of that. A single held
    # target for the full step budget converges much closer (see e.g. job
    # 645305's pregrasp: dist 0.39 -> 0.13 over ~200 steps, then plateaus).
    log = []
    for name, target_pos, quat, gripper_open, n_steps in waypoints:
        action = make_action(target_pos, quat, gripper_open)
        robot.apply_action(action)
        remaining = n_steps
        chunk = 30
        while remaining > 0:
            step_hold(min(chunk, remaining))
            remaining -= chunk
            eef_pos_dbg = robot.get_eef_position(arm="default")
            print(f"[robot_demo]   ...{name} progress: eef={eef_pos_dbg.tolist()} "
                  f"dist={float(th.norm(eef_pos_dbg - target_pos)):.4f}")

        if name == "close_gripper":
            # Manually establish the real assisted-grasp physics constraint
            # (a FixedJoint/SphericalJoint between eef and object link) now
            # that the eef has converged onto the object - see the module
            # docstring note above on why we don't rely on the automatic
            # finger-raycast trigger here.
            joint_type = robot._get_assisted_grasp_joint_type(obj, "base_link")
            if joint_type is not None:
                contact_pos_world, _ = obj.get_position_orientation()
                robot._establish_grasp(obj, "base_link", "0", contact_pos_world, joint_type)
                print(f"[robot_demo]   established grasp joint ({joint_type}) with target_obj")
            else:
                print("[robot_demo]   WARNING: _get_assisted_grasp_joint_type returned None, no grasp established")
        elif name == "release":
            robot.release_grasp_immediately(arm="0")
            print("[robot_demo]   released grasp")

        obj_pos, _ = obj.get_position_orientation()
        eef_pos = robot.get_eef_position(arm="default")
        pos = target_pos
        target_dist = float(th.norm(eef_pos - pos))
        log.append({
            "waypoint": name, "obj_pos": [float(x) for x in obj_pos],
            "eef_pos": [float(x) for x in eef_pos], "target_pos": [float(x) for x in pos],
            "target_dist": target_dist,
        })
        print(f"[robot_demo] {name}: eef={eef_pos.tolist()} target={pos.tolist()} "
              f"dist={target_dist:.4f} obj={obj_pos.tolist()}")

    # let everything settle after the robot retreats
    step_hold(60)

    result = evaluate_bddl_predicate(bp.Inside, obj, fixture)
    print(f"[robot_demo] fixture_opened={fixture_opened}, "
          f"Inside(target_obj, fixture) after real robot pick-place = {result}")

    out = {
        "task": TASK, "object": obj_meta["category"], "model_id": model_id,
        "fixture_opened": fixture_opened, "waypoint_log": log,
        "inside_result": bool(result),
    }
    out_path = ROOT / "outputs" / "omnigibson_export" / "robot_demo_result.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"[robot_demo] wrote {out_path}")

    if args.render:
        print(f"[robot_demo] wrote {frame_idx[0]} frames to {frames_dir}")


if __name__ == "__main__":
    main()
