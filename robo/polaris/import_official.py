"""Extract official PolaRiS task definitions into a plain YAML manifest,
WITHOUT launching Isaac Sim.

Why static-only: Isaac Sim 5.1.0's camera-enabled headless launch segfaults
on this cluster (see docs/POLARIS_INTEGRATION.md) -- driver 595.x vs Isaac
Sim 5.1.0's validated 580.65.06, a known upstream bug (github.com/isaac-sim/
IsaacSim discussions #648, #651). Every real PolaRiS gym env needs
`enable_cameras=True`, so no PolaRiS env can be exercised live here. This
script instead reads what the official install *declares* it will do:

  - `third_party/openpi`-style: no, wrong repo -- this reads
    `third_party/PolaRiS/src/polaris/environments/{__init__,droid_cfg,
    robot_cfg}.py` (Python source, static values transcribed with file:line
    provenance -- these classes are IsaacLab `@configclass`es that in
    practice need a live Isaac Sim process to instantiate/import cleanly, so
    we do NOT import them; we read them as text/constants instead) and
  - `third_party/PolaRiS/PolaRiS-Hub/<task>/scene.usda` +
    `initial_conditions.json` (plain files -- USD ASCII text + JSON, no USD
    library or Isaac Sim needed to read either).

Nothing here has been confirmed against a live running official
environment. That confirmation (Task 07 step 5: "Validate camera images,
robot zero pose, action units, gripper convention, control rate, and rubric
events") is exactly the part that's blocked. Every value below should be
read as "declared in the official source," not "observed."

Usage:
    python -m robo.polaris.import_official --task food_bussing \
        --out configs/polaris/tasks/food_bussing.yaml
    python -m robo.polaris.import_official --all \
        --out-dir configs/polaris/tasks
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
POLARIS_ROOT = REPO_ROOT / "third_party" / "PolaRiS"
HUB_ROOT = POLARIS_ROOT / "PolaRiS-Hub"

# ---------------------------------------------------------------------------
# Static robot/action/control facts, transcribed by hand from PolaRiS source.
# Every field cites the exact file it was read from so a future session can
# re-check these against a newer PolaRiS commit without re-deriving them.
# Source commit: third_party/PolaRiS is untracked (see docs/POLARIS_INTEGRATION.md
# for the pinned upstream commit hash of the vendored clone).
# ---------------------------------------------------------------------------

ROBOT_STATIC_FACTS: dict[str, Any] = {
    "source": "src/polaris/environments/robot_cfg.py:9-65 (NVIDIA_DROID ArticulationCfg)",
    "usd_path_relative_to_data_root": "nvidia_droid/noninstanceable.usd",
    "reset_joint_pos_rad": {
        "panda_joint1": 0.0,
        "panda_joint2": -1 / 5 * math.pi,
        "panda_joint3": 0.0,
        "panda_joint4": -4 / 5 * math.pi,
        "panda_joint5": 0.0,
        "panda_joint6": 3 / 5 * math.pi,
        "panda_joint7": 0.0,
        "finger_joint": 0.0,
    },
    "reset_joint_pos_rad_as_7_vector": [
        0.0, -1 / 5 * math.pi, 0.0, -4 / 5 * math.pi, 0.0, 3 / 5 * math.pi, 0.0,
    ],
    "note_vs_simany_frozen_fields": (
        "IDENTICAL to configs/experiments/frozen_fields.yaml robot.reset_pose_rad "
        "[0,-pi/5,0,-4pi/5,0,3pi/5,0] -- both trace to the same DROID convention. "
        "This is a genuine, checked frozen-field match, not a coincidence."
    ),
    "gravity_disabled": True,
    "self_collisions_enabled": False,
    "solver_position_iteration_count": 64,
    "solver_velocity_iteration_count": 0,
    "actuators": {
        "panda_shoulder (joints 1-4)": {"effort_limit": 87.0, "velocity_limit": 2.175, "stiffness": 400.0, "damping": 80.0},
        "panda_forearm (joints 5-7)": {"effort_limit": 12.0, "velocity_limit": 2.61, "stiffness": 400.0, "damping": 80.0},
        "gripper (finger_joint)": {"effort_limit": 200.0, "velocity_limit": 5.0, "stiffness": None, "damping": None},
    },
}

ACTION_STATIC_FACTS: dict[str, Any] = {
    "source": "src/polaris/environments/droid_cfg.py:230-244 (ActionCfg)",
    "arm": {
        "type": "JointPositionActionCfg",
        "joints": "panda_joint.*  (7 joints)",
        "use_default_offset": False,
        "convention": "absolute_joint_position",
    },
    "gripper": {
        "type": "BinaryJointPositionZeroToOneActionCfg (custom subclass, droid_cfg.py:196-227)",
        "input_range": "[0, 1] (or bool)",
        "binarize_threshold": 0.5,
        "open_command_rad": 0.0,
        "close_command_rad": math.pi / 4,
    },
    "action_dim": 8,
    "note_vs_simany_frozen_fields": (
        "IDENTICAL action_dim (8 = 7 absolute joint + 1 gripper) and "
        "action_convention (absolute_joint_position) and "
        "gripper_binarize_threshold (0.5) to "
        "configs/experiments/frozen_fields.yaml. Genuine, checked match."
    ),
}

CONTROL_STATIC_FACTS: dict[str, Any] = {
    "source": "src/polaris/environments/droid_cfg.py:339-364 (EnvCfg.__post_init__)",
    "decimation": 8,  # 4 * 2
    "sim_dt_s": 1 / (60 * 2),  # 1/120
    "control_dt_s": 8 * (1 / 120),  # decimation * sim_dt
    "control_rate_hz": round(1 / (8 * (1 / 120)), 4),  # 15.0
    "render_interval": 8,
    "episode_length_s": 30,
    "rerender_on_reset": True,
    "note_vs_simany_frozen_fields": (
        "control_rate_hz matches configs/experiments/frozen_fields.yaml "
        "(15 Hz) exactly -- genuine match, even though the two simulators "
        "reach it via different substep counts (PhysX 8 x 1/120 vs MuJoCo "
        "40 x 1/600). horizon_seconds does NOT match: official PolaRiS uses "
        "30s (episode_length_s above); SimAny's frozen_fields.yaml uses 32s "
        "(sourced from a documented docs/ROBOT.md gotcha about the MuJoCo "
        "suite's default being too short, not from PolaRiS). This is a real, "
        "declared difference -- if this pairing is ever run, either pin both "
        "to the same horizon or report it as an acknowledged non-frozen field."
    ),
}

# Per-task rubric criteria, transcribed from
# src/polaris/environments/__init__.py gym.register(...) calls. Each item is
# (criterion_fn_name, args, depends_on_indices). depends_on refers to the
# 0-based index of a PRIOR criterion in the same list (PolaRiS's Rubric class
# only counts a dependent criterion once all its dependencies have EVER been
# satisfied, not simultaneously -- see rubrics/base.py:37-73).
TASK_REGISTRY: dict[str, dict[str, Any]] = {
    "food_bussing": {
        "gym_id": "DROID-FoodBussing",
        "usd_scene_relpath": "food_bussing/scene.usda",
        "rubric_source": "src/polaris/environments/__init__.py:41-63",
        "rubric_criteria": [
            {"fn": "reach", "args": {"obj": "ice_cream_", "threshold": 0.2}, "depends_on": []},
            {"fn": "reach", "args": {"obj": "grapes", "threshold": 0.2}, "depends_on": []},
            {"fn": "lift", "args": {"obj": "ice_cream_", "threshold": 0.06}, "depends_on": [0]},
            {"fn": "lift", "args": {"obj": "grapes", "threshold": 0.06}, "depends_on": [1]},
            {"fn": "is_within_xy", "args": {"obj1": "ice_cream_", "obj2": "bowl", "percent_threshold": 0.8}, "depends_on": [2]},
            {"fn": "is_within_xy", "args": {"obj1": "grapes", "obj2": "bowl", "percent_threshold": 0.8}, "depends_on": [3]},
        ],
    },
    "block_stack_kitchen": {
        "gym_id": "DROID-BlockStackKitchen",
        "usd_scene_relpath": "block_stack_kitchen/scene.usda",
        "rubric_source": "src/polaris/environments/__init__.py:18-38",
        "rubric_criteria": [
            {"fn": "reach", "args": {"obj": "green_cube", "threshold": 0.2}, "depends_on": []},
            {"fn": "reach", "args": {"obj": "wood_cube", "threshold": 0.2}, "depends_on": []},
            {"fn": "lift", "args": {"obj": "green_cube", "default_height": 0.06, "threshold": 0.03}, "depends_on": [0]},
            {"fn": "lift", "args": {"obj": "wood_cube", "default_height": 0.06, "threshold": 0.03}, "depends_on": [1]},
            {"fn": "is_within_xy", "args": {"obj1": "green_cube", "obj2": "tray", "percent_threshold": 0.8}, "depends_on": [2]},
            {"fn": "is_within_xy", "args": {"obj1": "wood_cube", "obj2": "tray", "percent_threshold": 0.8}, "depends_on": [3]},
            {"fn": "is_within_xy", "args": {"obj1": "green_cube", "obj2": "wood_cube", "percent_threshold": 0.5}, "depends_on": [4, 5]},
        ],
    },
    "pan_clean": {
        "gym_id": "DROID-PanClean",
        "usd_scene_relpath": "pan_clean/scene.usda",
        "rubric_source": "src/polaris/environments/__init__.py:65-81",
        "rubric_criteria": [
            {"fn": "reach", "args": {"obj": "sponge", "threshold": 0.2}, "depends_on": []},
            {"fn": "lift", "args": {"obj": "sponge", "threshold": 0.09, "default_height": 0.0}, "depends_on": [0]},
            {"fn": "is_within_xy", "args": {"obj1": "sponge", "obj2": "pan", "percent_threshold": 0.8}, "depends_on": [1]},
        ],
    },
    "move_latte_cup": {
        "gym_id": "DROID-MoveLatteCup",
        "usd_scene_relpath": "move_latte_cup/scene.usda",
        "rubric_source": "src/polaris/environments/__init__.py:84-100",
        "rubric_criteria": [
            {"fn": "reach", "args": {"obj": "latteartcup_eval", "threshold": 0.2}, "depends_on": []},
            {"fn": "lift", "args": {"obj": "latteartcup_eval", "threshold": 0.04}, "depends_on": [0]},
            {"fn": "is_within_xy", "args": {"obj1": "latteartcup_eval", "obj2": "cuttingboard_eval", "percent_threshold": 0.8}, "depends_on": [1]},
        ],
    },
    "organize_tools": {
        "gym_id": "DROID-OrganizeTools",
        "usd_scene_relpath": "organize_tools/scene.usda",
        "rubric_source": "src/polaris/environments/__init__.py:102-118",
        "rubric_criteria": [
            {"fn": "reach", "args": {"obj": "scissor", "threshold": 0.2}, "depends_on": []},
            {"fn": "lift", "args": {"obj": "scissor", "threshold": 0.04}, "depends_on": [0]},
            {"fn": "is_within_xy", "args": {"obj1": "scissor", "obj2": "container_01", "percent_threshold": 0.8}, "depends_on": [1]},
        ],
    },
    "tape_into_container": {
        "gym_id": "DROID-TapeIntoContainer",
        "usd_scene_relpath": "tape_into_container/scene.usda",
        "rubric_source": "src/polaris/environments/__init__.py:120-136",
        "rubric_criteria": [
            {"fn": "reach", "args": {"obj": "tape_00", "threshold": 0.2}, "depends_on": []},
            {"fn": "lift", "args": {"obj": "tape_00", "threshold": 0.04}, "depends_on": [0]},
            {"fn": "is_within_xy", "args": {"obj1": "tape_00", "obj2": "container_02", "percent_threshold": 0.8}, "depends_on": [1]},
        ],
    },
}

RUBRIC_NOTE_VS_SIMANY = (
    "STRUCTURALLY DIFFERENT from SimAny's rubric, not just re-labeled: "
    "PolaRiS's Rubric (rubrics/base.py) takes a variable-length list of "
    "criteria (3-7 depending on task) with an explicit dependency graph "
    "between them, and progress = (count of criteria EVER satisfied) / "
    "(total criteria) -- success requires ALL criteria ever-satisfied. "
    "SimAny's TaskScorer (robo/tasks/pi05_tasks.py) is a fixed 4-stage "
    "{grasp,lift,hover,place} rubric at 0.25 credit/stage "
    "(configs/experiments/frozen_fields.yaml: rubric). These are NOT "
    "directly comparable progress scores without an explicit declared "
    "mapping (e.g. PolaRiS's reach+lift roughly correspond to SimAny's "
    "grasp+lift; PolaRiS has no 'hover' analog; PolaRiS's is_within_xy "
    "roughly corresponds to SimAny's place, but checked continuously "
    "against a persistent max-ever-reached flag rather than a 1s-hold "
    "window). task_adapter.py would need to either (a) implement one "
    "rubric on both sides, or (b) declare and test an explicit progress "
    "mapping -- neither is done here."
)

_CAMERA_PRIM_RE = re.compile(
    r'def Camera "(?P<name>[^"]+)"[^{]*\{(?P<body>.*?)\n    \}',
    re.DOTALL,
)
_FLOAT_RE = r"[-+0-9.eE]+"
_ATTR_RES = {
    "focal_length": re.compile(rf"float focalLength = ({_FLOAT_RE})"),
    "focus_distance": re.compile(rf"float focusDistance = ({_FLOAT_RE})"),
    "horizontal_aperture": re.compile(rf"float horizontalAperture = ({_FLOAT_RE})"),
    "vertical_aperture": re.compile(rf"float verticalAperture = ({_FLOAT_RE})"),
    "translate": re.compile(
        rf"double3 xformOp:translate = \(({_FLOAT_RE}), ({_FLOAT_RE}), ({_FLOAT_RE})\)"
    ),
    "orient": re.compile(
        rf"quatd xformOp:orient = \(({_FLOAT_RE}), ({_FLOAT_RE}), ({_FLOAT_RE}), ({_FLOAT_RE})\)"
    ),
}


def extract_camera_prims(usda_path: Path) -> list[dict[str, Any]]:
    """Regex-extract top-level `def Camera "..."` prims from a USD ASCII
    (.usda) file's text. No USD library needed -- .usda is plain text.

    This only reliably finds cameras declared at the SAME nesting depth as
    the pattern below (one closing brace at 4-space indent); PolaRiS's task
    scenes all declare `external_cam` directly under `/World`, matching this
    shape (verified by hand against all 6 PolaRiS-Hub task scenes on
    2026-08-16). A nested-camera scene would need a real USD parser -- this
    is intentionally the simple, auditable version, not a general USD
    reader.
    """
    text = usda_path.read_text()
    cams = []
    for m in _CAMERA_PRIM_RE.finditer(text):
        name = m.group("name")
        body = m.group("body")
        cam: dict[str, Any] = {"name": name}
        for key, pat in _ATTR_RES.items():
            am = pat.search(body)
            if am is None:
                continue
            groups = [float(g) for g in am.groups()]
            cam[key] = groups[0] if len(groups) == 1 else groups
        if "focal_length" in cam and "horizontal_aperture" in cam:
            cam["fov_x_deg"] = math.degrees(
                2 * math.atan(cam["horizontal_aperture"] / (2 * cam["focal_length"]))
            )
        if "focal_length" in cam and "vertical_aperture" in cam:
            cam["fov_y_deg"] = math.degrees(
                2 * math.atan(cam["vertical_aperture"] / (2 * cam["focal_length"]))
            )
        cams.append(cam)
    return cams


def load_initial_conditions(task_dir: Path) -> dict[str, Any]:
    path = task_dir / "initial_conditions.json"
    data = json.loads(path.read_text())
    if "instruction" not in data or "poses" not in data:
        raise ValueError(f"{path}: missing 'instruction' or 'poses' key")
    return {
        "language_instruction": data["instruction"],
        "num_initial_condition_sets": len(data["poses"]),
        "first_initial_condition_set": data["poses"][0],
        "objects_in_first_set": sorted(data["poses"][0].keys()),
    }


def build_task_manifest(task_name: str) -> dict[str, Any]:
    if task_name not in TASK_REGISTRY:
        raise KeyError(
            f"unknown task {task_name!r}; known tasks: {sorted(TASK_REGISTRY)}"
        )
    reg = TASK_REGISTRY[task_name]
    task_dir = HUB_ROOT / Path(reg["usd_scene_relpath"]).parent
    usda_path = HUB_ROOT / reg["usd_scene_relpath"]

    if not usda_path.exists():
        raise FileNotFoundError(
            f"{usda_path} not found -- run "
            "`uvx hf download owhan/PolaRiS-Hub --repo-type=dataset "
            "--local-dir third_party/PolaRiS/PolaRiS-Hub` first"
        )

    scene_cameras = extract_camera_prims(usda_path)
    ic = load_initial_conditions(task_dir)

    if scene_cameras:
        camera_note = (
            f"{len(scene_cameras)} camera prim(s) declared directly in "
            f"{reg['usd_scene_relpath']}; external_cam pose/intrinsics come "
            "from the USD prim itself (droid_cfg.py:143-152 reads pose "
            "only and passes spawn=None, i.e. reuses the prim's own lens "
            "attributes)."
        )
    else:
        camera_note = (
            "NO camera prim found in this task's scene.usda -- "
            "droid_cfg.py:172-189 falls back to a hardcoded default "
            "external_cam (focal_length=1.0476, h_aperture=2.5452, "
            "v_aperture=1.4721, pos=(-0.01,-0.33,0.48), "
            "rot=(0.76,0.43,-0.24,-0.42))."
        )

    return {
        "extraction_provenance": {
            "method": "static source/asset read, NO Isaac Sim launch",
            "reason": (
                "Isaac Sim 5.1.0 camera-enabled headless launch segfaults "
                "on this cluster -- see docs/POLARIS_INTEGRATION.md"
            ),
            "validated_against_live_environment": False,
            "polaris_repo": "third_party/PolaRiS (untracked vendored clone)",
        },
        "task_id": task_name,
        "gym_id": reg["gym_id"],
        "usd_scene_relpath": reg["usd_scene_relpath"],
        "language_instruction": ic["language_instruction"],
        "num_initial_condition_sets": ic["num_initial_condition_sets"],
        "objects_in_scene": ic["objects_in_first_set"],
        "example_initial_condition": ic["first_initial_condition_set"],
        "cameras": {
            "declared_in_scene_usda": scene_cameras,
            "note": camera_note,
        },
        "robot": ROBOT_STATIC_FACTS,
        "action": ACTION_STATIC_FACTS,
        "control": CONTROL_STATIC_FACTS,
        "rubric": {
            "source": reg["rubric_source"],
            "criteria": reg["rubric_criteria"],
            "note_vs_simany": RUBRIC_NOTE_VS_SIMANY,
        },
    }


def _dump_yaml(data: dict, out_path: Path) -> None:
    import yaml

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data, sort_keys=False, width=100))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", choices=sorted(TASK_REGISTRY), help="single task to extract")
    group.add_argument("--all", action="store_true", help="extract every registered task")
    ap.add_argument("--out", type=Path, help="output YAML path (with --task)")
    ap.add_argument("--out-dir", type=Path, help="output directory (with --all)")
    args = ap.parse_args(argv)

    if args.task:
        out = args.out or (REPO_ROOT / "configs" / "polaris" / "tasks" / f"{args.task}.yaml")
        manifest = build_task_manifest(args.task)
        _dump_yaml(manifest, out)
        print(f"wrote {out}")
    else:
        out_dir = args.out_dir or (REPO_ROOT / "configs" / "polaris" / "tasks")
        for task_name in sorted(TASK_REGISTRY):
            manifest = build_task_manifest(task_name)
            out = out_dir / f"{task_name}.yaml"
            _dump_yaml(manifest, out)
            print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
