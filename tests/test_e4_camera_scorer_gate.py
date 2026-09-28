from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from robo.eval import e4_camera_scorer_gate as gate


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_upstream_and_runtime_anchors_are_exact():
    assert gate.EXPECTED_CPU_FREEZE_ID.endswith("20260904T135257Z")
    assert gate.EXPECTED_CPU_PRODUCER_COMMIT == "0dda134578b25cd12d1791193bb437c6d310776f"
    assert gate.EXPECTED_CPU_GATE_SHA256 == "b4bb242ace4ce641051e58df003d126a1383ae0ddd003e3fe4dce3e0493ef355"
    assert gate.EXPECTED_CPU_CONFIG_SHA256 == "fea0c39c9fd789122e75721d9283f1d1c254835e8640c67267271d785d9f5b26"
    assert gate.EXPECTED_MENAGERIE_CLOSURE_SHA256 == "a275b73fb3d0b5fc52abbeb1cd58577c01a447c13ff5ba8d3f51199f27e7e46f"


def test_runtime_menagerie_copy_has_exact_recursive_closure():
    snapshot = gate._menagerie_snapshot(
        gate.EXPECTED_MENAGERIE_ROOT, gate.EXPECTED_MENAGERIE_COMMIT
    )
    assert snapshot["root"] == str(gate.CODE_ROOT / "third_party/mujoco_menagerie")
    assert snapshot["file_count"] == 94
    assert snapshot["size_bytes"] == 40_901_071
    assert snapshot["closure_sha256"] == gate.EXPECTED_MENAGERIE_CLOSURE_SHA256
    assert not any(Path(row["path"]).is_absolute() for row in snapshot["files"])


@pytest.mark.skipif(not gate.EXPECTED_OPENPI_ROOT.is_dir(), reason="sealed openpi checkout is unavailable")
def test_openpi_resize_source_is_clean_and_pinned():
    snapshot = gate._openpi_snapshot()
    assert snapshot["commit"] == gate.EXPECTED_OPENPI_COMMIT
    assert snapshot["dirty"] is False
    assert snapshot["image_tools"]["sha256"] == gate.EXPECTED_OPENPI_IMAGE_TOOLS_SHA256


@pytest.mark.parametrize("value", ["../escape", "/absolute", "bad id", "", "a/b"])
def test_gate_id_rejects_unsafe_path_spelling(value):
    with pytest.raises(gate.CameraScorerGateError):
        gate._validated_id(value, label="gate")


def test_atomic_directory_publishes_only_on_success(tmp_path):
    destination = tmp_path / "result"
    with gate._atomic_directory(destination) as staging:
        (staging / "member.txt").write_text("sealed", encoding="utf-8")
        assert not destination.exists()
    assert (destination / "member.txt").read_text(encoding="utf-8") == "sealed"
    with pytest.raises(gate.CameraScorerGateError):
        with gate._atomic_directory(destination):
            pass


def test_atomic_directory_removes_failed_staging(tmp_path):
    destination = tmp_path / "failed"
    with pytest.raises(RuntimeError):
        with gate._atomic_directory(destination) as staging:
            (staging / "partial").write_text("x", encoding="utf-8")
            raise RuntimeError("stop")
    assert not destination.exists()
    assert not list(tmp_path.glob(".failed.tmp.*"))


def test_reach_envelope_has_three_unique_checks_and_expected_margins():
    row = gate.reach_envelope([0.50, 0.0], [0.0, 0.0], 0.0)
    assert row["passed"] is True
    assert set(row["checks"]) == {
        "outside_inner_radius", "inside_outer_radius", "inside_front_cone"
    }
    assert row["margins"] == pytest.approx(
        {"front_cone_deg": 60.0, "inner_radius_m": 0.25, "outer_radius_m": 0.30}
    )


@pytest.mark.parametrize(
    "point",
    [[0.20, 0.0], [0.81, 0.0], [0.0, 0.50]],
)
def test_reach_envelope_fails_inner_outer_or_cone(point):
    assert gate.reach_envelope(point, [0.0, 0.0], 0.0)["passed"] is False


def test_module_has_no_duplicate_literal_dictionary_keys():
    tree = ast.parse(Path(gate.__file__).read_text(encoding="utf-8"))
    duplicates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [key.value for key in node.keys if isinstance(key, ast.Constant)]
        repeated = sorted({key for key in keys if keys.count(key) > 1})
        if repeated:
            duplicates.append((node.lineno, repeated))
    assert duplicates == []


def test_policy_mask_resize_has_exact_224_letterbox_geometry():
    mask = np.zeros((360, 640), dtype=bool)
    mask[100:120, 200:240] = True
    resized = gate._policy_resize_mask(mask)
    assert resized.shape == (224, 224)
    # 16:9 input becomes 224x126, centered with 49 top rows.
    assert not resized[:49].any()
    assert not resized[175:].any()
    assert resized.any()


def test_policy_visibility_threshold_is_predeclared_nine_pixels_three_by_three():
    passing = np.zeros((224, 224), dtype=bool)
    passing[10:13, 20:23] = True
    result = gate._mask_measurement(passing)
    assert result["pixels"] == 9
    assert all(result["checks"].values())
    narrow = np.zeros((224, 224), dtype=bool)
    narrow[10:19, 20] = True
    result = gate._mask_measurement(narrow)
    assert result["pixels"] == 9
    assert result["checks"]["bbox_width_at_least_3"] is False


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("move the leftmost stapler to the left side", "leftmost"),
        ("move the rightmost stapler to the right side", "rightmost"),
        ("move the nearest mouse", "nearest"),
        ("move the farthest mouse", "farthest"),
        ("move the stapler", None),
    ],
)
def test_task_qualifier_parsing(instruction, expected):
    task = {"instructions": {"default": instruction}}
    assert gate._task_qualifier(task) == expected


class _PoseEnv:
    def __init__(self, poses):
        self._poses = poses
        self.free_bodies = sorted(poses)

    def body_pose(self, name):
        return np.asarray(self._poses[name], dtype=float), np.asarray([1, 0, 0, 0])


def _graspable_row(name, label="stapler"):
    return {
        "name": name,
        "label": label,
        "tier": "A",
        "dims": np.asarray([0.10, 0.05, 0.03]),
        "mass": 0.3,
        "drift": 0.0,
    }


def test_disambiguation_margin_is_strictly_greater_than_30mm():
    suite = {"robot": {"base_pos": [0, 0, 0], "base_yaw": 0.0}}
    task = {
        "target": "obj_02",
        "target_label": "stapler",
        "any_instance": False,
        "instructions": {"default": "move the leftmost stapler"},
    }
    rows = [_graspable_row("obj_02"), _graspable_row("obj_03")]
    at_threshold = _PoseEnv({"obj_02": [0.5, 0.03, 0.8], "obj_03": [0.5, 0.0, 0.8]})
    assert gate.disambiguation_metric(at_threshold, task, suite, rows, {})["passed"] is False
    above = _PoseEnv({"obj_02": [0.5, 0.031, 0.8], "obj_03": [0.5, 0.0, 0.8]})
    assert gate.disambiguation_metric(above, task, suite, rows, {})["passed"] is True


def test_825_a0_extra_stapler_fails_leftmost_contract():
    suite = {"robot": {"base_pos": [0, 0, 0], "base_yaw": 0.0}}
    task = {
        "target": "obj_02",
        "target_label": "stapler",
        "any_instance": False,
        "instructions": {"default": "move the leftmost stapler"},
    }
    env = _PoseEnv(
        {"obj_01": [0.5, 0.08, 0.8], "obj_02": [0.5, 0.0, 0.8], "obj_03": [0.5, -0.04, 0.8]}
    )
    result = gate.disambiguation_metric(
        env, task, suite, [_graspable_row(name) for name in env.free_bodies], {}
    )
    assert result["competitor"] == "obj_01"
    assert result["observed_margin_m"] == pytest.approx(-0.08)
    assert result["passed"] is False


def test_headless_env_reuses_exact_900_step_reset_without_renderer(monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    from robo.rigs import pi05_rig

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.0016666666666666666"/>
          <worldbody>
            <geom name="floor" type="plane" size="2 2 .1"/>
            <body name="robot/link0" pos="0 0 .2">
              <joint name="j0" type="hinge" axis="0 0 1"/>
              <geom type="capsule" size=".02 .03"/>
              <body name="robot/link1" pos="0 0 .08">
                <joint name="j1" type="hinge" axis="0 1 0"/>
                <geom type="capsule" size=".02 .03"/>
                <body name="robot/link2" pos="0 0 .08">
                  <joint name="j2" type="hinge" axis="1 0 0"/>
                  <geom type="capsule" size=".02 .03"/>
                  <body name="robot/link3" pos="0 0 .08">
                    <joint name="j3" type="hinge" axis="0 1 0"/>
                    <geom type="capsule" size=".02 .03"/>
                    <body name="robot/link4" pos="0 0 .08">
                      <joint name="j4" type="hinge" axis="1 0 0"/>
                      <geom type="capsule" size=".02 .03"/>
                      <body name="robot/link5" pos="0 0 .08">
                        <joint name="j5" type="hinge" axis="0 1 0"/>
                        <geom type="capsule" size=".02 .03"/>
                        <body name="robot/link6" pos="0 0 .08">
                          <joint name="j6" type="hinge" axis="1 0 0"/>
                          <geom name="robot/robotiq_2f85_left_pad" size=".02"/>
                          <geom name="robot/robotiq_2f85_right_pad" size=".02" pos=".05 0 0"/>
                          <body name="robot/gripper" pos="0 0 .08">
                            <joint name="gripper_joint" type="hinge"/>
                            <geom size=".02"/>
                          </body>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
            <body name="obj_target" pos=".5 0 .1">
              <freejoint/><geom size=".03"/>
            </body>
          </worldbody>
          <actuator>
            <position name="a0" joint="j0"/><position name="a1" joint="j1"/>
            <position name="a2" joint="j2"/><position name="a3" joint="j3"/>
            <position name="a4" joint="j4"/><position name="a5" joint="j5"/>
            <position name="a6" joint="j6"/>
            <position name="gripper_actuator" joint="gripper_joint"/>
          </actuator>
        </mujoco>
        """
    )
    # Match pi05_rig's Python-assigned value exactly.  On this runtime the
    # naive quotient truncates to 899, which is the regression under test.
    model.opt.timestep = 1.0 / 600.0
    info = {
        "arm_joints": [f"j{index}" for index in range(7)],
        "arm_actuators": [f"a{index}" for index in range(7)],
        "gripper_actuator": "gripper_actuator",
        "gripper_driver_joint": "gripper_joint",
        "home": np.zeros(7),
    }
    monkeypatch.setattr(pi05_rig, "build_scene_model", lambda *_args, **_kwargs: (model, info))

    def reject_renderer(*_args, **_kwargs):
        raise AssertionError("qualifier preflight must never construct Renderer")

    monkeypatch.setattr(mujoco, "Renderer", reject_renderer)
    env = gate._build_headless_droid_env(
        scene_xml="unused.xml",
        base_pos=[0, 0, 0],
        base_yaw=0.0,
        table_box={},
        ext_cam={},
        exclude_objects=(),
        menagerie_root=Path("/unused"),
    )
    observation = env.reset(
        settle_s=1.5,
        jitter_body="obj_target",
        jitter_xy=0.01,
        jitter_uniform_draw=np.asarray([0.5, 0.5]),
    )
    assert observation == {}
    assert not hasattr(env, "renderer")
    assert env.last_reset_provenance["settle_protocol"]["step_count"] == 900


@pytest.mark.parametrize(
    ("duration_s", "timestep_s", "expected"),
    [(0.0, 1.0 / 600.0, 0), (0.1, 1.0 / 600.0, 60), (1.5, 1.0 / 600.0, 900)],
)
def test_settle_step_count_repairs_near_integer_float_drift(
    duration_s, timestep_s, expected
):
    from robo.envs.pi05_env import _settle_step_count

    assert _settle_step_count(duration_s, timestep_s) == expected


@pytest.mark.parametrize(
    ("duration_s", "timestep_s"),
    [(-0.1, 1.0 / 600.0), (float("nan"), 1.0 / 600.0),
     (1.5, 0.0), (1.5, float("inf")), (0.1001, 1.0 / 600.0)],
)
def test_settle_step_count_rejects_invalid_or_off_grid_duration(
    duration_s, timestep_s
):
    from robo.envs.pi05_env import _settle_step_count

    with pytest.raises(ValueError):
        _settle_step_count(duration_s, timestep_s)


@pytest.mark.parametrize(
    (
        "half_separation_m", "settle_steps", "expected_allowed",
        "expected_qualifier_passed", "expected_settle_passed",
    ),
    [
        (0.05, 900, True, 40, 40),
        (0.01, 900, False, 0, 40),
        (0.05, 899, False, 40, 0),
    ],
)
def test_cpu_qualifier_preflight_covers_40_cells_and_seals_result(
    monkeypatch, tmp_path, half_separation_m, settle_steps, expected_allowed,
    expected_qualifier_passed, expected_settle_passed
):
    from robo.tasks import pi05_tasks

    suites = {}
    factories = {}
    rows_by_scene = {}
    for scene_id in gate.SCENE_IDS:
        task_ids = list(gate.cpu_pilot.expected_task_ids(scene_id))
        first_target = task_ids[0].split("__", 1)[1].removesuffix("_to_region")
        task_rows = [
            _graspable_row(first_target),
            _graspable_row("obj_03"),
        ]
        rows_by_scene[scene_id] = task_rows
        tasks = [
            {
                "task_id": task_ids[0],
                "target": first_target,
                "target_label": "stapler",
                "any_instance": False,
                "instructions": {"default": "move the leftmost stapler"},
            },
            {
                "task_id": task_ids[1],
                "target": "obj_03",
                "target_label": "stapler",
                "any_instance": False,
                "instructions": {"default": "move the rightmost stapler"},
            },
        ]
        suites[scene_id] = {
            policy: {
                "scene_xml": scene_id,
                "robot": {"base_pos": [0, 0, 0], "base_yaw": 0.0},
                "table": {},
                "ext_cam": {},
                "exclude_objects": [],
                "tasks": tasks,
            }
            for policy in gate.POLICIES
        }
        factories[scene_id] = {
            policy: Path(f"{scene_id}-{policy}") for policy in gate.POLICIES
        }

    class FakeHeadlessEnv:
        def __init__(self, scene_id):
            first = rows_by_scene[scene_id][0]["name"]
            self.free_bodies = [first, "obj_03"]
            self.first = first
            self.last_reset_provenance = None
            self.positions = {}

        def reset(
            self, *, settle_s, jitter_body, jitter_xy,
            jitter_uniform_draw, reset_seed
        ):
            self.positions = {
                self.first: np.asarray([0.5, half_separation_m, 0.8]),
                "obj_03": np.asarray([0.5, -half_separation_m, 0.8]),
            }
            offset = (2.0 * np.asarray(jitter_uniform_draw) - 1.0) * jitter_xy
            self.positions[jitter_body] = self.positions[jitter_body].copy()
            self.positions[jitter_body][:2] += offset
            self.last_reset_provenance = {
                "jitter": {
                    "reset_seed": reset_seed,
                    "body": jitter_body,
                    "max_abs_xy_m": jitter_xy,
                    "uniform_draw_0_1": np.asarray(jitter_uniform_draw).tolist(),
                    "offset_xy_m": offset.tolist(),
                },
                "settle_protocol": {
                    "engine": "mujoco",
                    "step_function": "mujoco.mj_step",
                    "requested_duration_s": settle_s,
                    "model_timestep_s": 1.0 / 600.0,
                    "step_count": settle_steps,
                    "simulated_duration_s": settle_steps / 600.0,
                },
            }
            return {}

        def body_pose(self, name):
            return self.positions[name], np.asarray([1.0, 0.0, 0.0, 0.0])

    commit = "a" * 40
    snapshot = {
        "code_root": str(gate.CODE_ROOT),
        "commit": commit,
        "dirty": False,
    }
    upstream = {
        "factories": factories,
        "summary": {"closure": "sealed"},
        "suites": suites,
    }
    menagerie = {
        "closure_sha256": "b" * 64,
        "file_count": 94,
        "files": [],
        "root": str(tmp_path),
        "size_bytes": 40_901_071,
        "source_commit": gate.EXPECTED_MENAGERIE_COMMIT,
    }
    monkeypatch.setattr(gate, "_git_snapshot", lambda _commit: snapshot)
    monkeypatch.setattr(gate, "_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(gate, "_menagerie_snapshot", lambda *_args: menagerie)
    monkeypatch.setattr(gate, "validate_cpu_chain", lambda **_kwargs: upstream)
    monkeypatch.setattr(
        gate,
        "_build_headless_droid_env",
        lambda **kwargs: FakeHeadlessEnv(kwargs["scene_xml"]),
    )
    monkeypatch.setattr(
        pi05_tasks,
        "_load_objects",
        lambda factory: rows_by_scene[factory.name.split("-", 1)[0]],
    )

    gate_id = f"candidate-{int(expected_allowed)}-{settle_steps}"
    result = gate.run_qualifier_preflight(
        gate_id=gate_id,
        cpu_freeze_id=gate.EXPECTED_CPU_FREEZE_ID,
        cpu_producer_commit=gate.EXPECTED_CPU_PRODUCER_COMMIT,
        expected_cpu_gate_sha256=gate.EXPECTED_CPU_GATE_SHA256,
        expected_code_commit=commit,
        menagerie_root=tmp_path,
        expected_menagerie_commit=gate.EXPECTED_MENAGERIE_COMMIT,
    )
    output = tmp_path / "outputs/icra2027" / f"{gate_id}-qualifier-preflight"
    assert result["cell_coverage"]["cells"] == 40
    assert result["camera_job_submission_allowed"] is expected_allowed
    assert result["real_policy_infra_smoke_allowed"] is False
    assert result["renderer_constructed"] is False
    assert result["settle_contract_cells_passed"] == expected_settle_passed
    assert result["settle_contract_cells_failed"] == 40 - expected_settle_passed
    assert result["qualifier_margin_cells_passed"] == expected_qualifier_passed
    assert result["qualifier_margin_cells_failed"] == 40 - expected_qualifier_passed
    if settle_steps == 899:
        assert all(
            issue.startswith("settle_contract_failed:")
            for issue in result["blocking_issues"]
        )
    assert (output / "qualifier_metrics.jsonl").read_text().count("\n") == 40
    assert (output / "gate.json").is_file()
    assert (output / "manifest.json").is_file()
    assert (output / "seal.json").is_file()


def test_semantic_segmentation_preserves_target_and_each_distractor():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody>
          <geom name="tabletop" type="box" size="1 1 .1"/>
          <body name="obj_target"><freejoint/><geom name="target_geom" size=".1"/></body>
          <body name="obj_other"><freejoint/><geom name="other_geom" size=".1"/></body>
          <body name="robot/link"><geom name="robot_geom" size=".1"/></body>
        </worldbody></mujoco>
        """
    )
    env = type("Env", (), {"model": model})()
    segmentation = np.full((360, 640, 2), -1, dtype=np.int32)
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    target_id = int(model.geom("target_geom").id)
    other_id = int(model.geom("other_geom").id)
    robot_id = int(model.geom("robot_geom").id)
    table_id = int(model.geom("tabletop").id)
    for rows, columns, geom_id in (
        (slice(10, 20), slice(10, 20), target_id),
        (slice(20, 30), slice(20, 30), other_id),
        (slice(30, 40), slice(30, 40), robot_id),
        (slice(40, 50), slice(40, 50), table_id),
    ):
        segmentation[rows, columns, 0] = geom_id
        segmentation[rows, columns, 1] = geom_type
    semantic, masks = gate.semantic_segmentation(
        env, segmentation, "obj_target", ["obj_other"]
    )
    assert masks["target"].sum() == 100
    assert masks["obj_other"].sum() == 100
    assert (semantic == 1).sum() == 100
    assert (semantic == 5).sum() == 100
    assert (semantic == 2).sum() == 100
    assert (semantic == 3).sum() == 100


def test_workspace_gate_erodes_rotated_extent_and_requires_projection(monkeypatch):
    env = _PoseEnv({"obj_02": [0.50, 0.0, 0.80]})
    env.info = {"ext_cam": "ext_cam"}
    suite = {
        "robot": {"base_pos": [0, 0, 0], "base_yaw": 0.0},
    }
    task = {
        "target": "obj_02",
        "receptacle": None,
        "region": {"cx": 0.50, "cy": 0.30, "hx": 0.20, "hy": 0.20, "zlo": 0.7, "zhi": 1.0},
    }
    monkeypatch.setattr(
        gate,
        "_camera_pose",
        lambda *_args: {
            "fovy_deg": 60.0,
            "position_world_m": [0, 0, 1],
            "rotation_camera_to_world": np.eye(3).tolist(),
        },
    )
    monkeypatch.setattr(
        gate,
        "_project_with_depth",
        lambda *_args: {"visible": True},
    )
    result = gate.workspace_metric(
        env,
        task,
        suite,
        {"dims": np.asarray([0.10, 0.20, 0.05])},
        np.ones((360, 640), dtype=np.float32),
        {},
    )
    assert result["eroded_goal_core"]["nonempty"] is True
    assert result["eroded_goal_core"]["target_world_xy_half_diagonal_m"] == pytest.approx(
        np.hypot(0.05, 0.10)
    )
    assert result["exterior_projection"]["visible_count"] == 9
    assert result["passed"] is True


def test_scorer_positive_and_four_negative_replays_restore_state():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody>
          <body name="obj_target" pos=".5 0 .8"><freejoint/><geom size=".03"/></body>
          <body name="obj_wrong" pos=".6 0 .8"><freejoint/><geom size=".03"/></body>
        </worldbody></mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    class Env:
        free_bodies = ["obj_target", "obj_wrong"]

        def __init__(self):
            self.model, self.data = model, data
            self.start_pose = {
                name: self.body_pose(name) for name in self.free_bodies
            }

        def body_pose(self, name):
            body_id = self.model.body(name).id
            return self.data.xpos[body_id].copy(), self.data.xquat[body_id].copy()

        def grasped(self, _name):
            return False

        def at_rest(self, _name):
            return True

        def lifted(self, name, min_dz=0.05):
            return self.body_pose(name)[0][2] > self.start_pose[name][0][2] + min_dz

    env = Env()
    before_qpos = env.data.qpos.copy()
    result = gate.scorer_replay(
        env,
        {
            "target": "obj_target",
            "target_label": "thing",
            "any_instance": False,
            "receptacle": None,
            "region": {"cx": 0.5, "cy": 0.3, "hx": 0.1, "hy": 0.1, "zlo": 0.7, "zhi": 1.1},
        },
        {},
    )
    assert result["passed"] is True
    assert len(result["negative_replays"]) == 4
    assert result["oracle_validated"] is False
    assert np.array_equal(env.data.qpos, before_qpos)
    assert env.grasped("obj_target") is False


def test_pairing_requires_twenty_complete_a0_a4_pairs():
    observations = {}
    pose = {
        "fovy_deg": 60.0,
        "position_world_m": [0, 0, 0],
        "rotation_camera_to_world": np.eye(3).tolist(),
    }
    for scene_index, scene in enumerate(gate.SCENE_IDS):
        for task_index in range(2):
            task = f"{scene}__task{task_index}"
            for episode in range(5):
                reset = f"{task}__seed0__ep{episode}"
                observations[(scene, task, episode)] = {
                    policy: {
                        "cameras": {"exterior": pose, "wrist": pose},
                        "jitter": {"offset": [episode, scene_index]},
                        "reset_seed": episode,
                        "reset_state_id": reset,
                    }
                    for policy in gate.POLICIES
                }
    rows = gate._pairing_rows(observations)
    assert len(rows) == 20
    assert all(row["passed"] for row in rows)


def test_cpu_chain_rejects_unpinned_freeze_before_reading(tmp_path):
    with pytest.raises(gate.CameraScorerGateError, match="CPU freeze differs"):
        gate.validate_cpu_chain(
            root=tmp_path,
            cpu_freeze_id="wrong",
            cpu_producer_commit=gate.EXPECTED_CPU_PRODUCER_COMMIT,
            expected_cpu_gate_sha256=gate.EXPECTED_CPU_GATE_SHA256,
        )


def test_main_returns_three_for_sealed_semantic_failure(monkeypatch):
    monkeypatch.setattr(
        gate,
        "run_gate",
        lambda **_kwargs: {"real_policy_infra_smoke_allowed": False, "paper_ready": False},
    )
    args = [
        "--gate-id", "fresh", "--cpu-freeze-id", "cpu",
        "--cpu-producer-commit", "0" * 40,
        "--expected-cpu-gate-sha256", "0" * 64,
        "--expected-code-commit", "1" * 40,
        "--menagerie-root", "/tmp/m", "--expected-menagerie-commit", "2" * 40,
    ]
    assert gate.main(args) == 3


def test_main_returns_four_for_sealed_cpu_qualifier_failure(monkeypatch):
    monkeypatch.setattr(
        gate,
        "run_qualifier_preflight",
        lambda **_kwargs: {
            "camera_job_submission_allowed": False,
            "real_policy_infra_smoke_allowed": False,
        },
    )
    args = [
        "--qualifier-preflight-only",
        "--gate-id", "fresh", "--cpu-freeze-id", "cpu",
        "--cpu-producer-commit", "0" * 40,
        "--expected-cpu-gate-sha256", "0" * 64,
        "--expected-code-commit", "1" * 40,
        "--menagerie-root", "/tmp/m", "--expected-menagerie-commit", "2" * 40,
    ]
    assert gate.main(args) == 4


def test_submitter_is_one_ordinary_exact_a100_job_without_array_option():
    path = ROOT / "run/icra2027/submit_e4_camera_scorer_noarray.sh"
    text = path.read_text(encoding="utf-8")
    assert text.count('"$SBATCH_BIN" --parsable') == 1
    assert "--array" not in text
    assert text.index("--qualifier-preflight-only") < text.index('"$SBATCH_BIN" --parsable')
    assert "CPU qualifier preflight failed; refusing to call sbatch" in text
    assert "E4_QUALIFIER_PREFLIGHT_GATE_SHA256" in text
    for token in (
        '--nodelist="$GPU_NODE"', '--gpus="$GPU_GRES"',
        "--cpus-per-task=4", "--mem=32G", "--time=01:00:00",
        "--partition=batch", "--qos=normal",
    ):
        assert token in text
    assert "submitted 1 ordinary GPU job (no array)" in text


def test_worker_binds_split_roots_egl_openpi_and_refuses_arrays():
    path = ROOT / "run/slurm/icra2027_e4_camera_scorer_gpu.sbatch"
    text = path.read_text(encoding="utf-8")
    assert "--array" not in text
    assert "CODE_ROOT=${SIMANY_ROOT:-$PWD}/worktrees/e4-paired-pilot" in text
    assert "EVIDENCE_ROOT=${SIMANY_ROOT:-$PWD}" in text
    assert "OPENPI_ROOT=${OPENPI_ROOT:?set OPENPI_ROOT to the openpi checkout}/worktrees/e4-policy-server" in text
    assert 'export MUJOCO_GL=egl' in text
    assert 'export PYOPENGL_PLATFORM=egl' in text
    assert "E4_QUALIFIER_PREFLIGHT_GATE_SHA256" in text
    assert '[[ "${SLURM_CPUS_PER_TASK:-}" == 4 ]]' in text
    assert "EXPECTED_TIME=01:00:00" in text
    assert "exec \"$PYTHON\" -m robo.eval.e4_camera_scorer_gate" in text


def test_gate_source_never_authorizes_paper_or_large_fleet():
    text = Path(gate.__file__).read_text(encoding="utf-8")
    assert '"paper_ready": False' in text
    assert '"headline_eligible": False' in text
    assert '"large_rollout_launch_allowed": False' in text
    assert '"real_policy_infra_smoke_allowed": allowed' in text
