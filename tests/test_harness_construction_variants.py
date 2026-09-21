import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robo.eval import episode_log as elog
from robo.eval import e3_factory_materializer
from robo.eval import e4_task_freeze
from robo.eval import harness_runner
from robo.eval.harness_spec import load_harness_spec
from robo.eval.harness_validation import validate_resolved_scene_treatments


SCENE = "0123456789"


@pytest.fixture(autouse=True)
def _unit_materialization_validator(monkeypatch):
    """Keep harness tests focused; full E3 seal replay has dedicated tests."""
    def validate(factory_dir, *, expected_scene_id, expected_policy_id,
                 repository_root):
        factory = Path(factory_dir)
        manifest_path = factory / "materialization_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["scene_id"] != expected_scene_id:
            raise ValueError("materialization scene binding differs")
        if manifest["policy_id"] != expected_policy_id:
            raise ValueError("materialization policy binding differs")
        roster = manifest["roster"]
        return {
            "e3_claim_status": manifest["e3_claim_status"],
            "e3_code_commit": manifest["e3_code_commit"],
            "e3_freeze_id": manifest["e3_freeze_id"],
            "e3_root": manifest["e3_root"],
            "factory_dir": str(factory),
            "input_identities_sha256": hashlib.sha256(
                json.dumps(manifest["input_identities"], sort_keys=True).encode()
            ).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "policy_id": manifest["policy_id"],
            "roster": roster,
            "scene_id": manifest["scene_id"],
            "source_scene_sha256": hashlib.sha256(
                json.dumps(manifest["source_scene"], sort_keys=True).encode()
            ).hexdigest(),
            "study_scope": manifest["study_scope"],
        }

    monkeypatch.setattr(
        e3_factory_materializer, "validate_materialized_factory", validate
    )

    def validate_bundle(manifest_path, *, expected_scene_id, repository_root):
        path = Path(manifest_path)
        manifest = json.loads(path.read_text())
        if manifest["scene_id"] != expected_scene_id:
            raise ValueError("task-freeze scene binding differs")
        return {
            "bundle_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "planning_tasks": manifest["planning_tasks"],
            "factories": manifest["factories"],
            "variant_tasks": manifest["variant_tasks"],
            "scene_xml": manifest["scene_xml"],
        }

    monkeypatch.setattr(e4_task_freeze, "validate_task_bundle", validate_bundle)


def _task():
    return {
        "task_id": f"{SCENE}__obj_00_to_region",
        "target": "obj_00",
        "receptacle": None,
        "region": {"cx": 0.5, "cy": 0.0, "hx": 0.2, "hy": 0.2,
                   "zlo": 0.7, "zhi": 1.0},
        "instructions": {"default": "move the mug to the marked region"},
    }


def _suite(scene_xml, *, task=None):
    return {
        "scene": SCENE,
        "scene_xml": str(scene_xml),
        "robot": {"base_pos": [0.0, 0.0, 0.7], "base_yaw": 0.0},
        "table": {"cx": 0.5, "cy": 0.0, "hx": 0.5, "hy": 0.5,
                  "top_z": 0.7},
        "ext_cam": {"pos": [0.0, 0.0, 1.0]},
        "exclude_objects": [],
        "tasks": [copy.deepcopy(task or _task())],
    }


def _seal_factory(factory: Path, policy_id: str, repository_root: Path) -> None:
    objects = factory / "objects"
    objects.mkdir()
    objects_json = objects / "objects.json"
    objects_json.write_text("[]\n")
    object_identity = {
        "sha256": hashlib.sha256(objects_json.read_bytes()).hexdigest(),
        "size_bytes": objects_json.stat().st_size,
    }
    manifest = {
        "created_utc": "2026-09-03T00:00:00Z",
        "destination": factory.relative_to(repository_root).as_posix(),
        "e3_claim_status": {
            "freeze_id": "fixture-e3-freeze",
            "headline_eligible": False,
            "paper_ready": False,
            "retry_claim_status": "fixture_preliminary_only",
            "study_scope": "fixture",
        },
        "e3_code_commit": "c" * 40,
        "e3_freeze_id": "fixture-e3-freeze",
        "e3_root": "inputs/e3",
        "input_identities": {},
        "manifest_kind": "e4_construction_variant_materialization",
        "output_members": {"objects/objects.json": object_identity},
        "policy_id": policy_id,
        "roster": {
            "abstained_count": 0,
            "abstained_slots": [],
            "accepted_count": 1,
            "accepted_slots": ["obj_00"],
            "job_count": 1,
            "object_slots": ["obj_00"],
            "sha256": "d" * 64,
        },
        "scene_id": SCENE,
        "schema_version": 1,
        "selected_records": [],
        "source_scene": {},
        "study_scope": "fixture",
    }
    manifest_path = factory / "materialization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    seal = {
        "manifest_kind": "e4_construction_variant_materialization",
        "members": {
            "materialization_manifest.json": {
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "size_bytes": manifest_path.stat().st_size,
            }
        },
        "schema_version": 1,
    }
    (factory / "seal.json").write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n")


def _fixture(tmp_path, *, identical_xml=False, task_override=None):
    planning_path = tmp_path / "logical_tasks.json"
    a0_factory = tmp_path / "a0_factory"
    a4_factory = tmp_path / "a4_factory"
    a0_factory.mkdir()
    a4_factory.mkdir()
    _seal_factory(a0_factory, "A0", tmp_path)
    _seal_factory(a4_factory, "A4", tmp_path)
    a0_xml = a0_factory / "scene.xml"
    a4_xml = a4_factory / "scene.xml"
    a0_xml.write_text('<mujoco model="a0"/>')
    a4_xml.write_text(
        '<mujoco model="a0"/>' if identical_xml else '<mujoco model="a4"/>')
    planning_path.write_text(json.dumps(_suite(a0_xml)))
    a0_tasks = a0_factory / "tasks.json"
    a4_tasks = a4_factory / "tasks.json"
    a0_tasks.write_text(json.dumps(_suite(a0_xml)))
    a4_tasks.write_text(json.dumps(_suite(a4_xml, task=task_override)))
    task_freeze_manifest = tmp_path / "task_freeze_manifest.json"
    task_freeze_manifest.write_text(json.dumps({
        "scene_id": SCENE,
        "planning_tasks": str(planning_path),
        "factories": {"A0": str(a0_factory), "A4": str(a4_factory)},
        "variant_tasks": {"A0": str(a0_tasks), "A4": str(a4_tasks)},
        "scene_xml": {"A0": str(a0_xml), "A4": str(a4_xml)},
    }))
    scene_cfg = {
        "id": SCENE,
        "tasks_json": str(planning_path),
        "task_freeze_manifest": str(task_freeze_manifest),
        "construction_variants": {
            "fixed_single_path": {
                "factory_dir": str(a0_factory),
                "tasks_json": str(a0_tasks),
                "scene_xml": str(a0_xml),
            },
            "agentic": {
                "factory_dir": str(a4_factory),
                "tasks_json": str(a4_tasks),
                "scene_xml": str(a4_xml),
            },
        },
    }
    config = {
        "policy": "scripted_sinusoid",
        "paper_mode": False,
        "contract": {
            "policy": {
                "id": "scripted_sinusoid",
                "kind": "scripted_smoke",
                "checkpoint_hash": None,
                "checkpoint_hash_kind": "not_applicable",
            },
            "robot": {"id": "fixture_robot"},
            "cameras": {"id": "fixture_cameras"},
            "action_convention": "absolute_joint_position",
            "controller": {"id": "fixture_controller"},
            "control_rate_hz": 15,
            "horizon_s": 1,
            "task_instruction": "move the mug to the marked region",
            "rubric": {"id": "fixture_rubric"},
            "reset_ids": [f"{_task()['task_id']}__seed0__ep0"],
        },
        "scenes": [scene_cfg],
        "treatments": [
            {"id": "a0", "scene": "fixed_single_path"},
            {"id": "a4", "scene": "agentic"},
        ],
        "comparisons": [{
            "id": "construction", "axis": "scene",
            "treatments": ["a0", "a4"],
        }],
    }
    return config, scene_cfg


def test_cpu_invalid_arm_never_constructs_environment_and_keeps_all_resets(tmp_path, monkeypatch):
    from robo.eval import e4_reset_eligibility
    config, scene_cfg = _fixture(tmp_path)
    config["episodes"] = 5
    config["contract"]["reset_ids"] = [
        f"{_task()['task_id']}__seed0__ep{ep}" for ep in range(5)]
    monkeypatch.setattr(harness_runner, "ROOT", tmp_path)

    def eligibility(config, spec, states, *, root):
        return {(t.id, s.reset_state_id): {
            "passed": t.id == "a4", "failure_type": "cpu_reset_validity_failure",
            "failed_checks": ["stability_passed"]}
                for t in spec.treatments.values() for s in states}

    monkeypatch.setattr(e4_reset_eligibility, "load_reset_eligibility", eligibility)
    monkeypatch.setattr(e4_reset_eligibility, "certify_prebuild_failure",
        lambda record, spec, root: "construction_validity_evidence" in record)
    constructed = []

    def build(scene, condition, output):
        constructed.append(scene["factory_dir"])
        assert scene["factory_dir"].endswith("a4_factory")
        raise elog.BuildFailureError("test renderer unavailable")

    monkeypatch.setattr(harness_runner.legacy, "build_env", build)
    out = tmp_path / "run"
    harness_runner.run_matrix(config, out)
    rows = harness_runner.read_jsonl(out / harness_runner.LEDGER_NAME)
    assert len(rows) == 10
    assert len(constructed) == 1
    assert sum("construction_validity_evidence" in r for r in rows) == 5
    ledger_before = (out / harness_runner.LEDGER_NAME).read_bytes()
    harness_runner.run_matrix(config, out, resume=True)
    assert len(constructed) == 1
    assert (out / harness_runner.LEDGER_NAME).read_bytes() == ledger_before


def _resolved(scene_cfg, spec, out_dir):
    return {
        treatment.id: harness_runner._scene_config_for_treatment(
            scene_cfg, treatment, out_dir)
        for treatment in spec.treatments.values()
    }


def test_e4_variants_resolve_per_scene_with_shared_logical_tasks(tmp_path):
    config, scene_cfg = _fixture(tmp_path)
    spec = load_harness_spec(config)
    resolved = _resolved(scene_cfg, spec, tmp_path / "run")

    assert resolved["a0"]["factory_dir"].endswith("a0_factory")
    assert resolved["a4"]["factory_dir"].endswith("a4_factory")
    a0_suite = json.loads(Path(resolved["a0"]["tasks_json"]).read_text())
    a4_suite = json.loads(Path(resolved["a4"]["tasks_json"]).read_text())
    assert a0_suite["scene_xml"].endswith("a0_factory/scene.xml")
    assert a4_suite["scene_xml"].endswith("a4_factory/scene.xml")
    assert validate_resolved_scene_treatments(
        scene_cfg, resolved, spec, root=tmp_path)["ok"]


def test_split_root_paths_are_absolute_before_legacy_consumers(tmp_path, monkeypatch):
    config, scene_cfg = _fixture(tmp_path)
    monkeypatch.setattr(harness_runner, "ROOT", tmp_path)
    for variant in scene_cfg["construction_variants"].values():
        variant["factory_dir"] = str(
            Path(variant["factory_dir"]).relative_to(tmp_path)
        )
        variant["tasks_json"] = str(
            Path(variant["tasks_json"]).relative_to(tmp_path)
        )
        variant["scene_xml"] = str(
            Path(variant["scene_xml"]).relative_to(tmp_path)
        )
    scene_cfg["tasks_json"] = str(
        Path(scene_cfg["tasks_json"]).relative_to(tmp_path)
    )
    spec = load_harness_spec(config)

    resolved = _resolved(scene_cfg, spec, tmp_path / "run")
    for treatment in resolved.values():
        assert Path(treatment["factory_dir"]).is_absolute()
        assert Path(treatment["tasks_json"]).is_absolute()
        assert Path(treatment["scene_xml"]).is_absolute()
        assert Path(treatment["factory_dir"]).is_relative_to(tmp_path)

    planner_scenes = harness_runner._scene_configs_for_legacy_planner([scene_cfg])
    assert Path(planner_scenes[0]["tasks_json"]).is_absolute()
    assert Path(planner_scenes[0]["tasks_json"]).is_relative_to(tmp_path)
    # The frozen config object itself is not rewritten for provenance.
    assert not Path(scene_cfg["tasks_json"]).is_absolute()


def test_variant_local_collision_suite_keeps_its_embedded_xml(tmp_path):
    config, scene_cfg = _fixture(tmp_path)
    shim_xml = tmp_path / "a4_factory" / "scene_shims.xml"
    shim_xml.write_text('<mujoco model="a4_shims"/>')
    shim_tasks = tmp_path / "a4_factory" / "tasks_shims.json"
    shim_tasks.write_text(json.dumps(_suite(shim_xml)))
    scene_cfg["construction_variants"]["agentic"]["collision_variants"] = {
        "private_shims": {"tasks_json": str(shim_tasks)},
    }
    config["treatments"].append({
        "id": "a4_shims", "scene": "agentic", "collision": "private_shims",
    })
    config["comparisons"].append({
        "id": "collision", "axis": "collision",
        "treatments": ["a4", "a4_shims"],
    })
    spec = load_harness_spec(config)
    resolved = harness_runner._scene_config_for_treatment(
        scene_cfg, spec.treatments["a4_shims"], tmp_path / "run")
    suite = json.loads(Path(resolved["tasks_json"]).read_text())
    assert suite["scene_xml"] == str(shim_xml)
    assert "scene_xml" not in resolved


def test_preflight_rejects_byte_identical_construction_assets_before_build(
        tmp_path, monkeypatch):
    config, _ = _fixture(tmp_path, identical_xml=True)
    monkeypatch.setattr(harness_runner, "ROOT", tmp_path)
    called = {"env": False, "policy": False}

    def unexpected_env(*args, **kwargs):
        called["env"] = True
        raise AssertionError("environment construction must not start")

    def unexpected_policy(*args, **kwargs):
        called["policy"] = True
        raise AssertionError("policy construction must not start")

    monkeypatch.setattr(harness_runner.legacy, "build_env", unexpected_env)
    monkeypatch.setattr(harness_runner.legacy, "get_policy", unexpected_policy)
    with pytest.raises(ValueError, match="byte-identical scene_xml assets"):
        harness_runner.run_matrix(config, tmp_path / "run")
    assert called == {"env": False, "policy": False}


def test_preflight_rejects_variant_task_definition_drift(tmp_path):
    changed_task = _task()
    changed_task["instructions"]["default"] = "a treatment-specific prompt"
    config, scene_cfg = _fixture(tmp_path, task_override=changed_task)
    spec = load_harness_spec(config)
    resolved = _resolved(scene_cfg, spec, tmp_path / "run")
    report = validate_resolved_scene_treatments(
        scene_cfg, resolved, spec, root=tmp_path)
    assert not report["ok"]
    assert "task definition mismatch" in " | ".join(report["violations"])


def test_preflight_rejects_swapped_a0_a4_materialization_labels(tmp_path):
    config, scene_cfg = _fixture(tmp_path)
    variants = scene_cfg["construction_variants"]
    variants["fixed_single_path"], variants["agentic"] = (
        variants["agentic"], variants["fixed_single_path"])
    spec = load_harness_spec(config)
    resolved = _resolved(scene_cfg, spec, tmp_path / "run")
    report = validate_resolved_scene_treatments(
        scene_cfg, resolved, spec, root=tmp_path)
    assert not report["ok"]
    assert "materialization policy binding differs" in " | ".join(
        report["violations"])


def test_collision_axis_rejects_relabelled_full_room_xml(tmp_path):
    config, scene_cfg = _fixture(tmp_path)
    a4 = scene_cfg["construction_variants"]["agentic"]
    a4["collision_variants"] = {
        "private_shims": {
            "tasks_json": a4["tasks_json"],
            "scene_xml": a4["scene_xml"],
        }
    }
    config["treatments"].append(
        {"id": "a4_shims", "scene": "agentic", "collision": "private_shims"}
    )
    config["comparisons"].append(
        {
            "id": "collision",
            "axis": "collision",
            "treatments": ["a4", "a4_shims"],
        }
    )
    spec = load_harness_spec(config)
    resolved = _resolved(scene_cfg, spec, tmp_path / "run")
    report = validate_resolved_scene_treatments(
        scene_cfg, resolved, spec, root=tmp_path)
    assert not report["ok"]
    assert "do not use distinct scene_xml assets" in " | ".join(
        report["violations"])


def test_resume_refuses_resolved_config_drift_before_environment_build(tmp_path):
    config, _scene_cfg = _fixture(tmp_path)
    out = tmp_path / "run"
    out.mkdir()
    (out / "resolved_harness_config.json").write_text(
        json.dumps(config, sort_keys=True) + "\n")
    changed = copy.deepcopy(config)
    changed["episodes"] = 2
    with pytest.raises(ValueError, match="resolved harness config differs"):
        harness_runner.run_matrix(changed, out)


def test_saved_task_hash_is_checked_against_current_planning_task():
    task = _task()
    task_hash = harness_runner.task_definition_hash(task)
    artifacts = {
        "e3_code_commit": "c" * 40,
        "e3_freeze_id": "fixture",
        "materialization_manifest_sha256": "a" * 64,
        "materialization_policy_id": "A0",
        "object_slots": [],
        "roster_sha256": "b" * 64,
        "scene_xml_sha256": "d" * 64,
        "task_contract_sha256": "e" * 64,
    }
    record = {
        "episode_id": "fixture-episode",
        "scene_id": SCENE,
        "task_id": task["task_id"],
        "treatment_id": "a0",
        "contract": {
            "task_definition_hash": task_hash,
            "construction_artifacts": artifacts,
        },
    }
    resolved = {SCENE: {"a0": {"_e4_input_artifacts": artifacts}}}
    planning = {SCENE: {task["task_id"]: task}}
    assert not harness_runner._validate_record_inputs_against_current(
        [record], resolved, planning)
    changed = copy.deepcopy(task)
    changed["instructions"]["default"] = "mutated after the first run"
    violations = harness_runner._validate_record_inputs_against_current(
        [record], resolved, {SCENE: {task["task_id"]: changed}})
    assert any("task definition differs" in value for value in violations)


def test_e4_reset_preflight_rejects_persisted_seed_drift():
    task = _task()
    reset_id = f"{task['task_id']}__seed0__ep0"
    state = elog.ResetState(
        reset_state_id=reset_id,
        scene_id=SCENE,
        task_id=task["task_id"],
        ep=0,
        base_seed=0,
        reset_seed=123,
    )
    with pytest.raises(ValueError, match="reset seed mismatch"):
        harness_runner._validate_planned_resets(
            [state], {SCENE: {task["task_id"]: task}}, {SCENE})


def test_episode_reset_uses_explicit_seed_draw_and_offset(monkeypatch):
    captured = {}

    class Env:
        last_reset_provenance = None

        def reset(self, **kwargs):
            captured.update(kwargs)
            self.last_reset_provenance = {
                "jitter": {
                    "algorithm": (
                        "numpy.random.RandomState.random_sample_then_affine"
                    ),
                    "reset_seed": kwargs["reset_seed"],
                    "body": kwargs["jitter_body"],
                    "max_abs_xy_m": kwargs["jitter_xy"],
                    "uniform_draw_0_1": kwargs["jitter_uniform_draw"],
                    "offset_xy_m": kwargs["jitter_offset_xy"],
                    "applied": True,
                }
            }
            return {}

    class Policy:
        def reset(self):
            pass

        def __call__(self, observation, prompt):
            return np.full(8, np.nan)

    class Pipeline:
        def reset(self, episode_id):
            pass

        def get_obs(self):
            return SimpleNamespace(
                observation={}, latency_ms=0.0, per_camera={})

    monkeypatch.setattr(
        harness_runner.pi05_tasks, "TaskScorer", lambda env, task: object())
    result, _ticks, _frames, provenance = harness_runner._run_episode(
        Env(), _task(), Policy(), Pipeline(), episode_id="episode",
        prompt="prompt", horizon_s=1.0, reset_seed=123, jitter_xy=0.02,
        policy_timeout_s=1.0, capture_video=False)
    expected = harness_runner._planned_jitter(
        reset_seed=123, jitter_xy=0.02, target="obj_00")
    assert captured == {
        "jitter_body": expected["body"],
        "jitter_xy": expected["max_abs_xy_m"],
        "reset_seed": expected["reset_seed"],
        "jitter_uniform_draw": expected["uniform_draw_0_1"],
        "jitter_offset_xy": expected["offset_xy_m"],
    }
    assert provenance["jitter"] == expected
    assert result["outcome"] == "safety_termination"


def test_build_failure_keeps_discoverable_task_family():
    spec = load_harness_spec({"conditions": ["reference", "simany"]})
    task = _task()
    state = elog.ResetState(
        reset_state_id="r0", scene_id="room", task_id=task["task_id"],
        ep=0, base_seed=0, reset_seed=0)
    record = harness_runner._failure_record(
        spec.treatments["simany"], state, "broken", task=task)
    assert record["task_family"] == "object_to_region"


def test_preflight_rejects_task_target_absent_from_a4_acceptance(tmp_path):
    config, scene_cfg = _fixture(tmp_path)
    a4_factory = Path(
        scene_cfg["construction_variants"]["agentic"]["factory_dir"]
    )
    manifest_path = a4_factory / "materialization_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["roster"].update(
        {
            "accepted_count": 0,
            "accepted_slots": [],
            "abstained_count": 1,
            "abstained_slots": ["obj_00"],
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    spec = load_harness_spec(config)
    report = validate_resolved_scene_treatments(
        scene_cfg, _resolved(scene_cfg, spec, tmp_path / "run"), spec, root=tmp_path
    )
    assert not report["ok"]
    assert "target 'obj_00' is not an accepted object" in " | ".join(
        report["violations"]
    )


def test_preflight_rejects_cross_arm_source_provenance_drift(tmp_path):
    config, scene_cfg = _fixture(tmp_path)
    a4_factory = Path(
        scene_cfg["construction_variants"]["agentic"]["factory_dir"]
    )
    manifest_path = a4_factory / "materialization_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_scene"] = {"different": "sealed source"}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    spec = load_harness_spec(config)
    report = validate_resolved_scene_treatments(
        scene_cfg, _resolved(scene_cfg, spec, tmp_path / "run"), spec, root=tmp_path
    )
    assert not report["ok"]
    assert "source_scene_sha256" in " | ".join(report["violations"])


def test_paired_construction_requires_task_freeze_manifest(tmp_path):
    config, scene_cfg = _fixture(tmp_path)
    del scene_cfg["task_freeze_manifest"]
    spec = load_harness_spec(config)
    resolved = _resolved(scene_cfg, spec, tmp_path / "run")
    report = validate_resolved_scene_treatments(
        scene_cfg, resolved, spec, root=tmp_path)
    assert not report["ok"]
    assert any("requires task_freeze_manifest" in v for v in report["violations"])


@pytest.mark.parametrize('all_rejected', [True, False])
def test_automatic_prebuild_rows_precede_real_service_and_resume(tmp_path, monkeypatch, all_rejected):
    """Unavailable policy service cannot erase already measured rejection cells."""
    from robo.eval import e4_reset_eligibility
    config, _ = _fixture(tmp_path)
    config['episodes'] = 5
    config['contract']['reset_ids'] = [f"{_task()['task_id']}__seed0__ep{ep}" for ep in range(5)]
    spec = load_harness_spec(config)
    # Source/asset closure has dedicated negative tests; capture its valid
    # fixture result before stubbing the external registry and policy service.
    resolved = harness_runner._preflight_resolved_scenes(config, spec, tmp_path / 'preflight')
    config['policy'] = 'pi05_droid_jointpos'
    config['cpu_reset_eligibility'] = {'kind':'automatic_compact'}
    config['contract']['policy'] = {'id':config['policy'], 'kind':'real',
        'checkpoint_hash':'ac' * 32, 'training_config':'pi05_droid'}
    config['contract']['runtime_dependencies'] = {'mujoco_menagerie':{'root':str(tmp_path)}}
    monkeypatch.setattr(harness_runner, 'load_harness_spec', lambda value: spec)
    monkeypatch.setattr(harness_runner, '_preflight_resolved_scenes', lambda *a: resolved)
    monkeypatch.setattr(harness_runner, 'ROOT', tmp_path)
    monkeypatch.setattr(harness_runner, 'expected_server_identity_from_config', lambda value: {'expected':True})
    monkeypatch.setattr(harness_runner.legacy, '_policy_checkpoint_hash', lambda *a:'ac' * 32)
    def eligibility(config, spec, states, *, root):
        return {(t.id,s.reset_state_id):{'passed':False if all_rejected else t.id=='a0',
            'failure_type':'cpu_reset_validity_failure','failed_checks':['stability_passed']}
            for t in spec.treatments.values() for s in states}
    monkeypatch.setattr(e4_reset_eligibility, 'load_reset_eligibility', eligibility)
    calls=[]
    def unavailable(*a, **kw):
        calls.append('policy')
        raise RuntimeError('offline service identity unavailable')
    def forbidden_environment(*a, **kw):
        raise AssertionError('no environment may execute before service identity')
    monkeypatch.setattr(harness_runner.legacy, 'get_policy', unavailable)
    monkeypatch.setattr(harness_runner.legacy, 'build_env', forbidden_environment)
    monkeypatch.setattr(harness_runner, 'validate_saved_treatment_records',
        lambda *a, **kw:{'ok':True,'violations':[]})
    monkeypatch.setattr(harness_runner, 'generate_main_table', lambda *a, **kw:{})
    out=tmp_path/'run'
    for attempt in range(2):
        if all_rejected:
            harness_runner.run_matrix(config,out)
        else:
            with pytest.raises(RuntimeError,match='offline service identity'):
                harness_runner.run_matrix(config,out)
        rows=harness_runner.read_jsonl(out/harness_runner.LEDGER_NAME)
        assert len(rows)==(10 if all_rejected else 5)
        assert all(r['outcome']=='build_failure' for r in rows)
        assert all(r['contract']['policy_execution']=='not_invoked_prebuild' for r in rows)
        assert all('server_identity' not in r['contract']['policy'] for r in rows)
        assert all('reset_provenance' not in r for r in rows)
        if not all_rejected:
            assert {r['treatment_id'] for r in rows}=={'a4'}
        content=(out/harness_runner.LEDGER_NAME).read_bytes()
        if attempt: assert content==before
        before=content
    assert len(calls)==(0 if all_rejected else 2)
