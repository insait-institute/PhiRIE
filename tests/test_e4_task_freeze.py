from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from robo.eval import agentic_ablation as e3
from robo.eval import e4_task_freeze as freeze


SCENE = "0123456789"
VALIDATOR_COMMIT = "3" * 40


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _identity(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _reseal_bundle(out: Path, manifest: dict) -> None:
    manifest["files"] = {
        name: _identity(out / name)
        for name in ("planning_tasks.json", "a0_tasks.json", "a4_tasks.json")
    }
    _write_json(out / "manifest.json", manifest)
    _write_json(
        out / "seal.json",
        {
            "manifest_kind": "e4_paired_task_freeze",
            "members": {"manifest.json": _identity(out / "manifest.json")},
            "schema_version": freeze.SCHEMA_VERSION,
        },
    )


def _suite(factory: Path, *, cx: float) -> dict:
    return {
        "exclude_objects": [],
        "ext_cam": {"pos": [1, 2, 3]},
        "robot": {"base_pos": [0, 0, 0], "base_yaw": 0},
        "scene": f"{SCENE}_factory",
        "scene_xml": str(factory / "sim_export" / "scene.xml"),
        "table": {"cx": cx, "cy": 0, "hx": 1, "hy": 1, "top_z": 0.7},
        "tasks": [
            {
                "task_id": f"{SCENE}_factory__obj_00_to_region",
                "target": "obj_00",
                "receptacle": None,
                "region": {"cx": cx, "cy": 0, "hx": 0.2, "hy": 0.2},
                "instructions": {"default": "move the cup"},
            },
            {
                "task_id": f"{SCENE}_factory__obj_01_to_region",
                "target": "obj_01",
                "receptacle": None,
                "region": {"cx": cx, "cy": 0, "hx": 0.2, "hy": 0.2},
                "instructions": {"default": "move the box"},
            },
        ],
        "time_limit_s": 16.0,
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(e3, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(freeze, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        freeze,
        "_git_snapshot",
        lambda: {
            "code_root": str(freeze.CODE_ROOT),
            "commit": VALIDATOR_COMMIT,
            "dirty": False,
            "status": [],
        },
    )
    factories = {}
    suites = {}
    reports = {}
    for policy, cx, accepted in (
        ("A0", 1.0, ["obj_00", "obj_01"]),
        ("A4", 1.1, ["obj_00"]),
    ):
        factory = tmp_path / policy.lower()
        (factory / "sim_export").mkdir(parents=True)
        (factory / "sim_export" / "scene.xml").write_text(
            f'<mujoco model="{policy}"/>', encoding="utf-8"
        )
        suite_path = factory / "sim_export" / "pi05_tasks.json"
        suite_path.write_text(json.dumps(_suite(factory, cx=cx)), encoding="utf-8")
        factories[policy] = factory
        suites[policy] = suite_path
        reports[policy] = {
            "code_root": str(freeze.CODE_ROOT),
            "e3_claim_status": {
                "freeze_id": "freeze",
                "headline_eligible": False,
                "paper_ready": False,
                "retry_claim_status": "preliminary",
                "study_scope": "fixture",
            },
            "e3_code_commit": "c" * 40,
            "e3_producer_commit": "c" * 40,
            "e3_freeze_id": "freeze",
            "e3_root": "e3/agentic",
            "evidence_root": str(tmp_path),
            "factory_dir": str(factory),
            "input_identities_sha256": "1" * 64,
            "manifest_sha256": ("a" if policy == "A0" else "b") * 64,
            "materializer_commit": VALIDATOR_COMMIT,
            "policy_id": policy,
            "roster": {
                "accepted_slots": accepted,
                "object_slots": ["obj_00", "obj_01"],
            },
            "scene_id": SCENE,
            "source_scene_sha256": "2" * 64,
            "study_scope": "fixture",
            "validator_commit": VALIDATOR_COMMIT,
        }

    def validate(factory_dir, *, expected_scene_id, expected_policy_id,
                 repository_root):
        assert expected_scene_id == SCENE
        assert Path(factory_dir) == factories[expected_policy_id]
        return copy.deepcopy(reports[expected_policy_id])

    monkeypatch.setattr(freeze, "validate_materialized_factory", validate)
    return factories, suites, reports


def test_freeze_uses_one_explicit_source_and_only_common_supported_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factories, suites, _reports = _fixture(tmp_path, monkeypatch)
    candidate_root = tmp_path / "candidate_suites"
    candidate_root.mkdir()
    external_suites = {}
    for policy in ("A0", "A4"):
        external_suites[policy] = candidate_root / f"{policy.lower()}.json"
        external_suites[policy].write_bytes(suites[policy].read_bytes())
    out = tmp_path / "task_freezes" / SCENE
    manifest = freeze.freeze_task_bundle(
        scene_id=SCENE,
        a0_factory=factories["A0"],
        a4_factory=factories["A4"],
        a0_candidates=external_suites["A0"],
        a4_candidates=external_suites["A4"],
        planning_source="A4",
        out=out,
        repository_root=tmp_path,
    )
    assert manifest["logical_task_ids"] == [f"{SCENE}__obj_00_to_region"]
    planning = json.loads((out / "planning_tasks.json").read_text())
    a0 = json.loads((out / "a0_tasks.json").read_text())
    a4 = json.loads((out / "a4_tasks.json").read_text())
    assert planning["scene"] == SCENE
    assert planning["table"]["cx"] == 1.1
    assert planning["scene_xml"] is None
    assert a0["tasks"] == a4["tasks"] == planning["tasks"]
    assert {key: value for key, value in a0.items() if key != "scene_xml"} == {
        key: value for key, value in a4.items() if key != "scene_xml"
    }
    assert a0["scene_xml"] != a4["scene_xml"]
    assert manifest["provenance"] == {
        "code_root": str(freeze.CODE_ROOT),
        "evidence_root": str(tmp_path),
        "freezer_commit": VALIDATOR_COMMIT,
        "freezer_dirty": False,
        "validator_commit": VALIDATOR_COMMIT,
        "validator_dirty": False,
    }
    assert manifest["candidate_suites"]["A0"]["path"] == "candidate_suites/a0.json"
    assert not out.is_relative_to(factories["A0"])
    assert not out.is_relative_to(factories["A4"])
    report = freeze.validate_task_bundle(
        out / "manifest.json",
        expected_scene_id=SCENE,
        repository_root=tmp_path,
    )
    assert report["logical_task_ids"] == [f"{SCENE}__obj_00_to_region"]
    assert report["factories"] == {
        "A0": str(factories["A0"]),
        "A4": str(factories["A4"]),
    }
    assert report["planning_tasks"] == str(out / "planning_tasks.json")
    assert report["code_root"] == str(freeze.CODE_ROOT)
    assert report["evidence_root"] == str(tmp_path)
    assert report["freezer_commit"] == VALIDATOR_COMMIT
    assert report["validator_commit"] == VALIDATOR_COMMIT
    with pytest.raises(FileExistsError, match="overwrite"):
        freeze.freeze_task_bundle(
            scene_id=SCENE,
            a0_factory=factories["A0"],
            a4_factory=factories["A4"],
            a0_candidates=external_suites["A0"],
            a4_candidates=external_suites["A4"],
            planning_source="A4",
            out=out,
            repository_root=tmp_path,
        )


def test_validate_task_bundle_rejects_frozen_suite_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factories, suites, _reports = _fixture(tmp_path, monkeypatch)
    out = tmp_path / "frozen"
    freeze.freeze_task_bundle(
        scene_id=SCENE,
        a0_factory=factories["A0"],
        a4_factory=factories["A4"],
        a0_candidates=suites["A0"],
        a4_candidates=suites["A4"],
        planning_source="A4",
        out=out,
        repository_root=tmp_path,
    )
    suite = json.loads((out / "a0_tasks.json").read_text())
    suite["tasks"][0]["instructions"]["default"] = "tampered"
    (out / "a0_tasks.json").write_text(json.dumps(suite), encoding="utf-8")
    with pytest.raises(ValueError, match="bytes drifted"):
        freeze.validate_task_bundle(
            out / "manifest.json",
            expected_scene_id=SCENE,
            repository_root=tmp_path,
        )


def test_freeze_rejects_cross_arm_e3_provenance_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factories, suites, reports = _fixture(tmp_path, monkeypatch)
    reports["A4"]["e3_freeze_id"] = "other"
    with pytest.raises(ValueError, match="e3_freeze_id"):
        freeze.freeze_task_bundle(
            scene_id=SCENE,
            a0_factory=factories["A0"],
            a4_factory=factories["A4"],
            a0_candidates=suites["A0"],
            a4_candidates=suites["A4"],
            planning_source="A0",
            out=tmp_path / "frozen",
            repository_root=tmp_path,
        )


def test_freeze_rejects_when_no_task_is_supported_in_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factories, suites, reports = _fixture(tmp_path, monkeypatch)
    reports["A4"]["roster"]["accepted_slots"] = []
    with pytest.raises(ValueError, match="no common task"):
        freeze.freeze_task_bundle(
            scene_id=SCENE,
            a0_factory=factories["A0"],
            a4_factory=factories["A4"],
            a0_candidates=suites["A0"],
            a4_candidates=suites["A4"],
            planning_source="A4",
            out=tmp_path / "frozen",
            repository_root=tmp_path,
        )


def test_freeze_requires_receptacle_to_be_accepted_by_both_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factories, suites, _reports = _fixture(tmp_path, monkeypatch)
    for policy in ("A0", "A4"):
        suite = json.loads(suites[policy].read_text())
        suite["tasks"][0]["receptacle"] = "obj_01"
        _write_json(suites[policy], suite)
    with pytest.raises(ValueError, match="no common task"):
        freeze.freeze_task_bundle(
            scene_id=SCENE,
            a0_factory=factories["A0"],
            a4_factory=factories["A4"],
            a0_candidates=suites["A0"],
            a4_candidates=suites["A4"],
            planning_source="A0",
            out=tmp_path / "frozen",
            repository_root=tmp_path,
        )


def test_freeze_rejects_cross_arm_target_receptacle_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factories, suites, _reports = _fixture(tmp_path, monkeypatch)
    a4 = json.loads(suites["A4"].read_text())
    a4["tasks"][0]["receptacle"] = "obj_01"
    _write_json(suites["A4"], a4)
    with pytest.raises(ValueError, match="different target/receptacle"):
        freeze.freeze_task_bundle(
            scene_id=SCENE,
            a0_factory=factories["A0"],
            a4_factory=factories["A4"],
            a0_candidates=suites["A0"],
            a4_candidates=suites["A4"],
            planning_source="A4",
            out=tmp_path / "frozen",
            repository_root=tmp_path,
        )


def test_revalidation_replays_planning_source_after_self_reseal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factories, suites, _reports = _fixture(tmp_path, monkeypatch)
    out = tmp_path / "frozen"
    manifest = freeze.freeze_task_bundle(
        scene_id=SCENE,
        a0_factory=factories["A0"],
        a4_factory=factories["A4"],
        a0_candidates=suites["A0"],
        a4_candidates=suites["A4"],
        planning_source="A4",
        out=out,
        repository_root=tmp_path,
    )
    for name in ("planning_tasks.json", "a0_tasks.json", "a4_tasks.json"):
        suite = json.loads((out / name).read_text())
        suite["table"]["cx"] = 99.0
        _write_json(out / name, suite)
    _reseal_bundle(out, manifest)
    with pytest.raises(ValueError, match="planning-source replay"):
        freeze.validate_task_bundle(
            out / "manifest.json",
            expected_scene_id=SCENE,
            repository_root=tmp_path,
        )


def test_max_tasks_is_sealed_and_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factories, suites, reports = _fixture(tmp_path, monkeypatch)
    reports["A4"]["roster"]["accepted_slots"] = ["obj_00", "obj_01"]
    out = tmp_path / "frozen"
    manifest = freeze.freeze_task_bundle(
        scene_id=SCENE,
        a0_factory=factories["A0"],
        a4_factory=factories["A4"],
        a0_candidates=suites["A0"],
        a4_candidates=suites["A4"],
        planning_source="A0",
        max_tasks=1,
        out=out,
        repository_root=tmp_path,
    )
    assert manifest["max_tasks"] == 1
    report = freeze.validate_task_bundle(
        out / "manifest.json",
        expected_scene_id=SCENE,
        repository_root=tmp_path,
    )
    assert report["max_tasks"] == 1
    manifest["max_tasks"] = None
    _reseal_bundle(out, manifest)
    with pytest.raises(ValueError, match="task selection replay"):
        freeze.validate_task_bundle(
            out / "manifest.json",
            expected_scene_id=SCENE,
            repository_root=tmp_path,
        )


def test_freeze_rejects_dirty_code_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factories, suites, _reports = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        freeze,
        "_git_snapshot",
        lambda: {
            "code_root": str(freeze.CODE_ROOT),
            "commit": VALIDATOR_COMMIT,
            "dirty": True,
            "status": [" M robo/eval/e4_task_freeze.py"],
        },
    )
    out = tmp_path / "frozen"
    with pytest.raises(ValueError, match="clean code worktree"):
        freeze.freeze_task_bundle(
            scene_id=SCENE,
            a0_factory=factories["A0"],
            a4_factory=factories["A4"],
            a0_candidates=suites["A0"],
            a4_candidates=suites["A4"],
            planning_source="A4",
            out=out,
            repository_root=tmp_path,
        )
    assert not out.exists()


def test_revalidation_rejects_self_resealed_provenance_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factories, suites, _reports = _fixture(tmp_path, monkeypatch)
    out = tmp_path / "frozen"
    manifest = freeze.freeze_task_bundle(
        scene_id=SCENE,
        a0_factory=factories["A0"],
        a4_factory=factories["A4"],
        a0_candidates=suites["A0"],
        a4_candidates=suites["A4"],
        planning_source="A4",
        out=out,
        repository_root=tmp_path,
    )
    manifest["provenance"]["freezer_commit"] = "4" * 40
    manifest["provenance"]["validator_commit"] = "4" * 40
    _reseal_bundle(out, manifest)
    with pytest.raises(ValueError, match="differs from executing code"):
        freeze.validate_task_bundle(
            out / "manifest.json",
            expected_scene_id=SCENE,
            repository_root=tmp_path,
        )


def test_freeze_rejects_noncanonical_or_symlinked_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factories, suites, _reports = _fixture(tmp_path, monkeypatch)
    common = {
        "scene_id": SCENE,
        "a0_factory": factories["A0"],
        "a4_factory": factories["A4"],
        "a0_candidates": suites["A0"],
        "a4_candidates": suites["A4"],
        "planning_source": "A4",
        "repository_root": tmp_path,
    }
    aliased = f"{tmp_path}/not-created/../frozen"
    with pytest.raises(ValueError, match="dot segment"):
        freeze.freeze_task_bundle(out=aliased, **common)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        freeze.freeze_task_bundle(out=linked_parent / "frozen", **common)


def test_freeze_rejects_candidate_scene_xml_from_other_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factories, suites, _reports = _fixture(tmp_path, monkeypatch)
    a4 = json.loads(suites["A4"].read_text())
    a4["scene_xml"] = str(factories["A0"] / "sim_export" / "scene.xml")
    _write_json(suites["A4"], a4)
    with pytest.raises(ValueError, match="scene_xml differs from its factory"):
        freeze.freeze_task_bundle(
            scene_id=SCENE,
            a0_factory=factories["A0"],
            a4_factory=factories["A4"],
            a0_candidates=suites["A0"],
            a4_candidates=suites["A4"],
            planning_source="A4",
            out=tmp_path / "frozen",
            repository_root=tmp_path,
        )
