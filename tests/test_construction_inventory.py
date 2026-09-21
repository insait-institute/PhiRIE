from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import pytest

from robo.eval.construction_inventory import _read_scene_list, build_inventory
from robo.eval.construction_metrics import REQUIRED_REGIMES


REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = REPO_ROOT / "tests" / ".construction-inventory-work"


@contextmanager
def _repo_workspace():
    WORK_ROOT.mkdir(exist_ok=True)
    root = WORK_ROOT / uuid.uuid4().hex
    root.mkdir()
    try:
        yield root
    finally:
        shutil.rmtree(root)
        try:
            WORK_ROOT.rmdir()
        except OSError:
            pass


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _scene_ids(count: int) -> list[str]:
    return [f"{index:010x}" for index in range(count)]


def _make_report_scene(
    outputs: Path,
    scene: str,
    suffix: str,
    *,
    reported_inputs: int = 2,
    prepared_inputs: int = 2,
    with_eval: bool = True,
) -> Path:
    scene_dir = outputs / f"{scene}_{suffix}"
    objects_dir = scene_dir / "objects"
    objects = [{"index": index, "label": "box"} for index in range(prepared_inputs)]
    _write_json(objects_dir / "objects.json", objects)
    for index in range(prepared_inputs):
        mesh = objects_dir / f"obj_{index:02d}" / "trellis_mesh.ply"
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_bytes(b"ply\n")
    _write_json(objects_dir / "aligned_all.json", [{"index": index} for index in range(prepared_inputs)])

    report_objects = []
    for index in range(reported_inputs):
        tier = "A" if index == 0 else "C"
        report_objects.append({"index": index, "tier": tier, "f1_20": 0.75})
    _write_json(
        scene_dir / "report.json",
        {
            "n_instances": reported_inputs,
            "tier_A": 1 if reported_inputs else 0,
            "tier_B": 0,
            "tier_C_rejected": max(reported_inputs - 1, 0),
            "objects": report_objects,
        },
    )
    _write_json(
        scene_dir / "drop_v2.json",
        {
            "n_tested": 1 if reported_inputs else 0,
            "n_stable": 1 if reported_inputs else 0,
            "objects": ([{"stable": True}] if reported_inputs else []),
        },
    )
    (scene_dir / "timings.txt").write_text("prepare 60\nbuild 120\n", encoding="utf-8")
    if suffix in {"auto", "rowC2"}:
        (scene_dir / "auto_instances.npz").write_bytes(b"fixture")
    if suffix in {"rowC", "rowC2"}:
        (scene_dir / "derived_mesh.ply").write_bytes(b"ply\n")
    if with_eval:
        _write_json(
            scene_dir / "eval_vs_gt.json",
            {
                "n_objects": prepared_inputs,
                "objects": [
                    {
                        "name": f"obj_{index:02d}",
                        "tier": "A" if index == 0 else "C",
                        "f1_20_gt": 0.6 if index == 0 else 0.1,
                    }
                    for index in range(prepared_inputs)
                ],
            },
        )
    _write_json(scene_dir / "build_manifest.json", {"build_commit": "a" * 40})
    return scene_dir


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_checked_in_yaml_roster_pins_exact_fifty_unique_scenes():
    config = REPO_ROOT / "configs/experiments/icra2027/construction_regimes.yaml"
    scenes = _read_scene_list(config)

    assert len(scenes) == 50
    assert len(set(scenes)) == 50
    assert {"38d58a7a31", "c4c04e6d6c"} <= set(scenes)


def test_full_inventory_keeps_all_fifty_scenes_and_failed_rows():
    with _repo_workspace() as root:
        outputs = root / "outputs"
        outputs.mkdir()
        scenes = _scene_ids(50)

        result = build_inventory(
            outputs,
            root / "audit",
            "fixture-freeze",
            planned_scene_ids=scenes,
        )

        rows = _read_rows(root / "audit" / "scene_records.csv")
        assert len(rows) == 250
        assert Counter(row["regime"] for row in rows) == {
            regime: 50 for regime in REQUIRED_REGIMES
        }
        assert {row["scene_id"] for row in rows} == set(scenes)
        assert all(row["scene_status"] in {"failed", "unavailable"} for row in rows)
        assert all(row["record_valid"] == "false" for row in rows)
        assert result["population"]["full_population_preserved"] is True
        assert result["population"]["planned_record_count"] == 250
        assert result["paper_ready"] is False
        assert result["by_regime"][REQUIRED_REGIMES[0]]["completed_scenes"] == 0
        assert result["by_regime"][REQUIRED_REGIMES[0]]["contributing_scenes"] == 0


def test_inventory_rejects_unsafe_freeze_identifier():
    with _repo_workspace() as root:
        outputs = root / "outputs"
        outputs.mkdir()
        with pytest.raises(ValueError, match="freeze_id contains unsafe"):
            build_inventory(
                outputs,
                root / "audit",
                "../outside",
                expected_scenes=1,
                planned_scene_ids=_scene_ids(1),
                selected_regimes=[REQUIRED_REGIMES[0]],
                smoke=True,
            )


def test_smoke_selection_records_commit_sources_and_explicit_weights():
    with _repo_workspace() as root:
        outputs = root / "outputs"
        outputs.mkdir()
        scene = _scene_ids(1)[0]
        _make_report_scene(outputs, scene, "auto")

        result = build_inventory(
            outputs,
            root / "audit",
            "fixture-freeze",
            expected_scenes=1,
            planned_scene_ids=[scene],
            selected_scene_ids=[scene],
            selected_regimes=["auto_discovery_scan_mesh"],
            smoke=True,
        )

        row = _read_rows(root / "audit" / "scene_records.csv")[0]
        assert row["regime"] == REQUIRED_REGIMES[1]
        assert row["input_instances"] == "2"
        assert row["accepted_instances"] == "1"
        assert row["f1_20"] == "0.6"
        assert row["f1_weight"] == "1"
        assert row["stable_instances"] == "1"
        assert row["tested_instances"] == "1"
        assert row["runtime_minutes"] == "3.0"
        assert row["build_commit"] == "a" * 40
        assert row["source_artifact_hash"] != ""
        assert result["paper_ready"] is False
        manifest = json.loads((REPO_ROOT / row["build_manifest_path"]).read_text())
        assert manifest["import_kind"] == "legacy_evidence_import"
        assert manifest["source_build_manifest"].endswith("build_manifest.json")


def test_incoherent_denominators_are_visible_and_fail_closed():
    with _repo_workspace() as root:
        outputs = root / "outputs"
        outputs.mkdir()
        scene = _scene_ids(1)[0]
        _make_report_scene(
            outputs, scene, "auto", reported_inputs=2, prepared_inputs=1,
        )

        result = build_inventory(
            outputs,
            root / "audit",
            "fixture-freeze",
            expected_scenes=1,
            planned_scene_ids=[scene],
            selected_regimes=[REQUIRED_REGIMES[1]],
            smoke=True,
        )

        row = _read_rows(root / "audit" / "scene_records.csv")[0]
        reasons = json.loads(row["validity_reasons"])
        assert row["input_instances"] == "1"  # prepared population wins
        assert row["accepted_instances"] == "1"
        assert row["record_valid"] == "false"
        assert "prepared_and_report_input_denominators_disagree" in reasons
        assert result["denominator_coherent"] is False
        assert "prepared_and_report_input_denominators_disagree" in result["denominator_issues"]


def test_impossible_legacy_acceptance_is_diagnostic_only():
    with _repo_workspace() as root:
        outputs = root / "outputs"
        outputs.mkdir()
        scene = _scene_ids(1)[0]
        scene_dir = _make_report_scene(
            outputs, scene, "auto", reported_inputs=2, prepared_inputs=1,
        )
        report_path = scene_dir / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.update(tier_A=2, tier_B=0, tier_C_rejected=0)
        for item in report["objects"]:
            item["tier"] = "A"
        _write_json(report_path, report)

        build_inventory(
            outputs,
            root / "audit",
            "fixture-freeze",
            expected_scenes=1,
            planned_scene_ids=[scene],
            selected_regimes=[REQUIRED_REGIMES[1]],
            smoke=True,
        )

        row = _read_rows(root / "audit" / "scene_records.csv")[0]
        report = json.loads((root / "audit" / "missing_artifacts.json").read_text())
        assert row["input_instances"] == "1"
        assert row["accepted_instances"] == "0"
        assert row["f1_20"] == ""
        assert report["records"][0]["observed_legacy_metrics"]["accepted_instances"] == 2


def test_discovery_symlink_is_recorded_without_being_accepted():
    with _repo_workspace() as root:
        outputs = root / "outputs"
        outputs.mkdir()
        scene = _scene_ids(1)[0]
        scene_dir = _make_report_scene(outputs, scene, "auto")
        target = root / "symlink-target.npz"
        target.write_bytes(b"do not follow")
        auto_path = scene_dir / "auto_instances.npz"
        auto_path.unlink()
        auto_path.symlink_to(target)

        build_inventory(
            outputs,
            root / "audit",
            "fixture-freeze",
            expected_scenes=1,
            planned_scene_ids=[scene],
            selected_regimes=[REQUIRED_REGIMES[1]],
            smoke=True,
        )

        row = _read_rows(root / "audit" / "scene_records.csv")[0]
        reasons = json.loads(row["validity_reasons"])
        assert row["scene_status"] == "invalid"
        assert "automatic_instances_uses_symlink_path" in reasons


def test_e0_contract_hash_and_failed_checks_are_archived():
    with _repo_workspace() as root:
        outputs = root / "outputs"
        outputs.mkdir()
        scene = _scene_ids(1)[0]
        contract = root / "contract/freeze_manifest.json"
        _write_json(
            contract,
            {
                "freeze_id": "parent-freeze",
                "mode": "smoke",
                "contract_sha256": "f" * 64,
                "code": {"commit": "0" * 40, "dirty": True},
                "configs": [],
            },
        )

        result = build_inventory(
            outputs,
            root / "audit",
            "child-freeze",
            expected_scenes=1,
            planned_scene_ids=[scene],
            selected_regimes=[REQUIRED_REGIMES[0]],
            smoke=True,
            contract_manifest_path=contract,
        )

        inherited = result["upstream_e0_contract"]
        assert inherited["provided"] is True
        assert len(inherited["manifest_file_sha256"]) == 64
        assert inherited["checks"]["paper_mode"] is False
        assert "e0_contract_is_smoke_not_paper" in result["global_validity_reasons"]


def test_e0_short_commit_is_matched_as_an_unambiguous_git_prefix():
    with _repo_workspace() as root:
        outputs = root / "outputs"
        outputs.mkdir()
        scene = _scene_ids(1)[0]
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        config_hash = hashlib.sha256(
            (REPO_ROOT / "configs/experiments/icra2027/construction_regimes.yaml").read_bytes()
        ).hexdigest()
        contract = root / "contract/freeze_manifest.json"
        _write_json(
            contract,
            {
                "freeze_id": "parent-freeze",
                "mode": "smoke",
                "contract_sha256": "f" * 64,
                "code": {"commit": commit[:7], "dirty": False},
                "configs": [
                    {
                        "field": "construction_config",
                        "source_content_sha256": config_hash,
                    }
                ],
            },
        )

        result = build_inventory(
            outputs,
            root / "audit",
            "parent-freeze-child",
            expected_scenes=1,
            planned_scene_ids=[scene],
            selected_regimes=[REQUIRED_REGIMES[0]],
            smoke=True,
            contract_manifest_path=contract,
        )

        checks = result["upstream_e0_contract"]["checks"]
        assert checks["inventory_commit_matches_contract"] is True
        assert checks["construction_config_matches_contract"] is True
        assert checks["clean_contract_code"] is True


def test_rowd_counts_prepared_inputs_before_generator_failure():
    with _repo_workspace() as root:
        outputs = root / "outputs"
        outputs.mkdir()
        scene = _scene_ids(1)[0]
        scene_dir = outputs / scene
        objects_dir = scene_dir / "objects"
        _write_json(
            objects_dir / "objects.json",
            [{"index": index, "label": "box"} for index in range(3)],
        )
        for index in range(3):
            mesh = objects_dir / f"obj_{index:02d}" / "trellis_mesh.ply"
            mesh.parent.mkdir(parents=True, exist_ok=True)
            mesh.write_bytes(b"ply\n")
        _write_json(
            objects_dir / "aligned_all.json",
            [
                {
                    "index": 0,
                    "eval": {
                        "f1@20mm": {"f1": 0.5},
                        "f1@40mm": {"f1": 0.7},
                    },
                },
                {"index": 1, "rejected": "size gate"},
            ],
        )
        _write_json(
            outputs / "review_recompute_rowD.json",
            {
                "per_scene": {
                    scene: {
                        "n_instances": 2,
                        "n_tier_A": 1,
                        "n_tier_B": 0,
                        "status": "completed (DONE)",
                    }
                }
            },
        )

        build_inventory(
            outputs,
            root / "audit",
            "fixture-freeze",
            expected_scenes=1,
            planned_scene_ids=[scene],
            selected_regimes=[REQUIRED_REGIMES[4]],
            smoke=True,
        )

        row = _read_rows(root / "audit" / "scene_records.csv")[0]
        assert row["input_instances"] == "3"
        assert row["accepted_instances"] == "1"
        assert row["f1_weight"] == "0"
        assert row["f1_20"] == ""
        assert "gt_oracle_frame_selection" in json.loads(row["validity_reasons"])
        report = json.loads((root / "audit" / "missing_artifacts.json").read_text())
        assert report["records"][0]["observed_legacy_metrics"]["f1_20"] == 0.5


def test_rowc2_never_joins_stale_eval_to_current_acceptance_snapshot():
    with _repo_workspace() as root:
        outputs = root / "outputs"
        outputs.mkdir()
        scene = _scene_ids(1)[0]
        _make_report_scene(outputs, scene, "rowC2")

        build_inventory(
            outputs,
            root / "audit",
            "fixture-freeze",
            expected_scenes=1,
            planned_scene_ids=[scene],
            selected_regimes=[REQUIRED_REGIMES[3]],
            smoke=True,
        )

        row = _read_rows(root / "audit" / "scene_records.csv")[0]
        assert row["input_instances"] == "2"
        assert row["accepted_instances"] == "1"
        assert row["f1_20"] == ""
        assert row["f1_weight"] == "0"
        assert row["stable_instances"] == ""
        assert row["runtime_minutes"] == ""
        assert "incoherent_asset_eval_snapshots" in json.loads(row["validity_reasons"])


def test_injected_failure_leaves_no_partial_inventory():
    with _repo_workspace() as root:
        outputs = root / "outputs"
        outputs.mkdir()
        scene = _scene_ids(1)[0]
        destination = root / "audit"

        with pytest.raises(RuntimeError, match="injected construction inventory failure"):
            build_inventory(
                outputs,
                destination,
                "fixture-freeze",
                expected_scenes=1,
                planned_scene_ids=[scene],
                selected_regimes=[REQUIRED_REGIMES[0]],
                smoke=True,
                _inject_failure_after="manifests",
            )

        assert not destination.exists()
        assert list(root.glob(".audit.staging-*")) == []
