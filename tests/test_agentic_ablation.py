from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from agents.orchestrator import runtime as orchestrator_runtime
from agents.orchestrator.artifact import sha256_file
from robo.eval import agentic_ablation as e3


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "configs/experiments/icra2027/agentic_policies.yaml"


def _evidence() -> dict:
    return {
        "schema_frame_unit_valid": True,
        "symmetric_clipped_registration_residual_m": 0.01,
        "scale_ratio_vs_observation": 1.0,
        "observation_point_count": 500,
        "visible_fraction": 0.8,
        "support_gap_m": 0.005,
        "support_overlap_fraction": 0.8,
        "initial_penetration_m": 0.0,
        "collision_valid": True,
        "usable_convex_parts": 1,
        "settle_drift_m": 0.001,
        "settle_sunk": False,
        "settle_stable": True,
        "missing_evidence": [],
    }


def test_visible_observation_unprojects_only_valid_masked_pixels() -> None:
    depth = np.full((3, 4), 2.0)
    gaussian_alpha = np.ones_like(depth)
    rgba_alpha = np.array([[255, 0], [255, 255]], dtype=np.uint8)
    gaussian_alpha[2, 2] = 0.2
    intrinsics = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    points, stats = e3.visible_observation_points(
        depth,
        gaussian_alpha,
        rgba_alpha,
        [1, 1, 3, 3],
        intrinsics,
        np.eye(4),
    )
    assert points.dtype == np.float32
    assert points.shape == (2, 3)
    assert stats["mask_pixel_count"] == 3
    assert stats["observation_point_count"] == 2
    assert stats["observation_mask_fraction"] == pytest.approx(2 / 3)
    np.testing.assert_allclose(points[:, 2], 2.0)


def _fake_runtime(
    mesh_path,
    observation,
    *,
    out_dir,
    signed_source_up,
    producer_commit,
    input_hashes,
    **_kwargs,
):
    out_dir = Path(out_dir)
    for relative in e3.RUNTIME_ARTIFACT_ROLES:
        path = out_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((relative + "\n").encode())
    result = {
        "alignment": {
            "T": np.eye(4).tolist(),
            "source_up_hypothesis": "-z" if signed_source_up else "+z",
        },
        "evidence": _evidence(),
        "wall_s": 1.25,
        "artifact_files": list(e3.RUNTIME_ARTIFACT_ROLES),
    }
    (out_dir / "evidence.json").write_text(
        json.dumps(
            {
                "producer": "agents.orchestrator.runtime.align_and_probe",
                "producer_commit": producer_commit,
                "input_hashes": dict(sorted(input_hashes.items())),
                "raw_values": result["evidence"],
                "missing_flags": [],
                "started_utc": "2026-09-03T00:00:00Z",
                "finished_utc": "2026-09-03T00:00:01Z",
                "wall_s": result["wall_s"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _proposal(mesh: Path) -> dict:
    gaussian = mesh.with_name("raw_gaussian.ply")
    if not gaussian.exists():
        gaussian.write_bytes(b"raw Gaussian bytes")
    return {
        "freeze_id": "freeze",
        "scene_id": "5748ce6f01",
        "object_slot": "obj_00",
        "job_id": "5748ce6f01/obj_00",
        "proposal_id": "5748ce6f01/obj_00:trellis",
        "tool_id": "trellis",
        "artifact_paths": {
            "raw_mesh": mesh.relative_to(ROOT).as_posix(),
            "raw_gaussian": gaussian.relative_to(ROOT).as_posix(),
        },
        "artifact_hashes": {
            "raw_mesh": sha256_file(mesh),
            "raw_gaussian": sha256_file(gaussian),
        },
        "artifact_sizes": {
            "raw_mesh": mesh.stat().st_size,
            "raw_gaussian": gaussian.stat().st_size,
        },
    }


def test_candidate_closes_artifacts_and_does_not_backfill_generator_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mesh = tmp_path / "raw.ply"
    mesh.write_bytes(b"raw proposal bytes")
    staging = tmp_path / "staging"
    staging.mkdir()
    final = tmp_path / "final"
    monkeypatch.setattr(orchestrator_runtime, "align_and_probe", _fake_runtime)
    observation = {
        "observation_sha256": "a" * 64,
        "observation_mask_fraction": 0.8,
    }
    initial = e3._process_candidate(
        _proposal(mesh), observation, np.zeros((60, 3)), staging, final, "b" * 40
    )
    assert initial["tool"] == "trellis"
    assert initial["tool_commit"] == "legacy-unrecorded"
    assert initial["raw_generator_commit"] == "legacy-unrecorded"
    assert initial["registration_executor_commit"] == "b" * 40
    assert {"physics", "probe", "registration", "transform"}.issubset(
        initial["artifact_hashes"]
    )
    assert initial["input_hashes"]["raw_gaussian"] == initial["artifact_hashes"][
        "raw_gaussian"
    ]
    assert initial["artifact_paths"]["raw_gaussian"].endswith("raw_gaussian.ply")
    assert initial["artifact_sizes"]["raw_gaussian"] == len(b"raw Gaussian bytes")

    retry = e3._process_candidate(
        _proposal(mesh),
        observation,
        np.zeros((60, 3)),
        staging,
        final,
        "b" * 40,
        retry=True,
        parent_proposal_id=initial["proposal_id"],
    )
    assert retry["tool"] == "registration_retry"
    assert retry["tool_commit"] == "b" * 40
    assert retry["source_up_hypothesis"] == "-z"
    assert "retry" in retry["artifact_hashes"]


def test_candidate_failure_removes_partial_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mesh = tmp_path / "raw.ply"
    mesh.write_bytes(b"raw proposal bytes")
    staging = tmp_path / "staging"
    staging.mkdir()

    def fail_after_mkdir(*_args, out_dir, **_kwargs):
        Path(out_dir).mkdir(parents=True)
        (Path(out_dir) / "partial").write_text("partial")
        raise RuntimeError("injected")

    monkeypatch.setattr(orchestrator_runtime, "align_and_probe", fail_after_mkdir)
    with pytest.raises(RuntimeError, match="injected"):
        e3._process_candidate(
            _proposal(mesh),
            {"observation_sha256": "a" * 64, "observation_mask_fraction": 0.8},
            np.zeros((60, 3)),
            staging,
            tmp_path / "final",
            "b" * 40,
        )
    assert not any(staging.rglob("*"))


def test_smoke_uses_frozen_probe_and_exercises_terminal_actions(tmp_path: Path) -> None:
    policies = yaml.safe_load(POLICIES.read_text(encoding="utf-8"))
    out = tmp_path / "agentic"
    result = e3.run_smoke(policies, "smoke-freeze", out)
    assert result["actions_observed"] == ["abstain", "accept", "reject", "retry"]
    assert json.loads((out / "smoke/result.json").read_text()) == result


def test_aggregate_publication_rolls_back_partial_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "agentic"
    out.mkdir()
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "item.json").write_text("{}\n")
    real_link = os.link
    calls = 0

    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        return real_link(source, destination)

    monkeypatch.setattr(e3.os, "link", fail_second)
    with pytest.raises(OSError, match="injected"):
        e3._publish_aggregate(out, {"a.csv": b"a\n", "b.csv": b"b\n"}, selected)
    assert not (out / "a.csv").exists()
    assert not (out / "b.csv").exists()
    assert not (out / "selected_assets").exists()


def test_aggregate_seal_authenticates_selected_files_and_empty_directories(
    tmp_path: Path,
) -> None:
    out = tmp_path / "agentic"
    out.mkdir()
    selected = tmp_path / "selected"
    for policy_id in e3.POLICY_IDS:
        (selected / policy_id / "empty_scene").mkdir(parents=True)
    payload = b'{"selected": true}\n'
    item = selected / "A0" / "populated_scene" / "obj_00.json"
    item.parent.mkdir()
    item.write_bytes(payload)
    e3._publish_aggregate(out, {"table.csv": b"x\n"}, selected)
    seal = json.loads((out / "aggregate_seal.json").read_text())
    assert seal["selected_asset_members"] == {
        "A0/populated_scene/obj_00.json": {
            "sha256": sha256_file(item),
            "size_bytes": len(payload),
        }
    }
    assert set(seal["selected_asset_directories"]) >= {
        f"{policy_id}/empty_scene" for policy_id in e3.POLICY_IDS
    }


def test_empty_scene_materializes_all_selected_policy_directories(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    e3._write_controller_shard(
        staging,
        tmp_path / "final",
        {"schema_version": 1},
        [],
        {policy_id: {} for policy_id in e3.POLICY_IDS},
    )
    for policy_id in e3.POLICY_IDS:
        assert (staging / "selected_assets" / policy_id).is_dir()


def test_control_loader_rejects_undeclared_selected_asset(tmp_path: Path) -> None:
    out = tmp_path / "agentic"
    directory = out / "control" / "5748ce6f01"
    directory.mkdir(parents=True)
    selected = {policy_id: {} for policy_id in e3.POLICY_IDS}
    selected["A0"]["obj_00"] = {"selected_asset": None}
    e3._write_controller_shard(
        directory,
        directory,
        {
            "schema_version": 1,
            "scene_id": "5748ce6f01",
            "evaluation_geometry_read": False,
        },
        [],
        selected,
    )
    (directory / "selected_assets" / "A1" / "extra.json").write_text("{}\n")
    with pytest.raises(ValueError, match="seal coverage mismatch"):
        e3._load_control_scene(out, "5748ce6f01")


def test_control_loader_rejects_missing_or_tampered_selected_asset(
    tmp_path: Path,
) -> None:
    out = tmp_path / "agentic"
    directory = out / "control" / "5748ce6f01"
    directory.mkdir(parents=True)
    selected = {policy_id: {} for policy_id in e3.POLICY_IDS}
    selected["A0"]["obj_00"] = {"selected_asset": None}
    e3._write_controller_shard(
        directory,
        directory,
        {
            "schema_version": 1,
            "scene_id": "5748ce6f01",
            "evaluation_geometry_read": False,
        },
        [],
        selected,
    )
    item = directory / "selected_assets" / "A0" / "obj_00.json"
    original = item.read_bytes()
    item.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        e3._load_control_scene(out, "5748ce6f01")
    item.write_bytes(original)
    item.unlink()
    with pytest.raises(ValueError, match="seal coverage mismatch"):
        e3._load_control_scene(out, "5748ce6f01")


def test_coverage_sweep_retains_jobs_with_no_processed_proposal() -> None:
    policies = yaml.safe_load(POLICIES.read_text(encoding="utf-8"))
    control = {"scene": {"proposals": []}}
    evaluation = {
        "scene": {
            "proposal_metrics": {},
            "rows": [
                {"job_id": "scene/obj_00", "policy_id": policy_id}
                for policy_id in e3.POLICY_IDS
            ],
        }
    }
    rows = e3._coverage_sweep(policies, control, evaluation, planned_jobs=1)
    assert rows
    assert all(row["planned_jobs"] == 1 for row in rows)
    assert all(row["accepted_jobs"] == 0 for row in rows)
    assert all(row["build_coverage"] == 0.0 for row in rows)


def test_evaluation_seals_controller_before_opening_gt_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def load_control(*_args, **_kwargs):
        events.append("control")
        return (
            tmp_path,
            {"freeze_id": "freeze", "evaluation_geometry_read": False},
            {"members": {"controller_shard.json": "a" * 64}},
        )

    def open_full_inventory(*_args, **_kwargs):
        events.append("inventory_with_gt_references")
        raise RuntimeError("stop after order check")

    monkeypatch.setattr(e3, "_load_control_scene", load_control)
    monkeypatch.setattr(e3, "_load_inventory", open_full_inventory)
    with pytest.raises(RuntimeError, match="order check"):
        e3.run_evaluate("freeze", tmp_path, "5748ce6f01")
    assert events == ["control", "inventory_with_gt_references"]


def test_preliminary_aggregate_never_promotes_retry_count_to_headline() -> None:
    assert e3._preliminary_retry_claim_status(19, 20) == "case_study_only"
    assert (
        e3._preliminary_retry_claim_status(20, 20)
        == "retry_count_threshold_met_preliminary_only"
    )
    with pytest.raises(ValueError, match="retry counts"):
        e3._preliminary_retry_claim_status(-1, 20)


def test_policy_runtime_includes_failed_invoked_initials_and_retry() -> None:
    shard = {
        "proposals": [
            {"job_id": "job", "tool": "trellis", "wall_s": 1.0},
        ],
        "tool_failures": [
            {"job_id": "job", "tool": "reconviagen", "wall_s": 2.0},
            {"job_id": "job", "tool": "registration_retry", "wall_s": 3.0},
            # Typed unavailable inputs were not invoked and carry no wall time.
            {"job_id": "job", "failure_type": "typed_unavailable"},
        ],
    }
    ledger = {"job_id": "job"}
    assert e3._policy_runtime_seconds("A0", "job", shard, ledger) == 1.0
    assert e3._policy_runtime_seconds("A1", "job", shard, ledger) == 3.0
    assert e3._policy_runtime_seconds("A2", "job", shard, ledger) == 3.0
    assert e3._policy_runtime_seconds("A3", "job", shard, ledger) == 6.0
    assert e3._policy_runtime_seconds("A4", "job", shard, ledger) == 6.0


def test_controller_inventory_schema_rejects_undeclared_fields() -> None:
    identity = {"path": "/data/example", "size_bytes": 1, "sha256": "a" * 64}
    job = {
        "freeze_id": "freeze",
        "job_id": "5748ce6f01/obj_00",
        "scene_id": "5748ce6f01",
        "object_slot": "obj_00",
        "artifact_paths": {"rgba": "outputs/example.png"},
        "artifact_hashes": {"rgba": "b" * 64},
        "construction_evidence": {
            "source_frame": "frame.JPG",
            "bbox_px": [0, 0, 2, 2],
            "legacy_mask_provenance": "opaque_fixed_input",
        },
    }
    jobs = {
        "schema_version": 1,
        "freeze_id": "freeze",
        "study_scope": "conditional_fixed_gt_factory_jobs",
        "counts": {"scenes": 1, "jobs": 1, "policy_object_rows": 5},
        "source_contract": {
            "jobs_sha256": "c" * 64,
            "jobs_hash_method": "canonical_structured_sha256",
            "contract_sha256": "d" * 64,
            "contract_hash_method": "canonical_json_sha256",
            "contract_freeze_id": "contract",
            "code_commit": "e" * 40,
        },
        "observation_protocol": dict(e3.OBSERVATION_PROTOCOL),
        "scenes": [{
            "scene_id": "5748ce6f01",
            "source_scene_gaussian": dict(identity),
            "camera_artifacts": {
                "intrinsics": dict(identity), "poses": dict(identity)
            },
            "jobs": [job],
        }],
    }
    provenance = {
        "artifact_bytes_hash_frozen": True,
        "generator_commit": None,
        "checkpoint_identity_recorded": False,
        "runtime_manifest_recorded": False,
        "claim_status": "preliminary_only",
    }
    proposal_rows = []
    for tool in ("trellis", "reconviagen"):
        proposal_rows.append({
            "freeze_id": "freeze",
            "job_id": job["job_id"],
            "scene_id": job["scene_id"],
            "object_slot": job["object_slot"],
            "proposal_id": f"{job['job_id']}:{tool}",
            "tool_id": tool,
            "availability": "available",
            "artifact_paths": {"raw_mesh": "outputs/m.ply", "raw_gaussian": "outputs/g.ply"},
            "artifact_hashes": {"raw_mesh": "f" * 64, "raw_gaussian": "1" * 64},
            "artifact_sizes": {"raw_mesh": 1, "raw_gaussian": 1},
            "construction_evidence": {},
            "raw_generator_provenance": dict(provenance),
        })
    proposals = {
        "schema_version": 1,
        "freeze_id": "freeze",
        "counts": {
            "jobs": 1,
            "trellis_available": 1,
            "reconviagen_available": 1,
            "reconviagen_typed_unavailable": 0,
        },
        "proposals": proposal_rows,
    }
    e3._validate_controller_inventory_schema(jobs, proposals)
    jobs["scenes"][0]["jobs"][0]["secret_quality"] = 0.99
    with pytest.raises(ValueError, match="schema keys differ"):
        e3._validate_controller_inventory_schema(jobs, proposals)
