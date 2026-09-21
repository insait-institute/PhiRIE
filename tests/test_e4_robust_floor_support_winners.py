"""Contract tests for the sealed scene-scoped robust winner registry."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "run/icra2027/e4_robust_floor_support_winners.py"


def _load_registry():
    spec = importlib.util.spec_from_file_location(
        "e4_robust_floor_support_winners_test", REGISTRY
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _publish_bundle(
    directory: Path, *, module: Any, kind: str, gate: dict[str, Any]
) -> dict[str, str]:
    directory.mkdir(parents=True)
    gate_path = directory / "gate.json"
    _write_json(gate_path, gate)
    manifest = {
        "code": gate["code"],
        "files": {"gate.json": _identity(gate_path)},
        "manifest_kind": kind,
        "schema_version": 1,
        "study_scope": module.STUDY_SCOPE,
        "sweep_id": module.EXTERNAL_SWEEP_ID,
    }
    manifest_path = directory / "manifest.json"
    _write_json(manifest_path, manifest)
    seal_path = directory / "seal.json"
    _write_json(
        seal_path,
        {
            "manifest_kind": kind,
            "members": {
                "gate.json": _identity(gate_path),
                "manifest.json": _identity(manifest_path),
            },
            "schema_version": 1,
        },
    )
    return {
        "gate_sha256": _identity(gate_path)["sha256"],
        "manifest_sha256": _identity(manifest_path)["sha256"],
        "seal_sha256": _identity(seal_path)["sha256"],
    }


def _sealed_fixture(tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "evidence"
    variant = (
        root
        / "outputs/icra2027"
        / module.EXTERNAL_SWEEP_ID
        / "variants"
        / module.WINNER_VARIANT_ID
    )
    variant.mkdir(parents=True)
    spec_path = variant / "variant_spec.json"
    _write_json(spec_path, module.WINNER_VARIANT_SPEC)
    spec_sha256 = _identity(spec_path)["sha256"]

    package = variant / "common_static"
    collision = package / "room_collision"
    collision.mkdir(parents=True)
    (package / "background.obj").write_text("v 0 0 0\n", encoding="utf-8")
    _write_json(package / "background_carve.json", {"synthetic": True})
    (collision / "part.obj").write_text("v 0 0 0\n", encoding="utf-8")
    expected_variant = {
        "spec": module.WINNER_VARIANT_SPEC,
        "spec_file_sha256": spec_sha256,
    }
    payload = {
        "schema_version": 1,
        "scene_id": module.WINNER_SCENE_ID,
        "diagnostic_variant": expected_variant,
        "asset_xml_lines": [],
        "geom_xml_lines": [],
        "room_report": {},
    }
    _write_json(package / "payload.json", payload)
    members = {
        relative: _identity(package / relative)
        for relative in (
            "background.obj",
            "background_carve.json",
            "payload.json",
            "room_collision/part.obj",
        )
    }
    source_manifests = {"A0": "a" * 64, "A4": "b" * 64}
    common_manifest = {
        "schema_version": 1,
        "kind": module.COMMON_STATIC_KIND,
        "scene_id": module.WINNER_SCENE_ID,
        "diagnostic_variant": expected_variant,
        "source_manifest_sha256": source_manifests,
        "members": members,
        "content_sha256": _canonical_hash(members),
        "static_xml_sha256": _canonical_hash(
            {"asset_xml_lines": [], "geom_xml_lines": []}
        ),
        "background_sha256": members["background.obj"]["sha256"],
    }
    _write_json(package / "manifest.json", common_manifest)
    _write_json(
        package / "seal.json",
        {
            "schema_version": 1,
            "manifest_sha256": _identity(package / "manifest.json")["sha256"],
        },
    )
    common_identity = {
        "manifest_sha256": _identity(package / "manifest.json")["sha256"],
        "content_sha256": common_manifest["content_sha256"],
        "static_xml_sha256": common_manifest["static_xml_sha256"],
        "background_sha256": common_manifest["background_sha256"],
    }

    code = {
        "code_root": str(ROOT),
        "commit": module.EXTERNAL_CODE_COMMIT,
        "dirty": False,
    }
    comparison = {
        "code": code,
        "diagnostic_variant": expected_variant,
        "errors": {},
        "gpu_launch_allowed": False,
        "large_rollout_launch_allowed": False,
        "manifest_kind": module.COMPARISON_GATE_KIND,
        "paired_stable_at_least_two": True,
        "paired_stable_count": 2,
        "paired_stable_slots": list(module.PAIRED_STABLE_SLOTS),
        "paper_ready": False,
        "policy_scientific_pass": {"A0": True, "A4": True},
        "scene_id": module.WINNER_SCENE_ID,
        "static_package_identity_by_policy": {
            policy: common_identity for policy in module.POLICIES
        },
        "static_package_identity_equal": True,
        "status": "pass",
        "comparison_pass": True,
        "sweep_id": module.EXTERNAL_SWEEP_ID,
        "variant_id": module.WINNER_VARIANT_ID,
    }
    comparison_identity = _publish_bundle(
        variant / "comparison",
        module=module,
        kind=module.COMPARISON_KIND,
        gate=comparison,
    )

    variants = {variant_id: {} for variant_id in module.VARIANT_IDS}
    variants[module.WINNER_VARIANT_ID] = {
        **comparison,
        "bundle_manifest_sha256": comparison_identity["manifest_sha256"],
    }
    aggregate = {
        "code": code,
        "comparison_error_variants": [],
        "comparison_pass_count": 1,
        "complete_variant_count": len(module.VARIANT_IDS),
        "errors": {},
        "gpu_launch_allowed": False,
        "large_rollout_launch_allowed": False,
        "manifest_kind": module.AGGREGATE_GATE_KIND,
        "paper_ready": False,
        "scene_decisions": {
            "3db0a1c8f3": {"winner": None},
            module.WINNER_SCENE_ID: {
                "f_paired_stable_count": 1,
                "fs_paired_stable_count": 2,
                "reason": "f_fails_and_fs_passes",
                "winner": module.WINNER_VARIANT_ID,
            },
        },
        "scene_winners": {
            "3db0a1c8f3": None,
            module.WINNER_SCENE_ID: module.WINNER_VARIANT_ID,
        },
        "status": "complete",
        "sweep_id": module.EXTERNAL_SWEEP_ID,
        "variant_count": len(module.VARIANT_IDS),
        "variant_ids": list(module.VARIANT_IDS),
        "variants": variants,
    }
    aggregate_identity = _publish_bundle(
        variant.parents[1] / "aggregate",
        module=module,
        kind=module.AGGREGATE_KIND,
        gate=aggregate,
    )
    expected_identity = {
        "aggregate_manifest_sha256": aggregate_identity["manifest_sha256"],
        "aggregate_seal_sha256": aggregate_identity["seal_sha256"],
        "aggregate_gate_sha256": aggregate_identity["gate_sha256"],
        "comparison_manifest_sha256": comparison_identity["manifest_sha256"],
        "comparison_seal_sha256": comparison_identity["seal_sha256"],
        "comparison_gate_sha256": comparison_identity["gate_sha256"],
        "variant_spec_sha256": spec_sha256,
        "common_static_manifest_sha256": common_identity["manifest_sha256"],
        "common_static_seal_sha256": _identity(package / "seal.json")["sha256"],
        "common_static_payload_sha256": members["payload.json"]["sha256"],
        "common_static_content_sha256": common_manifest["content_sha256"],
        "common_static_xml_sha256": common_manifest["static_xml_sha256"],
        "common_static_background_sha256": common_manifest["background_sha256"],
        "source_manifest_sha256": source_manifests,
    }
    monkeypatch.setattr(module, "EXPECTED_WINNER_IDENTITY", expected_identity)
    return root, package, expected_identity


def test_registry_revalidates_and_returns_only_non_authorizing_d755_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_registry()
    root, _package, expected = _sealed_fixture(tmp_path, module, monkeypatch)

    registry = module.read_validated_winner_registry(root)

    assert set(registry) == {
        "schema_version",
        "registry_status",
        "source_sweep_id",
        "source_code_commit",
        "winners",
        "authorization",
    }
    assert registry["registry_status"] == "validated"
    assert set(registry["winners"]) == {module.WINNER_SCENE_ID}
    winner = registry["winners"][module.WINNER_SCENE_ID]
    assert winner["winner"] == "d755-fs"
    assert winner["room_surface_policy"] == "robust_floor_plus_support"
    assert winner["aggregate_bundle_manifest_sha256"] \
        == expected["aggregate_manifest_sha256"]
    assert winner["comparison_bundle_manifest_sha256"] \
        == expected["comparison_manifest_sha256"]
    assert winner["paired_stable_slots"] == ["obj_05", "obj_12"]
    assert registry["authorization"] == {
        "gpu_launch_allowed": False,
        "large_rollout_launch_allowed": False,
        "paper_ready": False,
    }


def test_registry_rejects_a_changed_common_static_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_registry()
    root, package, _expected = _sealed_fixture(tmp_path, module, monkeypatch)
    with (package / "room_collision/part.obj").open("ab") as handle:
        handle.write(b"v 1 1 1\n")

    with pytest.raises(module.WinnerRegistryError, match="member identity differs"):
        module.read_validated_winner_registry(root)


def test_registry_rejects_relative_roots() -> None:
    module = _load_registry()
    with pytest.raises(module.WinnerRegistryError, match="absolute non-symlink"):
        module.read_validated_winner_registry(Path("relative/evidence"))
