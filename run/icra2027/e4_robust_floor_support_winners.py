#!/usr/bin/env python3
"""Read the sealed, scene-scoped E4 robust-floor/support winner record.

This module records one diagnostic winner.  It does not make the winner a
global default and cannot authorize GPU, large-rollout, or paper use.  The
reader deliberately revalidates the frozen aggregate, the winning comparison,
the variant spec, and every member of the winning common-static package before
returning the record.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


EXPECTED_EVIDENCE_ROOT = Path(os.environ.get("SIMANY_EXPECTED_EVIDENCE_ROOT", "/opt/phirie/evidence/SimAny"))
EXTERNAL_SWEEP_ID = (
    "icra2027-contract-v1-e4-8c7b796b3f25-robust-floor-support-"
    "20260904T192849Z"
)
EXTERNAL_CODE_COMMIT = "8c7b796b3f25d0cdf3faf24bc6c8fe4ccfc424f0"
WINNER_SCENE_ID = "d755b3d9d8"
WINNER_VARIANT_ID = "d755-fs"

AGGREGATE_KIND = "e4_robust_floor_support_aggregate_artifacts"
AGGREGATE_GATE_KIND = "e4_robust_floor_support_aggregate_gate"
COMPARISON_KIND = "e4_robust_floor_support_pair_comparison_artifacts"
COMPARISON_GATE_KIND = "e4_robust_floor_support_pair_comparison_gate"
STUDY_SCOPE = "e4_robust_floor_support_diagnostic_cpu_only"
COMMON_STATIC_KIND = "simany_room_static_package"

VARIANT_IDS = ("3db-f", "3db-fs", "d755-f", "d755-fs")
POLICIES = ("A0", "A4")
PAIRED_STABLE_SLOTS = ("obj_05", "obj_12")

WINNER_VARIANT_SPEC: dict[str, Any] = {
    "carve_side_top_margin_m": 0.02,
    "hull_bottom": "clip_scan_aabb_bottom",
    "intrusive_primitive": "fail",
    "lower_carve_margin_m": 0.0,
    "min_support_area_m2": 0.05,
    "plane_residual_tol_m": 0.02,
    "room_surface_policy": "robust_floor_plus_support",
    "scene_id": WINNER_SCENE_ID,
    "schema_version": 2,
    "support_clip_offset_m": 0.005,
    "support_z": "fitted_mean_20mm",
    "variant_id": WINNER_VARIANT_ID,
}

# These identities freeze the only source that this registry accepts.  Keeping
# them together also lets unit tests exercise the validator with a tiny sealed
# fixture without depending on the live evidence mount.
EXPECTED_WINNER_IDENTITY: dict[str, Any] = {
    "aggregate_manifest_sha256": (
        "7d12d1cf7c212c97da0f97596970c0da983f450a89da7abe17035a98b4a7d5a6"
    ),
    "aggregate_seal_sha256": (
        "a151813e2709248dbafc62aa0b5c2c5400ba7bb8eed6ebe5e75bc2ef826d8a33"
    ),
    "aggregate_gate_sha256": (
        "842cc3ab4487e916fbb7bef629f5e723d01e21c73e35af20aee848bffd2ea381"
    ),
    "comparison_manifest_sha256": (
        "99b979ded06854426f2cab4942086401ef6a27f7b0c499ca4f80a1f341c0bc19"
    ),
    "comparison_seal_sha256": (
        "ef1b4497c772cde2ef4917234e26c472c306ab3ca334835fa59fc7d2a6d37674"
    ),
    "comparison_gate_sha256": (
        "88bc9c784ad9e422618b2435bca159c2f5046c179666974bfcc76f81907cd07d"
    ),
    "variant_spec_sha256": (
        "5df576ac5f365f916de164c5e9523a64adef855c7f222eb9474b6449564cfae3"
    ),
    "common_static_manifest_sha256": (
        "cd07da64e0648cdc6f685fda0a48cf4fa13a5eb877dd5caf87749799b21c3077"
    ),
    "common_static_seal_sha256": (
        "35031e07d269a3cf8e8aaecd137a8ffac7aaf16bc4e44d9de82b4419a0bd5d35"
    ),
    "common_static_payload_sha256": (
        "908ae0c81d1ce6038588a6ad5905671b298fbfc0c0b7d3f48a857208b0895c5a"
    ),
    "common_static_content_sha256": (
        "c5892adad046946451410f4fa03f83ebf0f88e70e56d1a990fba2801cc9ee0b7"
    ),
    "common_static_xml_sha256": (
        "7daaf11fb314cd05632345d1c384df5191104b0532209f0f9c3a49e077cb1577"
    ),
    "common_static_background_sha256": (
        "700bd682cc0935088473acf9f2f3d6b5b441fe1f57a4332d1cea09544ce99ff9"
    ),
    "source_manifest_sha256": {
        "A0": "72c4b4ce6c0addef514d49c8ab566f401da11b892a99ad04388d18ecbe69aa5a",
        "A4": "10efa8e0eda1cea3a8a3f5caec0b86424c287d3ee65cd4428d4613a407826e65",
    },
}


class WinnerRegistryError(RuntimeError):
    """Raised when the frozen diagnostic winner cannot be authenticated."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validated_root(root: Path) -> Path:
    root = Path(root)
    if not root.is_absolute() or root.is_symlink():
        raise WinnerRegistryError("evidence root must be an absolute non-symlink")
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise WinnerRegistryError(f"evidence root is unavailable: {exc}") from exc
    if resolved != root or not root.is_dir():
        raise WinnerRegistryError("evidence root is not a canonical directory")
    return root


def _inside(path: Path, *, root: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise WinnerRegistryError(f"{label} escapes the evidence root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise WinnerRegistryError(f"{label} contains a symlink: {current}")
    return path


def _regular_directory(path: Path, *, root: Path, label: str) -> Path:
    path = _inside(path, root=root, label=label)
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise WinnerRegistryError(f"{label} is unavailable: {exc}") from exc
    if not stat.S_ISDIR(mode) or path.resolve(strict=True) != path:
        raise WinnerRegistryError(f"{label} is not a canonical regular directory")
    return path


def _regular_file(path: Path, *, root: Path, label: str) -> Path:
    path = _inside(path, root=root, label=label)
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise WinnerRegistryError(f"{label} is unavailable: {exc}") from exc
    if not stat.S_ISREG(mode) or path.resolve(strict=True) != path:
        raise WinnerRegistryError(f"{label} is not a canonical regular file")
    return path


def _read_json(
    path: Path, *, root: Path, label: str
) -> tuple[Any, dict[str, Any]]:
    path = _regular_file(path, root=root, label=label)
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise WinnerRegistryError(f"{label} is invalid JSON: {exc}") from exc
    return value, {"sha256": _sha256_bytes(payload), "size_bytes": len(payload)}


def _read_identity(path: Path, *, root: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, root=root, label=label)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise WinnerRegistryError(f"cannot read {label}: {exc}") from exc
    return {"sha256": _sha256_bytes(payload), "size_bytes": len(payload)}


def _require_code(value: Any, *, label: str) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"code_root", "commit", "dirty"}
        or value.get("commit") != EXTERNAL_CODE_COMMIT
        or value.get("dirty") is not False
        or not isinstance(value.get("code_root"), str)
        or not Path(value["code_root"]).is_absolute()
    ):
        raise WinnerRegistryError(f"{label} code identity differs")


def _require_no_authorization(value: Mapping[str, Any], *, label: str) -> None:
    for key in ("gpu_launch_allowed", "large_rollout_launch_allowed", "paper_ready"):
        if value.get(key) is not False:
            raise WinnerRegistryError(f"{label} unexpectedly authorizes {key}")


def _validate_bundle(
    directory: Path,
    *,
    root: Path,
    kind: str,
    expected_prefix: str,
) -> dict[str, Any]:
    directory = _regular_directory(directory, root=root, label=f"{kind} bundle")
    actual_names = sorted(path.name for path in directory.iterdir())
    if actual_names != ["gate.json", "manifest.json", "seal.json"]:
        raise WinnerRegistryError(f"{kind} bundle inventory differs")

    manifest, manifest_identity = _read_json(
        directory / "manifest.json", root=root, label=f"{kind} manifest"
    )
    seal, seal_identity = _read_json(
        directory / "seal.json", root=root, label=f"{kind} seal"
    )
    gate, gate_identity = _read_json(
        directory / "gate.json", root=root, label=f"{kind} gate"
    )
    expected = EXPECTED_WINNER_IDENTITY
    if manifest_identity["sha256"] != expected[f"{expected_prefix}_manifest_sha256"]:
        raise WinnerRegistryError(f"{kind} manifest identity differs")
    if seal_identity["sha256"] != expected[f"{expected_prefix}_seal_sha256"]:
        raise WinnerRegistryError(f"{kind} seal identity differs")
    if gate_identity["sha256"] != expected[f"{expected_prefix}_gate_sha256"]:
        raise WinnerRegistryError(f"{kind} gate identity differs")

    expected_manifest_keys = {
        "code",
        "files",
        "manifest_kind",
        "schema_version",
        "study_scope",
        "sweep_id",
    }
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != expected_manifest_keys
        or manifest.get("manifest_kind") != kind
        or manifest.get("schema_version") != 1
        or manifest.get("study_scope") != STUDY_SCOPE
        or manifest.get("sweep_id") != EXTERNAL_SWEEP_ID
        or manifest.get("files") != {"gate.json": gate_identity}
    ):
        raise WinnerRegistryError(f"{kind} manifest binding differs")
    _require_code(manifest["code"], label=f"{kind} manifest")
    if seal != {
        "manifest_kind": kind,
        "members": {
            "gate.json": gate_identity,
            "manifest.json": manifest_identity,
        },
        "schema_version": 1,
    }:
        raise WinnerRegistryError(f"{kind} seal binding differs")
    if not isinstance(gate, Mapping):
        raise WinnerRegistryError(f"{kind} gate is not a mapping")
    return {
        "gate": dict(gate),
        "manifest_sha256": manifest_identity["sha256"],
        "seal_sha256": seal_identity["sha256"],
    }


def _safe_member_path(raw: Any) -> PurePosixPath:
    if not isinstance(raw, str) or not raw:
        raise WinnerRegistryError("common-static member path is invalid")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or path.as_posix() != raw
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise WinnerRegistryError(f"unsafe common-static member path: {raw!r}")
    return path


def _validate_common_static(
    directory: Path, *, root: Path, spec_sha256: str
) -> dict[str, Any]:
    directory = _regular_directory(directory, root=root, label="common-static package")
    manifest, manifest_identity = _read_json(
        directory / "manifest.json", root=root, label="common-static manifest"
    )
    seal, seal_identity = _read_json(
        directory / "seal.json", root=root, label="common-static seal"
    )
    expected = EXPECTED_WINNER_IDENTITY
    if (
        manifest_identity["sha256"]
        != expected["common_static_manifest_sha256"]
        or seal_identity["sha256"] != expected["common_static_seal_sha256"]
    ):
        raise WinnerRegistryError("common-static manifest or seal identity differs")
    manifest_keys = {
        "schema_version",
        "kind",
        "scene_id",
        "diagnostic_variant",
        "source_manifest_sha256",
        "members",
        "content_sha256",
        "static_xml_sha256",
        "background_sha256",
    }
    expected_variant = {
        "spec": WINNER_VARIANT_SPEC,
        "spec_file_sha256": spec_sha256,
    }
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != manifest_keys
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != COMMON_STATIC_KIND
        or manifest.get("scene_id") != WINNER_SCENE_ID
        or manifest.get("diagnostic_variant") != expected_variant
        or manifest.get("source_manifest_sha256")
        != expected["source_manifest_sha256"]
        or manifest.get("content_sha256")
        != expected["common_static_content_sha256"]
        or manifest.get("static_xml_sha256")
        != expected["common_static_xml_sha256"]
        or manifest.get("background_sha256")
        != expected["common_static_background_sha256"]
    ):
        raise WinnerRegistryError("common-static manifest binding differs")
    if seal != {
        "manifest_sha256": manifest_identity["sha256"],
        "schema_version": 1,
    }:
        raise WinnerRegistryError("common-static seal binding differs")

    members = manifest.get("members")
    if not isinstance(members, Mapping) or not members:
        raise WinnerRegistryError("common-static member inventory is empty")
    actual_members: set[str] = set()
    actual_directories: set[str] = set()
    for path in directory.rglob("*"):
        relative = path.relative_to(directory).as_posix()
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise WinnerRegistryError(
                f"cannot inspect common-static member: {exc}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise WinnerRegistryError(
                f"common-static package contains symlink: {relative}"
            )
        if stat.S_ISDIR(mode):
            actual_directories.add(relative)
        elif stat.S_ISREG(mode):
            if relative not in {"manifest.json", "seal.json"}:
                actual_members.add(relative)
        else:
            raise WinnerRegistryError(
                f"common-static package contains non-regular entry: {relative}"
            )
    if actual_directories != {"room_collision"}:
        raise WinnerRegistryError("common-static directory inventory differs")
    if actual_members != set(members):
        raise WinnerRegistryError("common-static member inventory differs")
    for raw, identity in members.items():
        relative = _safe_member_path(raw)
        if (
            not isinstance(identity, Mapping)
            or set(identity) != {"sha256", "size_bytes"}
            or _read_identity(
                directory / Path(*relative.parts),
                root=root,
                label=f"common-static member {raw}",
            )
            != identity
        ):
            raise WinnerRegistryError(f"common-static member identity differs: {raw}")
    if _sha256_bytes(_canonical_json_bytes(members)) != manifest["content_sha256"]:
        raise WinnerRegistryError("common-static content identity differs")
    background = members.get("background.obj")
    if (
        not isinstance(background, Mapping)
        or background.get("sha256") != manifest["background_sha256"]
    ):
        raise WinnerRegistryError("common-static background identity differs")

    payload, payload_identity = _read_json(
        directory / "payload.json", root=root, label="common-static payload"
    )
    if payload_identity["sha256"] != expected["common_static_payload_sha256"]:
        raise WinnerRegistryError("common-static payload identity differs")
    payload_keys = {
        "schema_version",
        "scene_id",
        "diagnostic_variant",
        "asset_xml_lines",
        "geom_xml_lines",
        "room_report",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != payload_keys
        or payload.get("schema_version") != 1
        or payload.get("scene_id") != WINNER_SCENE_ID
        or payload.get("diagnostic_variant") != expected_variant
        or not isinstance(payload.get("room_report"), Mapping)
    ):
        raise WinnerRegistryError("common-static payload binding differs")
    assets, geoms = payload.get("asset_xml_lines"), payload.get("geom_xml_lines")
    if (
        not isinstance(assets, list)
        or not isinstance(geoms, list)
        or any(not isinstance(line, str) for line in assets + geoms)
    ):
        raise WinnerRegistryError("common-static XML payload differs")
    xml_identity = _sha256_bytes(
        _canonical_json_bytes(
            {"asset_xml_lines": assets, "geom_xml_lines": geoms}
        )
    )
    if xml_identity != manifest["static_xml_sha256"]:
        raise WinnerRegistryError("common-static XML identity differs")
    return {
        "manifest_sha256": manifest_identity["sha256"],
        "seal_sha256": seal_identity["sha256"],
        "content_sha256": manifest["content_sha256"],
        "static_xml_sha256": manifest["static_xml_sha256"],
        "background_sha256": manifest["background_sha256"],
        "source_manifest_sha256": dict(manifest["source_manifest_sha256"]),
    }


def _validate_comparison(
    gate: Mapping[str, Any], *, common_identity: Mapping[str, Any], spec_sha256: str
) -> None:
    _require_code(gate.get("code"), label="winning comparison")
    _require_no_authorization(gate, label="winning comparison")
    comparable_common_identity = {
        key: common_identity[key]
        for key in (
            "manifest_sha256",
            "content_sha256",
            "static_xml_sha256",
            "background_sha256",
        )
    }
    if (
        gate.get("manifest_kind") != COMPARISON_GATE_KIND
        or gate.get("sweep_id") != EXTERNAL_SWEEP_ID
        or gate.get("variant_id") != WINNER_VARIANT_ID
        or gate.get("scene_id") != WINNER_SCENE_ID
        or gate.get("status") != "pass"
        or gate.get("comparison_pass") is not True
        or gate.get("errors") != {}
        or gate.get("paired_stable_at_least_two") is not True
        or gate.get("paired_stable_count") != len(PAIRED_STABLE_SLOTS)
        or gate.get("paired_stable_slots") != list(PAIRED_STABLE_SLOTS)
        or gate.get("diagnostic_variant")
        != {"spec": WINNER_VARIANT_SPEC, "spec_file_sha256": spec_sha256}
        or gate.get("policy_scientific_pass") != {"A0": True, "A4": True}
        or gate.get("static_package_identity_equal") is not True
        or gate.get("static_package_identity_by_policy")
        != {policy: comparable_common_identity for policy in POLICIES}
    ):
        raise WinnerRegistryError("winning comparison scientific binding differs")


def _validate_aggregate(
    gate: Mapping[str, Any], *, comparison: Mapping[str, Any], comparison_hash: str
) -> None:
    _require_code(gate.get("code"), label="robust aggregate")
    _require_no_authorization(gate, label="robust aggregate")
    if (
        gate.get("manifest_kind") != AGGREGATE_GATE_KIND
        or gate.get("sweep_id") != EXTERNAL_SWEEP_ID
        or gate.get("status") != "complete"
        or gate.get("errors") != {}
        or gate.get("comparison_error_variants") != []
        or gate.get("variant_count") != len(VARIANT_IDS)
        or gate.get("complete_variant_count") != len(VARIANT_IDS)
        or gate.get("variant_ids") != list(VARIANT_IDS)
        or gate.get("comparison_pass_count") != 1
        or gate.get("scene_winners")
        != {"3db0a1c8f3": None, WINNER_SCENE_ID: WINNER_VARIANT_ID}
    ):
        raise WinnerRegistryError("robust aggregate completion binding differs")
    variants = gate.get("variants")
    decisions = gate.get("scene_decisions")
    if (
        not isinstance(variants, Mapping)
        or set(variants) != set(VARIANT_IDS)
        or not isinstance(decisions, Mapping)
        or set(decisions) != {"3db0a1c8f3", WINNER_SCENE_ID}
    ):
        raise WinnerRegistryError("robust aggregate roster differs")
    recorded = variants.get(WINNER_VARIANT_ID)
    if not isinstance(recorded, Mapping):
        raise WinnerRegistryError("robust aggregate lacks the winning comparison")
    replay = dict(recorded)
    if (
        replay.pop("bundle_manifest_sha256", None) != comparison_hash
        or replay != comparison
    ):
        raise WinnerRegistryError("winning comparison does not replay from aggregate")
    decision = decisions[WINNER_SCENE_ID]
    if (
        not isinstance(decision, Mapping)
        or decision.get("winner") != WINNER_VARIANT_ID
        or decision.get("reason") != "f_fails_and_fs_passes"
        or decision.get("f_paired_stable_count") != 1
        or decision.get("fs_paired_stable_count") != 2
    ):
        raise WinnerRegistryError("winning scene decision differs")


def read_validated_winner_registry(root: Path) -> dict[str, Any]:
    """Authenticate and return the sole scene-scoped diagnostic winner.

    ``root`` is the evidence checkout root (normally
    :data:`EXPECTED_EVIDENCE_ROOT`), not the sweep directory.  Any missing,
    changed, symlinked, non-finite, or semantically inconsistent artifact
    raises :class:`WinnerRegistryError`.
    """
    root = _validated_root(root)
    sweep_root = _regular_directory(
        root / "outputs/icra2027" / EXTERNAL_SWEEP_ID,
        root=root,
        label="external robust-floor/support sweep",
    )
    spec, spec_identity = _read_json(
        sweep_root / "variants" / WINNER_VARIANT_ID / "variant_spec.json",
        root=root,
        label="winning variant spec",
    )
    if (
        spec != WINNER_VARIANT_SPEC
        or spec_identity["sha256"]
        != EXPECTED_WINNER_IDENTITY["variant_spec_sha256"]
    ):
        raise WinnerRegistryError("winning variant spec identity differs")

    common_identity = _validate_common_static(
        sweep_root / "variants" / WINNER_VARIANT_ID / "common_static",
        root=root,
        spec_sha256=spec_identity["sha256"],
    )
    comparison_bundle = _validate_bundle(
        sweep_root / "variants" / WINNER_VARIANT_ID / "comparison",
        root=root,
        kind=COMPARISON_KIND,
        expected_prefix="comparison",
    )
    _validate_comparison(
        comparison_bundle["gate"],
        common_identity=common_identity,
        spec_sha256=spec_identity["sha256"],
    )
    aggregate_bundle = _validate_bundle(
        sweep_root / "aggregate",
        root=root,
        kind=AGGREGATE_KIND,
        expected_prefix="aggregate",
    )
    _validate_aggregate(
        aggregate_bundle["gate"],
        comparison=comparison_bundle["gate"],
        comparison_hash=comparison_bundle["manifest_sha256"],
    )

    return {
        "schema_version": 1,
        "registry_status": "validated",
        "source_sweep_id": EXTERNAL_SWEEP_ID,
        "source_code_commit": EXTERNAL_CODE_COMMIT,
        "winners": {
            WINNER_SCENE_ID: {
                "scene_id": WINNER_SCENE_ID,
                "winner": WINNER_VARIANT_ID,
                "room_surface_policy": "robust_floor_plus_support",
                "variant_spec": dict(WINNER_VARIANT_SPEC),
                "variant_spec_sha256": spec_identity["sha256"],
                "aggregate_bundle_manifest_sha256": aggregate_bundle[
                    "manifest_sha256"
                ],
                "aggregate_bundle_seal_sha256": aggregate_bundle["seal_sha256"],
                "comparison_bundle_manifest_sha256": comparison_bundle[
                    "manifest_sha256"
                ],
                "comparison_bundle_seal_sha256": comparison_bundle["seal_sha256"],
                "common_static_manifest_sha256": common_identity["manifest_sha256"],
                "common_static_seal_sha256": common_identity["seal_sha256"],
                "common_static_identity": {
                    key: common_identity[key]
                    for key in (
                        "content_sha256",
                        "static_xml_sha256",
                        "background_sha256",
                        "source_manifest_sha256",
                    )
                },
                "paired_stable_count": len(PAIRED_STABLE_SLOTS),
                "paired_stable_slots": list(PAIRED_STABLE_SLOTS),
            }
        },
        "authorization": {
            "gpu_launch_allowed": False,
            "large_rollout_launch_allowed": False,
            "paper_ready": False,
        },
    }


if __name__ == "__main__":
    print(
        json.dumps(
            read_validated_winner_registry(EXPECTED_EVIDENCE_ROOT),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
