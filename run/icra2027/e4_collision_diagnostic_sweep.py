#!/usr/bin/env python3
"""Sealed, CPU-only diagnostics for the E4 room-collision repair sweep.

The sweep is intentionally incapable of releasing a GPU or paper run.  It
materializes fresh E3 factories, builds one immutable static-room package per
diagnostic variant, evaluates A0 and A4 against that exact shared package, and
then publishes pairwise and aggregate diagnostic gates.  Scientific failures
remain successful diagnostic jobs; provenance or artifact failures do not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_EVIDENCE_ROOT = Path("/group/worldcept/PhiRIE/code/SimAny")
EXPECTED_E3_ROOT = Path(
    "outputs/icra2027/"
    "icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic"
)
SCANNETPP_ROOT = Path("/data/ScanNetpp")
SPLATS_ROOT = Path("/data/ScanNetppv2_gsplat/splats")
POLICIES = ("A0", "A4")
EXPORT_STEPS = 1000
STABLE_DRIFT_LIMIT_M = 0.03
INITIAL_STATIC_PENETRATION_LIMIT_M = 0.005
COVERAGE_KEYS = {
    "n_points",
    "n_rays",
    "occupancy_agreement",
    "penetration_frac",
    "ray_dist_p95_m",
    "ray_hit_agreement",
}
SELECTION_RULE = (
    "eligible_requires_pair_comparison_pass_and_at_least_two_paired_stable_slots",
    "maximize_paired_stable_count",
    "prefer_raw_hulls_over_scan_aabb_clipping",
    "prefer_intrusive_primitive_fail_over_demotion",
    "prefer_plane_residual_20mm_then_25mm_then_30mm",
    "prefer_extent_support_over_fitted_mean_when_both_are_eligible",
    "prefer_higher_min_occupancy_agreement",
    "prefer_higher_min_ray_hit_agreement",
    "prefer_lower_max_penetration_fraction",
    "prefer_lower_max_ray_distance_p95",
    "break_remaining_ties_by_variant_id",
)
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SLOT_RE = re.compile(r"^obj_[0-9]+$")


class DiagnosticSweepError(RuntimeError):
    """A frozen sweep, provenance, or artifact contract is invalid."""


def _variant(
    variant_id: str,
    scene_id: str,
    *,
    hull_bottom: str,
    intrusive_primitive: str,
    plane_residual_tol_m: float,
    support_z: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "variant_id": variant_id,
        "scene_id": scene_id,
        "hull_bottom": hull_bottom,
        "intrusive_primitive": intrusive_primitive,
        "plane_residual_tol_m": plane_residual_tol_m,
        "support_z": support_z,
        "min_support_area_m2": 0.05,
        "carve_side_top_margin_m": 0.02,
        "lower_carve_margin_m": 0.0,
        "support_clip_offset_m": 0.005,
    }


def _frozen_variants() -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hull_id, hull_bottom in (
        ("hraw", "raw"),
        ("hclip", "clip_scan_aabb_bottom"),
    ):
        for primitive_id, intrusive_primitive in (
            ("pfail", "fail"),
            ("pdemote", "demote_to_residual"),
        ):
            rows.append(
                _variant(
                    f"3db-{hull_id}-{primitive_id}",
                    "3db0a1c8f3",
                    hull_bottom=hull_bottom,
                    intrusive_primitive=intrusive_primitive,
                    plane_residual_tol_m=0.02,
                    support_z="extent",
                )
            )
    for residual_id, residual in (("r20", 0.02), ("r25", 0.025), ("r30", 0.03)):
        for support_id, support_z in (
            ("zextent", "extent"),
            ("zmean", "fitted_mean_20mm"),
        ):
            for hull_id, hull_bottom in (
                ("hraw", "raw"),
                ("hclip", "clip_scan_aabb_bottom"),
            ):
                rows.append(
                    _variant(
                        f"d755-{residual_id}-{support_id}-{hull_id}",
                        "d755b3d9d8",
                        hull_bottom=hull_bottom,
                        intrusive_primitive="fail",
                        plane_residual_tol_m=residual,
                        support_z=support_z,
                    )
                )
    result = {row["variant_id"]: row for row in rows}
    if len(result) != 16:
        raise AssertionError("frozen diagnostic sweep must contain exactly 16 variants")
    return result


VARIANTS = _frozen_variants()
VARIANT_IDS = tuple(VARIANTS)


def _candidate():
    from robo.eval import e4_candidate_screen as candidate

    return candidate


def _validated_id(value: str) -> str:
    if not isinstance(value, str) or ID_RE.fullmatch(value) is None:
        raise DiagnosticSweepError("sweep ID is unsafe")
    return value


def _validated_variant(value: str) -> str:
    if value not in VARIANTS:
        raise DiagnosticSweepError(f"variant is outside the frozen sweep: {value}")
    return value


def _validated_policy(value: str) -> str:
    if value not in POLICIES:
        raise DiagnosticSweepError(f"invalid construction policy: {value}")
    return value


def _code_snapshot(expected_commit: str) -> dict[str, Any]:
    try:
        snapshot = _candidate()._code_snapshot(expected_commit)
    except Exception as exc:
        raise DiagnosticSweepError(str(exc)) from exc
    expected = {"code_root": str(CODE_ROOT), "commit": expected_commit, "dirty": False}
    if snapshot != expected:
        raise DiagnosticSweepError("diagnostic-sweep code snapshot differs")
    return snapshot


def _evidence_root() -> Path:
    try:
        root = _candidate().evidence_root()
    except Exception as exc:
        raise DiagnosticSweepError(str(exc)) from exc
    if root != EXPECTED_EVIDENCE_ROOT or root == CODE_ROOT:
        raise DiagnosticSweepError("diagnostic-sweep evidence/code root binding differs")
    return root


def _sweep_root(root: Path, sweep_id: str) -> Path:
    return root / "outputs" / "icra2027" / _validated_id(sweep_id)


def _variant_root(root: Path, sweep_id: str, variant_id: str) -> Path:
    return _sweep_root(root, sweep_id) / "variants" / _validated_variant(variant_id)


def _factory(
    root: Path, sweep_id: str, variant_id: str, policy_id: str
) -> Path:
    spec = VARIANTS[_validated_variant(variant_id)]
    return (
        _variant_root(root, sweep_id, variant_id)
        / "construction_variants"
        / _validated_policy(policy_id)
        / f"{spec['scene_id']}_factory"
    )


def _spec_path(root: Path, sweep_id: str, variant_id: str) -> Path:
    return _variant_root(root, sweep_id, variant_id) / "variant_spec.json"


def _package(root: Path, sweep_id: str, variant_id: str) -> Path:
    return _variant_root(root, sweep_id, variant_id) / "common_static"


def _policy_bundle(
    root: Path, sweep_id: str, variant_id: str, policy_id: str
) -> Path:
    return (
        _variant_root(root, sweep_id, variant_id)
        / "policy_evals"
        / _validated_policy(policy_id)
    )


def _comparison_bundle(root: Path, sweep_id: str, variant_id: str) -> Path:
    return _variant_root(root, sweep_id, variant_id) / "comparison"


def _aggregate_bundle(root: Path, sweep_id: str) -> Path:
    return _sweep_root(root, sweep_id) / "aggregate"


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    if not path.is_file() or path.is_symlink() or path.resolve(strict=True) != path:
        raise DiagnosticSweepError(f"not a regular canonical JSON file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticSweepError(f"invalid JSON {path}: {exc}") from exc


def _write_new_file(path: Path, payload: bytes) -> None:
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise DiagnosticSweepError(f"output parent is missing or symlinked: {path.parent}")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o400)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise DiagnosticSweepError(f"cannot publish fresh file {path}: {exc}") from exc


def _spec_and_hash(
    root: Path, sweep_id: str, variant_id: str
) -> tuple[dict[str, Any], str]:
    expected = VARIANTS[_validated_variant(variant_id)]
    path = _spec_path(root, sweep_id, variant_id)
    raw = _read_json(path)
    try:
        from robo.sim.room_collision import validate_room_diagnostic_spec

        canonical = validate_room_diagnostic_spec(
            raw, expected_scene_id=expected["scene_id"]
        )
    except (TypeError, ValueError) as exc:
        raise DiagnosticSweepError(str(exc)) from exc
    if canonical != expected:
        raise DiagnosticSweepError("variant spec differs from the frozen sweep cell")
    digest = _sha256(path)
    if SHA256_RE.fullmatch(digest) is None:
        raise DiagnosticSweepError("variant spec hash is invalid")
    return canonical, digest


def _materialization(
    factory: Path, *, scene_id: str, policy_id: str, root: Path, expected_commit: str
) -> dict[str, Any]:
    from robo.eval.e3_factory_materializer import validate_materialized_factory

    try:
        report = validate_materialized_factory(
            factory,
            expected_scene_id=scene_id,
            expected_policy_id=policy_id,
            repository_root=root,
        )
    except Exception as exc:
        raise DiagnosticSweepError(f"materialization validation failed: {exc}") from exc
    if report.get("validator_commit") != expected_commit:
        raise DiagnosticSweepError("materialization validator commit differs")
    roster = report.get("roster")
    if not isinstance(roster, Mapping):
        raise DiagnosticSweepError("materialization roster is absent")
    for key in ("accepted_slots", "object_slots"):
        slots = roster.get(key)
        if (
            not isinstance(slots, list)
            or len(slots) != len(set(slots))
            or any(not isinstance(slot, str) or SLOT_RE.fullmatch(slot) is None for slot in slots)
        ):
            raise DiagnosticSweepError(f"materialization {key} roster is invalid")
    if not set(roster["accepted_slots"]).issubset(roster["object_slots"]):
        raise DiagnosticSweepError("accepted roster is not a subset of discovered roster")
    return report


def _source_binding(report: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "e3_claim_status",
        "e3_code_commit",
        "e3_producer_commit",
        "e3_freeze_id",
        "e3_root",
        "input_identities_sha256",
        "materializer_commit",
        "policy_id",
        "provenance",
        "roster",
        "scene_id",
        "source_scene_sha256",
        "study_scope",
        "validator_commit",
    )
    try:
        return {key: report[key] for key in keys}
    except KeyError as exc:
        raise DiagnosticSweepError(
            f"materialization report lacks source binding {exc.args[0]}"
        ) from exc


def _tree_inventory(path: Path, *, factory: Path, root: Path, label: str) -> dict[str, Any]:
    from robo.eval import e4_region_pilot as sealed_cpu

    try:
        return sealed_cpu._tree_inventory(path, factory=factory, root=root, label=label)
    except Exception as exc:
        raise DiagnosticSweepError(str(exc)) from exc


def _object_tree(
    factory: Path, *, root: Path, scene_id: str, policy_id: str
) -> dict[str, Any]:
    return _tree_inventory(
        factory / "objects",
        factory=factory,
        root=root,
        label=f"diagnostic source objects {scene_id}/{policy_id}",
    )


def _export_environment(factory: Path, *, root: Path, scene_id: str) -> dict[str, str]:
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(CODE_ROOT),
        "SIMANY_EVIDENCE_ROOT": str(root),
        "SIMANY_OUT": str(factory),
        "SIMANY_ROOT": str(CODE_ROOT),
        "SIMANY_SCANNETPP_ROOT": str(SCANNETPP_ROOT),
        "SIMANY_SPLATS_ROOT": str(SPLATS_ROOT),
        "SIMANY_SCENE": scene_id,
        "OMP_NUM_THREADS": os.environ.get("SLURM_CPUS_PER_TASK", "1"),
    }
    for name in ("TMPDIR", "XDG_CACHE_HOME"):
        raw = os.environ.get(name)
        if not raw:
            continue
        path = Path(raw)
        try:
            path.resolve(strict=True).relative_to(root)
        except (FileNotFoundError, ValueError) as exc:
            raise DiagnosticSweepError(f"{name} escapes the evidence root") from exc
        if not path.is_dir() or path.is_symlink():
            raise DiagnosticSweepError(f"{name} is not a regular directory")
        environment[name] = str(path)
    return environment


def _exporter_argv(
    *,
    factories: Sequence[Path],
    spec_path: Path,
    package_out: Path | None = None,
    package_in: Path | None = None,
    test: bool = False,
) -> list[str]:
    if (package_out is None) == (package_in is None):
        raise DiagnosticSweepError("exactly one static package direction is required")
    argv = [sys.executable, "-m", "robo.sim.export_mjcf"]
    if test:
        argv.append("--test")
    argv.extend(["--collision-mode", "room"])
    for factory in factories:
        argv.extend(["--background-carve-factory", str(factory)])
    argv.extend(["--room-diagnostic-spec", str(spec_path)])
    if package_out is not None:
        argv.extend(["--room-static-package-out", str(package_out)])
    else:
        argv.extend(["--room-static-package", str(package_in)])
    return argv


def _static_identity(value: Any) -> dict[str, str]:
    keys = {
        "manifest_sha256",
        "content_sha256",
        "static_xml_sha256",
        "background_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise DiagnosticSweepError("static package identity schema differs")
    result = dict(value)
    if any(not isinstance(result[key], str) or SHA256_RE.fullmatch(result[key]) is None for key in keys):
        raise DiagnosticSweepError("static package identity hashes are invalid")
    return result


def _load_package(
    package_dir: Path, *, spec: Mapping[str, Any], spec_sha256: str
) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        from robo.sim import export_mjcf

        # export_mjcf's sealed loader intentionally binds against the active
        # robo.config scene.  This orchestration process handles 16 cells and
        # does not inherit SIMANY_SCENE; bind it only for this pure load and
        # restore the process-global value immediately afterwards.
        previous_scene = export_mjcf.C.SCENE_ID
        try:
            export_mjcf.C.SCENE_ID = str(spec["scene_id"])
            package = export_mjcf.load_common_room_static_package(
                package_dir, dict(spec), spec_sha256
            )
        finally:
            export_mjcf.C.SCENE_ID = previous_scene
    except Exception as exc:
        raise DiagnosticSweepError(f"static package validation failed: {exc}") from exc
    expected_keys = {
        "asset_xml_lines",
        "geom_xml_lines",
        "room_report",
        "source_manifest_sha256",
        "static_package_identity",
        "package_dir",
    }
    if not isinstance(package, Mapping) or set(package) != expected_keys:
        raise DiagnosticSweepError("static package loader return schema differs")
    return dict(package), _static_identity(package["static_package_identity"])


def _publish(
    destination: Path,
    *,
    kind: str,
    gate: Mapping[str, Any],
    code: Mapping[str, Any],
    sweep_id: str,
) -> dict[str, Any]:
    try:
        return _candidate()._publish_bundle(
            destination,
            manifest_kind=kind,
            payloads={"gate.json": _json_bytes(dict(gate))},
            manifest_fields={
                "code": dict(code),
                "study_scope": "e4_collision_diagnostic_sweep_cpu_only",
                "sweep_id": sweep_id,
            },
        )
    except Exception as exc:
        raise DiagnosticSweepError(str(exc)) from exc


def _validated_bundle_gate(
    directory: Path,
    *,
    root: Path,
    expected_kind: str,
    sweep_id: str,
    variant_id: str | None = None,
    policy_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        bundle = _candidate()._validate_bundle(
            directory, root=root, expected_kind=expected_kind
        )
    except Exception as exc:
        raise DiagnosticSweepError(str(exc)) from exc
    gate = _read_json(directory / "gate.json")
    if not isinstance(gate, Mapping) or gate.get("sweep_id") != sweep_id:
        raise DiagnosticSweepError(f"{expected_kind} sweep binding differs")
    for key, expected in (("variant_id", variant_id), ("policy_id", policy_id)):
        if expected is not None and gate.get(key) != expected:
            raise DiagnosticSweepError(f"{expected_kind} {key} binding differs")
    return dict(gate), bundle


def build_common(
    *, sweep_id: str, variant_id: str, e3_root: str | Path, expected_commit: str
) -> dict[str, Any]:
    """Materialize both arms and build one sealed shared static package."""
    sweep_id, variant_id = _validated_id(sweep_id), _validated_variant(variant_id)
    code = _code_snapshot(expected_commit)
    root = _evidence_root()
    source = Path(e3_root)
    if source.is_absolute() or source != EXPECTED_E3_ROOT:
        raise DiagnosticSweepError("E3 root differs from the frozen source")
    spec_expected = VARIANTS[variant_id]
    spec_path = _spec_path(root, sweep_id, variant_id)
    _write_new_file(spec_path, _json_bytes(spec_expected))
    spec, spec_sha256 = _spec_and_hash(root, sweep_id, variant_id)

    from robo.eval.e3_factory_materializer import materialize_factory_variant

    reports: dict[str, dict[str, Any]] = {}
    object_trees: dict[str, dict[str, Any]] = {}
    factories = [_factory(root, sweep_id, variant_id, policy) for policy in POLICIES]
    for policy, factory in zip(POLICIES, factories):
        try:
            materialize_factory_variant(
                e3_root=source,
                scene_id=spec["scene_id"],
                policy_id=policy,
                out=factory,
            )
        except Exception as exc:
            raise DiagnosticSweepError(
                f"cannot materialize {variant_id}/{policy}: {exc}"
            ) from exc
        reports[policy] = _materialization(
            factory,
            scene_id=spec["scene_id"],
            policy_id=policy,
            root=root,
            expected_commit=expected_commit,
        )
        object_trees[policy] = _object_tree(
            factory, root=root, scene_id=spec["scene_id"], policy_id=policy
        )

    package_dir = _package(root, sweep_id, variant_id)
    subprocess.run(
        _exporter_argv(
            factories=factories,
            spec_path=spec_path,
            package_out=package_dir,
        ),
        cwd=CODE_ROOT,
        env=_export_environment(factories[0], root=root, scene_id=spec["scene_id"]),
        check=True,
    )
    for policy, factory in zip(POLICIES, factories):
        replay = _materialization(
            factory,
            scene_id=spec["scene_id"],
            policy_id=policy,
            root=root,
            expected_commit=expected_commit,
        )
        if _source_binding(replay) != _source_binding(reports[policy]):
            raise DiagnosticSweepError("materialization changed while building package")
        if _object_tree(
            factory, root=root, scene_id=spec["scene_id"], policy_id=policy
        ) != object_trees[policy]:
            raise DiagnosticSweepError("materialized object tree changed while building package")
        for generated in (factory / "sim", factory / "sim_export"):
            if generated.exists() or generated.is_symlink():
                raise DiagnosticSweepError("common builder mutated a policy factory")
    package, identity = _load_package(
        package_dir, spec=spec, spec_sha256=spec_sha256
    )
    return {
        "code": code,
        "diagnostic_variant": {"spec": spec, "spec_file_sha256": spec_sha256},
        "gpu_launch_allowed": False,
        "large_rollout_launch_allowed": False,
        "manifest_kind": "e4_collision_diagnostic_common_build",
        "materializations": {
            policy: _source_binding(reports[policy]) for policy in POLICIES
        },
        "package_keys": sorted(package),
        "paper_ready": False,
        "scene_id": spec["scene_id"],
        "static_package_identity": identity,
        "status": "built",
        "sweep_id": sweep_id,
        "variant_id": variant_id,
    }


def _initial_contact_diagnostics(
    xml_path: Path, accepted_slots: Sequence[str]
) -> dict[str, Any]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    accepted = set(accepted_slots)
    max_static = 0.0
    pairs: dict[tuple[str, str, str, str], float] = {}
    for index in range(data.ncon):
        contact = data.contact[index]
        penetration = max(0.0, -float(contact.dist))
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1) or "world"
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2) or "world"
        if (body1 == 0 and name2 in accepted) or (body2 == 0 and name1 in accepted):
            max_static = max(max_static, penetration)
            if penetration > 0.0:
                geom_name1 = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
                    or f"geom_{geom1}"
                )
                geom_name2 = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2)
                    or f"geom_{geom2}"
                )
                key = (name1, geom_name1, name2, geom_name2)
                pairs[key] = max(penetration, pairs.get(key, 0.0))
    return {
        "contact_count": int(data.ncon),
        "max_static_penetration_m": max_static,
        "static_penetrations": [
            {
                "body_a": key[0],
                "geom_a": key[1],
                "body_b": key[2],
                "geom_b": key[3],
                "penetration_m": pairs[key],
            }
            for key in sorted(pairs)
        ],
    }


def _finite_json(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_json(item) for item in value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _finite_json(item) for key, item in value.items())
    return False


def _collision_coverage(value: Any) -> dict[str, Any] | None:
    """Canonicalize the frozen collision-coverage schema or fail it closed.

    Coverage is a scientific ranking input, not free-form diagnostics.  Keep
    malformed reports in the sealed policy gate, but make them ineligible.
    """
    if not isinstance(value, Mapping) or set(value) != COVERAGE_KEYS:
        return None
    n_rays, n_points = value.get("n_rays"), value.get("n_points")
    if (
        not isinstance(n_rays, int)
        or isinstance(n_rays, bool)
        or n_rays <= 0
        or not isinstance(n_points, int)
        or isinstance(n_points, bool)
        or n_points <= 0
    ):
        return None
    numeric: dict[str, float] = {}
    for key in (
        "occupancy_agreement",
        "penetration_frac",
        "ray_dist_p95_m",
        "ray_hit_agreement",
    ):
        observed = value.get(key)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
        ):
            return None
        numeric[key] = float(observed)
    if not (
        0.0 <= numeric["occupancy_agreement"] <= 1.0
        and 0.0 <= numeric["penetration_frac"] <= 1.0
        and 0.0 <= numeric["ray_hit_agreement"] <= 1.0
        and numeric["ray_dist_p95_m"] >= 0.0
    ):
        return None
    return {"n_points": n_points, "n_rays": n_rays, **numeric}


def _conservative_pair_coverage(
    policies: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    if set(policies) != set(POLICIES):
        return None
    rows = [_collision_coverage(policies[policy].get("collision_coverage")) for policy in POLICIES]
    if any(row is None for row in rows):
        return None
    valid = [row for row in rows if row is not None]
    return {
        "min_n_points": min(row["n_points"] for row in valid),
        "min_n_rays": min(row["n_rays"] for row in valid),
        "min_occupancy_agreement": min(
            row["occupancy_agreement"] for row in valid
        ),
        "max_penetration_frac": max(row["penetration_frac"] for row in valid),
        "max_ray_dist_p95_m": max(row["ray_dist_p95_m"] for row in valid),
        "min_ray_hit_agreement": min(row["ray_hit_agreement"] for row in valid),
    }


def _build_policy_gate(
    *, sweep_id: str, variant_id: str, policy_id: str, expected_commit: str
) -> dict[str, Any]:
    sweep_id = _validated_id(sweep_id)
    variant_id, policy_id = _validated_variant(variant_id), _validated_policy(policy_id)
    code = _code_snapshot(expected_commit)
    root = _evidence_root()
    spec, spec_sha256 = _spec_and_hash(root, sweep_id, variant_id)
    factory = _factory(root, sweep_id, variant_id, policy_id)
    materialization = _materialization(
        factory,
        scene_id=spec["scene_id"],
        policy_id=policy_id,
        root=root,
        expected_commit=expected_commit,
    )
    roster = materialization["roster"]
    accepted = list(roster["accepted_slots"])
    discovered = list(roster["object_slots"])
    package, expected_static_identity = _load_package(
        _package(root, sweep_id, variant_id), spec=spec, spec_sha256=spec_sha256
    )
    export_dir = factory / "sim_export"
    required = {
        name: export_dir / name
        for name in (
            "isaac_manifest.json",
            "mujoco_settle.json",
            "room_collision_report.json",
            "scene.xml",
        )
    }
    for path in required.values():
        if not path.is_file() or path.is_symlink():
            raise DiagnosticSweepError(f"policy export artifact is absent: {path}")
    collision = _read_json(required["room_collision_report.json"])
    settle = _read_json(required["mujoco_settle.json"])
    isaac = _read_json(required["isaac_manifest.json"])
    if not all(isinstance(value, Mapping) for value in (collision, settle, isaac)):
        raise DiagnosticSweepError("policy export JSON roots must be objects")
    diagnostic_variant = collision.get("diagnostic_variant")
    coverage = _collision_coverage(collision.get("coverage"))
    expected_variant = {"spec": spec, "spec_file_sha256": spec_sha256}
    raw = collision.get("raw_diagnostics")
    reported_identity = _static_identity(
        raw.get("static_package_identity") if isinstance(raw, Mapping) else None
    )
    drift_raw = settle.get("drift_m")
    drift = (
        {str(key): float(value) for key, value in drift_raw.items()}
        if isinstance(drift_raw, Mapping)
        else {}
    )
    stable_slots = sorted(
        slot for slot, value in drift.items() if value < STABLE_DRIFT_LIMIT_M
    )
    benchmark = collision.get("benchmark")
    exclusion = collision.get("collision_exclusion")
    carve = collision.get("background_carve")
    isaac_rows = isaac.get("objects")
    isaac_names = (
        [row.get("name") if isinstance(row, Mapping) else None for row in isaac_rows]
        if isinstance(isaac_rows, list)
        else []
    )
    parsed = ET.parse(required["scene.xml"])
    body_names = [body.get("name") for body in parsed.findall("./worldbody/body")]
    contacts = _initial_contact_diagnostics(required["scene.xml"], accepted)
    generated_trees = {
        name: _tree_inventory(
            factory / name,
            factory=factory,
            root=root,
            label=f"diagnostic eval {variant_id}/{policy_id}/{name}",
        )
        for name in ("sim", "sim_export")
    }
    checks = {
        "background_is_paired_common_roster": isinstance(carve, Mapping)
        and carve.get("mode") == "paired_policy_union"
        and carve.get("policy_ids") == ["A0", "A4"]
        and carve.get("discovered_slots") == discovered
        and carve.get("carved_slots") == sorted(discovered),
        "benchmark_is_finite_1000_steps": isinstance(benchmark, Mapping)
        and benchmark.get("finite") is True
        and benchmark.get("steps") == EXPORT_STEPS
        and isinstance(benchmark.get("state_hash"), str)
        and SHA256_RE.fullmatch(benchmark["state_hash"]) is not None,
        "collision_coverage_schema_valid": coverage is not None,
        "diagnostic_variant_matches": diagnostic_variant == expected_variant,
        "diagnostics_are_finite_json": isinstance(raw, Mapping) and _finite_json(raw),
        "raw_diagnostic_axes_present": isinstance(raw, Mapping)
        and {
            "common_carve_hulls",
            "room_collision",
            "feature_extraction",
            "dynamic_object_hulls",
            "static_package_identity",
        }.issubset(raw),
        "floor_is_scannetpp_world_z0": collision.get("floor_z_m") == 0.0
        and collision.get("floor_source") == "scannetpp_world_frame_z0",
        "initial_static_penetration_below_5mm": contacts[
            "max_static_penetration_m"
        ]
        <= INITIAL_STATIC_PENETRATION_LIMIT_M,
        "isaac_roster_matches_accepted": len(isaac_names) == len(set(isaac_names))
        and set(isaac_names) == set(accepted),
        "room_collision_mode_matches": collision.get("mode") == "room",
        "scene_body_roster_matches_accepted": len(body_names) == len(set(body_names))
        and set(body_names) == set(accepted),
        "settle_roster_matches_accepted": set(drift) == set(accepted)
        and all(math.isfinite(value) and value >= 0.0 for value in drift.values()),
        "settle_summary_replays": settle.get("n") == len(drift)
        and settle.get("stable_3cm") == len(stable_slots),
        "static_package_identity_matches": reported_identity == expected_static_identity,
    }
    unresolved = (
        exclusion.get("unresolved_intrusion_count")
        if isinstance(exclusion, Mapping)
        else None
    )
    checks["no_unresolved_static_intrusion"] = (
        isinstance(unresolved, int) and not isinstance(unresolved, bool) and unresolved == 0
    )
    scientific_pass = all(checks.values())
    return {
        "accepted_slots": accepted,
        "artifacts": {
            name: _candidate()._identity(path, root=root)
            for name, path in sorted(required.items())
        },
        "checks": checks,
        "code": code,
        "collision_exclusion": dict(exclusion) if isinstance(exclusion, Mapping) else None,
        "collision_coverage": coverage,
        "diagnostic_variant": expected_variant,
        "discovered_slots": discovered,
        "drift_m": drift,
        "generated_trees": generated_trees,
        "gpu_launch_allowed": False,
        "initial_contacts": contacts,
        "large_rollout_launch_allowed": False,
        "manifest_kind": "e4_collision_diagnostic_policy_eval_gate",
        "materialization": _source_binding(materialization),
        "package_keys": sorted(package),
        "paper_ready": False,
        "policy_id": policy_id,
        "raw_diagnostics": dict(raw),
        "scene_id": spec["scene_id"],
        "scientific_pass": scientific_pass,
        "stable_count": len(stable_slots),
        "stable_slots": stable_slots,
        "static_package_identity": reported_identity,
        "status": "pass" if scientific_pass else "diagnostic_fail",
        "sweep_id": sweep_id,
        "unresolved_static_intrusion_count": unresolved,
        "variant_id": variant_id,
    }


def evaluate_policy(
    *, sweep_id: str, variant_id: str, policy_id: str, expected_commit: str
) -> dict[str, Any]:
    """Consume the common package, run one policy, and seal its diagnostics."""
    sweep_id = _validated_id(sweep_id)
    variant_id, policy_id = _validated_variant(variant_id), _validated_policy(policy_id)
    code = _code_snapshot(expected_commit)
    root = _evidence_root()
    spec, spec_sha256 = _spec_and_hash(root, sweep_id, variant_id)
    factories = [_factory(root, sweep_id, variant_id, policy) for policy in POLICIES]
    factory = _factory(root, sweep_id, variant_id, policy_id)
    _materialization(
        factory,
        scene_id=spec["scene_id"],
        policy_id=policy_id,
        root=root,
        expected_commit=expected_commit,
    )
    for generated in (factory / "sim", factory / "sim_export"):
        if generated.exists() or generated.is_symlink():
            raise DiagnosticSweepError(f"refusing non-fresh policy export: {generated}")
    package_dir = _package(root, sweep_id, variant_id)
    _load_package(package_dir, spec=spec, spec_sha256=spec_sha256)
    subprocess.run(
        _exporter_argv(
            factories=factories,
            spec_path=_spec_path(root, sweep_id, variant_id),
            package_in=package_dir,
            test=True,
        ),
        cwd=CODE_ROOT,
        env=_export_environment(factory, root=root, scene_id=spec["scene_id"]),
        check=True,
    )
    gate = _build_policy_gate(
        sweep_id=sweep_id,
        variant_id=variant_id,
        policy_id=policy_id,
        expected_commit=expected_commit,
    )
    bundle = _publish(
        _policy_bundle(root, sweep_id, variant_id, policy_id),
        kind="e4_collision_diagnostic_policy_eval_artifacts",
        gate=gate,
        code=code,
        sweep_id=sweep_id,
    )
    return {**gate, "bundle_manifest_sha256": bundle["manifest_sha256"]}


def _build_comparison_gate(
    *, sweep_id: str, variant_id: str, expected_commit: str
) -> dict[str, Any]:
    sweep_id, variant_id = _validated_id(sweep_id), _validated_variant(variant_id)
    code = _code_snapshot(expected_commit)
    root = _evidence_root()
    spec, spec_sha256 = _spec_and_hash(root, sweep_id, variant_id)
    policies: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for policy in POLICIES:
        try:
            recomputed = _build_policy_gate(
                sweep_id=sweep_id,
                variant_id=variant_id,
                policy_id=policy,
                expected_commit=expected_commit,
            )
            published, bundle = _validated_bundle_gate(
                _policy_bundle(root, sweep_id, variant_id, policy),
                root=root,
                expected_kind="e4_collision_diagnostic_policy_eval_artifacts",
                sweep_id=sweep_id,
                variant_id=variant_id,
                policy_id=policy,
            )
            if not _candidate()._same_replay_structure(published, recomputed):
                raise DiagnosticSweepError("published policy gate does not replay")
            policies[policy] = {
                **recomputed,
                "bundle_manifest_sha256": bundle["manifest_sha256"],
            }
        except Exception as exc:
            errors[policy] = f"{type(exc).__name__}: {exc}"
    identities = {
        policy: gate["static_package_identity"] for policy, gate in policies.items()
    }
    coverages = {
        policy: gate.get("collision_coverage") for policy, gate in policies.items()
    }
    static_equal = (
        set(identities) == set(POLICIES)
        and identities["A0"] == identities["A4"]
    )
    coverage_equal = (
        set(coverages) == set(POLICIES)
        and coverages["A0"] is not None
        and coverages["A0"] == coverages["A4"]
    )
    discovered_roster_equal = (
        set(policies) == set(POLICIES)
        and policies["A0"]["discovered_slots"] == policies["A4"]["discovered_slots"]
    )
    stable_intersection = (
        sorted(set(policies["A0"]["stable_slots"]) & set(policies["A4"]["stable_slots"]))
        if set(policies) == set(POLICIES)
        else []
    )
    policy_pass = {
        policy: bool(gate["scientific_pass"]) for policy, gate in policies.items()
    }
    comparison_pass = (
        not errors
        and static_equal
        and coverage_equal
        and discovered_roster_equal
        and set(policy_pass) == set(POLICIES)
        and all(policy_pass.values())
        and len(stable_intersection) >= 2
    )
    return {
        "code": code,
        "collision_coverage_by_policy": coverages,
        "collision_coverage_equal": coverage_equal,
        "conservative_collision_coverage": _conservative_pair_coverage(policies),
        "comparison_pass": comparison_pass,
        "diagnostic_variant": {"spec": spec, "spec_file_sha256": spec_sha256},
        "discovered_roster_equal": discovered_roster_equal,
        "errors": errors,
        "gpu_launch_allowed": False,
        "large_rollout_launch_allowed": False,
        "manifest_kind": "e4_collision_diagnostic_pair_comparison_gate",
        "paired_stable_count": len(stable_intersection),
        "paired_stable_at_least_two": len(stable_intersection) >= 2,
        "paired_stable_slots": stable_intersection,
        "paper_ready": False,
        "policies": policies,
        "policy_scientific_pass": policy_pass,
        "scene_id": spec["scene_id"],
        "static_package_identity_by_policy": identities,
        "static_package_identity_equal": static_equal,
        "status": "pass" if comparison_pass else "diagnostic_fail",
        "sweep_id": sweep_id,
        "variant_id": variant_id,
    }


def compare_pair(
    *, sweep_id: str, variant_id: str, expected_commit: str
) -> dict[str, Any]:
    """Publish an afterany-safe A0/A4 comparison for one sweep cell."""
    sweep_id, variant_id = _validated_id(sweep_id), _validated_variant(variant_id)
    code = _code_snapshot(expected_commit)
    root = _evidence_root()
    gate = _build_comparison_gate(
        sweep_id=sweep_id, variant_id=variant_id, expected_commit=expected_commit
    )
    bundle = _publish(
        _comparison_bundle(root, sweep_id, variant_id),
        kind="e4_collision_diagnostic_pair_comparison_artifacts",
        gate=gate,
        code=code,
        sweep_id=sweep_id,
    )
    return {**gate, "bundle_manifest_sha256": bundle["manifest_sha256"]}


def _ranking_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    coverage = row.get("conservative_collision_coverage")
    valid_coverage = coverage if isinstance(coverage, Mapping) else {}

    def lower(key: str) -> float:
        value = valid_coverage.get(key)
        return math.inf if value is None else float(value)

    def higher(key: str) -> float:
        value = valid_coverage.get(key)
        return math.inf if value is None else -float(value)

    intervention = row["intervention_rank"]
    return (
        -int(bool(row["comparison_pass"])),
        -int(row["paired_stable_count"]),
        int(intervention["uses_clipped_hulls"]),
        int(intervention["uses_primitive_demotion"]),
        float(intervention["plane_residual_tol_m"]),
        int(intervention["uses_fitted_mean_support"]),
        higher("min_occupancy_agreement"),
        higher("min_ray_hit_agreement"),
        lower("max_penetration_frac"),
        lower("max_ray_dist_p95_m"),
        row["variant_id"],
    )


def aggregate(*, sweep_id: str, expected_commit: str) -> dict[str, Any]:
    """Recompute all 16 cells and seal a diagnostics-only aggregate."""
    sweep_id = _validated_id(sweep_id)
    code = _code_snapshot(expected_commit)
    root = _evidence_root()
    variants: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for variant_id in VARIANT_IDS:
        try:
            recomputed = _build_comparison_gate(
                sweep_id=sweep_id,
                variant_id=variant_id,
                expected_commit=expected_commit,
            )
            published, bundle = _validated_bundle_gate(
                _comparison_bundle(root, sweep_id, variant_id),
                root=root,
                expected_kind="e4_collision_diagnostic_pair_comparison_artifacts",
                sweep_id=sweep_id,
                variant_id=variant_id,
            )
            if not _candidate()._same_replay_structure(published, recomputed):
                raise DiagnosticSweepError("published comparison does not replay")
            variants[variant_id] = {
                **recomputed,
                "bundle_manifest_sha256": bundle["manifest_sha256"],
            }
        except Exception as exc:
            errors[variant_id] = f"{type(exc).__name__}: {exc}"
    scene_rankings: dict[str, list[dict[str, Any]]] = {}
    for scene_id in ("3db0a1c8f3", "d755b3d9d8"):
        rows = []
        for variant_id, comparison in variants.items():
            if comparison["scene_id"] != scene_id:
                continue
            policies = comparison["policies"]
            penetration_values = [
                float(gate["initial_contacts"]["max_static_penetration_m"])
                for gate in policies.values()
            ]
            max_penetration = max(penetration_values) if penetration_values else None
            spec = VARIANTS[variant_id]
            rows.append(
                {
                    "comparison_pass": comparison["comparison_pass"],
                    "conservative_collision_coverage": comparison.get(
                        "conservative_collision_coverage"
                    ),
                    "intervention_rank": {
                        "uses_clipped_hulls": spec["hull_bottom"] != "raw",
                        "uses_primitive_demotion": spec["intrusive_primitive"]
                        != "fail",
                        "plane_residual_tol_m": spec["plane_residual_tol_m"],
                        "uses_fitted_mean_support": spec["support_z"] != "extent",
                    },
                    "max_initial_static_penetration_m": max_penetration,
                    "paired_stable_count": comparison["paired_stable_count"],
                    "policy_pass_count": sum(
                        bool(value)
                        for value in comparison["policy_scientific_pass"].values()
                    ),
                    "variant_id": variant_id,
                }
            )
        rows.sort(
            key=_ranking_key
        )
        scene_rankings[scene_id] = rows
    sweep_complete = not errors and set(variants) == set(VARIANT_IDS)
    scene_winners = {
        scene_id: (
            rows[0]["variant_id"]
            if sweep_complete and rows and rows[0]["comparison_pass"]
            else None
        )
        for scene_id, rows in scene_rankings.items()
    }
    gate = {
        "code": code,
        "comparison_pass_count": sum(
            bool(comparison["comparison_pass"]) for comparison in variants.values()
        ),
        "complete_variant_count": len(variants),
        "errors": errors,
        # This is a diagnostic sweep by construction.  Even a clean winner is
        # evidence for the next CPU gate, never authority for a large/GPU run.
        "gpu_launch_allowed": False,
        "large_rollout_launch_allowed": False,
        "manifest_kind": "e4_collision_diagnostic_sweep_aggregate_gate",
        "paper_ready": False,
        "scene_winners": scene_winners,
        "scene_rankings": scene_rankings,
        "selection_rule": list(SELECTION_RULE),
        "status": "complete" if sweep_complete else "incomplete",
        "sweep_id": sweep_id,
        "variant_count": len(VARIANT_IDS),
        "variant_ids": list(VARIANT_IDS),
        "variants": variants,
    }
    bundle = _publish(
        _aggregate_bundle(root, sweep_id),
        kind="e4_collision_diagnostic_sweep_aggregate_artifacts",
        gate=gate,
        code=code,
        sweep_id=sweep_id,
    )
    return {**gate, "bundle_manifest_sha256": bundle["manifest_sha256"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sweep-id", required=True)
    common.add_argument("--expected-code-commit", required=True)
    build = subparsers.add_parser("build-common", parents=[common])
    build.add_argument("--variant-id", choices=VARIANT_IDS, required=True)
    build.add_argument("--e3-root", required=True)
    evaluate = subparsers.add_parser("eval-policy", parents=[common])
    evaluate.add_argument("--variant-id", choices=VARIANT_IDS, required=True)
    evaluate.add_argument("--policy-id", choices=POLICIES, required=True)
    compare = subparsers.add_parser("compare-pair", parents=[common])
    compare.add_argument("--variant-id", choices=VARIANT_IDS, required=True)
    subparsers.add_parser("aggregate", parents=[common])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build-common":
            report = build_common(
                sweep_id=args.sweep_id,
                variant_id=args.variant_id,
                e3_root=args.e3_root,
                expected_commit=args.expected_code_commit,
            )
        elif args.command == "eval-policy":
            report = evaluate_policy(
                sweep_id=args.sweep_id,
                variant_id=args.variant_id,
                policy_id=args.policy_id,
                expected_commit=args.expected_code_commit,
            )
        elif args.command == "compare-pair":
            report = compare_pair(
                sweep_id=args.sweep_id,
                variant_id=args.variant_id,
                expected_commit=args.expected_code_commit,
            )
        else:
            report = aggregate(
                sweep_id=args.sweep_id,
                expected_commit=args.expected_code_commit,
            )
    except (
        DiagnosticSweepError,
        FileNotFoundError,
        OSError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
        ET.ParseError,
    ) as exc:
        print(
            f"[e4-collision-diagnostic-sweep] FAIL: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
