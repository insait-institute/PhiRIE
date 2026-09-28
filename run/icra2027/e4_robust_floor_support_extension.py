#!/usr/bin/env python3
"""Sealed CPU-only robust-floor extension for four additional E4 scenes.

The eight local cells are rebuilt independently: robust floor alone (F) and
robust floor plus associated supports (FS) for 27dd4da69e, acd95847c5,
1ada7a0617, and 25f3b7a318.  The earlier sealed d755b3d9d8 FS winner is
authenticated only in the final aggregate readout; none of its factories or
generated files is used by a local build or evaluation.

This remains a diagnostic.  It can never authorize GPU work, a large rollout,
or paper use.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[2]
BASE_RUNNER = CODE_ROOT / "run/icra2027/e4_robust_floor_support_diagnostic.py"
WINNER_REGISTRY = CODE_ROOT / "run/icra2027/e4_robust_floor_support_winners.py"
EXPECTED_EVIDENCE_ROOT = Path(os.environ.get("SIMANY_EXPECTED_EVIDENCE_ROOT", "/opt/phirie/evidence/SimAny"))
EXPECTED_E3_ROOT = Path(
    "outputs/icra2027/"
    "icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic"
)
POLICIES = ("A0", "A4")
SCENE_VARIANTS = {
    "27dd4da69e": ("27dd-f", "27dd-fs"),
    "acd95847c5": ("acd-f", "acd-fs"),
    "1ada7a0617": ("1ada-f", "1ada-fs"),
    "25f3b7a318": ("25f3-f", "25f3-fs"),
}
SCENE_IDS = tuple(SCENE_VARIANTS)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sealed module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ROBUST = _load_module(BASE_RUNNER, "e4_robust_floor_support_extension_base")
DiagnosticSweepError = _ROBUST.DiagnosticSweepError


def _variant(
    variant_id: str, scene_id: str, room_surface_policy: str
) -> dict[str, Any]:
    return _ROBUST._variant(variant_id, scene_id, room_surface_policy)


VARIANTS = {
    "27dd-f": _variant("27dd-f", "27dd4da69e", "robust_floor_only"),
    "27dd-fs": _variant(
        "27dd-fs", "27dd4da69e", "robust_floor_plus_support"
    ),
    "acd-f": _variant("acd-f", "acd95847c5", "robust_floor_only"),
    "acd-fs": _variant(
        "acd-fs", "acd95847c5", "robust_floor_plus_support"
    ),
    "1ada-f": _variant("1ada-f", "1ada7a0617", "robust_floor_only"),
    "1ada-fs": _variant(
        "1ada-fs", "1ada7a0617", "robust_floor_plus_support"
    ),
    "25f3-f": _variant("25f3-f", "25f3b7a318", "robust_floor_only"),
    "25f3-fs": _variant(
        "25f3-fs", "25f3b7a318", "robust_floor_plus_support"
    ),
}
VARIANT_IDS = tuple(VARIANTS)

# Every reused helper resolves these module globals at call time.  Bind both
# layers explicitly so materialization, spec validation, package paths, and
# replay all use only this frozen eight-cell matrix.
_ROBUST.VARIANTS = VARIANTS
_ROBUST.VARIANT_IDS = VARIANT_IDS
_ROBUST._BASE.VARIANTS = VARIANTS
_ROBUST._BASE.VARIANT_IDS = VARIANT_IDS

POLICY_KIND = _ROBUST.POLICY_KIND
COMPARISON_KIND = _ROBUST.COMPARISON_KIND
AGGREGATE_KIND = "e4_robust_floor_support_extension_aggregate_artifacts"


def build_common(
    *, sweep_id: str, variant_id: str, e3_root: str | Path, expected_commit: str
) -> dict[str, Any]:
    return _ROBUST.build_common(
        sweep_id=sweep_id,
        variant_id=variant_id,
        e3_root=e3_root,
        expected_commit=expected_commit,
    )


def evaluate_policy(
    *, sweep_id: str, variant_id: str, policy_id: str, expected_commit: str
) -> dict[str, Any]:
    return _ROBUST.evaluate_policy(
        sweep_id=sweep_id,
        variant_id=variant_id,
        policy_id=policy_id,
        expected_commit=expected_commit,
    )


def _build_comparison_gate(
    *, sweep_id: str, variant_id: str, expected_commit: str
) -> dict[str, Any]:
    return _ROBUST._build_comparison_gate(
        sweep_id=sweep_id,
        variant_id=variant_id,
        expected_commit=expected_commit,
    )


def compare_pair(
    *, sweep_id: str, variant_id: str, expected_commit: str
) -> dict[str, Any]:
    return _ROBUST.compare_pair(
        sweep_id=sweep_id,
        variant_id=variant_id,
        expected_commit=expected_commit,
    )


def _choose_scene_winner(
    scene_id: str, variants: Mapping[str, Mapping[str, Any]]
) -> tuple[str | None, str]:
    if scene_id not in SCENE_VARIANTS:
        raise DiagnosticSweepError(f"scene is outside the extension: {scene_id}")
    floor_id, support_id = SCENE_VARIANTS[scene_id]
    floor = variants.get(floor_id)
    support = variants.get(support_id)
    floor_pass = bool(floor and floor.get("comparison_pass"))
    support_pass = bool(support and support.get("comparison_pass"))
    if floor_pass:
        if support_pass and int(support["paired_stable_count"]) > int(
            floor["paired_stable_count"]
        ):
            return support_id, "fs_strictly_increases_paired_stable_over_passing_f"
        return floor_id, "f_passes_and_fs_does_not_strictly_increase_paired_stable"
    if support_pass:
        return support_id, "f_fails_and_fs_passes"
    return None, "neither_f_nor_fs_passes"


def _validated_external_winners(root: Path) -> Mapping[str, Any]:
    registry = _load_module(
        WINNER_REGISTRY, "e4_robust_floor_support_extension_winner_registry"
    )
    reader = getattr(registry, "read_validated_winner_registry", None)
    if not callable(reader):
        raise DiagnosticSweepError("winner registry lacks its sealed reader")
    try:
        result = reader(root)
    except Exception as exc:
        raise DiagnosticSweepError(f"external winner registry is invalid: {exc}") from exc
    if not isinstance(result, Mapping):
        raise DiagnosticSweepError("external winner registry result is not a mapping")
    expected_keys = {
        "schema_version",
        "registry_status",
        "source_sweep_id",
        "source_code_commit",
        "winners",
        "authorization",
    }
    winners = result.get("winners")
    authorization = result.get("authorization")
    if (
        set(result) != expected_keys
        or result.get("registry_status") != "validated"
        or not isinstance(winners, Mapping)
        or set(winners) != {"d755b3d9d8"}
        or not isinstance(winners["d755b3d9d8"], Mapping)
        or winners["d755b3d9d8"].get("scene_id") != "d755b3d9d8"
        or winners["d755b3d9d8"].get("winner") != "d755-fs"
        or not isinstance(authorization, Mapping)
        or set(authorization)
        != {"gpu_launch_allowed", "large_rollout_launch_allowed", "paper_ready"}
        or any(value is not False for value in authorization.values())
    ):
        raise DiagnosticSweepError("external winner registry schema/binding differs")
    return dict(result)


def _publish(
    destination: Path,
    *,
    gate: Mapping[str, Any],
    code: Mapping[str, Any],
    sweep_id: str,
) -> dict[str, Any]:
    try:
        return _ROBUST._BASE._candidate()._publish_bundle(
            destination,
            manifest_kind=AGGREGATE_KIND,
            payloads={"gate.json": _ROBUST._BASE._json_bytes(dict(gate))},
            manifest_fields={
                "code": dict(code),
                "study_scope": (
                    "e4_robust_floor_support_four_scene_extension_cpu_only"
                ),
                "sweep_id": sweep_id,
            },
        )
    except Exception as exc:
        raise DiagnosticSweepError(str(exc)) from exc


def aggregate(*, sweep_id: str, expected_commit: str) -> dict[str, Any]:
    sweep_id = _ROBUST._BASE._validated_id(sweep_id)
    code = _ROBUST._BASE._code_snapshot(expected_commit)
    root = _ROBUST._BASE._evidence_root()
    variants: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for variant_id in VARIANT_IDS:
        try:
            recomputed = _build_comparison_gate(
                sweep_id=sweep_id,
                variant_id=variant_id,
                expected_commit=expected_commit,
            )
            published, bundle = _ROBUST._BASE._validated_bundle_gate(
                _ROBUST._BASE._comparison_bundle(root, sweep_id, variant_id),
                root=root,
                expected_kind=COMPARISON_KIND,
                sweep_id=sweep_id,
                variant_id=variant_id,
            )
            if not _ROBUST._BASE._candidate()._same_replay_structure(
                published, recomputed
            ):
                raise DiagnosticSweepError("published comparison does not replay")
            variants[variant_id] = {
                **recomputed,
                "bundle_manifest_sha256": bundle["manifest_sha256"],
            }
        except Exception as exc:
            errors[variant_id] = f"{type(exc).__name__}: {exc}"

    external_registry: Mapping[str, Any] = {}
    try:
        external_registry = _validated_external_winners(root)
    except Exception as exc:
        errors["external:d755b3d9d8"] = f"{type(exc).__name__}: {exc}"

    comparison_error_variants = sorted(
        variant_id
        for variant_id, comparison in variants.items()
        if not isinstance(comparison.get("errors"), Mapping)
        or bool(comparison["errors"])
    )
    complete = (
        not errors
        and set(variants) == set(VARIANT_IDS)
        and not comparison_error_variants
        and bool(external_registry)
    )

    winners: dict[str, str | None] = {}
    decisions: dict[str, dict[str, Any]] = {}
    for scene_id, (floor_id, support_id) in SCENE_VARIANTS.items():
        winner, reason = _choose_scene_winner(scene_id, variants)
        if not complete:
            winner = None
            reason = (
                "comparison_artifact_or_runtime_error"
                if comparison_error_variants
                else "aggregate_or_external_reference_incomplete"
            )
        winners[scene_id] = winner
        decisions[scene_id] = {
            "f_paired_stable_count": variants.get(floor_id, {}).get(
                "paired_stable_count"
            ),
            "fs_paired_stable_count": variants.get(support_id, {}).get(
                "paired_stable_count"
            ),
            "reason": reason,
            "winner": winner,
        }

    combined_winners: dict[str, Any] = {}
    if complete:
        combined_winners["d755b3d9d8"] = external_registry["winners"][
            "d755b3d9d8"
        ]["winner"]
        combined_winners.update(winners)
    else:
        combined_winners = {
            scene_id: None for scene_id in ("d755b3d9d8", *SCENE_IDS)
        }

    gate = {
        "code": code,
        "combined_scene_winners": combined_winners,
        "comparison_error_variants": comparison_error_variants,
        "comparison_pass_count": sum(
            bool(row["comparison_pass"]) for row in variants.values()
        ),
        "complete_variant_count": len(variants),
        "errors": errors,
        "external_winner_registry": external_registry,
        "gpu_launch_allowed": False,
        "large_rollout_launch_allowed": False,
        "manifest_kind": "e4_robust_floor_support_extension_aggregate_gate",
        "paper_ready": False,
        "scene_decisions": decisions,
        "scene_winners": winners,
        "selection_rule": [
            "eligible_requires_pair_comparison_pass_and_at_least_two_paired_stable_slots",
            "choose_f_when_f_passes_unless_fs_also_passes_and_strictly_increases_paired_stable",
            "choose_fs_when_f_fails_and_fs_passes",
            "all_eight_local_comparisons_must_replay_without_artifact_or_runtime_errors",
            "sealed_d755_fs_winner_is_an_authenticated_final_readout_reference_only",
            "external_factories_are_never_reused",
            "never_release_gpu_large_rollout_or_paper_from_this_diagnostic",
        ],
        "status": "complete" if complete else "incomplete",
        "sweep_id": sweep_id,
        "variant_count": len(VARIANT_IDS),
        "variant_ids": list(VARIANT_IDS),
        "variants": variants,
    }
    bundle = _publish(
        _ROBUST._BASE._aggregate_bundle(root, sweep_id),
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
    ) as exc:
        print(
            f"[e4-robust-floor-extension] FAIL: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
