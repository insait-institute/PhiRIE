"""Construction-only evidence gates and deterministic candidate ordering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


REASON_CODES_IN_ORDER = (
    "invalid_schema_frame_or_unit",
    "implausible_scale",
    "registration_residual_too_high",
    "insufficient_observation_support",
    "invalid_support_gap",
    "insufficient_support_overlap",
    "initial_penetration",
    "invalid_collision",
    "unusable_convex_decomposition",
    "sank_during_settle",
    "unstable_settle",
    "excessive_settle_drift",
    "missing_evidence",
)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason_codes: tuple[str, ...]


def _finite_number(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return numeric == numeric and abs(numeric) != float("inf")


def evaluate_gate(evidence: Mapping[str, Any], gates: Mapping[str, Any]) -> GateResult:
    """Apply only construction-time evidence; missing values always fail closed."""
    reasons: list[str] = []
    if evidence.get("schema_frame_unit_valid") is not True:
        reasons.append("invalid_schema_frame_or_unit")

    ratio = evidence.get("scale_ratio_vs_observation")
    if not _finite_number(ratio) or not (
        float(gates["min_scale_ratio"]) <= float(ratio) <= float(gates["max_scale_ratio"])
    ):
        reasons.append("implausible_scale")

    residual = evidence.get("symmetric_clipped_registration_residual_m")
    if not _finite_number(residual) or float(residual) > float(gates["max_registration_residual_m"]):
        reasons.append("registration_residual_too_high")

    n_points = evidence.get("observation_point_count")
    visible = evidence.get("visible_fraction")
    if (
        not _finite_number(n_points)
        or int(n_points) < int(gates["min_observation_points"])
        or not _finite_number(visible)
        or float(visible) < float(gates["min_visible_fraction"])
    ):
        reasons.append("insufficient_observation_support")

    gap = evidence.get("support_gap_m")
    if not _finite_number(gap) or abs(float(gap)) > float(gates["max_abs_support_gap_m"]):
        reasons.append("invalid_support_gap")

    overlap = evidence.get("support_overlap_fraction")
    if not _finite_number(overlap) or float(overlap) < float(gates["min_support_overlap_fraction"]):
        reasons.append("insufficient_support_overlap")

    penetration = evidence.get("initial_penetration_m")
    if not _finite_number(penetration) or float(penetration) > float(
        gates.get("max_initial_penetration_m", 0.002)
    ):
        reasons.append("initial_penetration")

    if gates.get("require_collision_valid", True) and evidence.get("collision_valid") is not True:
        reasons.append("invalid_collision")
    parts = evidence.get("usable_convex_parts")
    if (
        not _finite_number(parts)
        or int(parts) < 1
        or int(parts) > int(gates.get("max_collision_parts", 64))
    ):
        reasons.append("unusable_convex_decomposition")

    if evidence.get("settle_sunk") is not False:
        reasons.append("sank_during_settle")
    if gates.get("require_drop_stable", True) and evidence.get("settle_stable") is not True:
        reasons.append("unstable_settle")

    drift = evidence.get("settle_drift_m")
    if not _finite_number(drift) or float(drift) > float(gates["max_settle_drift_m"]):
        reasons.append("excessive_settle_drift")

    missing = evidence.get("missing_evidence")
    if not isinstance(missing, list) or missing:
        reasons.append("missing_evidence")

    return GateResult(not reasons, tuple(dict.fromkeys(reasons)))


def selection_key(candidate: Mapping[str, Any], gates: Mapping[str, Any]) -> tuple:
    """Stable lexicographic key. Evaluation quality is intentionally absent."""
    evidence = candidate["evidence"]
    gate = evaluate_gate(evidence, gates)
    residual = evidence.get("symmetric_clipped_registration_residual_m")
    residual = float(residual) if _finite_number(residual) else float("inf")
    ratio = evidence.get("scale_ratio_vs_observation")
    scale_error = abs(float(ratio) - 1.0) if _finite_number(ratio) else float("inf")
    drift = evidence.get("settle_drift_m")
    drift = float(drift) if _finite_number(drift) else float("inf")
    tool_rank = {name: rank for rank, name in enumerate(gates["selection_tool_order"])}
    return (
        0 if gate.passed else 1,
        len(gate.reason_codes),
        residual,
        scale_error,
        drift,
        tool_rank.get(candidate.get("tool"), len(tool_rank)),
        candidate["proposal_id"],
    )


def choose_best(candidates: list[Mapping[str, Any]], gates: Mapping[str, Any], *, passing_only: bool = False):
    eligible = list(candidates)
    if passing_only:
        eligible = [c for c in eligible if evaluate_gate(c["evidence"], gates).passed]
    return min(eligible, key=lambda c: selection_key(c, gates)) if eligible else None
