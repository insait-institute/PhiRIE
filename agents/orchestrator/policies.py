"""Frozen A0--A4 policy semantics for the E3 construction ablation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .evidence import REASON_CODES_IN_ORDER, choose_best, evaluate_gate


POLICY_IDS = ("A0", "A1", "A2", "A3", "A4")


@dataclass(frozen=True)
class PolicyOutcome:
    policy_id: str
    terminal_action: str
    selected_proposal_id: str | None
    retry_invoked: bool
    reason_codes: tuple[str, ...]
    support_label: str


def validate_policy_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("agentic policy schema_version must equal 1")
    policies = config.get("policies")
    if not isinstance(policies, list) or [p.get("policy_id") for p in policies] != list(POLICY_IDS):
        raise ValueError("policies must be exactly A0,A1,A2,A3,A4 in order")
    gates = config.get("gates")
    required = {
        "min_observation_points", "max_registration_residual_m",
        "min_scale_ratio", "max_scale_ratio", "require_collision_valid",
        "max_collision_parts", "require_drop_stable",
        "max_initial_penetration_m", "selection_tool_order",
        "min_visible_fraction", "max_abs_support_gap_m",
        "min_support_overlap_fraction", "max_settle_drift_m",
    }
    if not isinstance(gates, Mapping) or not required.issubset(gates):
        raise ValueError(f"gates are missing keys: {sorted(required - set(gates or {}))}")
    if not 0 < float(gates["max_registration_residual_m"]) < 1:
        raise ValueError("registration residual gate must be expressed in metres")
    priority = config.get("fixed_priority")
    if priority != ["reconviagen", "trellis"]:
        raise ValueError("A1 fixed priority must be reconviagen then trellis")
    retry = config.get("retry")
    if not isinstance(retry, Mapping) or retry.get("max_retries_per_job") != 1:
        raise ValueError("retry policy must be bounded to exactly one retry")
    primary_gate = config.get("primary_gate")
    if (
        not isinstance(primary_gate, Mapping)
        or primary_gate.get("reason_codes_in_order") != list(REASON_CODES_IN_ORDER)
    ):
        raise ValueError("primary gate reason-code order differs from the executor")


def fixed_priority(candidates: list[Mapping[str, Any]], priority: list[str]):
    by_tool = {candidate["tool"]: candidate for candidate in candidates}
    return next((by_tool[tool] for tool in priority if tool in by_tool), None)


def outcome_for_policy(
    policy_id: str,
    initial_candidates: list[Mapping[str, Any]],
    retry_candidate: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> PolicyOutcome:
    gates = config["gates"]
    trellis = fixed_priority(initial_candidates, ["trellis"])
    retry_invoked = retry_candidate is not None

    if policy_id == "A0":
        if trellis is None:
            return PolicyOutcome(policy_id, "reject", None, False,
                                 ("trellis_missing",), "unsupported")
        return PolicyOutcome(policy_id, "accept", trellis["proposal_id"], False,
                             ("fixed_single_path",), "not_evaluated")

    if policy_id == "A1":
        chosen = fixed_priority(initial_candidates, config["fixed_priority"])
        if chosen is None:
            return PolicyOutcome(policy_id, "reject", None, False,
                                 ("initial_pool_empty",), "unsupported")
        return PolicyOutcome(policy_id, "accept", chosen["proposal_id"], False,
                             ("fixed_priority",), "not_evaluated")

    if policy_id == "A2":
        chosen = choose_best(initial_candidates, gates)
        if chosen is None:
            return PolicyOutcome(policy_id, "reject", None, False,
                                 ("initial_pool_empty",), "unsupported")
        gate = evaluate_gate(chosen["evidence"], gates)
        reason = "frozen_gate_pass" if gate.passed else "best_initial_below_gate"
        support = "supported" if gate.passed else "unsupported"
        return PolicyOutcome(
            policy_id, "accept", chosen["proposal_id"], False,
            ("construction_evidence_selection", reason), support,
        )

    candidates = list(initial_candidates)
    if retry_candidate is not None:
        candidates.append(retry_candidate)
    chosen = choose_best(candidates, gates)
    if policy_id == "A3":
        if chosen is None:
            return PolicyOutcome(policy_id, "reject", None, retry_invoked,
                                 ("proposal_pool_empty_after_retry",), "unsupported")
        gate = evaluate_gate(chosen["evidence"], gates)
        reasons = ("bounded_retry" if retry_invoked else "retry_not_triggered",)
        reasons += ("frozen_gate_pass" if gate.passed else "best_available_below_gate",)
        support = "supported" if gate.passed else "unsupported"
        return PolicyOutcome(
            policy_id, "accept", chosen["proposal_id"], retry_invoked, reasons, support
        )
    if policy_id == "A4":
        passing = choose_best(candidates, gates, passing_only=True)
        if passing is None:
            reasons = ["no_proposal_passed_frozen_gate"]
            if retry_invoked:
                reasons.append("bounded_retry_exhausted")
            return PolicyOutcome(
                policy_id, "abstain", None, retry_invoked, tuple(reasons), "unsupported"
            )
        return PolicyOutcome(policy_id, "accept", passing["proposal_id"], retry_invoked,
                             ("frozen_gate_pass",), "supported")
    raise ValueError(f"unknown policy_id: {policy_id}")
