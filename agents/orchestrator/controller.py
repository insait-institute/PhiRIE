"""Pure controller: no filesystem access and no evaluation-reference access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .evidence import REASON_CODES_IN_ORDER, evaluate_gate, selection_key
from .policies import POLICY_IDS, PolicyOutcome, outcome_for_policy, validate_policy_config


@dataclass(frozen=True)
class ControllerResult:
    outcomes: tuple[PolicyOutcome, ...]
    retry_required: bool
    retry_parent_proposal_id: str | None
    retry_reason_codes: tuple[str, ...]


def plan_retry(initial_candidates: list[Mapping[str, Any]], config: Mapping[str, Any]):
    gates = config["gates"]
    if not initial_candidates:
        return True, None, ("initial_pool_empty",)
    gate_results = [(candidate, evaluate_gate(candidate["evidence"], gates))
                    for candidate in initial_candidates]
    if any(result.passed for _, result in gate_results):
        return False, None, ()
    parent, _ = min(gate_results, key=lambda item: selection_key(item[0], gates))
    observed_reasons = {
        reason
        for _, result in gate_results
        for reason in result.reason_codes
    }
    trigger_reasons = tuple(
        reason for reason in REASON_CODES_IN_ORDER if reason in observed_reasons
    )
    return True, parent["proposal_id"], trigger_reasons


def run_policies(
    initial_candidates: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    retry_candidate: Mapping[str, Any] | None = None,
) -> ControllerResult:
    validate_policy_config(config)
    proposal_ids = [candidate.get("proposal_id") for candidate in initial_candidates]
    if any(not value for value in proposal_ids) or len(proposal_ids) != len(set(proposal_ids)):
        raise ValueError("initial proposal IDs must be non-empty and unique")
    retry_required, parent_id, retry_reasons = plan_retry(initial_candidates, config)
    if retry_candidate is not None:
        if not retry_required:
            raise ValueError("retry candidate supplied for a job that did not trigger retry")
        if retry_candidate.get("proposal_id") in proposal_ids:
            raise ValueError("retry must create a new proposal ID")
        parents = retry_candidate.get("parent_proposal_ids") or []
        if parents != [parent_id]:
            raise ValueError("retry proposal must name the selected initial parent")
    outcomes = tuple(
        outcome_for_policy(policy_id, initial_candidates,
                           retry_candidate if retry_required else None, config)
        for policy_id in POLICY_IDS
    )
    return ControllerResult(outcomes, retry_required, parent_id, retry_reasons)
