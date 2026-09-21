from __future__ import annotations

from pathlib import Path

import pytest

from agents.orchestrator.artifact import ProposalRecord, proposal_digest
from agents.orchestrator.controller import run_policies
from agents.orchestrator.evidence import (
    REASON_CODES_IN_ORDER,
    choose_best,
    evaluate_gate,
)
from agents.orchestrator.job_graph import descendants, validate_ledger


def policy_config():
    return {
        "schema_version": 1,
        "policies": [{"policy_id": name} for name in ("A0", "A1", "A2", "A3", "A4")],
        "fixed_priority": ["reconviagen", "trellis"],
        "gates": {
            "min_observation_points": 200,
            "max_registration_residual_m": 0.02,
            "min_scale_ratio": 0.4,
            "max_scale_ratio": 2.0,
            "require_collision_valid": True,
            "max_collision_parts": 1,
            "require_drop_stable": True,
            "max_initial_penetration_m": 0.002,
            "min_visible_fraction": 0.5,
            "max_abs_support_gap_m": 0.015,
            "min_support_overlap_fraction": 0.15,
            "max_settle_drift_m": 0.03,
            "selection_tool_order": ["trellis", "reconviagen", "registration_retry"],
        },
        "retry": {"max_retries_per_job": 1},
        "primary_gate": {"reason_codes_in_order": list(REASON_CODES_IN_ORDER)},
    }


def evidence(*, residual=0.01, ratio=1.0, stable=True, missing=()):
    return {
        "schema_frame_unit_valid": True,
        "symmetric_clipped_registration_residual_m": residual,
        "scale_ratio_vs_observation": ratio,
        "observation_point_count": 400,
        "visible_fraction": 0.8,
        "support_gap_m": 0.0,
        "support_overlap_fraction": 1.0,
        "initial_penetration_m": 0.0,
        "collision_valid": True,
        "usable_convex_parts": 1,
        "settle_drift_m": 0.001,
        "settle_sunk": False,
        "settle_stable": stable,
        "missing_evidence": list(missing),
    }


def candidate(pid, tool, **kwargs):
    return {"proposal_id": pid, "tool": tool, "evidence": evidence(**kwargs)}


def test_a0_to_a4_have_distinct_frozen_semantics():
    trellis = candidate("job:trellis", "trellis", residual=0.015)
    rvg = candidate("job:rvg", "reconviagen", residual=0.005)
    result = run_policies([trellis, rvg], policy_config())
    outcomes = {row.policy_id: row for row in result.outcomes}
    assert outcomes["A0"].selected_proposal_id == "job:trellis"
    assert outcomes["A1"].selected_proposal_id == "job:rvg"
    assert outcomes["A2"].selected_proposal_id == "job:rvg"
    assert outcomes["A3"].selected_proposal_id == "job:rvg"
    assert outcomes["A4"].terminal_action == "accept"
    assert result.retry_required is False


def test_fixed_priority_does_not_inspect_evidence():
    class ForbiddenEvidence(dict):
        def __getattribute__(self, name):
            if name in {"get", "__getitem__", "items", "keys", "values"}:
                raise AssertionError("A1 inspected evidence")
            return super().__getattribute__(name)

    trellis = {"proposal_id": "t", "tool": "trellis", "evidence": ForbiddenEvidence()}
    rvg = {"proposal_id": "r", "tool": "reconviagen", "evidence": ForbiddenEvidence()}
    from agents.orchestrator.policies import outcome_for_policy

    result = outcome_for_policy("A1", [trellis, rvg], None, policy_config())
    assert result.selected_proposal_id == "r"


def test_retry_is_bounded_new_and_parented():
    initial = [candidate("t", "trellis", residual=0.05),
               candidate("r", "reconviagen", residual=0.04)]
    first = run_policies(initial, policy_config())
    assert first.retry_required and first.retry_parent_proposal_id == "r"
    retry = candidate("r:retry", "registration_retry", residual=0.01)
    retry["parent_proposal_ids"] = ["r"]
    final = run_policies(initial, policy_config(), retry_candidate=retry)
    outcomes = {row.policy_id: row for row in final.outcomes}
    assert outcomes["A3"].retry_invoked
    assert outcomes["A4"].selected_proposal_id == "r:retry"
    retry["proposal_id"] = "r"
    with pytest.raises(ValueError, match="new proposal ID"):
        run_policies(initial, policy_config(), retry_candidate=retry)


def test_retry_preserves_canonical_union_of_all_initial_failure_reasons():
    residual_failure = candidate("t", "trellis", residual=0.05)
    stability_failure = candidate("r", "reconviagen", residual=0.01, stable=False)
    result = run_policies([residual_failure, stability_failure], policy_config())
    assert result.retry_required is True
    assert result.retry_reason_codes == (
        "registration_residual_too_high",
        "unstable_settle",
    )


def test_abstention_does_not_turn_into_acceptance():
    initial = [candidate("t", "trellis", residual=0.08, stable=False)]
    retry = candidate("t:retry", "registration_retry", residual=0.04, stable=False)
    retry["parent_proposal_ids"] = ["t"]
    outcomes = {row.policy_id: row for row in
                run_policies(initial, policy_config(), retry_candidate=retry).outcomes}
    assert outcomes["A3"].terminal_action == "accept"
    assert "best_available_below_gate" in outcomes["A3"].reason_codes
    assert outcomes["A3"].support_label == "unsupported"
    assert outcomes["A4"].terminal_action == "abstain"
    assert outcomes["A4"].support_label == "unsupported"
    assert outcomes["A4"].selected_proposal_id is None


def test_selection_is_invariant_to_input_order():
    values = [candidate("z", "trellis", residual=0.01),
              candidate("a", "reconviagen", residual=0.01)]
    a = choose_best(values, policy_config()["gates"])
    b = choose_best(list(reversed(values)), policy_config()["gates"])
    assert a["proposal_id"] == b["proposal_id"] == "z"


def test_missing_evidence_fails_closed():
    row = evidence(missing=("support_gap_m",))
    assert not evaluate_gate(row, policy_config()["gates"]).passed


def test_multi_failure_reason_vocabulary_and_order_are_frozen():
    row = evidence(
        residual=0.2,
        ratio=9.0,
        stable=False,
        missing=("unavailable_field",),
    )
    row.update(
        {
            "schema_frame_unit_valid": False,
            "observation_point_count": 0,
            "visible_fraction": 0.0,
            "support_gap_m": 0.2,
            "support_overlap_fraction": 0.0,
            "initial_penetration_m": 0.2,
            "collision_valid": False,
            "usable_convex_parts": 0,
            "settle_sunk": True,
            "settle_drift_m": 0.2,
        }
    )
    assert evaluate_gate(row, policy_config()["gates"]).reason_codes == (
        REASON_CODES_IN_ORDER
    )

    drifted = policy_config()
    drifted["primary_gate"]["reason_codes_in_order"] = list(
        reversed(REASON_CODES_IN_ORDER)
    )
    with pytest.raises(ValueError, match="reason-code order"):
        run_policies([], drifted)


def test_proposal_contract_is_canonical_and_complete():
    record = ProposalRecord(
        freeze_id="freeze", scene_id="scene", object_id="obj_00", job_id="scene:obj_00",
        proposal_id="scene:obj_00:trellis", tool="trellis", tool_commit="a" * 40,
        input_hashes={"mesh": "b" * 64}, artifact_paths={"mesh": "outputs/mesh.ply"},
        evidence=evidence(), decision="accept", reason_codes=("fixed_single_path",),
    )
    assert proposal_digest(record) == proposal_digest(record.as_dict())
    broken = dict(record.as_dict())
    del broken["evidence"]["settle_stable"]
    with pytest.raises(ValueError, match="missing fields"):
        ProposalRecord(**{
            key: (tuple(value) if key in {"reason_codes", "parent_proposal_ids"} else value)
            for key, value in broken.items()
        }).validate()


def test_duplicate_ledger_rows_and_graph_scope():
    row = {"policy_id": "A0", "job_id": "j", "proposal_id": "p"}
    with pytest.raises(ValueError, match="duplicate"):
        validate_ledger([row, row])
    assert descendants([("a", "b"), ("b", "c"), ("x", "y")], "a") == {"b", "c"}


def test_controller_package_contains_no_evaluation_reference_access():
    package = Path(__file__).resolve().parents[1] / "agents" / "orchestrator"
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    assert "gt_points" not in source
    assert "hybrid.json" not in source
    assert "aligned.json" not in source
