"""Ledger validation and descendant-only invalidation for E3 jobs."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable, Mapping


def validate_ledger(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    seen: set[tuple[str, str]] = set()
    for row in materialized:
        policy_id = row.get("policy_id", "")
        job_id = row.get("job_id", "")
        proposal_id = row.get("proposal_id", "")
        key = (policy_id, job_id)
        if not all((policy_id, job_id, proposal_id)):
            raise ValueError("every ledger row needs policy_id, job_id, and proposal_id")
        if key in seen:
            raise ValueError(f"duplicate policy/job ledger row: {key}")
        seen.add(key)
        action = row.get("terminal_action")
        if action is not None and action not in {"accept", "reject", "abstain"}:
            raise ValueError(f"invalid terminal action: {action!r}")
        selected = row.get("selected_proposal_id")
        if action == "accept" and not selected:
            raise ValueError("accepted ledger rows require a selected proposal")
        if action in {"reject", "abstain"} and selected is not None:
            raise ValueError("rejected/abstained ledger rows cannot select a proposal")
        support = row.get("support_label")
        if support is not None and support not in {
            "supported", "unsupported", "not_evaluated"
        }:
            raise ValueError(f"invalid support label: {support!r}")
    return materialized


def descendants(edges: Iterable[tuple[str, str]], failed_id: str) -> set[str]:
    children: dict[str, list[str]] = defaultdict(list)
    for parent, child in edges:
        if parent == child:
            raise ValueError("job graph self-edge")
        children[parent].append(child)
    invalidated: set[str] = set()
    queue = deque(children.get(failed_id, ()))
    while queue:
        node = queue.popleft()
        if node in invalidated:
            continue
        invalidated.add(node)
        queue.extend(children.get(node, ()))
    return invalidated
