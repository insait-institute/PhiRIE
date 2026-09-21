"""Typed, deterministic proposal records for construction orchestration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


VALID_DECISIONS = frozenset({"pending", "accept", "retry", "reject", "abstain"})
REQUIRED_EVIDENCE = (
    "schema_frame_unit_valid",
    "symmetric_clipped_registration_residual_m",
    "scale_ratio_vs_observation",
    "observation_point_count",
    "visible_fraction",
    "support_gap_m",
    "support_overlap_fraction",
    "initial_penetration_m",
    "collision_valid",
    "usable_convex_parts",
    "settle_drift_m",
    "settle_sunk",
    "settle_stable",
    "missing_evidence",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProposalRecord:
    freeze_id: str
    scene_id: str
    object_id: str
    job_id: str
    proposal_id: str
    tool: str
    tool_commit: str
    input_hashes: Mapping[str, str]
    artifact_paths: Mapping[str, str]
    evidence: Mapping[str, Any]
    decision: str = "pending"
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    parent_proposal_ids: tuple[str, ...] = field(default_factory=tuple)
    started_utc: str = ""
    finished_utc: str = ""
    wall_s: float = 0.0

    def validate(self) -> None:
        identifiers = {
            "freeze_id": self.freeze_id,
            "scene_id": self.scene_id,
            "object_id": self.object_id,
            "job_id": self.job_id,
            "proposal_id": self.proposal_id,
            "tool": self.tool,
            "tool_commit": self.tool_commit,
        }
        for key, value in identifiers.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty string")
        if self.decision not in VALID_DECISIONS:
            raise ValueError(f"invalid proposal decision: {self.decision!r}")
        if not isinstance(self.wall_s, (int, float)) or self.wall_s < 0:
            raise ValueError("wall_s must be non-negative")
        for role, digest in self.input_hashes.items():
            if not role or not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"invalid input hash for {role!r}")
            int(digest, 16)
        missing = [key for key in REQUIRED_EVIDENCE if key not in self.evidence]
        if missing:
            raise ValueError(f"proposal evidence is missing fields: {missing}")
        flags = self.evidence.get("missing_evidence")
        if not isinstance(flags, list) or any(not isinstance(v, str) for v in flags):
            raise ValueError("missing_evidence must be a list of strings")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("reason_codes must be unique")
        if len(self.parent_proposal_ids) != len(set(self.parent_proposal_ids)):
            raise ValueError("parent_proposal_ids must be unique")
        if self.proposal_id in self.parent_proposal_ids:
            raise ValueError("a proposal cannot be its own parent")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "freeze_id": self.freeze_id,
            "scene_id": self.scene_id,
            "object_id": self.object_id,
            "job_id": self.job_id,
            "proposal_id": self.proposal_id,
            "tool": self.tool,
            "tool_commit": self.tool_commit,
            "input_hashes": dict(sorted(self.input_hashes.items())),
            "artifact_paths": dict(sorted(self.artifact_paths.items())),
            "evidence": dict(self.evidence),
            "decision": self.decision,
            "reason_codes": list(self.reason_codes),
            "parent_proposal_ids": list(self.parent_proposal_ids),
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "wall_s": float(self.wall_s),
        }


def proposal_digest(record: ProposalRecord | Mapping[str, Any]) -> str:
    payload = record.as_dict() if isinstance(record, ProposalRecord) else dict(record)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
