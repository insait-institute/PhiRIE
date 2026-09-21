"""Typed manifest schemas for Task 01 (plan/01_EXPERIMENT_MANIFEST.md).

pydantic 2.13.4 is already vendored in .venv (checked before writing this --
see the task brief), so these are pydantic BaseModel subclasses rather than
hand-validated dataclasses; that gets us field-level type coercion, explicit
`extra="forbid"` typo-catching, and JSON-mode dumping for free instead of
reimplementing all three.

Two manifest kinds, both carrying the eight fields
`docs/ICRA_RESEARCH_CONTRACT.md` / plan/01 require on "every output":
`scene_build_commit`, `scene_manifest_hash`, `policy_checkpoint_hash`,
`controller_config_hash`, `camera_config_hash`, `task_id`,
`initial_state_id`, `rollout_seed`.

  - SceneBuildManifest: describes one reconstructed scene. Not tied to a
    task/seed/policy, so those three (+ policy_checkpoint_hash) are legally
    null; `scene_manifest_hash` on a build manifest is its OWN content hash
    (see hash.manifest_content_hash), computed after construction and
    copied into the field so downstream RolloutManifests can cite "which
    scene build did this rollout use" via that same value.
  - RolloutManifest: describes one policy episode against one scene build.
    All eight fields are required and non-null (enforced by validators
    below) because these are exactly the fields the mujoco_paired /
    oracle_causal protocols need frozen-and-identical across the conditions
    being compared (configs/experiments/frozen_fields.yaml).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

SCHEMA_VERSION = 1

_STRICT = ConfigDict(extra="forbid")


# --------------------------------------------------------------- submodels --

class CaptureInputs(BaseModel):
    """Capture/reconstruction inputs feeding one scene build."""

    model_config = _STRICT

    source_type: Literal["scannetpp_scan", "phone_video", "synthetic"]
    source_path: str
    n_frames: int | None = None
    fps: float | None = None
    duration_s: float | None = None
    capture_device: str | None = None


class ScaleAlignmentResiduals(BaseModel):
    """How far the reconstructed scene's metric frame drifted from its
    registration target (laser scan / colmap), per docs/ICRA_RESEARCH_CONTRACT.md
    provenance requirements."""

    model_config = _STRICT

    rotation_deg: float
    translation_m: float
    scale_ratio: float
    rmse_m: float


class InstanceInventoryItem(BaseModel):
    """One simulated object instance, tier-labeled the same way
    agents/eval/aggregate_results.py reads report.json (`tier_A/B/C_rejected`)."""

    model_config = _STRICT

    object_id: str
    label: str
    tier: Literal["A", "B", "C"]
    asset_hash: str | None = None


class SimulatorVersions(BaseModel):
    """Software versions that can silently change physics/rendering
    behavior between two otherwise-identical manifests."""

    model_config = _STRICT

    python: str
    mujoco: str | None = None
    gsplat: str | None = None
    torch: str | None = None
    cuda: str | None = None


class VerificationResults(BaseModel):
    """Outcome of whatever automated checks the build pipeline ran on this
    scene (collision sanity, drop-test stability, etc.)."""

    model_config = _STRICT

    passed: bool
    checks: dict[str, bool] = {}
    notes: str = ""


class ObservationPreprocessing(BaseModel):
    """configs/experiments/frozen_fields.yaml: observation_preprocessing.
    'composite mode is one scene per process; never mix modes within one
    comparison' -- `mode` is exactly the `--obs raster|composite` flag on
    robo/eval/pi05_eval.py."""

    model_config = _STRICT

    mode: Literal["raster", "composite"]
    resize_hw: tuple[int, int] = (224, 224)
    notes: str = ""


class StagedProgress(BaseModel):
    """Mirrors robo/tasks/pi05_tasks.py TaskScorer.stages exactly (one
    target's {grasp,lift,hover,place} dict) -- 0.25 credit/stage per
    configs/experiments/frozen_fields.yaml: rubric."""

    model_config = _STRICT

    grasp: bool = False
    lift: bool = False
    hover: bool = False
    place: bool = False


# -------------------------------------------------------------- base class --

class ManifestBase(BaseModel):
    """Fields shared by every manifest kind."""

    model_config = _STRICT

    schema_version: int = SCHEMA_VERSION
    manifest_kind: Literal["scene_build", "rollout"]
    created_utc: str  # ISO-8601, informational only -- excluded from content_hash
    git_dirty: bool

    # --- the eight required-on-every-output provenance/frozen fields -----
    scene_build_commit: str
    scene_manifest_hash: str
    policy_checkpoint_hash: str | None = None
    controller_config_hash: str | None = None
    camera_config_hash: str | None = None
    task_id: str | None = None
    initial_state_id: str | None = None
    rollout_seed: int | None = None

    @field_validator("scene_build_commit", "scene_manifest_hash")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must not be empty")
        return v


# ---------------------------------------------------------- concrete kinds --

class SceneBuildManifest(ManifestBase):
    manifest_kind: Literal["scene_build"] = "scene_build"

    scene_id: str
    capture_inputs: CaptureInputs
    scale_alignment_residuals: ScaleAlignmentResiduals
    instance_inventory: list[InstanceInventoryItem] = []
    asset_hashes: dict[str, str] = {}
    full_room_collision_hash: str
    simulator_versions: SimulatorVersions
    verification_results: VerificationResults


class RolloutManifest(ManifestBase):
    manifest_kind: Literal["rollout"] = "rollout"

    scene_id: str
    language: str
    rubric_version: str
    horizon_s: float
    observation_preprocessing: ObservationPreprocessing
    action_convention: str
    action_dim: int
    video_path: str | None = None
    state_path: str | None = None
    contact_path: str | None = None
    success: bool
    staged_progress: StagedProgress
    failure_label: str | None = None

    @field_validator("task_id", "initial_state_id", "controller_config_hash",
                      "camera_config_hash")
    @classmethod
    def _required_for_rollout(cls, v, info):
        if v is None:
            raise ValueError(
                f"{info.field_name} is required (non-null) for a rollout "
                "manifest -- only scene_build manifests may leave it null")
        return v

    @field_validator("rollout_seed")
    @classmethod
    def _seed_required(cls, v):
        if v is None:
            raise ValueError("rollout_seed is required (non-null) for a "
                              "rollout manifest")
        return v


# Fields the mujoco_paired / oracle_causal protocols require byte-identical
# between the conditions being compared (configs/experiments/frozen_fields.yaml:
# control rate+action convention+robot reset pose -> controller_config_hash;
# camera intrinsics/extrinsics -> camera_config_hash; rubric -> rubric_version).
# `python -m robo.manifest.io diff` restricts its output to this set by
# default so a diff between two rollout manifests reads as "which declared
# frozen field differs" rather than a noisy full dict diff.
FROZEN_ROLLOUT_FIELDS = [
    "controller_config_hash",
    "camera_config_hash",
    "action_convention",
    "action_dim",
    "rubric_version",
    "horizon_s",
    "observation_preprocessing",
]

MANIFEST_KIND_TO_CLASS: dict[str, type[ManifestBase]] = {
    "scene_build": SceneBuildManifest,
    "rollout": RolloutManifest,
}
