"""robo.policy.registry: load `configs/policies/*.yaml` into typed
`PolicyEntry` records, verify their pinned checkpoint hashes and control
contract, and construct runnable policy clients.

Two real checkpoints are expected locally under
`${OPENPI_DATA_HOME}/openpi-assets-simeval/` (`pi05_droid_jointpos`
and `droid_pi05_jointpos_with_web_and_sim/80000`), plus the checkpoint-free
`scripted_sinusoid` smoke test -- these three are `configs/policies/
frozen_fields.yaml`'s `policies` list verbatim. Policies without a local
checkpoint can be declared as registry entries with `status: unavailable`
rather than working adapters -- `make_client` on any of those raises
`PolicyUnavailableError` naming the missing checkpoint, never a generic
crash.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from robo.manifest import hash as manifest_hash
from robo.policy.control_contract import (
    FROZEN_FIELDS_PATH,
    ControlContract,
    ControlContractMismatchError,
    validate_against_frozen,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = ROOT / "configs" / "policies"


# --------------------------------------------------------------- errors ---

class PolicyRegistryError(Exception):
    """Base class for robo.policy.registry problems."""


class UnknownPolicyError(PolicyRegistryError, KeyError):
    """`PolicyRegistry.get`/`make_client` called with an id that is not in
    any loaded `configs/policies/*.yaml` file."""


class PolicyUnavailableError(PolicyRegistryError):
    """`make_client` called on a `status: unavailable` policy (no local
    checkpoint). Always names the missing checkpoint URI so this is a clear,
    specific error rather than a generic crash somewhere downstream."""


class CheckpointHashMismatchError(PolicyRegistryError):
    """A policy's pinned `checkpoint_hash` (recorded in its
    `configs/policies/*.yaml` entry when the config was authored) does not
    match `robo.manifest.hash.hash_checkpoint_path` computed fresh against
    `checkpoint_path` right now -- the checkpoint on disk changed size or
    mtime since the config was pinned, so this run must not silently enter
    the main benchmark matrix (a hash mismatch prevents a run from entering
    the main matrix)."""


# ---------------------------------------------------------- entry schema --

class ImagePreprocessing(BaseModel):
    """`docs/ROBOT.md` / `robo/eval/pi05_eval.py::ServerPolicy._request`:
    every client resizes both cameras with `openpi_client.image_tools.
    resize_with_pad` to `resize_hw` before it ever reaches the model."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["resize_with_pad"] = "resize_with_pad"
    resize_hw: tuple[int, int] = (224, 224)


class PolicyEntry(BaseModel):
    """One `configs/policies/*.yaml` entry: checkpoint identity + pinned
    hash, training config, action semantics (model-internal AND
    env-facing, kept as two separate fields -- see
    `robo.policy.control_contract` module docstring for why), image
    preprocessing, language template, chunking/temporal aggregation, and
    a `status` reflecting whether this checkpoint's behavior has actually
    been matched to a published/reproduced real-world (or documented
    real-vs-sim) score (`verified`), exists locally but is unmatched so
    far (`exploratory`), or has no local checkpoint at all (`unavailable`).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    status: Literal["verified", "exploratory", "unavailable"]
    client_kind: Literal["pi05_server", "scripted", "unavailable"]

    # --- checkpoint identity -------------------------------------------
    checkpoint_uri: str | None = None
    # Local path used for hashing (robo.manifest.hash.hash_checkpoint_path).
    # None for scripted (no checkpoint) and for unavailable policies (no
    # local copy exists -- checkpoint_uri may still record the aspirational
    # gs:// path for documentation).
    checkpoint_path: str | None = None
    # Hash pinned at config-authoring time; PolicyRegistry.verify_checkpoint_hash
    # recomputes and compares. None = nothing pinned to check (e.g. scripted).
    checkpoint_hash: str | None = None
    training_config: str | None = None  # openpi `--policy.config` name

    # --- action semantics -----------------------------------------------
    # What the model predicts internally (informational; NOT validated
    # against the frozen contract -- e.g. frozen_fields.yaml's
    # "joint_delta_chunk_15x8" for the two real pi0.5 checkpoints).
    model_action_convention: str | None = None
    # What the CLIENT hands to the environment, downstream of any
    # server-side transform (e.g. openpi's AbsoluteActions). This IS
    # validated against configs/policies/frozen_fields.yaml.
    env_action_convention: Literal["absolute_joint_position", "joint_delta"] = (
        "absolute_joint_position"
    )
    action_dim: int = 8

    # --- gripper convention ---------------------------------------------
    gripper_range: tuple[float, float] = (0.0, 1.0)
    gripper_open_value: float = 0.0
    gripper_closed_value: float = 1.0
    gripper_binarize_threshold: float = 0.5

    # --- control rate / clipping ----------------------------------------
    control_rate_hz: int = 15
    per_tick_joint_delta_clamp_rad: float = 0.2

    # --- chunking / temporal aggregation ---------------------------------
    chunk_size: int = 15
    temporal_aggregation: Literal["requery_after_full_chunk"] = (
        "requery_after_full_chunk"
    )

    # --- observation / language ------------------------------------------
    image_preprocessing: ImagePreprocessing = ImagePreprocessing()
    language_template: str = "{instruction}"

    # --- provenance / reporting bookkeeping -------------------------------
    published_real_score_ref: str | None = None
    notes: str = ""


# ------------------------------------------------------------- registry --

class PolicyRegistry:
    """In-memory registry of `PolicyEntry` records, keyed by `id`."""

    def __init__(self, entries: dict[str, PolicyEntry]):
        self._entries = dict(entries)

    @classmethod
    def from_config_dir(cls, config_dir: "str | Path" = DEFAULT_CONFIG_DIR) -> "PolicyRegistry":
        """Load every `*.yaml` file under `config_dir`. Each file's
        top-level `policies:` key is a list of `PolicyEntry`-shaped dicts
        (see `configs/policies/*.yaml`). The frozen control contract
        (`frozen_fields.yaml`) lives in the same directory but is not a
        registry file, so it is skipped."""
        entries: dict[str, PolicyEntry] = {}
        for path in sorted(Path(config_dir).glob("*.yaml")):
            if path.name == FROZEN_FIELDS_PATH.name:
                continue
            doc = yaml.safe_load(path.read_text()) or {}
            raw_entries = doc.get("policies", [])
            for raw in raw_entries:
                entry = PolicyEntry(**raw)
                if entry.id in entries:
                    raise PolicyRegistryError(
                        f"duplicate policy id {entry.id!r}: defined in both "
                        f"a previously-loaded file and {path}")
                entries[entry.id] = entry
        return cls(entries)

    @classmethod
    def from_entries(cls, entries: "list[PolicyEntry] | dict[str, PolicyEntry]") -> "PolicyRegistry":
        """Build a registry directly from in-memory entries (tests, or a
        caller assembling a one-off registry without touching disk)."""
        if isinstance(entries, dict):
            return cls(entries)
        return cls({e.id: e for e in entries})

    def list_ids(self) -> list[str]:
        return sorted(self._entries)

    def get(self, policy_id: str) -> PolicyEntry:
        try:
            return self._entries[policy_id]
        except KeyError:
            raise UnknownPolicyError(
                f"no policy {policy_id!r} in this registry; known ids: "
                f"{self.list_ids()}") from None

    # ------------------------------------------------------- verification --

    def resolved_checkpoint_hash(self, policy_id: str) -> str | None:
        """Recompute `robo.manifest.hash.hash_checkpoint_path` for this
        policy's `checkpoint_path` right now. Returns None if the entry has
        no checkpoint_path, or the path does not exist on this machine
        (distinct from a hash *mismatch* -- an absent checkpoint is
        `PolicyUnavailableError` territory, handled in `make_client`, not a
        hash problem)."""
        entry = self.get(policy_id)
        if not entry.checkpoint_path:
            return None
        p = Path(entry.checkpoint_path)
        if not p.exists():
            return None
        return manifest_hash.hash_checkpoint_path(p)

    def verify_checkpoint_hash(self, policy_id: str) -> None:
        """Raise `CheckpointHashMismatchError` if this entry has a pinned
        `checkpoint_hash` that disagrees with the hash resolved from disk
        right now. No-op if nothing is pinned, or if the checkpoint isn't
        present locally (nothing to compare against)."""
        entry = self.get(policy_id)
        if entry.checkpoint_hash is None:
            return
        actual = self.resolved_checkpoint_hash(policy_id)
        if actual is None:
            return
        if actual != entry.checkpoint_hash:
            raise CheckpointHashMismatchError(
                f"policy {policy_id!r}: declared checkpoint_hash "
                f"{entry.checkpoint_hash!r} does not match the hash "
                f"resolved from {entry.checkpoint_path!r} right now "
                f"({actual!r}) -- the checkpoint on disk changed (size or "
                f"mtime) since this config was pinned; this run must not "
                f"enter the main benchmark matrix until re-pinned "
                f"deliberately")

    def validate_control_contract(
        self, policy_id: str, frozen: "ControlContract | None" = None,
    ) -> list[str]:
        """Return the list of frozen-contract mismatches for this policy
        (empty = compliant). See
        `robo.policy.control_contract.validate_against_frozen`."""
        return validate_against_frozen(self.get(policy_id), frozen)

    # --------------------------------------------------------- client ------

    def make_client(self, policy_id: str, *, strict_contract: bool = True, **kwargs):
        """Construct a runnable `PolicyClient` for `policy_id`.

        Order of checks, each with a specific, named error rather than a
        downstream crash:
          1. `status`/`client_kind` unavailable -> `PolicyUnavailableError`
             naming the missing checkpoint (never a generic exception).
          2. pinned checkpoint hash mismatch -> `CheckpointHashMismatchError`.
          3. (if `strict_contract`, the default) control-contract mismatch
             against `configs/policies/frozen_fields.yaml` ->
             `ControlContractMismatchError`.

        Remaining `**kwargs` are forwarded to the client class's
        `from_entry(entry, **kwargs)` classmethod (e.g. `home=` for
        `scripted`, `connector=`/`host=`/`port=` for `pi05_server`).
        """
        entry = self.get(policy_id)
        if entry.status == "unavailable" or entry.client_kind == "unavailable":
            raise PolicyUnavailableError(
                f"policy {policy_id!r} is unavailable: no local checkpoint "
                f"exists (checkpoint_uri={entry.checkpoint_uri!r}). This "
                f"checkpoint has not been downloaded locally -- populate "
                "$OPENPI_DATA_HOME/ (and give the entry a real "
                f"checkpoint_path + client_kind) before requesting a "
                f"runnable client for it.")

        if entry.client_kind == "pi05_server":
            missing = [name for name, value in (
                ("checkpoint_path", entry.checkpoint_path),
                ("checkpoint_hash", entry.checkpoint_hash),
                ("training_config", entry.training_config),
            ) if not value]
            if missing:
                raise PolicyUnavailableError(
                    f"policy {policy_id!r} cannot be served because its "
                    f"registry entry lacks {missing}")
            if not Path(str(entry.checkpoint_path)).exists():
                raise PolicyUnavailableError(
                    f"policy {policy_id!r} cannot be served because its "
                    f"checkpoint_path does not exist: {entry.checkpoint_path!r}")

        self.verify_checkpoint_hash(policy_id)

        if strict_contract:
            issues = self.validate_control_contract(policy_id)
            if issues:
                raise ControlContractMismatchError(
                    f"policy {policy_id!r} fails control-contract "
                    f"validation against configs/policies/"
                    f"frozen_fields.yaml: " + "; ".join(issues))

        from robo.policy.clients import CLIENT_KIND_TO_CLASS

        try:
            client_cls = CLIENT_KIND_TO_CLASS[entry.client_kind]
        except KeyError:
            raise PolicyRegistryError(
                f"policy {policy_id!r} has unknown client_kind "
                f"{entry.client_kind!r}") from None
        return client_cls.from_entry(entry, **kwargs)


_DEFAULT_REGISTRY: PolicyRegistry | None = None


def load_default_registry(config_dir: "str | Path" = DEFAULT_CONFIG_DIR) -> PolicyRegistry:
    """Cache-and-return a `PolicyRegistry` loaded from `config_dir`
    (default `configs/policies/`). Separate from `PolicyRegistry.
    from_config_dir` so repeated calls in one process don't re-parse every
    yaml file, while `from_config_dir` callers that want a fresh,
    uncached load still can."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None or Path(config_dir) != DEFAULT_CONFIG_DIR:
        registry = PolicyRegistry.from_config_dir(config_dir)
        if Path(config_dir) == DEFAULT_CONFIG_DIR:
            _DEFAULT_REGISTRY = registry
        return registry
    return _DEFAULT_REGISTRY


__all__ = [
    "PolicyEntry",
    "ImagePreprocessing",
    "PolicyRegistry",
    "PolicyRegistryError",
    "UnknownPolicyError",
    "PolicyUnavailableError",
    "CheckpointHashMismatchError",
    "load_default_registry",
    "DEFAULT_CONFIG_DIR",
]
