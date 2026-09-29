"""robo.eval.episode_log: outcome taxonomy, deterministic ID derivation, and
the append-only per-episode ledger for paired policy rollouts.

This module owns three things an episode runner needs and would otherwise
reimplement ad hoc:

  1. `Outcome` + `classify_exception` -- the requirement that
     environment crash, build failure, policy timeout, safety termination,
     and task failure are DISTINCT categories, never collapsed into one
     generic "failure" bucket. Every outcome funnels through
     `classify_exception`, so the taxonomy lives in exactly one place.
  2. `ResetState` + `derive_reset_seed` -- the single most
     important invariant: reset states are generated ONCE and every
     condition (reference/simany) consumes the SAME persisted list,
     including the SAME per-episode jitter seed. `derive_reset_seed` is a
     pure hash of (base_seed, task_id, ep), not a running RandomState
     advanced by call order, precisely so the draw cannot silently depend
     on which condition happened to run first.
  3. `EpisodeLedger` -- the append-only JSONL that makes resume-at-episode-
     granularity (step 5) and coverage accounting that includes failed
     builds (step 2) mechanical: every PLANNED episode gets exactly one
     line, regardless of outcome, keyed by a deterministic `episode_id`.
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
from enum import Enum
from pathlib import Path


# --------------------------------------------------------------- outcomes --

class Outcome(str, Enum):
    """Every planned episode ends in exactly one of these six values.
    SUCCESS and TASK_FAILURE are ordinary rollout endings (the scorer ran
    to completion); the other four are the required DISTINCT
    failure categories:

      - BUILD_FAILURE: the scene/env could not even be constructed
        (missing asset, uninstantiable geometry) -- step 2's "coverage
        failure," counted, never silently dropped from any denominator.
      - POLICY_TIMEOUT: a policy call did not return within its wall-clock
        budget, or the client itself raised a timeout (e.g. openpi
        websocket retry exhaustion in robo/eval/pi05_eval.py's
        ServerPolicy).
      - SAFETY_TERMINATION: an explicit safety check (non-finite action or
        simulator state) stopped the rollout before it could reach a
        normal task outcome.
      - ENV_CRASH: the residual bucket for anything else raised during a
        successfully-built env's rollout -- a genuine simulator/env bug,
        distinct from all of the above.
    """

    SUCCESS = "success"
    TASK_FAILURE = "task_failure"
    BUILD_FAILURE = "build_failure"
    POLICY_TIMEOUT = "policy_timeout"
    SAFETY_TERMINATION = "safety_termination"
    ENV_CRASH = "env_crash"


#: The five outcomes that must be independently
#: distinguishable (SUCCESS is deliberately excluded -- it is not a
#: failure category, it is the thing every failure category is
#: distinguished FROM).
FAILURE_OUTCOMES = (
    Outcome.TASK_FAILURE,
    Outcome.BUILD_FAILURE,
    Outcome.POLICY_TIMEOUT,
    Outcome.SAFETY_TERMINATION,
    Outcome.ENV_CRASH,
)


class BuildFailureError(RuntimeError):
    """A scene/env could not be constructed for a (scene_id, condition)."""


class PolicyTimeoutError(RuntimeError):
    """A policy call exceeded its wall-clock budget, or the client itself
    signaled a timeout."""


class SafetyTerminationError(RuntimeError):
    """An explicit safety check (non-finite action/state) fired."""


def classify_exception(exc: BaseException) -> Outcome:
    """The single dispatch point every failure outcome funnels through.
    Order matters: our three custom exceptions are checked before the
    generic fallback, and builtin `TimeoutError` is treated the same as
    our `PolicyTimeoutError` (a real served policy's client library may
    raise the builtin, not ours). Anything unrecognized is `ENV_CRASH` --
    the deliberately residual bucket, never silently reclassified as one
    of the more specific categories.
    """
    if isinstance(exc, BuildFailureError):
        return Outcome.BUILD_FAILURE
    if isinstance(exc, PolicyTimeoutError):
        return Outcome.POLICY_TIMEOUT
    if isinstance(exc, TimeoutError):
        return Outcome.POLICY_TIMEOUT
    if isinstance(exc, SafetyTerminationError):
        return Outcome.SAFETY_TERMINATION
    return Outcome.ENV_CRASH


# ------------------------------------------------------- deterministic IDs

def derive_reset_seed(base_seed: int, task_id: str, ep: int) -> int:
    """One deterministic uint32 seed per (base_seed, task_id, ep), shared
    byte-for-byte by EVERY condition -- the core invariant.

    A plain running `np.random.RandomState` threaded across a whole suite
    (what robo/eval/pi05_eval.py does today: one `rng` object advanced by
    every episode/task/scene it processes, in whatever order the process
    happens to iterate them) makes episode N's jitter draw depend on every
    call before it *in that process's call order*. Two conditions run as
    separate loops (or separate processes) would silently diverge in that
    order, breaking "byte-identical reset states across conditions" even
    though both used the "same seed." Hashing the identifying triple
    instead makes the draw a pure function of (seed, task, episode) alone,
    independent of iteration order, process, or which condition runs
    first.
    """
    h = hashlib.sha256(f"{base_seed}|{task_id}|{ep}".encode()).hexdigest()
    return int(h[:8], 16)


@dataclasses.dataclass(frozen=True)
class ResetState:
    """One planned reset. Generated ONCE for the whole matrix and then shared
    read-only by every condition -- no condition may resample this."""

    reset_state_id: str
    scene_id: str
    task_id: str
    ep: int
    base_seed: int
    reset_seed: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResetState":
        return cls(**{k: d[k] for k in
                      ("reset_state_id", "scene_id", "task_id", "ep",
                       "base_seed", "reset_seed")})


def episode_id_for(condition: str, reset_state_id: str) -> str:
    """Deterministic episode_id, unique per (condition, reset_state) --
    the ledger's join key for resume and coverage accounting."""
    return f"{condition}__{reset_state_id}"


def save_reset_states(states: "list[ResetState]", path: "str | Path") -> None:
    Path(path).write_text(
        json.dumps([s.to_dict() for s in states], indent=1) + "\n")


def load_reset_states(path: "str | Path") -> "list[ResetState]":
    data = json.loads(Path(path).read_text())
    return [ResetState.from_dict(d) for d in data]


# --------------------------------------------------------- episode record --

@dataclasses.dataclass
class EpisodeRecord:
    """One ledger line. `stages`/`score`/`success` are the scorer's own
    output (robo/tasks/pi05_tasks.py TaskScorer.summary()) when the
    episode reached SUCCESS/TASK_FAILURE; for the four exception-derived
    outcomes they are the zeroed placeholder required so a
    build failure still occupies exactly one row with a well-defined
    (non-missing) score, not a null/absent one."""

    episode_id: str
    condition: str
    scene_id: str
    task_id: str
    reset_state_id: str
    ep: int
    base_seed: int
    reset_seed: int
    outcome: Outcome
    success: bool
    score: float
    stages: dict
    ticks: int
    error: "str | None"
    video_path: "str | None"
    timeseries_path: "str | None"
    manifest_path: "str | None"
    wall_s: float

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["outcome"] = self.outcome.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeRecord":
        d = dict(d)
        d["outcome"] = Outcome(d["outcome"])
        return cls(**d)


class EpisodeLedger:
    """Append-only JSONL at `path`: the single source of truth for

      (a) RESUME -- which episode_ids are already terminal, so a
          restarted matrix run continues at the next incomplete episode
          without duplicating or reordering IDs; and
      (b) COVERAGE -- every planned episode gets exactly one line here
          regardless of outcome, so a build failure is counted just like
          a success or an ordinary task failure.

    Appends are flushed immediately (no buffering) so a process crash
    right after a line is written never loses that line, and a crash
    right before never produces a partial one -- `append` writes one
    complete `json.dumps(...) + "\\n"` call, so a truncated write can only
    ever drop the last, not-yet-durable line, which `load_completed`
    already treats as "not completed."
    """

    def __init__(self, path: "str | Path"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_completed(self) -> "dict[str, EpisodeRecord]":
        if not self.path.exists():
            return {}
        out: dict[str, EpisodeRecord] = {}
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = EpisodeRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, ValueError):
                # A partially-flushed final line from a killed process --
                # never trust it as completed; the episode gets re-run.
                continue
            out[rec.episode_id] = rec
        return out

    def append(self, record: EpisodeRecord) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            f.flush()


# ------------------------------------------------------------- time series --

def write_timeseries(path: "str | Path", ticks: "list[dict]") -> None:
    """gzip-compressed JSON array of one dict per control tick (joint
    position, gripper position, action, per-object pose, contacts, staged-
    rubric booleans) -- the "full state/contact/rubric time
    series," not just the final staged-score summary
    robo/eval/pi05_eval.py's results.json captures today."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(ticks, f)


def read_timeseries(path: "str | Path") -> "list[dict]":
    with gzip.open(path, "rt") as f:
        return json.load(f)


# --------------------------------------------------- degenerate diagnostic

def check_degenerate(records: "list[EpisodeRecord]", min_n: int = 4,
                      max_score: float = 1.0) -> "str | None":
    """A loud diagnostic for the historical all-zero
    failure mode (docs/ROBOT.md: "Baseline (pre geometry fix): all zero --
    32 episodes across raster and composite, zero successes"). Only looks
    at episodes that actually ran the scorer (SUCCESS/TASK_FAILURE) --
    build failures/crashes/timeouts are a different, already-flagged
    problem, not a scoring degeneracy, and mixing them in would let a run
    that is mostly build failures with one all-zero success masquerade as
    "not degenerate."

    Returns (and prints) a warning string once at least `min_n` scored
    episodes exist and every single one scored exactly 0.0 or exactly
    `max_score`; returns None otherwise. Never removes or mutates
    `records` -- the caller keeps every completed episode regardless of
    what this reports.
    """
    scored = [r for r in records
              if r.outcome in (Outcome.SUCCESS, Outcome.TASK_FAILURE)]
    if len(scored) < min_n:
        return None
    scores = [r.score for r in scored]
    if all(s == 0.0 for s in scores):
        msg = (f"[episode_log] DEGENERATE RUN WARNING: all {len(scored)} "
               f"scored episodes scored exactly 0.0 -- this is the historical "
               f"failure mode documented in docs/ROBOT.md ('Baseline (pre "
               f"geometry fix): all zero'). Data is PRESERVED, not discarded; "
               f"investigate before trusting this matrix's numbers.")
        print(msg, flush=True)
        return msg
    if all(s == max_score for s in scores):
        msg = (f"[episode_log] DEGENERATE RUN WARNING: all {len(scored)} "
               f"scored episodes scored exactly {max_score} (max) -- a "
               f"saturated rubric is just as suspicious as all-zero (it hides "
               f"any real behavior difference between conditions/policies). "
               f"Data is PRESERVED, not discarded; investigate before trusting "
               f"this matrix's numbers.")
        print(msg, flush=True)
        return msg
    return None


__all__ = [
    "Outcome",
    "FAILURE_OUTCOMES",
    "BuildFailureError",
    "PolicyTimeoutError",
    "SafetyTerminationError",
    "classify_exception",
    "derive_reset_seed",
    "ResetState",
    "episode_id_for",
    "save_reset_states",
    "load_reset_states",
    "EpisodeRecord",
    "EpisodeLedger",
    "write_timeseries",
    "read_timeseries",
    "check_degenerate",
]
