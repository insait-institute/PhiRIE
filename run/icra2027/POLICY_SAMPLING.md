# Deterministic policy sampling for paired engineering rollouts

The previous canonical client reset only its local action chunk. OpenPI's
persistent JAX RNG advanced on every inference, including warmup and retried
requests. Simulator reset seeds therefore did not also pair policy sampling.

New compact real-policy configs declare `contract.policy.sampling`. The bound
server must be launched with `run/pi05_serve_bound.sh --deterministic-sampling`.
Its authenticated identity includes the exact algorithm, NumPy version, model
noise shape, and serving/client implementation file hashes. Legacy configs and
server invocations retain their existing behavior; they do not acquire a paired
sampling guarantee.

The canonical runner calls `begin_episode(reset_seed)` before actions. Each
inference chunk has an envelope containing that seed, its zero-based chunk
index, the frozen sampling contract, and an `episode` domain. A separate fixed
`warmup` domain cannot consume an episode chunk. SHA256 of canonical envelope
JSON seeds NumPy PCG64; `standard_normal(..., dtype=float32)` generates noise.
Both installed Python environments currently have NumPy 1.26.4; a different
version fails the contract check. The server wrapper calls the pinned public
`Policy.infer(observation, noise=noise)` API. The current upstream Pi0 sampler
uses its RNG only when explicit noise is absent. No private RNG state is reset.

The client independently computes the noise hash and verifies the server's
receipt before returning any action. Retries resend the same envelope; changing
connection, arm ordering, warmup count, or the length of a previous episode does
not change later per-episode noise. The noise shape is read from the declared
training configuration and checked against the loaded model: 15 by 32 for
`pi05_droid_jointpos`, 15 by 8 for `pi05_droid_jointpos_sim`. Noise equality does
not imply action equality when observations differ, or successful manipulation.

The existing canonical ledger and episode manifests retain each consumed chunk
receipt. Validation binds receipt seeds and chunk counts to the frozen reset
seed and executed ticks; early termination can change the number of consumed
chunks. The complete sampling contract remains part of paired frozen fields.
No model checkpoint, environment, rollout format, metric, rubric, or treatment
axis is replaced.

## Validation and release gates

Run focused CPU tests with the primary SimAny Python:

```
python -m pytest -q tests/test_policy_sampling.py tests/test_policy_runtime_identity.py tests/test_control_contract.py tests/test_e4_compact_policy.py tests/test_harness_validation.py tests/test_harness_prebuild_policy_identity.py tests/test_harness_construction_variants.py
```

Tests cover lost replies after inference, warmup isolation, reseeded arm replay,
changed observations, malformed seeds/contracts, missing or changed receipts,
authenticated server capability, changed serving source hashes, and canonical
runner seed propagation. These are engineering tests, not checkpoint evaluation.
A genuine loaded-policy GPU smoke remains required before performance promotion.

Existing qualification source `03f1e7f2d9c8225b330f8610781189625e677c83` and its
freeze remain immutable. Scripted and all-rejected stages can still execute
there without this change. A later eligible real-policy stage needs a new source
freeze and either newly produced qualification or an explicit typed bridge
which invokes the original producer's validator from its exact clean pinned
worktree and authenticates its complete sealed outputs. Merely relabeling the
old camera or CPU source as the new runtime commit is forbidden. This patch
contains no cross-source bridge and launches no jobs.
