# Source integration and readiness boundaries

Reviewed baseline main: `472fd32f28a6d55b31e59becbf60c0b457e62cef`.
Reviewed result release: `results/gaussian-system-20260908` at
`524a4d08f633e7558fc0ac95fd5afcc7f7d46071`.

Result root:
`plan/icra2027/13_gaussian_system/results/20260908T222513Z-gaussian-system/`.
Authoritative prior mechanism closure: `snapshot-010/prior_f1_final/`.
Later local inpainting: `p06-local-dev-complete-20260909T0300Z/`.
Later official PolaRiS: `claude-session-20260910/`.
Commercial image editing / native SfM: `claude-session-20260910-round2/`.
Read all relevant snapshots: one agent's "no local baseline" statement must not
erase another published local baseline. Compare artifact IDs before combining.

## Reuse known working fragments, not blanket branch merges

- `7c637d1dfe473bf7f37e126e97a027ba568818aa`: observed-RGB marker contract and
  texture-binding import. Its complete branch contains other changes; review
  the relevant diff before cherry-picking.
- `03e733ebe51e0dd9318448f28dadea24b3f3f1ad`: genuine multi-reset DEV native pair.
- `c2fba3ec4eefb6283a346f307e41807d739549d2`: identity refresh fix. It was committed
  but not fully runtime-validated in the reviewed release. Re-test, do not label
  it as a proven resolution of every identity mismatch.
- `agent/gemini-transport-20260910`: inspect its current exact SHA if paid editing
  must be resumed. Reuse already authorized outputs first; preserve provider
  identity, call accounting and image-sharing constraints.

Do not copy output snapshots into running code or force-reset another worktree.
Commit the integrated code before preparing final job bundles. A main SHA need
not own every old artifact, but every reused artifact must keep its actual
producer source/config/checkpoint receipts.

## Required narrow native integration

The old `robo.roundtrip.spec` and `paired.run_resolved_episode` intentionally
restrict methods/sensors/renderers. New final arm IDs are **not** admitted by
removing validation or renaming all methods B3. Add a versioned final-contract
path in the native worker; leave old v1/v2 behavior and recorded metadata intact.

Use the existing `harness_runner.run_native_episode`, same native robot/policy,
canonical state/reset bank, and native result/ledger producers. The final block
JSON fixes methods, resets, task horizons, sensor and replacement scope.

The integration must:
1. Resolve each native asset bundle before a policy process starts. Use genuine
   SF/PolaRiS geometry and estimated placement, not our fixed-TRELLIS proxy.
2. Instantiate/restore the canonical reference with identical observation-cache
   freshness on both sides. Test physical-state/import identity and separately
   report image-repeatability. Do not tune success thresholds to pass the gate.
3. Run fresh reference and all admitted methods in one process per block; record
   exact source/policy runtime identity. Use the original result schema.
4. For observation arms, wrap `get_policy_observation()` using
   `PolicyObservationAdapter`. Supply a correct calibrated GS renderer, visible
   buffers and the existing actual H/correction callbacks. Preserve native image
   preprocessing, actions and success. This wrapper is not a GS renderer itself.
5. Export a native plan, ledger, unit crosswalk and execution contracts, then run
   `finalize link`. No manual success transcription is accepted.

Final marker-video sensor name is `rgb_video_public_marker_v1`. Metres come from
declared public calibration, never evaluation mesh/depth/poses. Camera estimates
must still come from RGB. A source PLY alone does not establish robot alignment.

If only a posed/RGB-D path is operational, preserve it as a separately named
control; do not relabel it to satisfy the final-video protocol.

## Genuine blockers versus method outcomes

Missing package/API glue: implement it. Unaccepted upstream license or unavailable
hardware: record external blocker. Finite poor reconstruction: run the admitted
baseline and measure failure. Undefined/nonfinite simulation: do not run unsafe
numerics, preserve the failure. Do not apply strict B4 gates to every baseline.

If the new final system cannot be admitted, return PARTIAL with exact missing
blocks and preserve the existing paper evidence. Do not silently declare old
RGB-D/native results to complete the new GS study.
