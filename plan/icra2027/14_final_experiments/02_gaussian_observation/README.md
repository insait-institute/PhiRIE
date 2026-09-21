# F2 — Gaussian observation and state-constrained Harmonizer

Fixed budget:12 source-selected instances x10 resets x6 arms=720 units,
including fresh references. Main physics is the same immutable SimAnyRoom asset
bundle for every non-reference arm.

Arms: REF native, SimAnyRoom native raster, raw GS, official temporal H,
official H+existing robot restoration, same H+fixed state correction.
No new model training, learned cache, or extra enhancer in this final round.
An affine-color baseline can be evaluated offline using the same saved frames.

The new `StateBoundObserver` commits all cameras atomically. It rejects stale
snapshots, cross-episode history, changed canvas, advanced physics and hidden
fallback. Repeat reads of the same state do not advance temporal conditioning.
`PolicyObservationAdapter` replaces only admitted camera keys, preserving the
native proprioception, actions, dynamics and success.

Integration TODO:
1. Bind the actual GS state, camera calibration, visible ID/depth, robot and
   target layers. Verify double-object removal and move/reveal on DEV.
2. Provide render/enhance/constrain callbacks using existing backend APIs, not
   synthetic frames or identity callbacks labeled as the real model.
3. Use the exact native camera preprocessing. Confirm the central runner calls
   get_policy_observation, not a bypassed raw env observation.
4. Export runtime state-sync events and canonical physics definition per reset.
5. Evaluate offline standard room/object PSNR,SSIM,LPIPS, visible mask IoU,
   valid-correspondence temporal error, protected-pixel equality and p95 latency.
6. Run every frozen arm even if its visual/policy result is negative.

Same state/camera for paired image references. No matching divergent rollouts by
frame index. No semantic instruction that paints a desired outcome. Buffers come
from the reconstructed simulator, not the hidden reference. Unknown/occluded
pixels are not temporal matches. The bridge's pixel protection is a local
invariant, not proof of globally correct image geometry.

Acceptance: complete valid current-state policy results plus independent image
metrics and coverage. Enhancement improves a paper claim only if the relevant
contrast and state-consistency evidence support it. Merely acquiring Cosmos
weights, copying robot pixels or producing an attractive image is insufficient.
