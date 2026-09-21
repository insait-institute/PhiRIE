## 2026-09-06T12:11:37.653497+00:00 — pinned Cosmos access recheck

Owner root; main baseline83efa8e. Authenticated metadata-only request for nvidia/Cosmos-Predict2-0.6B-Text2Image/model.pt at dd55b6858b22ad569976bff207880b8fea839da7 remains HTTP403/GatedRepoError (2026-09-06T12:06:23Z). Both Harmonizer weights exist locally; base config.json/model.pt/tokenizer/tokenizer.pth are absent. No credentials logged, download, model smoke, pilot, full run or Slurm job. Evidence: main-worktree outputs/e4-root-cause-diagnosis-20260906/cosmos_access.json. StateBLOCKED; real-model claimNOT_RUN. This access is required for E5 and the enhanced E4 observation arm, not for E4 construction/physics diagnosis. See ../PROJECT_LIMITATIONS_AND_TODO.md for all current project limitations and next actions.

---

## 2026-09-05 22:11 UTC access recheck

Pinned Cosmos model.pt metadata request with existing cluster credentials still returns HTTP403/GatedRepoError. No model download and no credentials logged. Receipt: `/group/worldcept/code/SimAny/outputs/icra2027/orchestration-status-20260905T221103Z/harmonizer_access_recheck.json`. StateBLOCKED; real-model pilot/full and claim remainNOT_RUN.

# E5 Harmonizer Option C status

Updated: 2026-09-04 UTC. Owner: root orchestrator.
Branch: `agent/icra-e3-agentic`; audited source: `c3b97ffe4db235cff8728bb5a836ae5f6f02a2c0` plus recorded prior dirty work.
State: BLOCKED. `MAIN_PAPER_GATE=NOT_RUN`. No scientific full run.

- Smoke verification: `.venv/bin/python -m pytest -q tests/test_harmonizer_protocol.py tests/test_harmony_visual_metrics.py tests/test_robot_restore.py tests/test_audit_loso.py tests/test_audit_metrics.py tests/test_task_support_smoke.py`: 13 passed in 0.63 s on Hala CPU. This is unit coverage, not the required real-model smoke.
- Pilot/full commands: `bash run/harness/serve_harmonizer.sh` and `python -m robo.eval.harmony_visual_metrics --manifest configs/experiments/icra2027/harmony_visual_manifest.json --out <fresh-freeze>/harmony/visual_table --lpips-device cuda`; NOT_RUN.
- Slurm jobs: none. Real-model runtime and latency: unavailable.
- Available weights: `checkpoints/harmonizer/models/diffusion_harmonizer.pkl` (5,042,242,222 bytes) and `harmonizer_nontemporal.pt` (1,448,843,112 bytes). These have not been promoted to a complete checkpoint closure.
- Missing base model/tokenizer: `nvidia/Cosmos-Predict2-0.6B-Text2Image`, pinned revision `dd55b6858b22ad569976bff207880b8fea839da7`.
- Acquisition attempted through the existing authenticated Hugging Face client: `snapshot_download(..., allow_patterns=['config.json','model.pt','tokenizer/tokenizer.pth'])` returned HTTP 403 `GatedRepoError`. Classification: missing authorized private/gated model access. User was asked to obtain model access/update cluster login; no token requested in chat.
- Durable acquisition record: `outputs/icra2027/20260904-07e8b05-v1/initial_audit/harmonizer_acquisition.json`.
- Config/checkpoint identities: E0 resolved config hashes are in `outputs/icra2027/20260904-07e8b05-v1/contract/hashes.csv`; no complete real-service checkpoint hash exists.
- Failed criteria: real two-camera/two-episode smoke, five-condition aligned exporter, motion-compensated temporal pairs, target preservation, real-model coverage/latency, and matched E4 observation episodes.
- Known limitation: checked manifest has no records and only three legacy variant names; it does not implement the required five-condition evaluation. Identity backend unit success cannot pass the model or paper gate.

Harmonizer geometry/physics improvement and policy improvement are not supported claims.
# 2026-09-06 planned-frame visual contract

Owner: root; branch `agent/icra-e5-visual-contract`; source is the clean commit
containing this entry (exact-source E0 recorded separately before merge).
State: `IMPLEMENTING`; visual input/metric contract smoke `SMOKE_PASSED`;
real-model pilot/full `NOT_RUN`; `MAIN_PAPER_GATE=FAIL` (not demonstrated).

The existing visual evaluator now has an explicit schema-2 five-condition
planned frame contract, documented in `PLANNED_VISUAL_CONTRACT.md`. Missing
and enhancer-failed frames stay in the denominator. Hashes, filenames and shapes
are checked before model loading; Option C's nonempty core is compared directly
to raw pixels, and a failed invariant is retained. Per-frame end-to-end latency
includes enhancer failures without combining nested timing fields. Generated
LaTeX shows coverage before conditional quality. Legacy metrics and published
outputs are unchanged; no new metric formula or rollout format was introduced.

Smoke command from this worktree after the existing headless setup:
`python -m pytest -q tests/test_harmony_planned_frames.py
tests/test_harmony_visual_metrics.py tests/test_demo_final.py
tests/test_paper_pipeline_audit.py --basetemp=outputs/e5-visual-tests/pt6`.
Result: **89 passed in 8.99 s**, with actual synthetic-image PSNR/SSIM and null
LPIPS (no model loaded). Exact logs: `outputs/e5-visual-tests/pt6.log`.
Initial test invocation lacked the basetemp parent (environment setup); a
subsequent invocation named a nonexistent test file; a synthetic LPIPS stub then
needed the canonical optional mask argument. These development failures were
corrected before publication and no real data/model job was run.

Pilot/full reproducible command is in the contract document; result `NOT_RUN`.
The current ICRA template intentionally has no planned frames and is rejected,
rather than producing an apparently complete empty five-way evaluation.
Slurm IDs: none; hardware: CPU tests only; runtime above is test runtime.
New checkpoint hashes: none; no weights loaded or downloaded. Existing model
access was rechecked at 05:33:07 UTC, fixed Cosmos revision
`dd55b6858b22ad569976bff207880b8fea839da7`, and still returns HTTP403
`GatedRepoError`, classification `missing_authorized_gated_model_access`.
Receipt: main-worktree
`outputs/icra2027/orchestration-status-20260906T053500Z/harmonizer_access_recheck.json`.

Passed criteria: planned coverage, actual core-byte negative test, no hidden
fallback, dimension/filename/hash rejection and no-overwrite contract.
Unmet: official model service smoke, aligned genuine E4 sequences, five-way real
quality/temporal/latency measurements and manipulation preservation. These unit
checks do not certify a real model, stream history, metric input separation, or
policy result. Scientific claim gate `NOT_RUN`.
# 2026-09-06 atomic visual publication follow-up

Owner root; branch `agent/icra-e5-visual-publication`. The earlier source
`bf32e12fcf9a450a2e7930d97177aab931f06313` remains clean with exact E0
`dfcd0249e9493ae88a96f78e88cce8e33e07a29e1ca3bbf1c701543044ab62b5`
(130 passed, 27.36 s). This follow-up uses the existing shared atomic directory
publisher, archives the exact visual input manifest, and rejects missing or
empty appearance masks instead of silently evaluating the whole frame.

The same four-file focused suite, `--basetemp=outputs/e5-visual-tests/pt1`, now
passes **91 tests in 9.39 s**, including injected publication failure and absent
evaluation-mask negatives. Log: `outputs/e5-visual-tests/pt1.log`. No actual
experiment or real-model pilot ran between these revisions. Pilot/full remain
`NOT_RUN`, scientific gate `NOT_RUN`, missing credentials/footage unchanged.
Exact-source E0 is required before merge and recorded in the central status.
# 2026-09-06 deterministic color baseline

Owner root; branch `agent/icra-e5-color-baseline`. The construction-TRAIN fitted
per-camera affine baseline is implemented through the existing observation
pipeline; see `COLOR_MATCH.md` for exact formula, split boundary and command.
Smoke: 48 passed in 9.79 s, log `outputs/e5-color-tests/pt1.log`.
No fitted real calibration or scientific output has been produced. Pilot/full
remain NOT_RUN; no GPU or Slurm job. Missing model access and aligned E4 footage
remain the blockers to the actual five-condition experiment. Scope: numerical
baseline and observation integration only; E5 scientific gate NOT_RUN.
Exact-source E0 runs after the source commit, with receipt recorded centrally.
