# Raw-only E2 room smoke

This adapter renders the fresh v30 input Gaussian on the eight official TEST
cameras already selected in the source-bound E2 readiness audit. It calls
`agents.core.common.load_gaussians` and `render_view` unchanged, at full source
resolution with the original K/w2c, SH degree, default black background and
the existing clip-to-uint8 PNG convention. It does not load a factory, GT or
automatic replacement composite, register an object, select a new view, or
compute any metric formula itself.

The plan revalidates the v30/v34 and fresh initial-tool receipt chain using the
existing readiness builder. The shared canonical freeze stores E0 and execution
receipts. Rendered/reference PNGs, the canonical metric manifest and table stay
under this checkout's `outputs/icra2027/<freeze>/raw_room`, satisfying the
unchanged evaluator's repository-local/no-symlink path contract. Shared receipts
point to those products by path and hash.

The metric phase calls `robo.eval.fidelity_metrics.evaluate_manifest` with CUDA
LPIPS and the existing AlexNet cache. The input Gaussian row has one scene and
eight views. The other seven Table II method rows are explicitly empty/null.
Every output is `paper_ready=false`; this engineering smoke is not a full
50-scene room comparison or object-fidelity evaluation.

Before execution, commit this source/config, run `run/icra2027/preflight.sh
--smoke` using the main Python, and publish a new canonical E0 using
`e2_raw_room_freeze.yaml` plus that passing preflight log. Then:

```bash
${SIMANY_ROOT}/.venv/bin/python -m run.icra2027.e2_raw_room \
  --config configs/experiments/icra2027/e2_raw_room.yaml \
  --freeze-root /absolute/shared/canonical/freeze --phase plan
```

Only after that CPU plan passes, submit `e2_raw_room.sbatch` with the exact
source checkout in `E2_RAW_CODE` and canonical root in `E2_RAW_FREEZE`. The
launcher requests Hala/debug with explicit QoS debug, one A6000, four CPUs,
64 GB and ten minutes. Verify Slurm did not rewrite the requested QoS.

The renderer uses the proven mini-viewer prebuilt gsplat runtime with the
existing scoped Python 3.10 pydantic cache. Metrics use the main Python 3.11
environment. Both disable user-site imports; CPU probes and actual GPU workers
check the same targeted executable/module/RECORD/native identities. These are
explicitly targeted checks, not a closed full dependency byte inventory.

No phase overwrites a result. The execution claim prevents duplicate runs;
failed stage logs, receipts and partial render staging remain available. A new
attempt requires a new frozen output rather than altering the failed evidence.
