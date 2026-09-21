# Fresh E2 input readiness

This CPU audit consumes the frozen v30 TRAIN-only Gaussian, v34 automatic
discovery, fresh TRELLIS `20260904-4d0787c-v2` and RVG
`20260905-4d0787c-v1` receipts. The config pins existing completion audits,
input manifests and pools. It reuses the existing fresh-source and RVG view
validators and E2 camera selector. It does not invoke a model or evaluator.

Run from a clean checkout, choosing a new output file:

```bash
/group/worldcept/code/SimAny/.venv/bin/python -m run.icra2027.e2_fresh_readiness \
  --config configs/experiments/icra2027/e2_fresh_readiness.json \
  --out /absolute/new/path/readiness.json
```

The output records all five discovered objects, including the three preparation
failures. Eight prospective official-test views use the unchanged E2 selector,
exclude the entire official TRAIN split, and have image hashes compared with all
48 selected optimization inputs. Complete 48-frame optimization provenance and
the actual generation-view union are kept separately. The public camera source
may reflect dataset-wide SfM; this does not imply that calibration was estimated
only from the 48 images.

Room input availability is not a completed render/metric bundle. The next step
would be a minimal raw-Gaussian producer using the existing render function and
canonical `robo.eval.fidelity_metrics`, gated by a new source/E0/config and smoke.
The old room exporter requires factory and automatic composite builds, so it
cannot directly consume these raw pools. This one-scene readiness audit does
not fill the fixed 50-scene Table II population.

Existing object masks are generation inputs. Existing `gt_points.ply` files in
this fresh chain are samples of the TRAIN-derived mesh despite their legacy
filenames; they are excluded as independent evaluation surfaces. Resampling that
mesh or changing its file hash would not make it independent. Held-out object
visibility, photo masks, registered predictions, independent evaluation surfaces
and metric scale proof remain missing. No scans, private labels or GT-controlled
construction are used. All metric readiness flags remain false.

Focused CPU checks (the existing export tests require a repository-local temp):

```bash
mkdir -p .tmp
/group/worldcept/code/SimAny/.venv/bin/python -m pytest -q \
  tests/test_e2_fresh_readiness.py tests/test_fidelity_room_export.py \
  --basetemp .tmp/e2-fresh-readiness-tests
```
