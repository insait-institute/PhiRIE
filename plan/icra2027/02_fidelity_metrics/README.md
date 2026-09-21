# E2 — Held-Out Appearance and Geometry Fidelity

**Priority:** P0  
**Paper output:** Table II  
**Depends on:** a fresh E0 contract at the exact E2 commit and frozen scene/object builds

## Outcome and evaluation units

E2 measures whether SimAnyRoom preserves a captured room while making its
objects independently movable. Room appearance and object fidelity have
different units and are never pooled:

- room appearance: official held-out DSLR views, reported with PSNR, SSIM, and
  LPIPS;
- object appearance: held-out masked object views;
- object geometry: independently registered/evaluated object surfaces,
  reported with symmetric CD in centimeters and F1@20 mm.

The checked-in population is the exact 50-scene ScanNet++ `nvs_sem_val`
roster in `configs/experiments/icra2027/fidelity_manifest.json`. The same list
must match E1's `construction_regimes.yaml`. Failed scenes remain in the
denominator; the producer may not replace them with whatever directories are
available at run time.

## Full run, not a smoke

The accepted dependency chain is:

```text
fresh E0 -> 2 individual smoke renders -> smoke inventory -> smoke metrics
         -> 5 individual replacement jobs -> 50 individual full renders
         -> full CPU inventory -> full A100 metrics
```

The smoke is end-to-end and uses its own output ID, so it validates rendering,
inventory, and metrics without colliding with the full output. After it passes,
five independent replacement jobs regenerate eight leakage-contaminated
objects in five scene/method bundles using official-train images only. Every
full render depends on all five replacements. No legacy object is silently
substituted, and none of the selected evaluation images may be a generation
input.

`run/slurm/icra2027_e2_fidelity.sbatch` then has three resource-distinct
phases.

1. `render-room` is submitted as 50 ordinary one-scene GPU jobs. Each job
   loads the background once and produces the factory and automatic composites
   in the same process.
2. `inventory` is one CPU job with an `afterok` dependency on all 50 render
   job IDs. It validates all 50 scene bundles and builds the canonical
   manifest.
3. `metrics` is one GCP A100-80G job dependent on inventory. It computes
   frame-level PSNR/SSIM/LPIPS and the final table; full-resolution LPIPS is
   not sent through a prohibitively slow CPU path.

For every scene, the render task selects exactly eight unique registered
frames from the official DSLR test split and rejects train/test overlap. It
publishes these paired directories atomically:

```text
outputs/icra2027/<freeze_id>/fidelity/room_runs/<scene_id>/
  gt/                                  # 8 PNGs
  input_scene_gaussian/                # 8 PNGs
  factorized_gt_discovery/             # 8 PNGs
  factorized_auto_discovery/           # 8 PNGs
  manifest.json
```

Thus a valid full run contains exactly 50 scene manifests and 400 PNGs per
non-Harmonizer method, plus 400 paired GT PNGs. Legacy
`render_metrics_v2.json` summaries and contact sheets are never accepted as
substitutes for these fresh frame-level products.

## Required clean provenance

Before submission:

- commit the tested E2 code and use its exact 40-character SHA;
- run a fresh E0 preflight at that commit;
- pass the resulting repository-local `freeze_manifest.json` to E2;
- ensure that the E0 contract's frozen E2-config hash matches the current
  schema-v2 config;
- choose distinct smoke and full `E2_FREEZE_ID` values; no phase overwrites a
  prior output.

The launcher fails closed on a dirty tree, abbreviated/drifted commit, stale E0
config hash, roster drift, a symlinked/out-of-repository contract, any Slurm
array context, a missing or out-of-roster per-job scene/bundle ID, missing
source build, legacy split, absent frame, filename mismatch, or artifact-hash
mismatch. Temporary and cache files are kept under this checkout's `.tmp/` and
`.cache/` directories.

The render phase is resumable without overwriting. Re-submitting an individual
scene under the same freeze ID re-hashes and reuses its complete scene bundle
(`E2_RENDER_ROOM=REUSED`); missing scenes are rendered, while a corrupt or
provenance-drifted existing bundle fails closed for manual audit.

The two-scene smoke and full run deliberately use different IDs. This costs two
repeated scene renders, but makes the smoke a real end-to-end gate with its own
canonical inventory/table and leaves the full 50-scene bundle atomic.

LPIPS is pinned to the already proven repository-local AlexNet cache:

```text
.cache/icra2027/e2-render-compat/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
size_bytes = 244408911
sha256 = 7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02
```

For the render and metrics GPU phases the launcher sets
`TORCH_HOME=.cache/icra2027/e2-render-compat/torch` and refuses to start if the
directory or weight is missing/symlinked, or if size/hash differs. It neither
downloads nor copies the checkpoint. The metrics subprocess receives the
path, size, and SHA-256 through `E2_LPIPS_WEIGHTS_*`; the final
`run_manifest.json` records all three alongside `lpips_device=cuda`.

The explicit `E2_GPU_PROFILE=gcp-a100` profile is validated on
`gcp-eu1-a100-80g-qrfh`. Compatibility job `816031` completed with exit
`0:0` in `5:33`; relative to the A6000 compatibility reference `814563`, its
rendered JPEG SHA-256 was identical, PSNR and SSIM were exactly equal, and the
maximum LPIPS difference was approximately `7.4e-5`. That bounded
device-dependent LPIPS difference is accepted for scheduling render and
metrics on A100-80G. Replacement dependency/profile probe `816065` and the
five pre-fix diagnostic jobs `816198`, `816241`, `816242`, `816243`, and
`816244` then exercised the repository-local PyTorch/ninja/spconv/SAM3/TRELLIS
stack on the same A100 profile. Jobs `816243` and `816244` completed and
strictly validated their bundles; the other three reached final alignment and
were rejected by the unchanged scientific gates. These results establish that
the A100 hardware/runtime profile is ready, but they are not replacement
evidence for the revised method.

Post-failure inspection showed that the old factory alignment initialized only
a source-frame `+z` up axis even though image-to-3D assets can use any signed
canonical axis. The commit-bound revision evaluates six fixed, ordered proper
rotations mapping source `+z`, `-z`, `+x`, `-x`, `+y`, and `-y` to world up,
then selects with the existing `sym_score`. It does not change the yaw/scale
search, ICP, tier-A `F1@20 mm`, tier-B `F1@40 mm`, or size-ratio gates. The
replacement factory-alignment path records `source_up_hypothesis` plus
deterministic GT-surface and mesh-sample seeds `42`; the legacy four-value
`align_object` API and unrelated historical hybrid/SHARP paths remain
`+z`-only. The five ordinary replacement jobs in the canonical DAG are the
formal post-fix rerun. Each depends on smoke metrics, and all five must pass
before any full render dependency is released.

Replacement generation is offline and pinned to these repository-local model
inputs (all paths are relative to this checkout):

```text
TRELLIS executable source:
  .cache/icra2027/e2-replacements/trellis-source-442aa1e1afb9014e80681d3bf604e8d728a86ee7
  upstream_commit = 442aa1e1afb9014e80681d3bf604e8d728a86ee7
  flexicubes_commit = 815e075a2a400d06c48d94c347674344ed6ae5c5
  file_count = 223
  tree_sha256 = df4059255d1a72a6d82a76f2b9c97f034e1c1fd9c06ddec782bd8282be3f8401

TRELLIS snapshot:
  .cache/icra2027/e2-replacements/trellis-image-large-25e0d31ffbebe4b5a97464dd851910efc3002d96
  file_count = 13
  tree_sha256 = 8daaba378d461e5ac0a7771c92d8ee70abfe7a95b0bb54149b44d3f12ea79d41

DINOv2 source tree:
  .cache/icra2027/e2-replacements/torch/hub/facebookresearch_dinov2_main
  file_count = 203
  tree_sha256 = bad25746e5bccefe9a039b7ebe5537c72765edc9be6654881699298f37f0cc4d

DINOv2 checkpoint:
  .cache/icra2027/e2-replacements/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
  size_bytes = 1217607321
  sha256 = 36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51

SAM3 executable source:
  .cache/icra2027/e2-replacements/sam3-source-8e451d5eb43c817b64ae7577fb7b9ae223db88a9
  upstream_commit = 8e451d5eb43c817b64ae7577fb7b9ae223db88a9
  package_version = 0.1.0
  file_count = 523
  tree_sha256 = 73c418359155da5da839853613260e84f48e5bd8e3d494c6614ca9e561a187ae

SAM3 checkpoint:
  .cache/icra2027/e2-replacements/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt
  size_bytes = 3450062241
  sha256 = 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
```

The tree digests bind every sorted relative path, byte size, and per-file
SHA-256. The replacement launcher validates all six inputs, rejects symlinks
or any identity drift, and does not download or copy model data.

Replacement alignment keeps the established acceptance policy unchanged:
tier A requires `F1@20 mm >= 0.40`, tier B requires
`F1@40 mm >= 0.20`, and the observed-size ratio must remain in `[0.4, 2.5]`.
The six signed source-up hypotheses and deterministic GT-surface and mesh
sample seeds affect only initialization/reproducibility; they do not weaken
these scientific gates.

Create the Slurm log directory before submission because Slurm opens log files
before the script begins. Run these commands only after the final tested E2
change is committed and the worktree is clean. Every job is submitted under
the audited `runyi_yang` account with automatic requeue disabled. A failed or
preempted attempt therefore becomes terminal and can be audited before the
same frozen output is resumed explicitly:

```bash
cd /group/worldcept/code/SimAny
set -euo pipefail
mkdir -p outputs/icra2027/slurm
CODE_COMMIT=$(git rev-parse HEAD)
[[ "$CODE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
SMOKE_FREEZE_ID="icra2027-contract-v1-e2-${CODE_COMMIT:0:12}-smoke"
FULL_FREEZE_ID="icra2027-contract-v1-e2-${CODE_COMMIT:0:12}-full"
```

First submit a fresh CPU-only E0 at that commit. Wait for terminal
`COMPLETED/0:0` before reading its log: a running or failed attempt is not a
contract. The log must contain exactly one repository-local `run_root`, and
the resulting manifest must record the exact clean 40-character code SHA and
a 64-character contract digest before any downstream job is submitted.

```bash
E0_JOB=$(sbatch --parsable --no-requeue \
  --account=runyi_yang \
  --job-name=icra-e0-contract-e2 \
  --partition=batch --nodelist=sof1-h200-2 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=00:30:00 \
  --output=/group/worldcept/code/SimAny/outputs/icra2027/slurm/e0-%j.out \
  --error=/group/worldcept/code/SimAny/outputs/icra2027/slurm/e0-%j.err \
  run/slurm/icra2027_e0_preflight.sbatch)
E0_JOB=${E0_JOB%%;*}
while :; do
  E0_STATE=$(sacct -nX -j "$E0_JOB" --format=State | awk 'NF {print $1; exit}')
  case "$E0_STATE" in
    COMPLETED) break ;;
    FAILED*|CANCELLED*|TIMEOUT*|PREEMPTED*|NODE_FAIL*|OUT_OF_MEMORY*|BOOT_FAIL*|DEADLINE*|REVOKED*)
      echo "E0 failed: job=$E0_JOB state=$E0_STATE" >&2
      exit 2
      ;;
  esac
  sleep 5
done
E0_EXIT_CODE=$(sacct -nX -j "$E0_JOB" --format=ExitCode | awk 'NF {print $1; exit}')
[[ "$E0_EXIT_CODE" == 0:0 ]] || {
  echo "E0 completed with unexpected exit code: $E0_EXIT_CODE" >&2
  exit 2
}
E0_LOG="/group/worldcept/code/SimAny/outputs/icra2027/slurm/e0-${E0_JOB}.out"
[[ -s "$E0_LOG" && ! -L "$E0_LOG" ]]
mapfile -t E0_ROOTS < <(sed -n 's/^run_root=//p' "$E0_LOG")
[[ ${#E0_ROOTS[@]} -eq 1 ]] || {
  echo "expected exactly one E0 run_root, found ${#E0_ROOTS[@]}" >&2
  exit 2
}
E0_ROOT=${E0_ROOTS[0]}
case "$E0_ROOT" in
  /group/worldcept/code/SimAny/outputs/icra2027/preflight-smoke-*) ;;
  *) echo "unexpected E0 run root: $E0_ROOT" >&2; exit 2 ;;
esac
CONTRACT="$E0_ROOT/contract/freeze_manifest.json"
[[ -s "$CONTRACT" && ! -L "$CONTRACT" ]]
CONTRACT_COMMIT=$(jq -er \
  '.code.commit | select(type == "string" and test("^[0-9a-f]{40}$"))' \
  "$CONTRACT")
[[ "$CONTRACT_COMMIT" == "$CODE_COMMIT" ]]
jq -e \
  '.code.dirty == false and .code.dirty_override_for_smoke == false' \
  "$CONTRACT" >/dev/null
jq -e \
  '.contract_sha256 | type == "string" and test("^[0-9a-f]{64}$")' \
  "$CONTRACT" >/dev/null
"$PWD/.venv/bin/python" - "$CONTRACT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
declared = payload["contract_sha256"]
canonical = json.dumps(
    {
        key: value for key, value in payload.items()
        if key not in {"created_utc", "environment", "contract_sha256"}
    },
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
observed = hashlib.sha256(canonical).hexdigest()
if observed != declared:
    raise SystemExit(
        f"E0 contract digest mismatch: declared={declared} observed={observed}"
    )
PY
```

Load the exact scene and replacement rosters from the frozen config. The count
checks prevent a truncated submission loop from creating a plausible-looking
dependency chain:

```bash
CONFIG="$PWD/configs/experiments/icra2027/fidelity_manifest.json"
mapfile -t SMOKE_SCENES < <(jq -er '.population.smoke_scene_ids[]' "$CONFIG")
mapfile -t FULL_SCENES < <(jq -er '.population.scene_ids[]' "$CONFIG")
mapfile -t REPLACEMENT_BUNDLES < <(
  jq -er '.leakage_remediation.bundles[].bundle_id' "$CONFIG"
)
[[ ${#SMOKE_SCENES[@]} -eq 2 ]]
[[ ${#FULL_SCENES[@]} -eq 50 ]]
[[ ${#REPLACEMENT_BUNDLES[@]} -eq 5 ]]
```

Run the pinned two-scene smoke end to end first. Each scene is an ordinary
Slurm job with its explicit `E2_SCENE_ID`; smoke inventory depends on both job
IDs, and smoke metrics depends on that inventory:

```bash
SMOKE_RENDER_JOBS=()
for SCENE_ID in "${SMOKE_SCENES[@]}"; do
  JOB=$(sbatch --parsable --no-requeue \
    --account=runyi_yang \
    --job-name="icra-e2-smoke-${SCENE_ID}" \
    --partition=batch --qos=normal --nodelist=gcp-eu1-a100-80g-qrfh \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=04:00:00 \
    --gpus=a100-80g:1 \
    --dependency="afterok:${E0_JOB}" --kill-on-invalid-dep=yes \
    --output="/group/worldcept/code/SimAny/outputs/icra2027/slurm/e2-smoke-${SCENE_ID}-%j.out" \
    --error="/group/worldcept/code/SimAny/outputs/icra2027/slurm/e2-smoke-${SCENE_ID}-%j.err" \
    --export=ALL,E2_GPU_PROFILE=gcp-a100,E2_PHASE=render-room,E2_MODE=smoke,E2_SCENE_ID="$SCENE_ID",E2_FREEZE_ID="$SMOKE_FREEZE_ID",E2_CODE_COMMIT="$CODE_COMMIT",E2_CONTRACT_MANIFEST="$CONTRACT" \
    run/slurm/icra2027_e2_fidelity.sbatch)
  SMOKE_RENDER_JOBS+=("${JOB%%;*}")
done
[[ ${#SMOKE_RENDER_JOBS[@]} -eq 2 ]]
SMOKE_RENDER_DEP=$(IFS=:; printf '%s' "${SMOKE_RENDER_JOBS[*]}")

SMOKE_INVENTORY_JOB=$(sbatch --parsable --no-requeue \
  --account=runyi_yang \
  --job-name=icra-e2-inventory-smoke --partition=batch --nodelist=sof1-h200-2 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=01:00:00 \
  --dependency="afterok:${SMOKE_RENDER_DEP}" --kill-on-invalid-dep=yes \
  --output=/group/worldcept/code/SimAny/outputs/icra2027/slurm/e2-smoke-inventory-%j.out \
  --error=/group/worldcept/code/SimAny/outputs/icra2027/slurm/e2-smoke-inventory-%j.err \
  --export=ALL,E2_PHASE=inventory,E2_MODE=smoke,E2_FREEZE_ID="$SMOKE_FREEZE_ID",E2_CODE_COMMIT="$CODE_COMMIT",E2_CONTRACT_MANIFEST="$CONTRACT" \
  run/slurm/icra2027_e2_fidelity.sbatch)
SMOKE_INVENTORY_JOB=${SMOKE_INVENTORY_JOB%%;*}

SMOKE_METRICS_JOB=$(sbatch --parsable --no-requeue \
  --account=runyi_yang \
  --job-name=icra-e2-metrics-smoke \
  --partition=batch --qos=normal --nodelist=gcp-eu1-a100-80g-qrfh \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=01:00:00 \
  --gpus=a100-80g:1 \
  --dependency="afterok:${SMOKE_INVENTORY_JOB}" --kill-on-invalid-dep=yes \
  --output=/group/worldcept/code/SimAny/outputs/icra2027/slurm/e2-smoke-metrics-%j.out \
  --error=/group/worldcept/code/SimAny/outputs/icra2027/slurm/e2-smoke-metrics-%j.err \
  --export=ALL,E2_GPU_PROFILE=gcp-a100,E2_PHASE=metrics,E2_MODE=smoke,E2_FREEZE_ID="$SMOKE_FREEZE_ID",E2_CODE_COMMIT="$CODE_COMMIT",E2_CONTRACT_MANIFEST="$CONTRACT" \
  run/slurm/icra2027_e2_fidelity.sbatch)
SMOKE_METRICS_JOB=${SMOKE_METRICS_JOB%%;*}
```

Enqueue the five bundles immediately after recording the smoke job IDs. Each
is an ordinary job with one explicit `E2_BUNDLE_ID`; its dependency makes it
wait for the smoke metric gate rather than requiring a second manual submit:

```bash
REPLACEMENT_JOBS=()
for BUNDLE_ID in "${REPLACEMENT_BUNDLES[@]}"; do
  JOB=$(sbatch --parsable --no-requeue \
    --account=runyi_yang \
    --job-name="icra-e2-replace-${BUNDLE_ID}" \
    --partition=batch --qos=normal --nodelist=gcp-eu1-a100-80g-qrfh \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=100G --time=03:50:00 \
    --gpus=a100-80g:1 \
    --dependency="afterok:${SMOKE_METRICS_JOB}" --kill-on-invalid-dep=yes \
    --output="/group/worldcept/code/SimAny/outputs/icra2027/slurm/e2-replace-${BUNDLE_ID}-%j.out" \
    --error="/group/worldcept/code/SimAny/outputs/icra2027/slurm/e2-replace-${BUNDLE_ID}-%j.err" \
    --export=ALL,E2_GPU_PROFILE=gcp-a100,E2_BUNDLE_ID="$BUNDLE_ID",E2_FREEZE_ID="$FULL_FREEZE_ID",E2_CODE_COMMIT="$CODE_COMMIT",E2_CONTRACT_MANIFEST="$CONTRACT" \
    run/slurm/icra2027_e2_replacements.sbatch)
  REPLACEMENT_JOBS+=("${JOB%%;*}")
done
[[ ${#REPLACEMENT_JOBS[@]} -eq 5 ]]
REPLACEMENT_DEP=$(IFS=:; printf '%s' "${REPLACEMENT_JOBS[*]}")
```

Submit 50 ordinary one-scene render jobs. Every render carries an `afterok`
dependency on all five replacement job IDs:

```bash
FULL_RENDER_JOBS=()
for SCENE_ID in "${FULL_SCENES[@]}"; do
  JOB=$(sbatch --parsable --no-requeue \
    --account=runyi_yang \
    --job-name="icra-e2-full-${SCENE_ID}" \
    --partition=batch --qos=normal --nodelist=gcp-eu1-a100-80g-qrfh \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=04:00:00 \
    --gpus=a100-80g:1 \
    --dependency="afterok:${REPLACEMENT_DEP}" --kill-on-invalid-dep=yes \
    --output="/group/worldcept/code/SimAny/outputs/icra2027/slurm/e2-full-${SCENE_ID}-%j.out" \
    --error="/group/worldcept/code/SimAny/outputs/icra2027/slurm/e2-full-${SCENE_ID}-%j.err" \
    --export=ALL,E2_GPU_PROFILE=gcp-a100,E2_PHASE=render-room,E2_MODE=full,E2_SCENE_ID="$SCENE_ID",E2_FREEZE_ID="$FULL_FREEZE_ID",E2_CODE_COMMIT="$CODE_COMMIT",E2_CONTRACT_MANIFEST="$CONTRACT" \
    run/slurm/icra2027_e2_fidelity.sbatch)
  FULL_RENDER_JOBS+=("${JOB%%;*}")
done
[[ ${#FULL_RENDER_JOBS[@]} -eq 50 ]]
FULL_RENDER_DEP=$(IFS=:; printf '%s' "${FULL_RENDER_JOBS[*]}")
```

There is deliberately no client-side concurrency throttle. Slurm schedules
the ordinary A100 replacement and render jobs against current capacity on
`gcp-eu1-a100-80g-qrfh`; excess ordinary jobs remain pending until A100
capacity is released. The account's separate `gres/gpu:a6000=4` association
limit does not throttle this A100 profile.

Submit inventory as a CPU-only dependent job. Do not pass `--gpus=0`:
some Slurm configurations reject it; omitting all GPU options is the portable
way to request no GPU. The dependency names all 50 render job IDs, so any one
failed render prevents inventory from starting.

```bash
INVENTORY_JOB=$(sbatch --parsable --no-requeue \
  --account=runyi_yang \
  --job-name=icra-e2-inventory-full \
  --partition=batch --nodelist=sof1-h200-2 \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=04:00:00 \
  --dependency="afterok:${FULL_RENDER_DEP}" --kill-on-invalid-dep=yes \
  --output=/group/worldcept/code/SimAny/outputs/icra2027/slurm/e2-inventory-%j.out \
  --error=/group/worldcept/code/SimAny/outputs/icra2027/slurm/e2-inventory-%j.err \
  --export=ALL,E2_PHASE=inventory,E2_MODE=full,E2_FREEZE_ID="$FULL_FREEZE_ID",E2_CODE_COMMIT="$CODE_COMMIT",E2_CONTRACT_MANIFEST="$CONTRACT" \
  run/slurm/icra2027_e2_fidelity.sbatch)
INVENTORY_JOB=${INVENTORY_JOB%%;*}
```

Submit metrics on one GCP A100-80G only after inventory succeeds:

```bash
METRICS_JOB=$(sbatch --parsable --no-requeue \
  --account=runyi_yang \
  --job-name=icra-e2-metrics-full \
  --partition=batch --qos=normal --nodelist=gcp-eu1-a100-80g-qrfh \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=04:00:00 \
  --gpus=a100-80g:1 \
  --dependency="afterok:${INVENTORY_JOB}" --kill-on-invalid-dep=yes \
  --output=/group/worldcept/code/SimAny/outputs/icra2027/slurm/e2-metrics-%j.out \
  --error=/group/worldcept/code/SimAny/outputs/icra2027/slurm/e2-metrics-%j.err \
  --export=ALL,E2_GPU_PROFILE=gcp-a100,E2_PHASE=metrics,E2_MODE=full,E2_FREEZE_ID="$FULL_FREEZE_ID",E2_CODE_COMMIT="$CODE_COMMIT",E2_CONTRACT_MANIFEST="$CONTRACT" \
  run/slurm/icra2027_e2_fidelity.sbatch)
METRICS_JOB=${METRICS_JOB%%;*}
printf 'e0=%s smoke_renders=%s smoke_inventory=%s smoke_metrics=%s replacements=%s full_renders=%s full_inventory=%s full_metrics=%s\n' \
  "$E0_JOB" "$SMOKE_RENDER_DEP" "$SMOKE_INVENTORY_JOB" \
  "$SMOKE_METRICS_JOB" "$REPLACEMENT_DEP" "$FULL_RENDER_DEP" \
  "$INVENTORY_JOB" "$METRICS_JOB"
```

The inventory uses the reviewed object snapshot at
`outputs/review_recompute_hybrid.json` by default. Override
`E2_REVIEWED_OBJECT_AGGREGATE` only with another regular file inside this
checkout; the inventory still rejects incoherent or leakage-prone evidence.

## Monitoring and completion checks

```bash
ALL_JOB_IDS=(
  "$E0_JOB"
  "${SMOKE_RENDER_JOBS[@]}"
  "$SMOKE_INVENTORY_JOB"
  "$SMOKE_METRICS_JOB"
  "${REPLACEMENT_JOBS[@]}"
  "${FULL_RENDER_JOBS[@]}"
  "$INVENTORY_JOB"
  "$METRICS_JOB"
)
[[ ${#ALL_JOB_IDS[@]} -eq 62 ]]
ALL_JOBS=$(IFS=,; printf '%s' "${ALL_JOB_IDS[*]}")
squeue -j "$ALL_JOBS" \
  -o '%.20i %.28j %.10T %.10M %.10l %.24R %b'
sacct -j "$ALL_JOBS" \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,NodeList,AllocTRES
```

For recurring monitoring, run the same `squeue` command with `watch -n 15`.
For example:

```bash
watch -n 15 "squeue -j '$ALL_JOBS' -o '%.20i %.28j %.10T %.10M %.10l %.24R %b'"
```

After completion, inspect all 62 ordinary jobs in `sacct`; replacement
generation cannot begin unless the complete smoke passes, every full render
depends on all five replacements, full inventory depends on all 50 renders,
and each metrics job depends on its matching validated inventory. The
authoritative full output is:

```text
outputs/icra2027/<freeze_id>/fidelity/
  replacements/                       # 5 atomic bundles, 8 regenerated objects
  room_runs/                         # 50 atomic scene bundles
  inventory/
    manifests/fidelity_manifest.json
    validation_report.json
  table/
    fidelity_table.json
    fidelity_table.csv
  inventory_run_manifest.json
  run_manifest.json
  artifact_hashes.sha256
```

Success of the full room job means the pinned 50-scene appearance evaluation
was generated and validated. It does not make all of Table II paper-ready:
the Harmonizer row depends on E5, and object rows remain unavailable until
held-out masks and registration-independent evaluation surfaces pass the
strict manifest checks. Missing cells remain null with reasons; known paper
anchors are comparison checks only and are never copied into output files.

## Two-scene developer smoke

The config pins scenes `38d58a7a31` and `5748ce6f01` solely for the fast
pre-full regression gate. The accepted chain above submits one ordinary job
per scene with `E2_MODE=smoke`, its explicit `E2_SCENE_ID`, and a distinct
smoke ID, then runs inventory only after both render IDs and metrics only after
inventory. It produces exactly two manifests and 64 total PNGs: 48 method
renders plus 16 paired GT targets, along with a non-paper smoke table. A smoke
is never accepted as the requested full experiment and cannot satisfy the
50-scene coverage gate.

Job `815356` was a successful one-scene, pre-freeze development pilot on
`5748ce6f01`. Because it predates the final clean commit, fresh E0 contract,
replacement chain, and pinned two-scene smoke, it is diagnostic only and is
not accepted E2 evidence.

## Fixed Table II rows

Room rows, in order:

1. Input scene Gaussian, reconstruction ceiling
2. Factorized composite, GT discovery
3. Factorized composite, automatic discovery
4. Composite + Harmonizer Option C

Object rows, in order:

1. TRELLIS, best single view
2. ReconViaGen, multi-view
3. Evidence-selected proposal
4. Evaluation-only oracle candidate

No room-level CD or object-level full-frame PSNR is reported. Dashes/nulls are
used for scientifically undefined comparisons.

## Acceptance gates

- [ ] All 50 individual full-render jobs succeed, not only the smoke subset.
- [ ] Each scene has one manifest and 32 hash-verified PNGs.
- [ ] Each non-Harmonizer room row has 50 scenes and 400 paired views.
- [ ] Every evaluation frame is official-test and absent from the complete
      accepted-object generation-input union.
- [ ] Room and object units remain separate in JSON, CSV, and bootstrap output.
- [ ] Registration and evaluation surfaces are independent for every non-null
      object geometry record.
- [ ] Every metric carries sample count and source-artifact hashes.
- [ ] Empty paper cells remain explicit and no known anchor is copied.
- [ ] `STATUS.md` records exact Slurm IDs, terminal states, commit, contract,
      output root, sample counts, claim gate, and artifact-hash verification.
