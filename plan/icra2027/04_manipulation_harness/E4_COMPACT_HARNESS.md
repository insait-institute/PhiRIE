# Compact E4 harness preparation

Owner: E4 compact handoff agent. Branch: `agent/icra-e4-compact-harness`.
Scope: engineering preparation only; no learned rollout or paper claim.

The preregistered protocol at
`configs/experiments/icra2027/e4_compact_canonical/protocol.yaml` is owned by the
canonical upstream task. It fixes the two rooms, all discovered objects and all
prospective semantic queries before construction and qualification outcomes.
The four selected tasks are `27dd4da69e__obj_1001_to_region`,
`27dd4da69e__obj_1001_to_obj_1000`, `40aec5fffa__obj_1001_to_region`, and
`40aec5fffa__obj_1001_to_obj_1005`. No failed task may be replaced by a qualifier
winner. A0/A4 share five reset definitions per task and the same pi05 policy.

## Implemented preparation

`run/icra2027/e4_compact_harness.py` checks the protocol hash, exact task endpoints,
semantic first-task selection, cohort, policy, seeds and jitter. It uses
`robo.eval.episode_log.ResetState`, `derive_reset_seed` and `episode_id_for` to
preregister 20 reset identities and 40 arm/reset cells. These contain **no
measured or invented initial pose**. Actual common-frame reset geometry must
come from the existing task/reset producers after canonical endpoints exist.
The script does not write `reset_states.json`, a runnable harness config, or
`harness_ledger.jsonl`, and cannot launch a policy or GPU job.

With an exact-source qualification ID, it delegates full source, static export,
geometry, Menagerie, reset, drift and denominator replay to
`e4_candidate_screen._validate_qualifier_output` and `_load_prepare` before
extracting the prechosen cells. It verifies all 280 prospective qualification
cells against the preregistration, retains each selected failure and its original
cell ID, and does not rank or substitute tasks. Failed qualification is not
reported as an attempted rollout. Unattempted outcomes remain null.

```bash
mkdir -p .t
/group/worldcept/code/SimAny/.venv/bin/python -m pytest -q \
  tests/test_e4_compact_harness.py tests/test_e4_reset_eligibility.py \
  --basetemp=.t/compact
/group/worldcept/code/SimAny/.venv/bin/python -m run.icra2027.e4_compact_harness \
  --protocol /group/worldcept/code/SimAny-wt/e4-compact-canonical/configs/experiments/icra2027/e4_compact_canonical/protocol.yaml \
  --protocol-sha256 <replace-with-authenticated-protocol-sha256> \
  --out outputs/icra2027/<new-audit-directory>/compact_handoff
```

The placeholder hash must come from the upstream sealed input inventory. To authenticate later CPU results,
add `--qualification-screen-id <E4-freeze-id>` and
`--expected-code-commit <exact-qualification-source-sha>` while running from that
same source worktree. Inputs are read-only; output directories refuse overwrite.
The qualification source must contain this wrapper; do not alias a historical
producer commit to the current source.

Smoke: 24 tests PASS (0.73s), including missing/duplicate selected cells, source
replay failure, protocol drift, changed policy/reset/jitter/cohort, no overwrite,
paired canonical reset IDs, null unattempted outcomes and all-failed qualification.
First test invocation failed because repository-local `.t` did not exist;
creating its parent fixed the invocation without changing any gate.
No Slurm IDs or GPU allocations. Pilot/full learned execution: NOT_RUN.

## Exact next producer/validator extensions

The complete path cannot be safely enabled by a wrapper alone:

1. **Automatic camera/scorer producer** (`robo/eval/e4_camera_scorer_gate.py`).
   `validate_cpu_chain` is tied to the old region pilot and `validate_winner_chain`
   to a single legacy winner. Add an explicit automatic paired-task scope that
   consumes sealed task manifests and original qualification rows for these
   fixed tasks; reuse the current rig, visibility, robot/target mask, scorer
   positives/negatives and state-restoration checks. The same common observed
   frame must define both arms; do not search cameras or relax pixel/drift limits.
2. **Eligibility and nonrollout certification**
   (`robo/eval/e4_reset_eligibility.py`). `load_reset_eligibility` and
   `certify_prebuild_failure` currently reject real pi05 and automatic qualifier
   bundles. Add a typed branch backed by the full canonical qualifier replay
   and the camera/scorer producer above. Preserve endpoint/precondition failure
   rows with null reset telemetry. Bind every failure to original source/hash,
   task bundle and the fixed reset. Keep the legacy scripted branch unchanged.
3. **Canonical real-policy launch boundary** (`robo/eval/harness_runner.py`).
   `run_matrix` connects to a real server before writing CPU-rejected cells.
   Support authenticated rejected cells without requiring a running policy,
   then gate every eligible rollout on the new camera/scorer closure plus
   unchanged `expected_server_identity_from_config`. Keep one canonical ledger,
   one shared reset bank, original checkpoint/robot/control/camera hashes and
   all 40 planned arm/reset pairs. No per-arm reset transforms or quiet fallback.

The existing runner already reads a canonical persisted reset list, so no new
reset planner is required. Generate the actual bank from the fixed common
layout and frozen task definitions after qualification artifacts exist. Never
filter tasks by learned-policy outcomes. Full automatic qualifier population
and the fixed manipulation subset are distinct denominators and both remain
explicit. The current wrapper leaves launch flags false even when CPU replay
passes; real camera, scorer, scripted smoke and policy identity remain separate
unexecuted gates. The central eligibility, camera, runner and metric files are
unchanged by this bounded handoff.

## Automatic camera/scorer extension

The first extension above is now implemented in `e4_camera_scorer_gate` under
`study_scope: automatic_compact_camera_diagnostic`. It authenticates the entire
sealed automatic qualification population, then evaluates the predeclared 40
cells without selecting replacement tasks. Both arms retain identical task,
robot, table, camera and exclusion fields. CPU-rejected cells keep their original
failure checks and null camera/reset telemetry. Diagnostic exceptions remain
unexecuted typed failures; other planned cells are still processed.

The public interfaces are `validate_automatic_chain(config, root=...)`,
`write_automatic_camera_config(...)`, and
`validate_automatic_camera_output(config_path=..., output=...,
expected_code_commit=...)`. The chain returns original CPU `rows`, per-scene
`suites`, `factories`, and `task_bundle`, plus authenticated upstream identities.
The output validator returns `gate`, `cells`, and `manifest_sha256`. Every cell
has the original CPU `cell_id`, `executed`, `passed`, and an explicit outcome;
completed cells additionally have boolean `checks` and actual rendered/scorer
measurements. A camera failure is preserved in the planned denominator.

Create the config using `write_automatic_camera_config` only after sealed
qualification exists, passing `protocol_path`, `protocol_sha256`,
`qualifier_screen_id`, a newly reserved `freeze_id`, `menagerie_root`,
`expected_code_commit`, and a fresh `out` path. Qualifier and camera execution
must use the same final integrated source/worktree: the existing source validator
checks both. Historical materialization retains its original authenticated
source; it must not be relabeled as the camera producer.

```bash
source /group/worldcept/code/SimAny/outputs/test-headless-setup/env.sh
export LP_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=/group/worldcept/code/openpi-wt/e4-policy-server/packages/openpi-client/src:.
/group/worldcept/code/SimAny/.venv/bin/python -m robo.eval.e4_camera_scorer_gate \
  --automatic-config <authenticated-camera-config.json> \
  --expected-code-commit <final-integrated-source> --automatic-preflight-only
/group/worldcept/code/SimAny/.venv/bin/python -m robo.eval.e4_camera_scorer_gate \
  --automatic-config <authenticated-camera-config.json> \
  --expected-code-commit <final-integrated-source>
```

The full producer requires OSMesa. It replays each eligible CPU reset exactly
(including the 900-step settle protocol, stability and workspace evidence), then
uses the existing camera, workspace, object-disambiguation and direct-state
scorer diagnostics with unchanged thresholds. It seals `cells.jsonl`, per-cell
records, rendered images, `gate.json`, `manifest.json` and `seal.json` under
`outputs/icra2027/<freeze_id>/harness/automatic_camera_scorer`; overwrite is
refused. The global diagnostic result grants no learned-policy permission.
Eligibility and canonical runner integration remain separate gates.

Smoke: 132 tests PASS (7.51 s), including real OSMesa renders, exact CPU reset
replay, scorer negative paths, paired-field drift, missing qualification,
changed source/config/artifacts, repeated cell IDs, fabricated telemetry,
retained diagnostic failures, and legacy camera/winner regressions. Command:

```bash
# With the OSMesa, threading and PYTHONPATH environment above:
/group/worldcept/code/SimAny/.venv/bin/python -m pytest -q \
  tests/test_e4_automatic_camera.py tests/test_e4_winner_camera.py \
  tests/test_e4_camera_scorer_gate.py tests/test_e4_receptacle_contract.py \
  tests/test_e4_compact_harness.py --basetemp=.t/auto-camera-final-r2
```

Real pilot/full: NOT_RUN pending sealed actual qualification artifacts from the
integrated source. No GPU allocation, policy service, policy rollout or success
measurement was produced by this extension. Synthetic fixture visibility results
are engineering checks, not scientific evidence or a basis for tuning cameras.

## Source-bound automatic qualification launcher

`run.icra2027.e4_compact_qualification` consumes the original materialization
handoff from freeze `20260905-e8277d2-v1`, source
`14964859267ac986cd595344737570b0d733a313`, E0 digest
`bfe82199cd6103514e02025d19dfb7da65a47ef7392b339d2d11bbca9a035230`.
It authenticates that original E0, config, both scene receipts/descriptors and
all four destination-bound materialization manifests, then replays the sealed
canonical source closure. It does not relocate materializations or alter their
manifests. The existing candidate producer rematerializes the original sealed
E3 selections at its required stage paths; no asset generation or observation
rendering is repeated. Full-room export, planning and qualification remain the
existing producers with unchanged gates.

After the launcher, camera and eligibility changes are integrated, reserve a
new freeze through the shared contract API. Prepare its config from that final
worktree, commit the config if tracked, and run E0 at the exact resulting source.
E0 must bind file resources named `e4_compact_qualification_config` and
`qualification_python`. The config records original source identities, model
and resize identities, interpreter and new freeze ID; it does not claim any
new qualification result.

```bash
# In the final integrated clean worktree, before stage E0:
python -m run.icra2027.e4_compact_qualification \
  --inspect-handoff /group/worldcept/code/SimAny/outputs/icra2027/20260905-e8277d2-v1/materialization/handoff.json
python -m run.icra2027.e4_compact_qualification \
  --prepare-config /group/worldcept/code/SimAny/outputs/icra2027/20260905-e8277d2-v1/materialization/handoff.json \
  --freeze-id <new-reserved-id> --menagerie-root <final-worktree>/third_party/mujoco_menagerie \
  --out <new-stage-config.json>
# After exact-source E0, independent ordinary CPU jobs, one per fixed scene:
python -m run.icra2027.e4_compact_qualification \
  --config <new-stage-config.json> --freeze-root <new-stage-root> \
  --expected-code-commit <exact-final-source> --scene-id 27dd4da69e
python -m run.icra2027.e4_compact_qualification \
  --config <new-stage-config.json> --freeze-root <new-stage-root> \
  --expected-code-commit <exact-final-source> --scene-id 40aec5fffa
# After both sealed qualifiers, using the OSMesa/OpenPI environment above:
python -m run.icra2027.e4_compact_qualification \
  --config <new-stage-config.json> --freeze-root <new-stage-root> \
  --expected-code-commit <exact-final-source> --camera
```

The canonical outputs live directly under the new stage's
`automatic_candidates/<scene>`, `scene_prepares/<scene>`,
`task_freezes/<scene>` and `scene_qualifiers/<scene>` directories. The full
qualification denominator is 280; the camera subset retains the fixed 40 cells.
Camera config is generated only after full qualification authenticates, and
replayed against the E0-bound source, models and scope. The launcher resumes
completed qualifiers by validating their original sealed output, never by
rewriting results. Missing upstream readiness returns WAITING without fabricated
build failures. Arrays and GPU reservations are rejected. Policy services and
learned execution are outside this launcher.

Launcher/compact smoke: 43 tests PASS (0.79 s), covering original source and
artifact tampering, E0/config/model/interpreter drift, retained failed results,
full-population delegation, replay-only resume, WAITING without execution,
fixed-scene enforcement and GPU/array rejection. Reproduce with
`python -m pytest -q tests/test_e4_compact_qualification.py tests/test_e4_compact_harness.py`.
Real qualification/camera pilot remains NOT_RUN until a new integrated freeze
executes these commands. Original materialization jobs 832171 and 832172 completed;
their handoff is input evidence, not a qualification or policy-success result.

Historical validator closure: the original materializer pins both validator
commit and worktree. The launcher therefore invokes all four original factory
validations in one read-only subprocess using the original E0-bound interpreter
and exact clean `1496485` worktree. It never aliases that provenance to the new
qualifier source. The new qualifier rematerializes through its existing public
producer. Launcher/compact coverage now passes 48 tests (0.64 s), including
historical source/dirty-tree rejection and exact interpreter/worktree routing.
