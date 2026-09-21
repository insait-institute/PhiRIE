# D0 demo status

## 2026-09-06 05:47 UTC — full-cohort engineering clips delivered

Owner: `trellis2_environment`; branch `agent/icra-d0-final-mode`; immutable
rendering source `399aee3403428c6daf54402beaac329cc2e94f68`. State:
`IMPLEMENTING`; engineering smoke and pilot `PILOT_PASSED`; final films `NOT_RUN`.
Freeze `20260906-aa67bf0-v2`; actual E0 SHA
`1432efe9191234aeeba5184e2cfa4343252343ebfe1d470f6441331191ce5bbf`.
Config SHA `7b6e729dc6250f24b230974de64c09800aa4194a71475d0370c9b6a9ea9390af`.

Focused smoke: primary Python `pytest -q tests/test_demo_final.py
tests/test_demo_manifest.py tests/test_demo_timeline.py tests/test_demo_evidence.py
tests/test_freeze.py tests/test_contract_validation.py`, **55 passed in 19.21 s**.
Exact-source E0: **130 passed**, audit
`86d82f9720b86c888e5123ec2b8d6885f3ea9d024be4e553acd2f5f304c23c00`.
Regression checks include immutable source comparison after rendering, forged
episode/frame identities, missing gates, overwritten outputs and bounded fallback.

Real engineering render command from the clean source worktree:
`SIMANY_PY=/group/worldcept/code/SimAny/.venv/bin/python bash run/demo/render_icra2027.sh
--config configs/demo/icra2027_full_evidence.yaml
--out /group/worldcept/code/SimAny/outputs/icra2027/20260906-aa67bf0-v2/demo`.
Existing headless OSMesa environment and four native threads are required.
Result: exit 0, CPU only, no Slurm job or loaded checkpoint. Encoding runtime was
not separately recorded and is not reported as experimental runtime.

Delivered `demo/evidence_clips/{selection,retry,frozen_results}_1080p.{mp4,png,srt}`:
three 6-second, 1920x1080, 30 fps clips with 180 frames each. Full decode and
independent source/hash validation PASS. Both agent and root visually inspected
all three posters: complete geometry, legible labels, no clipping; failed
construction probes remain explicit. Selection is original `09c1414f1b/obj_1000`;
genuine changed-registration retry is `09c1414f1b/obj_1002`, both chosen by the
predeclared lexicographic rule. The result clip uses the exact whole published
E9 figure from `20260906-6dcb0e7-v2`.

Independent QA: `20260906-aa67bf0-v2/independent_demo_qa.json`, SHA
`a09b5cc334e340639f085a1885669073fa2528652574b642ab8a5774959e9bf7`.
Source manifest SHA `5ce8a5443592fb63af18e66f57b65b91e5623e8a88fb995232fa6e625a6ade70`.
The unexecuted v1 reservation remains preserved; the source mutation repair used
a new freeze before rendering. Final 90/30/8-second products still require
authentic synchronized E4 footage, E5 certified restoration footage, and a common
scientific submission freeze. Claim gate **NOT_RUN**; `FULL_DEMO_READY=false`.

Current implementation and source-bound draft results are in the milestone below. Final films remain NOT_RUN.

## Initial audit (superseded by the implementation milestone)

Updated: 2026-09-04 UTC. Owner: root orchestrator.
Branch: `agent/icra-e3-agentic`; audited code: `c3b97ffe4db235cff8728bb5a836ae5f6f02a2c0`.
State: IMPLEMENTING. Final demo gate: NOT_RUN.

Scaffolding is now assigned to the D0 agent in isolated worktree `/group/worldcept/code/SimAny-wt/d0-demo`, branch `agent/icra-d0-demo`, based on `52d696f`. Audit ID `20260904-07e8b05-v6` is reserved; it may only hold a clearly labeled engineering draft after the source/config is committed. Final film gates remain unmet.

- Smoke/pilot/full commands: NOT_RUN. Required `interface/demo_agentic.py`, `configs/demo/icra2027.yaml` and four `run/demo/*icra2027.sh` launchers are absent in this source state.
- Existing reusable framework: `interface/demo_movie.py`, `interface/demo_session.py`, viewer and rendering utilities.
- E3 genuine selection and retry evidence exists at `outputs/icra2027/icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic`; it is preliminary and cannot be labeled a final paper result.
- Missing final inputs: eligible full-room collision/task suite and genuine matched E4 policy episode, E5 preservation-certified frames, complete real-capture hero, and E9 final claim-valid freeze.
- Slurm jobs: none. Render hardware/runtime, config hashes, source episode IDs and output checksums: not yet available.
- Passed criteria: none of the requested 90 s, 30 s, 8 s deliverables is claimed complete.
- Blocker classification: implementation and upstream scientific/dependency gates. No final shot selection or successful-policy montage has been fabricated.
- Required smoke: 12 s at 960x540/15 fps from declared source hashes, plus tested three-second local-video fallback. Required final outputs remain 1080p hero/teaser/loop, poster, subtitles, source manifest and live launcher.


## D0 framework and engineering draft, 2026-09-04

- Owner: D0 demo agent. Branch: `agent/icra-d0-demo`.
- Implementation/source commit: `8aa18ebde5f24716c7da8e7922a33ed925642a98`; clean source at draft rendering.
- State: `IMPLEMENTING`; framework tests `SMOKE_PASSED`; required complete storyboard smoke, pilot, and final hero/teaser/loop `NOT_RUN`.
- Freeze: `20260904-07e8b05-v6`; source worktree `/group/worldcept/code/SimAny-wt/d0-demo`.
- Output: `outputs/icra2027/20260904-07e8b05-v6/demo/`.
- CPU hardware: hala; no GPU allocation or Slurm jobs.
- Tests: `TMPDIR=/group/worldcept/code/SimAny-wt/d0-demo/.tmp /group/worldcept/code/SimAny/.venv/bin/python -m pytest -q tests/test_demo_manifest.py tests/test_demo_timeline.py --basetemp=/group/worldcept/code/SimAny-wt/d0-demo/.tmp/d0-clean-tests`; exit 0, **10 passed in 2.66 s**.
- Durable test/preflight/render/assembly logs: `outputs/icra2027/20260904-07e8b05-v6/engineering_audit/`.
- Initial development failures: test helper import path (code bug, fixed before source commit); absent pytest temporary parent directory (environment setup, fixed). No remaining framework test failures.

Reproducible commands from the source worktree (set `SIMANY_PY=/group/worldcept/code/SimAny/.venv/bin/python`):

```bash
bash run/demo/preflight_icra2027.sh --config /group/worldcept/code/SimAny/outputs/icra2027/20260904-07e8b05-v6/resolved_demo.yaml
bash run/demo/render_icra2027.sh --config /group/worldcept/code/SimAny/outputs/icra2027/20260904-07e8b05-v6/resolved_demo.yaml --out /group/worldcept/code/SimAny/outputs/icra2027/20260904-07e8b05-v6/demo
bash run/demo/assemble_icra2027.sh --demo-dir /group/worldcept/code/SimAny/outputs/icra2027/20260904-07e8b05-v6/demo
```

All returned exit 0. The render command intentionally refuses the existing
output; a rerun requires a fresh freeze/config. The encoded artifact is
`engineering_draft_12s.mp4`: 12.000 seconds, 180 frames, 960×540, 15 fps,
H.264/yuv420p. Poster, subtitles, source manifest, artifact hashes, and local
video presentation manifest accompany it. Encoding wall time was not separately
instrumented; movie duration is not presented as compute runtime.

The four shots are a real held-out DSLR frame; a genuine preliminary E3
evidence-selected object; a genuine E3 retry that **remains unsupported**;
and an explicit missing-manipulation card. No policy episode, physical
success, synchronized physics view, or final scientific result is depicted.
Selection follows first lexicographic E3 evidence-selected A2 record and
first lexicographic A3 produced retry, plus first lexicographic E2 frame in
the same scene. All selected-asset artifact hashes are verified.

Sources: E2 `icra2027-contract-v1-e2-2c7ed67df534-full`; E3
`icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2`. Source freezes are
explicitly retained in the new draft manifest, never relabeled as new
experiments. E3 records are scene `09c1414f1b`, A2/obj_00 and A3/obj_01.
No checkpoint is loaded by D0; imported artifact identities and source
closure hashes are archived, with inherited preliminary provenance limits.

Negative checks cover altered hashes, mixed freezes, spliced episode IDs,
invalid numeric result overlays, unsupported paper mode, rereading a proposal
labeled as retry, output overwrites, wrong duration, changed fallback file,
hung health checks, and service failure after live launch.

Live watchdog integration against unreachable `http://127.0.0.1:9/health`
dispatched the prevalidated local video in **0.114614486 s**; actual MP4
headless decoding exited 0. Evidence: `engineering_audit/watchdog_result.json`.
Package preflight checks content hashes; live playback checks pinned size/mtime
and uses bounded health subprocesses. Physical display/player startup was
not verified. No service or policy inference is claimed live.

The four-shot contact sheet was visually inspected: captions are readable,
all shots carry a preliminary-draft watermark, retry failure remains visible,
and the missing-manipulation card is unambiguous. Contact sheet:
`engineering_audit/contact_sheet.jpg`.

Passed criteria: tested framework, immutable source closure, genuine selection
and retry records, 12-second draft encoding, subtitles/poster, honest
dependency card, and bounded fallback dispatch. Unmet criteria: full
four-stage D0 smoke story, rendered proposal comparisons, synchronized E4
physics/appearance footage, E5 preservation-certified frames, final E9
scientific freeze, 80–95 second hero, 30-second teaser, seamless loop, and
conference-device QA. Claim gate: **NOT_RUN**; `FULL_DEMO_READY=false`.

Hashes relative to the v6 freeze:

- `resolved_demo.yaml`: `6f5d80c8f8c8a881e8bb96b494aa64cb720bc243574ef2b6db25ff30bd2bab33`
- `demo/source_manifest.json`: `3bf38632bc320fd6fc8b0675bcc8a39da23540a281d7dcd8d864f9734b5949a7`
- `demo/engineering_draft_12s.mp4`: `76ebe3b7d6bcc7dac6677b88bdcf0eae1f1e8a6fd39a1ae589ee7f08ab343b6c`
- `demo/poster.png`: `ebb8774b842aef164a5b45709d466727ca6072f608e2c1270fa079da203976e7`
- `engineering_audit/contact_sheet.jpg`: `89225a3c8a2bc180e5b191fe714c8165fdfb1a3f5295e5acf057b2e5fc660705`

## Canonical evidence comparison upgrade, 2026-09-05

Owner: D0 evidence-shots agent. Branch: `agent/icra-d0-evidence-shots`, based on
main `61d8ba4a61b1cec2da0aa8f668339091cffb1c2f`. Implementation, pinned config and
reproducible commands: `EVIDENCE_SHOTS.md`. No model environment or old freeze
is modified. Focused comparison tests include actual MuJoCo CPU mesh rendering,
missing/tampered artifacts, forged probe verdicts, wrong retry parents and
re-encoded unchanged transforms. Rendering/QA results are appended after the
committed source executes. Full hero, teaser and loop remain NOT_RUN;
`FULL_DEMO_READY=false`; scientific claim gate NOT_RUN.


### Evidence clips complete; final D0 remains NOT_RUN

Owner: D0 evidence-shots agent. Branch: `agent/icra-d0-evidence-shots`.
Frozen rendering source commit: `5a8f293d85cf129be5db049b641aaf303fb13e8b`.
Implementation commits: `bcadcb5`, `db3fbce`, `5a8f293`; subsequent status-only
commit records this completed run. Hardware: CPU OSMesa, four software-rendering
threads; Slurm job IDs: none; checkpoints: none loaded by D0.

Smoke/schema command (from this worktree):

```bash
source /group/worldcept/code/SimAny/outputs/test-headless-setup/env.sh
env -u SIMANY_EVIDENCE_ROOT LP_NUM_THREADS=4 PYTHONPATH=. \
 /group/worldcept/code/SimAny/.venv/bin/python -m pytest -q \
 tests/test_demo_manifest.py tests/test_demo_timeline.py tests/test_demo_evidence.py \
 tests/test_freeze.py tests/test_contract_validation.py \
 --basetemp=outputs/d0-evidence-tests/pt6
```

Result: **33 passed in 8.80 s**. Includes actual CPU mesh rendering, complete
union-vertex camera bounds, identical camera distance across proposal panels,
and missing/tampered artifacts, forged probe evidence, wrong retry parent,
re-encoded unchanged transform, overwrite and fallback negative checks.

Real-data engineering pilot: preflight, render and assemble commands in
`EVIDENCE_SHOTS.md` all exited 0. Render wall time **11.56017805985175 s**.
Output: `/group/worldcept/code/SimAny/outputs/icra2027/20260905-61d8ba4-v3/demo`.
Package contains the 12.000 s / 960x540 / 15 fps draft, plus genuine selection
and retry 6.000 s / 1920x1080 / 30 fps clips and posters in `evidence_clips/`,
subtitles, source manifest and recursive artifact hashes. Config hash is pinned
by `demo/artifact_hashes.json` for `resolved_demo_config.yaml`; original selected
artifact/checkpoint provenance remains in `demo/source_manifest.json`. No model
checkpoint is introduced by these visualization changes.

Machine-readable QA and full logs: sibling `engineering_audit/review.json`.
All three videos pass profile/hash checks and full decode/black-frame detection.
Both full-resolution comparison posters were visually inspected: readable text,
seven-percent safe margins, all fragments visible, actual relative scale retained.
A shared camera fits every vertex from both proposals within the central 60%
frame, with near/far/frustum validation recorded per panel in the source manifest.
Both selection and retry remain explicitly **UNSUPPORTED**. Original construction
evidence is from `20260905-33bd974-v1`; this engineering draft does not display
current E9 aggregate numbers and is not a final common-freeze D0 deliverable.

The v1 and v2 drafts remain intact. v1 required a footer safe-margin correction;
v2 passed text QA but left proposal geometry too small. v3 improves only the
shared camera framing, without changing objects, geometry, registration or evidence.

Passed criteria: authenticated registered-mesh comparisons, genuinely changed
retry action and transform, complete-geometry shared camera, negative tests,
encoded draft and 1080p comparison clips, inspected posters and source closure.
Full command/result: **NOT_RUN**, prerequisites detailed in `EVIDENCE_SHOTS.md`.
Remaining: eligible hero room, one synchronized policy episode, certified Option C
frames if shown, final common scientific freeze and real-capture footage.
90 s hero / 30 s teaser / 8 s loop: **NOT_RUN**. Scientific claim gate: **NOT_RUN**;
`FULL_DEMO_READY=false`. Read-only E6 blockers: sibling audit `E6_READINESS.md`.

## Final-mode framework and full-E3 engineering source upgrade

Owner: `trellis2_environment`; branch `agent/icra-d0-final-mode`.
State: IMPLEMENTING (framework validation; actual final film NOT_RUN).
Source commit: exact task branch commit containing this entry; execution E0 and
render receipts will bind that SHA independently without editing the frozen source.
Smoke command: primary Python `pytest -q tests/test_demo_final.py
 tests/test_demo_manifest.py tests/test_demo_timeline.py tests/test_demo_evidence.py
 tests/test_freeze.py tests/test_contract_validation.py` from the task worktree,
with existing OSMesa environment; logs `outputs/d0-final-mode/tests-r*.log`.
Final expanded suite: 55 passed in 28.05 seconds. Exact-source E0 is required before any newly frozen evidence render.
Pilot/full commands: prerequisite-gated `interface.demo_agentic render` schema 2;
NOT_RUN. No Slurm job, GPU allocation or new model checkpoint.

Framework now declares exact 90/30/8-second 1080p products, authentic evidence
subtitles/posters, source manifests, E9 numeric/figure binding and live fallback.
Hero room ordering is lexicographic scene ID, independent of acceptance counts,
held-out quality and policy success. First genuine full-E3 selection/retry IDs
are selected lexicographically from the complete original ledger. A synchronized
shot must authenticate a single canonical real-policy episode and each original
tick's two render views; the existing policy-observation video alone cannot pass.
E5/Harmonizer footage must be an evaluated restored frame from canonical E5
visual metrics. Missing scenes, source bytes, reset/camera records, gates or
failed-source identities cannot be replaced by illustrative footage.

Final unmet gates: E4 synchronized frame provenance, E5 preservation footage,
eligible full-room hero, and the common scientific submission freeze. Current
paper publication `20260906-6dcb0e7-v2` enables narrow E3 construction claims but
has scientific submission FAIL. Existing draft v3 is preserved. New engineering
clips may display authenticated full-E3 evidence and exact current E9 PNG only;
they explicitly remain outside the final common-freeze deliverable.
Claim-gate conclusion: final D0 NOT_RUN; no manipulation/physics success implied.
