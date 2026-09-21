# Fast Validation Matrix

This file is the reviewer-facing and agent-handoff companion to the task files in `plan/`. Each task still owns its full acceptance criteria. The checks below are the **fastest admissible validation**: they should finish in minutes, emit one machine-readable artifact, and catch the most likely integration error before a reviewer or downstream agent spends time on the branch.

## Universal handoff rule

An agent handoff is accepted only when it provides:

1. branch and exact commit SHA;
2. the exact command below, or a task-file-approved replacement;
3. wall time, hostname/GPU, exit code, and artifact path;
4. a manifest or report containing the implementation commit;
5. one intentionally failing negative test;
6. unresolved limitations and the next blocked task.

Use `validation/<task_id>/<commit_sha>/` for smoke outputs. Never overwrite an earlier validation directory.

## Wave 0 — research contract and provenance

### Task 00 — Research contract
- **Command:** `python -m agents.eval.validate_contract configs/experiments/icra_contract_v1.yaml --strict`
- **Target time / hardware:** <30 s, CPU.
- **Pass artifact:** `validation/00/<sha>/contract_report.json` with zero errors and every planned table column mapped to a metric.
- **Negative check:** mutate one frozen camera field; validator must fail.
- **Escalate when:** a primary policy checkpoint cannot be matched to the real-result source.

### Task 01 — Experiment manifest
- **Command:** `pytest -q tests/test_manifest_roundtrip.py && python -m robo.manifest.io validate configs/experiments/example_manifest.yaml`
- **Target:** <60 s, CPU.
- **Pass artifact:** canonical manifest plus stable hash in `validation/01/<sha>/`.
- **Negative check:** change checkpoint bytes or camera extrinsics; the hash must change and output reuse must be refused.

### Task 10 — Metrics and power
- **Command:** `pytest -q tests/test_predictive_metrics.py && python -m agents.eval.power --config configs/experiments/icra_contract_v1.yaml --dry-run`
- **Target:** <2 min, CPU.
- **Pass artifact:** toy-metric report, episode-budget JSON, and bootstrap seed list.
- **Negative check:** a synthetic dataset with high pooled Pearson but reversed within-task ranking must be detected.

## Wave 1 — phone capture to metric reconstruction

### Task 02 — Phone capture protocol
- **Command:** `python -m capture.validate_video tests/data/phone/clean.mp4 --config configs/capture/phone_default.yaml --out validation/02/<sha>`
- **Target:** <2 min, CPU.
- **Pass artifact:** capture report with blur, parallax, exposure, marker visibility, and actionable warnings.
- **Negative check:** static-camera or blurred fixture must fail the corresponding threshold.

### Task 03 — Phone reconstruction front end
- **Command:** `bash run/run_phone2sim.sh tests/data/phone/miniscene.mp4 --config configs/recon/phone_smoke.yaml --stop-after discover`
- **Target:** <15 min on one 24 GB GPU; CPU fallback may stop after poses/mesh.
- **Pass artifact:** build manifest, posed frames, splat/mesh paths, held-out render metrics, and discovered-instance inventory.
- **Negative check:** alter one input frame or config; stale dependent stages must be invalidated.

### Task 04 — Metric scale and robot alignment
- **Command:** `pytest -q tests/test_robot_alignment_synthetic.py && python -m agents.recon.alignment_report tests/data/alignment/synthetic.yaml`
- **Target:** <60 s, CPU.
- **Pass artifact:** held-out translation/rotation/scale residual JSON and overlay image.
- **Negative check:** collinear markers or insufficient trajectory baseline must return a degeneracy error, not a transform.

### Task 05 — Scene quality audit
- **Command:** `pytest -q tests/test_build_audit.py && python -m agents.eval.build_audit tests/data/builds/audit_fixture --out validation/05/<sha>`
- **Target:** <3 min, one GPU optional for renders.
- **Pass artifact:** versioned scene/object audit JSON and diagnostic contact/render sheet.
- **Negative check:** ghost object, penetration, missing support, and bad scale fixtures must trigger distinct failures.

## Wave 2 — fair robot simulation

### Task 06 — Full-room collision export
- **Command:** `pytest -q tests/test_room_collision.py && python -m robo.sim.room_collision --scene tests/data/scenes/collision_fixture --smoke-steps 1000`
- **Target:** <5 min, CPU.
- **Pass artifact:** collision-coverage/penetration report and deterministic state hash after 1000 steps.
- **Negative check:** an object pushed outside its original support footprint must still collide with the room; support-shim mode must fail this test.

### Task 07 — PolaRiS adapter
- **Command:** `python -m robo.polaris.validate_pair --task DROID-FoodBussing --official <path> --simany <path> --strict`
- **Target:** <5 min after assets, one Isaac-capable GPU.
- **Pass artifact:** frozen-field diff showing only scene-construction fields differ, plus one reset image from each environment.
- **Negative check:** camera, gripper convention, control rate, or rubric change must fail validation.

### Task 08 — Policy/control/checkpoint matrix
- **Command:** `pytest -q tests/test_control_contract.py && python -m robo.policy.registry verify --config configs/policies/smoke.yaml`
- **Target:** <5 min excluding first checkpoint download, one policy-capable GPU.
- **Pass artifact:** checkpoint/preprocessing hashes and golden observation-action trace.
- **Negative check:** absolute/delta joint or reversed gripper convention must be detected.

### Task 11 — Task interaction graph
- **Command:** `pytest -q tests/test_task_graph.py && python -m robo.certification.task_graph --task tests/data/tasks/food_bussing.yaml --scene tests/data/scenes/task_graph_fixture`
- **Target:** <60 s, CPU.
- **Pass artifact:** graph JSON and policy-view overlay with robot, cameras, manipulated object, target, support, and obstacles.
- **Negative check:** same-category distractor and distant irrelevant furniture must not enter the graph without geometric/rubric evidence.

## Wave 3 — paired evaluation and self-awareness

### Task 09 — Paired rollout runner
- **Command:** `python -m robo.eval.paired_runner --config configs/experiments/paired_smoke.yaml --episodes 2`
- **Target:** <10 min plus model startup, one simulation/policy GPU.
- **Pass artifact:** paired episode manifests, videos, state/contact/rubric logs, and reset-state equality report.
- **Negative check:** crash/restart must neither duplicate episode IDs nor change seeds.

### Task 12 — Task-conditioned certificate
- **Command:** `pytest -q tests/test_certificate.py && python -m robo.certification.calibrate --fixture tests/data/certificate/toy.json --leave-one-scene-out`
- **Target:** <2 min, CPU.
- **Pass artifact:** risk-coverage JSON/plot, calibration report, bad-pair AUROC, and per-feature explanations.
- **Negative check:** scene ID or real success included as a feature must be rejected as leakage.

### Task 13 — Targeted repair loop
- **Command:** `pytest -q tests/test_repair_routing.py && bash run/run_repair.sh tests/data/builds/repair_fixture --max-attempts 1`
- **Target:** <10 min for fixture, one GPU when appearance repair is invoked.
- **Pass artifact:** parent/child manifests, selected repair reason, before/after audit, cost, and abstention status.
- **Negative check:** manual pose/asset placement or an undeclared repair action must be refused.

## Wave 4 — evidence tracks

### Task 14 — Oracle benchmark
- **Command:** `python -m oracle.capture_generator --config configs/oracle/smoke.yaml && python -m oracle.evaluate_reconstruction --identity-export`
- **Target:** <10 min, one simulation GPU.
- **Pass artifact:** identity row with exact unit scores, hidden-GT access audit, and paired rollout fixture.
- **Negative check:** reconstruction process attempting to read oracle state must fail sandbox validation.

### Task 15 — Construction baselines
- **Command:** `python -m baselines.release_status --strict && python -m baselines.raw_reconstruction --fixture tests/data/oracle/miniscene`
- **Target:** <3 min for status/raw baseline; full baseline runtime logged separately.
- **Pass artifact:** release/commit/license table, adapter manifest, human-intervention count, and explicit `N/A` cells.
- **Negative check:** unavailable or partial methods must not be converted to zero-valued results.

### Task 16 — DROID track
- **Command:** `python -m agents.recon.droid_static_select --fixture tests/data/droid/episode_fixture && python -m agents.eval.droid_alignment_eval --fixture tests/data/droid/episode_fixture`
- **Target:** <5 min, CPU; reconstruction smoke may require one GPU.
- **Pass artifact:** selected static-frame list, contamination metrics, held-out FK alignment residual, and episode/checkpoint provenance.
- **Negative check:** dynamic robot/object frames must not silently train the static background.

### Task 17 — Phone-room collection
- **Command:** `python -m capture.validate_bundle <room_bundle> --contract configs/experiments/icra_contract_v1.yaml --strict`
- **Target:** <3 min per room, CPU before reconstruction.
- **Pass artifact:** consent/redaction status, raw-file hashes, capture report, surveyed evaluation-only checks, tasks, and reset distribution.
- **Negative check:** room captured after task-specific simulator tuning or with missing release status must be marked ineligible.

### Task 18 — Real-robot paired evaluation
- **Command:** `python -m agents.eval.paired_real_sim --config configs/real_robot/smoke.yaml --validate-only`
- **Target:** <2 min, CPU; physical trials are session-budgeted separately.
- **Pass artifact:** session calibration, randomized trial order, reset IDs, rubric reliability, safety/exclusion rules, and matching sim manifests.
- **Negative check:** unlogged trial exclusion or policy/camera/controller mismatch must invalidate the cell.

## Wave 5 — ablations and paper artifacts

### Task 19 — Ablations and failure taxonomy
- **Command:** `python -m agents.eval.ablation_matrix --config configs/ablations/smoke.yaml --validate-only && pytest -q tests/test_failure_taxonomy.py`
- **Target:** <2 min, CPU.
- **Pass artifact:** one-variable diff report, paired seed/reset map, label schema, and annotator-agreement fixture.
- **Negative check:** a row changing two independent variables must be rejected.

### Task 20 — Tables, figures, and video
- **Command:** `python -m agents.eval.make_icra_tables --freeze <freeze_id> --check-only && python -m agents.eval.make_icra_figures --freeze <freeze_id> --check-only`
- **Target:** <5 min, CPU.
- **Pass artifact:** regenerated LaTeX/plots, provenance JSON per artifact, no manual numeric cells, and 200 dpi figure previews.
- **Negative check:** edited table cell or missing CI/sample count must fail provenance validation.

### Task 21 — Release and submission audit
- **Command:** `python release/check_release.py --strict && latexmk -pdf -halt-on-error conference.tex`
- **Target:** <10 min excluding external data download, CPU.
- **Pass artifact:** submission audit with page count, font embedding, anonymity, citation/release verification, clean-checkout reproduction, and unresolved finding count.
- **Negative check:** inject an author path/token, overfull box, unsupported “first/any/general” claim, or provenance mismatch; audit must fail.

## Reviewer spot-check order

A reviewer or lead should validate in this order:

1. Task 00 contract and Task 01 hashes;
2. Task 06 full-room collision;
3. Tasks 07–09 fair paired runner;
4. Task 10 ranking/calibration metrics;
5. Task 12 false accepts and risk-coverage;
6. Tasks 17–18 actual phone/real evidence;
7. Task 20 provenance and Task 21 clean build.

This order prevents impressive rendering or correlation plots from masking a broken control contract, invalid collision model, or selective coverage.