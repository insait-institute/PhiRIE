# SR8 — Generated tables, synchronized demo and paper integration

**Owner:** result/demo agent, with one designated paper integrator. **Priority:** framework in parallel, final after measurements. Read ../TABLES_AND_METRICS.md and ../PROTOCOL.md. Do not block native experiments waiting for polished graphics.

## Table producers

Extend `robo.eval.paper_pipeline` with a separate `sim_recon_sim` evidence section. Reuse `fidelity_metrics`, canonical harness ledger/counting, standard serialization and bootstrap helpers. Add adapters/formatters, not competing metric formulas. Native tasks need their native success values rather than old DROID stage inference.

First generate the complete empty table skeletons T1-T4 from a manifest listing all planned methods. Each cell has `measured`, `not_run`, `blocked`, or `not_applicable` metadata. Do not silently remove a method when a file is absent. During submission integration, retain scientifically relevant coverage/failures and remove unsupported prose, not factual unfavorable outcomes.

T4 verification-controller native-completion cells are **fixed-action replay**, not additional learned-policy episodes. Schedule REF/B0/B3/V1/B4/BM replay arms on the same action/reset bank: 6 x 320 = 1,920 traces for the full RoboCasa core. The SR5 initial four-arm pilot grows to this declared six-arm analysis once SR3 supplies V1/BM. This costs no policy inference for the extra arms. T3 remains the 1,600-episode five-arm learned-policy matrix. Label T4 completion explicitly as replay in generated captions and headers. If extra replay arms are not scheduled, their cells remain NOT_RUN and the claim is narrowed.

Outputs: full and common-evaluable counts, raw metric CSVs, comparison intervals, per-scene/task results, runtime ledger, failure taxonomy, generated LaTeX and a claim-to-source map. Test tables on hand-calculable synthetic data with missing/failed units before consuming real results.

## Paper placement and claims

This is a NEW controlled evaluation track. Keep prior ScanNet++ evidence and its negative outcomes; do not pretend the new experiment repairs previous frozen rows. Preserve real capture/real robot as separate sections. Native simulator reference is not 'real-world GT'.

Recommended main evidence:

1. Native-source reconstruction fidelity (T1).
2. Native-task closed-loop success and observation intervention (T3).
3. Room-context verification/repair ablation with replay evidence (T4).

T2 full trajectory errors and supplementary controls can move to supplementary material when page space is limited. Exact arrangement should follow measured evidence, not a requirement to print many empty tables.

Allowed claim: 'On a declared same-engine reconstruction benchmark, method X preserves/improves native task performance relative to construction baseline Y at stated coverage and cost.'

Not allowed from this track: 'predicts real-world robot success', 'beats BEHAVIOR's physics engine', 'solves all BEHAVIOR tasks', or 'full-room reconstruction' from target-only/oracle-context rows. Agentic mechanism can remain in method with its actual action space. Task-Conditioned System Verification is construction-time checking/repair, not a formal correctness certificate.

## Demo: reference -> capture -> reconstruction -> interaction

Produce a 60-90 second comparison segment plus a 20-30 second presentation cut. Reuse `interface/demo_agentic.py` and existing demo launchers, not another rendering framework. A successful episode is useful, but a faithful failure/repair comparison is also a strong scientific demo.

Storyboard:

- 0-7s: native room and a native robot task. Caption `Reference simulation` and engine name. No claim of real footage.
- 7-16s: native camera path, observed RGB/RGB-D and source GS at a matched viewpoint. Caption the exact sensor access.
- 16-28s: one automatically discovered object, competing real proposals, evidence selection and genuine registration/room repair action. Values come from its ledger, never a mock LLM chat.
- 28-40s: synchronized collision/GS replay. Move the object and reveal the clean background. Preserve actual support geometry and visible failures.
- 40-65s: reference versus reconstruction paired-policy videos. Common initial reset intervention, same task/checkpoint. Both display their OWN native elapsed simulation time; after divergence do not falsely label them the same state. Do not retime one grasp to match the other. Keep each episode continuous.
- 65-80s: before/after room-context repair and an automatically selected informative failure; show a plain failure reason, not an unexplained red score.
- 80-90s: three measured numbers only: evaluated/planned tasks, native success gap, and an actual fidelity/repair result. Read from the final result manifest.

For a PURE appearance wipe, use renders at identical saved state/time/camera and label `fixed-state rendering comparison`. It is not a policy comparison. For fixed-action replay label `same robot actions`; for closed loop label `same policy, paired reset`. These three types must never be visually conflated.

## Visual specifications

1920x1080/30fps presentation master, native policy images archived separately. Calm camera motion, one readable sans-serif family, labels >=28px at 1080p, no more than three simultaneous labels, consistent colors, 7% safe area. No generative video interpolation, hidden failure removal or fabricated contact. Scaling/cropping for layout must be disclosed where it changes the shown field of view. Do not distribute font files.

Every shot links capture/build/episode/state IDs, renderer, frame range, display speed, manual crop and source hashes. Curated success examples are explicitly qualitative; quantitative tables always use the full fixed roster. Also include a median case and one failure selected by recorded rules. No splice of different trials into one apparent successful attempt.

Live demo uses cached native/reference and reconstructed assets. Do not generate assets or launch gated models on stage. Keep an offline video fallback and visibly identify prerecorded replay. Harmonizer footage is optional; lack of Cosmos access does not prevent a complete raw-GS comparison demo.

## Commands and validation

Implement a workload-specific config for existing `paper_pipeline` and demo CLI; document the exact tested command in STATUS.md. Do not present unimplemented module names as runnable commands.

Tests: every table cell has source/count/unit; missing rows are preserved; native success labels are not inferred from video captions; replay and policy cannot be mixed; paired differences use matching IDs; scene/episode count reconciles; source hash changes invalidate stale figures; all videos decode and have readable labels; GT vault is absent from public package.

## Definition of done

- [ ] Native identity/import controls documented.
- [ ] Frozen captures, sensor assistance, task scope and native policy explicitly recorded.
- [ ] T1-T4 generated, with missingness and coverage honest.
- [ ] At least one continuous native/reference versus reconstructed robot experiment exists, or exact scientific failure reported.
- [ ] All planned core units have terminal records; optional blocked arms do not become zeros.
- [ ] Figures/video distinguish fixed-state, fixed-action and closed-loop evidence.
- [ ] Paper source updates are generated/reviewed, PDF compiled and visually checked after any actual paper edit.
- [ ] A portable nonrestricted result bundle and commit/run manifest are available.

A publication gate failing is not permission to end with bookkeeping alone. First diagnose and repair independently identified development bugs, execute the bounded pilot and report the measured result. No guarantees about paper awards or acceptance.
