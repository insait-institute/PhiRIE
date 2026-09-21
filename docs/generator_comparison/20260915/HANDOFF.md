# Generator comparison: all available objects completed

Verified at 2026-09-15T06:09:15.387043+00:00. **116/116 runnable slots completed generation, common registration, export, collision import and stability evaluation. Zero task jobs remain pending or running.** The complete planned denominator is still **128 slots**: three original objects lack inputs, leaving 12 explicit `MISSING_INPUT` rows. `summary.json` remains `PARTIAL` for that full denominator; `completed_four_backend_comparison.json` records completion on available inputs.

The frozen cohort is **32 ScanNet++ objects × four backends × seed 0**. These are historically observed DEV scenes, not an untouched TEST set. The outside-roster engineering object is `fb5a96b1a2-a1037`. Selection, prior exposure and original input identities are in [cohort.json](cohort.json).

Geometry uses the **same 10 independently reference-matched objects** for every method. The other 19 generated objects per method have no independent geometry match. CD is in centimetres (lower is better); F1 is at 20 and 40 mm (higher is better). Stability uses all 29 generated objects per method. These are descriptive measurements with one seed, not evidence of a statistically significant ranking.

| Method | Generated / planned | Geometry N | CD cm ↓ | F1@20 mm ↑ | F1@40 mm ↑ | Stable / tested |
|---|---:|---:|---:|---:|---:|---:|
| phi-RIE + TRELLIS | 29/32 | 10 | 7.1939 | 0.3087 | 0.5311 | 15/29 |
| phi-RIE + TRELLIS.2 | 29/32 | 10 | 10.2237 | 0.2305 | 0.4482 | 10/29 |
| phi-RIE + SAM 3D Objects | 29/32 | 10 | 9.4079 | 0.1847 | 0.3724 | 16/29 |
| phi-RIE + ReconViaGen (multi-view) | 29/32 | 10 | 8.6694 | 0.2883 | 0.5064 | 14/29 |

TRELLIS has the highest observed geometry F1 on this shared support. All methods have unstable assets and poor or partial reconstructions. `EVALUATED` means the pipeline finished and recorded its outcome; it does not mean every asset reconstructed well or passed stability. No object was regenerated to improve a scientific score.

**True multi-view ReconViaGen:** all 29 available cohort objects use 5–12 distinct TRAIN frames, preserving the original anchor. Extra masks come from the existing occlusion-aware projection of the original TRAIN-derived construction mesh and automatic memberships. Evaluator geometry is not read during view preparation or registration. The construction file named `gt_points.ply` is authenticated as derived construction-surface samples. All 30 source-view sets, including the smoke, passed image hashes, unique-frame and alpha-mask checks. The historical 29 single-view ReconViaGen outputs remain separately listed in [recovery.json](recovery.json) and are not pooled.

**Reuse and recovery:** 58 compatible prior generations (29 TRELLIS and 29 TRELLIS.2) were reused, preserving model, seed, input and artifact receipts. All 58 received the common CPU evaluation. New multi-view ReconViaGen and SAM each contributed 29 generations. Twenty historical independent measurements remain recorded with their original provenance.

**Final Hala continuation:** SAM job **901688** was cancelled by Slurm UID 0 after 28 generations and 27 evaluations. The log records SIGTERM; scheduler metadata gives no further cause. CPU job **902359** evaluated the saved `fb5a96b1a2-a1020-sam3d` generation, and one-A6000 job **902360** generated and evaluated only the never-started `fb5a96b1a2-a1070-sam3d`. Both completed on `hala` with exit 0. The 28 completed generations and 27 completed evaluations were retained without regeneration. Tail resources were 8 CPUs, 64 GiB, 15 minutes each; only 902360 allocated a GPU.

Historical infrastructure failures are preserved in [results/failure_reconciliation.json](results/failure_reconciliation.json): 900911 (offline MoGe resolution), 900930 (wrapper keyword signature), 900913 (non-importable child CLI), and 900945 (Gaussian export keyword after inference). Corrected SAM smoke 901678 passed under the renewed user request; corrected CPU evaluation 900931 and ReconViaGen production 901002/901003 completed. The lost original ReconViaGen smoke Gaussian state remains unavailable; all 29 production Gaussian exports were verified.

| Job | Work | Final state | GPUs | Requested limit |
|---|---|---|---:|---|
| 900911 | sam3d-smoke | FAILED | 1 | 00:30:00 |
| 900912 | views | COMPLETED | 0 | 02:00:00 |
| 900913 | reference-evaluation | FAILED | 0 | 04:00:00 |
| 900930 | sam3d-smoke-retry1 | FAILED | 1 | 00:30:00 |
| 900931 | reference-evaluation-retry1 | COMPLETED | 0 | 02:00:00 |
| 900945 | reconviagen-smoke | FAILED | 1 | 00:30:00 |
| 900980 | reconviagen-smoke-recovery1 | COMPLETED | 0 | 00:30:00 |
| 901002 | reconviagen-production-00 | COMPLETED | 1 | 01:58:00 |
| 901003 | reconviagen-production-01 | COMPLETED | 1 | 00:21:00 |
| 901678 | sam3d-smoke-retry2 | COMPLETED | 1 | 00:30:00 |
| 901688 | sam3d-production-00 | CANCELLED | 1 | 01:58:00 |
| 902359 | sam3d-tail-evaluation | COMPLETED | 0 | 00:15:00 |
| 902360 | sam3d-tail-generation | COMPLETED | 1 | 00:15:00 |

All submissions are ordinary `sbatch`, without arrays or nested GPU allocations. GPU jobs used one H200 each except the final Hala A6000 continuation. Exact commands, sbatch arguments, source commits, result paths and logs are in [jobs.json](jobs.json); terminal scheduler receipts are in [audits/sacct.txt](audits/sacct.txt). Original CPU job 900913 was initially submitted with four hours and corrected to two hours before failure; its correction receipt is retained.

Cumulative additional allocation cost: **2.412500 / 8 GPU-hours spent, 0 reserved**, including failed and cancelled jobs. CPU-only jobs consume no GPU-hours. Submission enforced at most two active GPU jobs, one GPU per job, and the existing project cap including known unrelated work. No unrelated job was cancelled.

Final artifact audit: **116 slots, 1613 checked receipts across 1380 unique files**. Checks include original generation/evaluation/input hashes, native mesh and world-surface finiteness, face indices, render/URDF receipts, collision import, seed 0, evaluator-isolated registration and distinct ReconViaGen inputs. All 29 SAM and 29 ReconViaGen production Gaussian files exist with matching receipts. See [audits/completed-four-backend-artifacts.json](audits/completed-four-backend-artifacts.json). Actual generated-mesh diagnostic views were inspected; this does not measure Gaussian or texture appearance.

**15 focused CPU tests passed** after the Hala routing change. Earlier 13-test and 12-test receipts are retained. Actual GPU smoke, full production files and rendered views provide separate execution evidence; CPU tests alone do not validate model output quality.

The shared protocol uses observation-only signed-source-up registration, scale baked once, a 40,000-triangle budget, common CoACD limits, URDF import and the existing 240 Hz factory plane-drop check. Imported mass is 0.3 kg, friction 0.5 and restitution 0.0. The stored restitution prior 0.1 is not applied by the existing writer/drop function; [audits/actual-physics.json](audits/actual-physics.json) records the actual common defaults.

Per-object inference, generation/export, registration, collision export and evaluation times, plus recorded peak GPU memory, are kept distinct in [results/per_object.csv](results/per_object.csv). Batch timing includes initialization. SAM has one A6000 object and 28 H200 objects, so its timing is not a homogeneous GPU benchmark. Historical generation wall time is not total conversion latency. Appearance is unmeasured for all methods; Gaussian-specific appearance is not applicable to TRELLIS.2.

Frozen latest worker checkout: `/group/worldcept/code/SimAny/worktrees/generator-comparison-final-20260915` at `d061e6636f77b36a7cb3a82b58c81c2badd4eada`. Run: `/group/worldcept/code/SimAny/outputs/icra2027/generator-comparison-bounded-20260914`. Per-job earlier worker commits remain frozen. Publication branch: `agent/generator-comparison-publication-20260915`. The original dirty implementation checkout was preserved; `git pull --ff-only` was already up to date on its branch, and origin/main was fetched to `7b0273bb` for this isolated work. The paper repository was not modified.

Final tables: [JSON](results/completed_four_backend_comparison.json), [CSV](results/completed_four_backend_comparison.csv), [all 128 rows](results/per_object.csv), [LaTeX fragment](results/generator_comparison.tex). The earlier three-backend files are timestamped historical snapshots; the four-backend files supersede their completion status.

Remaining limitations: the three missing-input objects are `5eb31827b7-a1006`, `5ee7c22ba0-a1027`, `5ee7c22ba0-a1041`. They remain in the full denominator and were not substituted. Independent geometry is unavailable for 19 generated objects per method; appearance remains unmeasured. There is no unresolved runnable work in this bounded invocation.

Inspection and idempotent continuation commands:

```bash
run_dir=/group/worldcept/code/SimAny/outputs/icra2027/generator-comparison-bounded-20260914
sacct -X -j 900911,900912,900913,900930,900931,900945,900980,901002,901003,901678,901688,902359,902360 --format=JobID,State,ExitCode,Elapsed,AllocTRES,NodeList
bash "$run_dir/continue.sh" status
bash "$run_dir/continue.sh" collect
# No remaining eligible work; dispatch reconciles receipts and will not repeat completed work.
bash "$run_dir/continue.sh" dispatch
```

Only compact measurements, manifests, code and bounded log excerpts are published. Excerpt trailing whitespace is normalized; original logs remain at the paths in jobs.json. Dataset images, meshes, weights and environments remain on the cluster. No background monitoring service is installed.
