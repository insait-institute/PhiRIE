# Final presentation framework (sources still incomplete)

Owner: D0 final-mode agent. Branch `agent/icra-d0-final-mode`.
Existing draft assets/source freezes remain unchanged. No new policy is run by
this renderer. CPU encoding and read-only source authentication only.

Schema 2 (`mode: final`) adds exactly 1920×1080, 30 fps timelines of 90, 30 and
8 seconds. The prospective durations and construction-only hero ordering live
in `configs/demo/icra2027_final_storyboard.yaml`. This is a protocol template,
not an executed/frozen config. A final materialized config supplies `e3`, `e9`,
`sources`, `timelines`, `execution_contract` and `harmonizer_gate`.

- `e3`: exact full completion-audit path/SHA and the fixed `HERO_RULE`. Complete
  50-scene / 1871-object / 9355-row ledger coverage is checked. Selection is the
  first lexicographic genuine cross-tool A2 choice, retry the first genuine A3
  changed registration. Their raw meshes, transforms, evidence, parent IDs and
  probe verdicts use the existing authenticated comparison renderer. Held-out
  quality and policy success do not select shots. Hero candidates follow lexicographic scene ID; A4 acceptance count never ranks scenes. The first candidate with an actually recorded
  canonical main-system episode is used, and that episode is lexicographically
  first regardless of success; users cannot supply only a winning episode.
- `e9`: one publication freeze, its `paper_table_provenance.json` identity and
  exact `submission_audit.json` identity. Formatter source/config, every table
  and figure, and every numeric source JSON are rehashed. The E9 agentic source
  must bind the same full E3 completion audit. Numeric overlays specify a source
  name, exact JSON key/index pointer, label and constrained rounding format.
  No manually supplied experimental number is accepted.
- Sources: checked image/video references, genuine E3 `selection` records,
  exact E9 `figure` PNGs or `result_card`, and `synchronized_episode`. Missing
  source paths are reported by plan mode and block encoding before output.
- Synchronized episodes consume the existing canonical harness ledger, config,
  reset bank, manifest and compressed time series. The canonical record and
  paired-manifest validators run first. Real policy, agentic scene, full-room
  collision and a real clean background are required. D0's renderer sidecar
  describes existing per-tick raster/photoreal images, both tied to the same
  original tick/body-state digest. This is rendering provenance, not another
  rollout ledger. The movie is composed directly from these checked frame
  pairs; it does not trust a pre-spliced encoded video. No simulation is rerun.
  Complete recorded tick coverage is required, and excerpts may not skip,
  overlap, reverse, or mix episodes. Failed episodes are allowed with faithful
  captions; a successful-placement caption requires the actual success flag.
- Harmonizer admission requires the exact E9-bound canonical five-condition
  visual table, its original manifest, non-null measured preservation/temporal
  fields, byte-exact robot-core telemetry and exact evaluated restored frames.
  An arbitrary `status: PASS` receipt or raw-RGB fallback is insufficient.

The current harness writes policy-observation video plus ticks but does not
produce those synchronized frame pairs. E5 is also blocked on genuine model
access. The current E9 publication is preliminary. Therefore final D0 remains
**NOT_RUN**, and the framework cannot mark the movie or paper ready today.
The final source/config must be committed, reviewed and E0-bound before any
actual final render; do not reserve or reuse an old output ID.

Commands after supplying the exact frozen config:

```bash
SIMANY_PY=/group/worldcept/code/SimAny/.venv/bin/python \
  bash run/demo/plan_icra2027.sh --config /absolute/frozen_demo.yaml
SIMANY_PY=/group/worldcept/code/SimAny/.venv/bin/python \
  bash run/demo/preflight_icra2027.sh --config /absolute/frozen_demo.yaml
SIMANY_PY=/group/worldcept/code/SimAny/.venv/bin/python \
  bash run/demo/render_icra2027.sh --config /absolute/frozen_demo.yaml \
  --out /group/worldcept/code/SimAny/outputs/icra2027/NEW_FREEZE/demo
bash run/demo/assemble_icra2027.sh --demo-dir /absolute/NEW_FREEZE/demo
```

Products: `master/` 90-second MP4/MOV and SRT, `teaser/` 30-second MP4/SRT,
`loop/` 8-second MP4/WebP/SRT, poster, source manifest, resolved config, profile
QA, recursive hashes and the existing three-second-watchdog presentation
manifest. The seamless loop explicitly holds authentic evidence; it never
loops or reverses a policy trajectory. It is not a claim of periodic motion.
Partial encoding is preserved on failure; an existing output is never replaced.
All original sources are revalidated before atomic publication.

Validation: `tests/test_demo_final.py` covers profile/denominator, source hash,
missing footage, exact E9 fields, unsupported success captions, mixed/reversed
or skipped episode excerpts, source gates and no-output-on-blocking behavior.
Existing draft/comparison and watchdog tests continue to apply. Actual full
scientific encoding and synchronized recorder closure remain prerequisite-gated.

Engineering evidence package (separate from final mode):

```bash
python -m interface.demo_agentic evidence-clips --config /absolute/engineering.yaml \
  --out /group/worldcept/code/SimAny/outputs/icra2027/NEW_FREEZE/demo
```

This command requires `mode: engineering_evidence_clips`, schema 2, a newly
allocated freeze, exact `execution_contract`, full `e3` and current `e9`
references, an authenticated `publication_receipt`, and `result_figure` naming
an exact E9 PNG. It emits three six-second 1080p evidence holds: the first
lexicographic genuine selection, genuine retry, and the exact results figure.
Source identity, E0, changed parent/registration, original negative probe labels,
full clip decode, dimensions, duration, subtitles and no-overwrite are checked.
`full_demo_ready` and `paper_ready` are always false. This package deliberately
makes no synchronized-rollout, Harmonizer or completed final-film claim.
