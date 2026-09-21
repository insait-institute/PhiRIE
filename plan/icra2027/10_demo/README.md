# D0 — SimAnyRoom Hero Demo and Presentation Package

**Priority:** P0  
**Goal:** a visually polished, technically honest demo that explains the paper in under 90 seconds and survives a live conference presentation  
**Depends on:** frozen hero scenes from E3–E5; final numbers from E9

## Deliverables

Produce all four:

1. **Paper video:** `simanyroom_hero_90s_1080p.mp4`, 80–95 s, 1920×1080, 30 fps.
2. **Teaser:** `simanyroom_teaser_30s_1080p.mp4`, self-contained and understandable without narration.
3. **Loop:** `simanyroom_loop_8s.mp4` plus GIF/WebP for a project page.
4. **Presentation package:** one-click live/pre-rendered launcher with a guaranteed fallback video.

Also export a lossless or visually lossless master, the first-frame poster, subtitles, shot manifest, and source hashes.

## The one-sentence story

> A short room capture becomes an editable simulator because an agentic constructor proposes, tests, selects, retries, and assembles object, physics, and appearance artifacts before a frozen robot policy interacts with the result.

The demo should emphasize **agentic construction and the finished simulator**. Task-local support and Harmonizer are supporting evidence, not the emotional center of the video.

## Read first

- `interface/demo_movie.py`
- `interface/demo_session.py`
- `interface/viewer.py`
- `interface/mujoco_live_viewer.py`
- `docs/DEMO_STORYBOARD.md` as legacy footage guidance
- `robo/eval/harness_runner.py`
- `robo/eval/episode_log.py`
- `robo/rendering/harness_composite_obs.py`
- `robo/rendering/robot_restore.py`
- `agents/eval/build_audit.py`
- E3, E4, E5, E7, and E9 `STATUS.md` files

Reuse rendering, camera, label, and assembly utilities from `interface/demo_movie.py`. Do not break the legacy command. Implement the new film in a separate entry point.

## Required implementation

Add:

```text
interface/demo_agentic.py
configs/demo/icra2027.yaml
run/demo/render_icra2027.sh
run/demo/assemble_icra2027.sh
run/demo/present_icra2027.sh
run/demo/preflight_icra2027.sh
tests/test_demo_manifest.py
tests/test_demo_timeline.py
```

Output:

```text
outputs/icra2027/<freeze_id>/demo/
  source_manifest.json
  resolved_demo_config.yaml
  shots/
  overlays/
  audio/
  master/
    simanyroom_hero_90s_master.mov
    simanyroom_hero_90s_1080p.mp4
    simanyroom_hero_90s.srt
  teaser/
    simanyroom_teaser_30s_1080p.mp4
  loop/
    simanyroom_loop_8s.mp4
    simanyroom_loop_8s.webp
  poster/
    simanyroom_poster.png
  qa/
    frame_contact_sheet.jpg
    black_frame_report.json
    freeze_frame_report.json
    text_safe_area_report.json
    source_hashes.csv
```

## Visual language

Use the same conceptual palette as the paper teaser:

- construction and capture: blue;
- appearance/Harmonizer: violet;
- simulation and successful decisions: green;
- agent decisions/retries: orange;
- failures/rejections: muted red;
- background/UI: warm white or near-black, never a saturated gradient behind scientific content.

Typography:

- one clean sans-serif family available on the rendering machine;
- title 64–84 px at 1080p;
- section cards 40–52 px;
- labels at least 28 px;
- no paragraph-sized text in the video;
- no overlay longer than 7 words, except the paper title;
- all captions remain inside a 7% edge-safe region.

Motion:

- slow camera movement, no fast pan or artificial shake;
- use ease-in/ease-out and hold important states for at least 0.8 s;
- hard cuts between conceptual acts, gentle match cuts within an act;
- avoid constant zooming, glowing particles, lens flares, or generic AI visual effects;
- never place more than three simultaneous labels on a scene.

## Hero-scene selection

Freeze three visually distinct rooms before final rendering:

1. **Hero room:** strongest complete story: clean background, at least four movable objects, valid full-room collision, and one usable manipulation episode.
2. **Scale room:** visually different layout with many discovered objects.
3. **Real-capture room/workspace:** DROID or phone capture that completed construction and alignment.

Select using a declared config over E2/E3/E4 outputs, not manual aesthetic preference. The selection function should score only predeclared evidence such as complete-build status, accepted-object count, median object F1, clean-background availability, and existence of a valid rollout. Save all candidates and scores.

The final video may manually choose camera paths inside a frozen hero room, but may not replace the room after viewing final manipulation success without updating the selection manifest.

## 90-second storyboard

### Act 0 — Promise, 0:00–0:04

Visual:

- black or warm-white background;
- title: `SimAnyRoom`;
- subtitle: `Making Rooms Simulatable through Agentic Real-to-Sim`;
- one clean hero frame fades in behind the title.

Narration/overlay:

> Capture a room. Build a simulator.

Do not show a dense pipeline diagram here.

### Act 1 — Capture becomes a room model, 0:04–0:15

Visual sequence:

1. 1.5 s phone/video capture strip with subtle camera-path line;
2. point cloud/mesh emerges;
3. smooth match cut to the Gaussian reconstruction;
4. camera performs one slow, physically plausible fly-through.

Overlay sequence:

- `Casual multi-view capture`
- `Metric room reconstruction`

Use real captured frames and a real reconstructed scene. The scan-wave effect in `demo_movie.py` can be reused, but reduce the cyan glow and keep it physically grounded.

### Act 2 — The agent constructs one difficult object, 0:15–0:31

This is the signature shot that justifies the title.

Use one object for which the two generators differ meaningfully, ideally a mug or bottle. Show a clean horizontal sequence:

```text
object observation
      ↓
TRELLIS proposal     ReconViaGen proposal
      ↓                     ↓
registered residual / support evidence
      ↓
SELECT one proposal
```

Use real meshes/renders and real evidence values from E3. The selected proposal receives a restrained green outline. The rejected proposal fades to 40% opacity with one reason, for example `multi-view collapse` or `poor registration`.

Then show a second difficult object in 4–5 s:

```text
proposal fails → RETRY → alternative accepted
```

A retry must correspond to an actual E3 job-ledger action. Do not animate a fake agent chat or show an LLM text box. The agent is represented through artifacts, evidence, and actions.

Overlays:

- `Propose`
- `Register`
- `Compare evidence`
- `Select or retry`

### Act 3 — One identity connects physics and appearance, 0:31–0:45

Use the selected object and split the screen into two synchronized views:

Left:

- canonical mesh;
- convex collision parts;
- mass/friction values shown briefly.

Right:

- object Gaussians;
- original room location;
- clean background after removal.

Animate the object moving away. The old location must reveal a clean support surface with no duplicated object. Then recombine to one photoreal room view.

Overlay:

> One object identity, two branches

This shot must clearly show why the factorized representation is necessary. It is more important than another generic room fly-through.

### Act 4 — Full-room simulation, 0:45–0:58

Show the same action in two synchronized panels:

- left: MuJoCo physics/collision view;
- right: SimAnyRoom photoreal rendering.

Use the same body poses and frame timestamps. Draw a small shared-state link between them, not a large explanatory paragraph. Show one contact-rich action: grasp and lift, slide across support, or placement in a tray.

Overlay sequence:

- `Full-room collision`
- `One live state`
- `Physics ↔ appearance`

Use actual full-room collision. Do not use private support shims in the hero shot.

### Act 5 — Frozen-policy evaluation and Option C, 0:58–1:12

Show one matched reset from E4. Use a three-panel strip for a short interval:

1. MuJoCo raster;
2. raw photoreal composite;
3. Harmonizer + robot restoration.

Then enlarge the main photoreal policy view while a minimal stage bar updates:

```text
reach · grasp · lift · place
```

The displayed episode must be selected by the frozen demo-candidate rule from E4. If no successful placement exists, show genuine grasp/lift progress and label it accurately. Never splice stages from different episodes into one apparent success.

Briefly flash the Option C construction:

```text
full-frame enhance + exact robot restore
```

Do not spend more than 3 s on the mask. The visual story is that policy observations become coherent while the robot remains unchanged.

### Act 6 — Scale and real inputs, 1:12–1:23

Use a four-tile montage:

- two ScanNet++ rooms;
- one DROID workspace;
- one phone room.

Each tile follows the same 1.5–2.0 s micro-sequence: capture → constructed twin → simulation view. Keep all tiles synchronized and avoid tiny object labels.

Overlay:

> One system, diverse rooms

If a real-input build failed, it may appear only in a clearly labeled failure montage, not as a successful twin.

### Act 7 — Result card and close, 1:23–1:32

Use a clean final card over a slow hero-room orbit. Populate values directly from E9's frozen JSON:

- `50 rooms`
- `1,082 discovered instances`
- `17 min / room`
- `0.783 F1@20 mm`
- one agentic/manipulation result only if its gate passed.

Finish with:

> Real-to-sim as agentic construction

Do not display six decimals, confidence intervals too small to read, or any value from a different freeze.

## 30-second teaser

Structure:

- 0–4 s: title + phone capture;
- 4–11 s: reconstruction and object discovery;
- 11–18 s: two proposals → evidence selection/retry;
- 18–25 s: synchronized physics and photoreal manipulation;
- 25–30 s: four headline numbers and title.

The teaser must work muted. Burn in concise English captions and also export an `.srt`.

## 8-second loop

Use a seamless sequence:

```text
phone frame → reconstructed room → object lifted in physics/photoreal split → return to matched phone frame
```

Avoid title cards and result text in the loop. It should function as a project-page hero visual.

## Source manifest

Every shot must be declared in `configs/demo/icra2027.yaml` and resolved into `source_manifest.json`:

```yaml
shot_id:
act:
source_type: capture|render|mesh|job_ledger|rollout|result_json
source_paths: []
source_hashes: []
scene_id:
object_ids: []
episode_id:
freeze_id:
frame_range:
camera_path:
overlay_text: []
metric_fields: []
manual_edits: []
```

Manual edits may include crop, speed ramp, color-neutral fade, and text timing. They may not alter object geometry, remove failures, fabricate contact, or combine different episodes into one result.

## Rendering requirements

Master:

- 1920×1080, 30 fps;
- render source at equal or higher resolution;
- lossless/visually lossless intermediate, preferably ProRes 422 HQ, DNxHR HQX, or FFV1;
- final H.264 High Profile, CRF 16–18, `yuv420p`, AAC only if narration is used;
- no copyrighted music; silence or original/cleared audio is acceptable;
- no frame interpolation that changes physical timing.

Use exact camera trajectories stored in config. Never rerender a failed shot with a different reset under the same episode label.

## UI overlays

Implement reusable components rather than drawing ad hoc text in every shot:

- title/subtitle card;
- section label;
- proposal card;
- evidence bar with units;
- decision chip: `SELECT`, `RETRY`, `REJECT`, `ABSTAIN`;
- stage progress bar;
- synchronized-view label;
- result counter;
- failure card.

All components should share spacing, corner radius, line thickness, and typography. Avoid drop shadows except a subtle 2–3 px shadow for readability over images.

## Live presentation mode

`run/demo/present_icra2027.sh` should provide:

```text
--mode video       guaranteed conference mode
--mode live        interactive viewer + prewarmed services
--mode auto        live when healthy, otherwise video
```

### Live preflight

Check:

- GPU and display/EGL availability;
- hero build files and hashes;
- MuJoCo scene loads;
- viewer port free;
- policy server reachable when policy playback is requested;
- Harmonizer service healthy when enabled;
- cached camera path/session available;
- fallback MP4 exists and passes checksum.

The script must fall back to the local video within 3 seconds, without exposing a terminal error to the audience.

### Live interaction sequence

Keep the live portion under 90 seconds:

1. open the frozen hero room;
2. click one object to reveal its object record and selected proposal;
3. toggle collision/appearance split;
4. drag or trigger a scripted physical interaction;
5. show synchronized MuJoCo and photoreal views;
6. optionally replay one frozen policy episode.

Do not run object generation or full reconstruction live. Present cached, manifest-backed artifacts and make that clear.

## Presentation guidance

For a 3-minute oral/demo explanation:

1. **20 s:** `A reconstruction is not yet a simulator.`
2. **35 s:** explain heterogeneous tools and agentic evidence selection with the proposal shot.
3. **35 s:** explain persistent object identity with the move-and-reveal shot.
4. **35 s:** explain full-room physics and photoreal policy observation.
5. **25 s:** show manipulation and one honest result.
6. **20 s:** show scale and real captures.
7. **10 s:** close on `making rooms simulatable`.

Do not narrate every module. Explain the two insights: evidence-driven construction and a shared physics/appearance object identity.

## Quality assurance

Automate these checks:

- exact duration and frame rate;
- no all-black frame longer than the intended transitions;
- no frozen duplicated frame longer than 0.5 s outside title/result holds;
- no missing source file/hash;
- no text outside safe area;
- no overlay below minimum font size;
- no episode IDs reused with different source hashes;
- no result number absent from E9 provenance;
- no hero object with known failed background removal;
- physics and photoreal synchronized views have matching body-pose hashes;
- Option C robot-core equality passes in every displayed frame;
- final output plays in Chrome, QuickTime, VLC, and a standard conference laptop.

Human review checklist:

- at 100% size, all text is readable;
- at 50% size, the story remains understandable;
- first-time viewer can state the input, agentic decision, and output after one viewing;
- no visual suggests a real-world success that was only simulated;
- failure/retry shot is understandable without narration;
- no repeated room shot feels like filler;
- color and exposure remain stable across cuts;
- motion is calm enough for a paper presentation.

## Fast smoke

Generate a 12-second draft using cached assets:

```text
capture/reconstruction 3 s
proposal selection 3 s
shared identity 3 s
physics/photoreal 3 s
```

Render at 960×540, 15 fps. The smoke must validate config parsing, source hashes, overlays, assembly, and fallback playback in under 10 minutes.

## Full acceptance criteria

- [ ] 80–95 s 1080p hero video completed.
- [ ] 30 s teaser and 8 s loop derived from the same source manifest.
- [ ] Agentic proposal/selection and genuine retry are visible using real E3 evidence.
- [ ] Moving-object shot reveals a clean background with no double rendering.
- [ ] Physics and photoreal panels use synchronized states.
- [ ] Policy footage is one genuine matched episode, not a montage presented as one trial.
- [ ] All result numbers come from E9's freeze.
- [x] Live launcher has a tested 3-second fallback.
- [ ] Automated and human QA reports are archived.
- [ ] Video remains visually strong without narration.

## Handoff

Create `STATUS.md` containing the freeze ID, hero-scene selection, all source episode/job IDs, render commands, codec metadata, output checksums, QA failures, and links to the hero, teaser, loop, poster, and live launcher.

## Engineering scaffold commands

The current CPU scaffold deliberately accepts only `mode: draft|smoke`,
960×540 at 15 fps, and an exact 12-second timeline. It does not manufacture
the missing policy/Harmonizer story, final result cards, or full film formats.
The empty checked-in config fails closed until immutable source hashes and
an assigned freeze ID are supplied. Draft imported E3 records must declare
their original freeze and authenticate every selected-asset artifact.

```bash
bash run/demo/preflight_icra2027.sh --config <resolved-demo.yaml>
bash run/demo/render_icra2027.sh --config <resolved-demo.yaml> --out outputs/icra2027/<freeze_id>/demo
bash run/demo/assemble_icra2027.sh --demo-dir outputs/icra2027/<freeze_id>/demo
bash run/demo/preflight_icra2027.sh --demo-dir outputs/icra2027/<freeze_id>/demo
bash run/demo/present_icra2027.sh --manifest outputs/icra2027/<freeze_id>/demo/presentation.json --mode video
```

Set `SIMANY_PY` when using an existing interpreter from another worktree.
Package preflight checks SHA256 of the MP4, source files, artifact closure,
and supporting outputs before presentation. The live watchdog uses this
prevalidated fallback's size/mtime identity, one-second aggregate health
timeouts, and a 150 ms polling interval. Configure actual local command
arrays in `presentation.json` for live mode; an absent/unhealthy service or
exited live process starts the local video. A player/display must already
work on the presentation machine. Tests verify player dispatch within three
seconds, not the physical display's decoding/startup latency.

Real capture plus preliminary evidence cards and an explicit missing-footage
card constitute an engineering draft, **not** the required four-stage smoke
storyboard. The final 90-second, 30-second, and seamless-loop deliverables
remain gated on E4/E5 and the final E9 scientific freeze.

Fallback acceptance evidence: the existing launcher component smoke measured
0.114614486seconds in the archived D0 verification (see STATUS.md); current
source399aee3 final framework preserves the local-video fallback and55focused
tests pass. This checkbox covers launcher engineering only. The authentic
90/30/8-second final package, synchronized policy footage and final submission
freeze remain incomplete and their checkboxes stay open.
