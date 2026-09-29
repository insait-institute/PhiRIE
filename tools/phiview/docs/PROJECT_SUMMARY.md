# Project summary — PhysicalView 0.2

PhysicalView connects observation-to-simulation tools through one server application.
PhiView is its image-only demo frontend: it sends input events and displays rendered
frames. The server holds the full Gaussian scene, visible object IDs, generated variants,
MuJoCo state and composited robot imagery. The original viser studio remains available
for detailed construction work and has an optional client-splat inspection mode.

## Contributions and ownership

The ten [pipeline blocks](../pipelines/README.md) expose dataset adaptation, reconstruction,
discovery, generation, registration, inpainting, simulation, robotics, viewing and paper
export through `physicalview plan/run`. Each has a branch, tool manifest, input/output
contract and documented call. Runtime supplies portable config and recorded local/Slurm
jobs; release supplies packaging, CI and contribution rules.

PhysicalView integrates the separate SimAny/PhiRoom backend; it does not claim the
third-party algorithms as new methods. SAM3, TRELLIS, SAM 3D, Qwen, gsplat, MuJoCo and
OpenPI retain their upstream sources/licenses. The backend source revision is pinned
in `tools/backends/sources.json` and requires repository access.

## Requested feature map

| Request | Implementation / call | Evidence boundary |
|---|---|---|
| a: original full Gaussian scene | PhiView `Original`, viewer block | All rows remain loaded; image output resolution is separately recorded |
| b: automatically highlight objects | discovery block, highlight-all | Existing proposals may be GT-assisted; highlights do not measure exhaustive recall |
| c: mouse selection | server-rendered ID buffer | Visibility-aware click tests; browser transfers images and events |
| d: make objects simulatable | enable constructed MuJoCo bodies | Requires valid generated/registered collision assets |
| e: physical parameters | object inspector and simulation block | Shows active mass, inertia and friction; estimates are not measured ground truth |
| f: generated alternatives | generation/registration and variant selector | Registered alternatives can be selected; generation quality varies |
| g: clean selected object | removal plus selected clean version | Residual geometry and image artifacts require visual review |
| f2: clean all simulatable objects | clean-all state | Constructed/active object set is explicit; incomplete removals remain known |
| h: prompted inpainting | inpainting block; Qwen 2511 demo adapter | Prompt/model receipts retained; successful execution does not establish semantic removal |
| i: fall and friction | MuJoCo interaction probes | Results depend on estimated collision geometry and coefficients |
| j: throw | selected-body impulse and trajectories | A simulator interaction, not validated real-world dynamics |
| k: shoot | camera-ray projectiles | Virtual projectiles use the active simulator contact model |
| l: robot arm commands | composited Franka, scripted IK / external policy client | GT assistance plus scripted IK, not a learned policy; general manipulation unverified |
| m: navigation | WASDQE, mouse look and server camera | Historical browser checks exist; this release does not claim a new browser performance benchmark |

## Datasets and paper delivery

ScanNet++ uses scanned scenes and source Gaussians. LIBERO adapts native simulator XML,
recorded state and GT geometry; articulated fixtures are frozen and the native robot is
omitted. BEHAVIOR WDS preparation selects views with compatible object state and preserves
native RGB bytes/resolution. DROID and video reconstruction remain supported through the
existing scene contract and studio dataset selector. See the historical
[native](NATIVE_DEMO.md), [BEHAVIOR/DROID](DEMO_BEHAVIOR_DROID.md) and
[paper campaign](PHIVIEW_PAPER_CAPTURE.md) notes.

The paper request is 10 scenes × 14 feature groups × 3 datasets = **420 groups**.
The archived campaign summary, last updated 2026-09-14 03:26:48 UTC, records:

| Dataset | Complete scenes / requested | Captured groups / requested | Original PNGs | Approved groups |
|---|---:|---:|---:|---:|
| ScanNet++ | 10 / 10 | 140 / 140 | 609 | 0 |
| LIBERO | 7 / 10 | 98 / 140 | 420 | 0 |
| BEHAVIOR | 0 / 10 | 0 / 140 | 0 | 0 |

This snapshot is included as [campaign-coverage.json](releases/0.2.0/campaign-coverage.json).
It is not a live scheduler dashboard. The release integrates the tooling and preserves
these partial results; it does not mark the remaining campaign or paper review complete.

The paper block builds lossless PNG/SVG packs, lightweight JPEG previews and self-contained
ZIP galleries. Each archive verifies CRCs, local links and content hashes, and derives
coverage/approval counts from its input snapshot. Download artifacts remain outside Git.

## Validation and next research work

See [release validation](RELEASE.md) for CPU, backend and H200 checks using the new uv
profiles. Model weights, TRELLIS native extensions, dedicated SAM3D/OpenPI server/Isaac Sim
environments and licensed assets remain explicit prerequisites. Further research work is
needed for clean novel views, reliable manipulation, the remaining dataset campaign and
human approval of publication figures.
