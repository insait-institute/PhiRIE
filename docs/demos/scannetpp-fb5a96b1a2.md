# ScanNet++ scene fb5a96b1a2 demo

[Download the red-mask demo ZIP](https://github.com/RunyiYang/PhiRoom/releases/download/v2.0.0/PhiRIE_ScanNetpp_fb5a96b1a2_RedMask_Demo.zip)
(29,945,015 bytes), extract it, then open `PhiRIE_ScanNetpp_Demo/index.html`.

[Download the 54-second demo video](https://github.com/RunyiYang/PhiRoom/releases/download/v2.0.0/PhiRIE_fb5a96b1a2_RedMask_Demo.mp4).
The package contains the offline page, four videos, images, generated asset, physical
traces, development outcomes, implementation scripts and source provenance.

The SAM3 target and Chorus query highlights now use vivid red masks and white outlines.
The selection chapter at 8–12 seconds uses the same updated overlays. The recorded
masks and Chorus scores are retained, including the query's false-positive regions;
this update changes presentation. The other three replay videos and physical traces
remain byte-for-byte unchanged. Live PhiView on main uses the same red-overlay module.

- [Red-mask ZIP checksum](https://github.com/RunyiYang/PhiRoom/releases/download/v2.0.0/PhiRIE_ScanNetpp_fb5a96b1a2_RedMask_Demo.zip.sha256)
- [Unchanged original demo ZIP](https://github.com/RunyiYang/PhiRoom/releases/download/v2.0.0/PhiRIE_ScanNetpp_Demo.zip) (25,730,916 bytes)
- [Original ZIP checksum](https://github.com/RunyiYang/PhiRoom/releases/download/v2.0.0/PhiRIE_ScanNetpp_Demo.zip.sha256)

The red-mask archive has SHA-256
`c3b57701c7045679e0f4e9e4ba3e6cc2c1dd045c2858c9b464052f42a1341b9f`.
Validation checked all 63 original manifest entries, all 72 updated entries, ZIP CRC
and complete decoding of the four videos. The package's `validation.json` and
`evidence/highlight_update.json` record the checks and presentation changes.

This is **GT assistance plus scripted IK, not a learned policy**. The selected bottle
transfer is one successful upright placement among three development attempts;
the 18 kitchen failures remain in the evidence. These counts are development history,
not independent benchmark estimates. Full datasets, Gaussian training inputs and model
checkpoints remain outside this viewable package.

The demo assets are separate downloads attached to v2.0.0. The existing v2.0.0 source
tag and source archives remain unchanged; the newer viewer overlay is on main.

For reproducible presentation updates, see [tools/demo](../../tools/demo/).

## PhiView shooting video

[Download the shooting MP4](https://github.com/RunyiYang/PhiRoom/releases/download/v2.0.0/PhiView_fb5a96b1a2_Shooting_Demo.mp4)
or [its recording and browser evidence ZIP](https://github.com/RunyiYang/PhiRoom/releases/download/v2.0.0/PhiView_fb5a96b1a2_Shooting_Evidence.zip).

This separate 12-second recording shows actual PhiView server frames and three
projectile commands on an NVIDIA RTX A6000. It loads all 1,893,903 original Gaussians
and records at 1920 × 1080. The video contains 180 encoded frames, including repeated
frames while waiting for the server: 51 distinct rendered frames over 12.22 seconds
of wall time, with 5.166 seconds of simulated time. It does not demonstrate 15 FPS
rendering performance. MuJoCo projectile contacts were observed; the final contact
window records floor collisions, not a target-hit success metric.

The live scene uses existing factory proposals (10 objects, eight collision bodies).
Its bottle is factory object `obj_03` / GT 84, distinct from GT 85 in the scripted
transfer above. Its object masks use approximate proposal regions. This physics
demo uses a floor and a GT-derived table proxy; full room collision was not validated.
All four tabletop objects drifted under 3 cm in a two-second settling check; upper
shelf objects can fall when all bodies are enabled. Shooting enables available bodies.

Simulation works over the original observed background. Newly exposed regions remain
unfilled, and clean-view controls stay disabled until actual inpainting is available.
The browser check passed image picking, keyboard movement, right-drag rotation,
shooting, pause and clean-view gating, with no page errors, canvas or geometry downloads.
Model generation, prompted completion and robot commands were not validated in this
live session. The recording evidence retains the source commit, runtime manifest,
collision setup, commands, frame timing and browser screenshots.

## Click an unprepared object and make it simulatable

In live PhiView, click a reconstructed object outside the existing red regions.
SAM3 segments the clicked region on the server and highlights its visible Gaussians.
When selection finishes, click **Make simulatable** to create a collision body, then
use **Fall**, **Friction** or **Throw**. Existing prepared objects still select immediately.

The A6000 browser check added a previously unlisted chair region as `obj_10`, increasing
the object count from 10 to 11. It selected 1,549 visible Gaussians, verified that the
object had no body before enabling, built a body and ran a finite fall interaction.
The masks, indices, collision proxy and physical defaults persist in the session output.
Collision geometry is a convex approximation of the observed surface; mass and friction
are explicitly unmeasured defaults. Create new bodies before placing a robot.

See the [PhiView controls and environment guide](https://github.com/RunyiYang/PhysicalView/blob/main/docs/PHIVIEW.md).
