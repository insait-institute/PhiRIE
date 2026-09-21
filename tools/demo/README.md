# Demo presentation tools

`red_mask_package.py` verifies an existing ScanNet++ offline package, regenerates
high-contrast overlays from the recorded SAM3 mask and Chorus score map, replaces
only the selection chapter before video encoding, then validates and packages the result.
It uses `physicalview.highlight` from the pinned PhiView submodule. Run it with the
PhiView CPU environment; FFmpeg and DejaVu fonts must be available.

```bash
integrations/phiview/.venv/bin/python tools/demo/red_mask_package.py \
  --source-zip /path/to/PhiRIE_ScanNetpp_Demo.zip \
  --sam3-rgb /path/to/DSC03413.JPG --sam3-mask /path/to/recorded_mask.png \
  --chorus-rgb /path/to/rendered_rgb.png --chorus-scores /path/to/scoremap.npy \
  --out outputs/new-red-mask-package
```

The output path must be new. Original evidence is checked before changes. The tool
preserves the prediction masks, records inputs and new hashes, and retains the
original physical outcomes. It does not run models or claim new task success.

`record_shooting.py` connects to a running PhiView server, enables and focuses a
selected object, fires three projectiles and encodes actual server JPEG frames.
It requires exclusive use of the shared camera while recording, the PhiView CPU
environment and FFmpeg. It restores the original view and native resolution afterward.

```bash
integrations/phiview/.venv/bin/python tools/demo/record_shooting.py \
  --url http://localhost:8100 --object obj_03 --seconds 12 \
  --out outputs/new-shooting-recording
```

The output directory must be new. Its evidence records commands, distinct server
frames, repeated presentation frames, wall time, simulated time and contacts.
A contact indicates a physical collision; it does not by itself establish a hit
on the selected object. This recorder currently restores the fb5a96b1a2 source
camera `DSC03413.JPG` and is intended for that demo scene.
