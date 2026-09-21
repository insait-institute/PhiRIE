# F3 — Object-region image fidelity on frozen held-out views

Owner: rendering/evaluation agent. Implemented module `robo.roundtrip.object_fidelity`.

The original native full-frame comparison is dominated by retained room pixels. Do not delete or clip infinite original PSNR values. This is a separately declared object-region analysis of the same frozen rendered data.

## Mask generation

Use unchanged reference native scenes and original held-out cameras. Replay the declared TRAIN capture rendering prefix before the held-out suffix. A reference RGB render must match its original byte-exact snapshot before its native segmentation mask is admitted. Segmentation is RoboSuite `MjSim.render` output: object TYPE in channel 0, object ID in channel 1. Only visible target visual geoms/descendants count. Occluding room/robot geometry remains in the scene.

No prediction-derived masks, GT registration of generated targets, exposure alignment, image rescaling, or favorable held-out view replacement. Invisible reference targets get `TARGET_NOT_VISIBLE`; native identity failures and render failures remain explicit. Reference masks and goal geometry stay evaluator-only, outside the constructor allowlist.

## Standard metrics

- `masked_psnr`: the existing PSNR implementation on the reference target mask.
- `crop_ssim`: existing unmasked SSIM on a fixed reference-derived crop.
- `crop_lpips`: pinned AlexNet LPIPS on that same crop, NOT a newly invented masked LPIPS.

Crop = target reference bbox padded by 10% on each side, minimum side 32 pixels, clipped to image, no resizing. Reuse `appearance_render.json` predictions whose original native identity controls passed. The tool checks image/mask hashes, complete mask shards and duplicate/conflicting sources before measuring. Missing method renders remain unavailable rather than yielding zero error.

All methods use the same independent mask/crop. Aggregate by canonical instance on common supported views, while separately reporting available/planned instances. Exact identical target pixels remain `POSITIVE_INFINITY` PSNR with explicit status. `--skip-lpips` means partial diagnostic, never fill a paper LPIPS cell from it.

## Scale

`masks` supports deterministic instance sharding with `--shard-index` and `--shard-count`. Use ordinary separate jobs, not arrays. `metrics` accepts repeated `--masks` and `--renders`; require all declared mask shards. Cache generated masks once and reuse across method comparisons. Existing held-out renders are reused; missing output may be regenerated only from the unchanged frozen asset/camera/renderer contract, preserving original failures and new runtime lineage.

Outputs: `mask_manifest.json` and mask PNGs; `object_region_records.jsonl`, summary JSON/CSV and provenance. Any global fidelity conclusion must report coverage and not extrapolate a 12-instance intersection to 48 successful reconstructions.
