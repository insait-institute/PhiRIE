#!/usr/bin/env python3
"""Refresh the ScanNet++ offline demo with recorded masks and strong red overlays."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / "integrations/phiview"))
from physicalview.highlight import red_mask


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def render_overlay(image, mask, output, outline):
    rgb = np.asarray(Image.open(image).convert("RGB"), dtype=np.float32) / 255
    pixels = np.asarray(mask, dtype=bool)
    result = red_mask(rgb, pixels, opacity=0.86, outline_px=outline)
    Image.fromarray(np.uint8(result * 255 + 0.5)).save(output)
    return int(pixels.sum())


def replace_chapter(source, output, chorus, sam3):
    """Only replace the original 8..12 second selection chapter before encoding."""
    fontdir = Path("/usr/share/fonts/truetype/dejavu")

    def font(n, bold=False):
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(str(fontdir / name), n)

    card = Image.new("RGB", (1280, 720), "#10282f")
    draw = ImageDraw.Draw(card)
    draw.text((38, 20), "PhiRIE", font=font(28, True), fill="#79dcc0")
    draw.text(
        (178, 23),
        "Query the scene, select the object",
        font=font(28, True),
        fill="#e6f1f0",
    )
    draw.line((38, 68, 1242, 68), fill="#365058", width=1)
    draw.text(
        (38, 681),
        "Actual Chorus query and SAM3 masks | High-contrast red selection overlays",
        font=font(18),
        fill="#aac3c7",
    )
    for label, path, x in [
        ("Chorus query: bottle", chorus, 38),
        ("SAM3 target mask", sam3, 652),
    ]:
        draw.text((x + 12, 92), label, font=font(22), fill="#e6f1f0")
        image = ImageOps.contain(
            Image.open(path).convert("RGB"), (590, 515), Image.Resampling.LANCZOS
        )
        card.paste(
            image, (x + (590 - image.width) // 2, 138 + (515 - image.height) // 2)
        )
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps != 15 or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != 810:
        raise ValueError("Expected the original 54-second, 15 fps demo")
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1280x720",
            "-r",
            "15",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-threads",
            "4",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    count = changed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if 120 <= count < 180:
                rgb = np.asarray(card)
                changed += 1
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encoder.stdin.write(rgb.tobytes())
            count += 1
        encoder.stdin.close()
        if encoder.wait() != 0 or count != 810 or changed != 60:
            raise RuntimeError("Selection chapter encoding failed")
    finally:
        cap.release()
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
    return dict(
        frames=count,
        fps=15,
        duration_s=54,
        replaced_frames=changed,
        changed_interval_s=[8, 12],
        other_frames="original decoded frames before video re-encoding",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-zip",
        "sam3-rgb",
        "sam3-mask",
        "chorus-rgb",
        "chorus-scores",
        "out",
    ):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(a.source_zip) as z:
        if z.testzip() is not None:
            raise ValueError("Source archive CRC failed")
        for n in z.namelist():
            if Path(n).is_absolute() or ".." in Path(n).parts:
                raise ValueError("Unsafe archive member")
        z.extractall(a.out)
    package = a.out / "PhiRIE_ScanNetpp_Demo"
    assets, evidence = package / "assets", package / "evidence"
    old = json.loads((package / "manifest.json").read_text())
    for row in old:
        file = package / row["path"]
        if file.stat().st_size != row["bytes"] or sha(file) != row["sha256"]:
            raise ValueError("Original asset hash mismatch: " + row["path"])
    inputs = evidence / "highlight_inputs"
    inputs.mkdir()
    Image.open(a.sam3_rgb).convert("RGB").save(inputs / "sam3_rgb.png")
    mask = np.asarray(Image.open(a.sam3_mask).convert("L")) > 0
    Image.fromarray(np.uint8(mask) * 255).save(inputs / "sam3_mask.png")
    scores = np.load(a.chorus_scores, allow_pickle=False)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite Chorus scores")
    threshold = float(np.percentile(scores, 99))
    hot = scores >= threshold
    Image.open(a.chorus_rgb).convert("RGB").resize(
        (scores.shape[1], scores.shape[0])
    ).save(inputs / "chorus_rgb.png")
    Image.fromarray(np.uint8(hot) * 255).save(inputs / "chorus_mask.png")
    shutil.copy2(a.chorus_scores, inputs / "chorus_scoremap.npy")
    count_sam = render_overlay(
        inputs / "sam3_rgb.png", mask, assets / "sam3_highlight.png", 4
    )
    count_chorus = render_overlay(
        inputs / "chorus_rgb.png", hot, assets / "chorus_highlight.png", 2
    )
    source_movie = a.out / "original-demo.mp4"
    (assets / "PhiRIE_demo.mp4").replace(source_movie)
    movie = replace_chapter(
        source_movie,
        assets / "PhiRIE_demo.mp4",
        assets / "chorus_highlight.png",
        assets / "sam3_highlight.png",
    )
    update = dict(
        scene="fb5a96b1a2",
        source_archive=str(a.source_zip),
        source_archive_sha256=sha(a.source_zip),
        source_manifest_assets_verified=len(old),
        presentation_only=True,
        color_rgb=[255, 0, 8],
        opacity=0.86,
        sam3_mask_pixels=count_sam,
        chorus_mask_pixels=count_chorus,
        chorus_score_percentile=99,
        chorus_threshold=threshold,
        mask_semantics="Recorded masks unchanged; false positive Chorus regions retained",
        movie=movie,
        physics="Original states, outcomes and three other replay videos retained byte-for-byte",
        robotics_scope="GT assistance plus scripted IK, not a learned policy",
    )
    (evidence / "highlight_update.json").write_text(json.dumps(update, indent=2) + "\n")
    shutil.copy2(package / "validation.json", evidence / "original_validation.json")
    shutil.copy2(__file__, evidence / "scripts/red_mask_package.py")
    shutil.copy2(
        ROOT / "integrations/phiview/physicalview/highlight.py",
        evidence / "scripts/highlight.py",
    )
    text = (
        (package / "index.html")
        .read_text()
        .replace(
            "渲染像素余弦分数前 1% 高亮",
            "渲染像素余弦分数前 1% 用醒目的红色 mask 和白色轮廓高亮",
        )
        .replace(
            "SAM3 图像检测，目标 confidence",
            "红色 mask 与白色轮廓显示原始 SAM3 目标；confidence",
        )
    )
    (package / "index.html").write_text(text)
    with (package / "README.md").open("a") as stream:
        stream.write(
            "\n## Red-mask presentation update\n\nThe SAM3 and Chorus highlights now use strong red masks with white outlines.\nOnly the original selection chapter (8–12 seconds) was replaced in the 54-second\nvideo. The original physical replays and outcomes are preserved. Recorded source\nmasks, RGB inputs and Chorus scores are included in `evidence/highlight_inputs/`;\nsee `evidence/highlight_update.json`. This remains GT assistance plus scripted IK,\nnot a learned policy.\n"
        )
    records = {r["path"]: r for r in old}
    manifest = []
    for file in sorted(package.rglob("*")):
        if not file.is_file() or file.relative_to(package).as_posix() in {
            "index.html",
            "README.md",
            "manifest.json",
            "validation.json",
        }:
            continue
        name = file.relative_to(package).as_posix()
        record = records.get(name, {})
        source = record.get(
            "source", "red-mask update; see evidence/highlight_update.json"
        )
        if record and record["sha256"] != sha(file):
            source = "presentation regenerated from recorded inputs; see evidence/highlight_update.json"
        manifest.append(
            dict(path=name, source=source, bytes=file.stat().st_size, sha256=sha(file))
        )
    (package / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    videos = []
    for file in sorted(assets.glob("*.mp4")):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(file), "-f", "null", "-"], check=True
        )
        videos.append(file.name)
    validation = dict(
        status="PASS",
        source_archive_crc=True,
        source_manifest_assets_verified=len(old),
        manifest_assets_verified=len(manifest),
        videos_fully_decoded=videos,
        highlight_update=update,
        scientific_results="unchanged historical development demo",
    )
    (package / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    archive = a.out / "PhiRIE_ScanNetpp_fb5a96b1a2_RedMask_Demo.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(package.rglob("*")):
            if file.is_file():
                z.write(file, file.relative_to(a.out))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    archive.with_suffix(".zip.sha256").write_text(
        sha(archive) + "  " + archive.name + "\n"
    )
    print(
        json.dumps(
            dict(
                archive=str(archive),
                bytes=archive.stat().st_size,
                sha256=sha(archive),
                validation=validation,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
