#!/usr/bin/env python3
"""Record actual PhiView server frames and projectile commands as an MP4."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--object", default="obj_03")
    parser.add_argument("--seconds", type=int, default=12)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    def get(path):
        with urllib.request.urlopen(
            args.url.rstrip("/") + path, timeout=15
        ) as response:
            return json.load(response)

    def command(message):
        req = urllib.request.Request(
            args.url.rstrip("/") + "/api/command",
            data=json.dumps(message).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=40) as response:
            result = json.load(response)
        events.append(
            {
                "elapsed_s": time.monotonic() - start,
                "command": message,
                "result": result,
            }
        )
        return result

    before = get("/api/health")
    if not before.get("ready"):
        raise RuntimeError("Viewer is not ready")
    events = []
    start = time.monotonic()
    command({"op": "reset"})
    command({"op": "select", "object": args.object})
    command({"op": "enable"})
    command({"op": "focus"})
    command({"op": "resolution", "value": "1080p"})
    command({"op": "highlight", "value": True})
    time.sleep(0.5)
    fps = 15
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "1920x1080",
            "-r",
            str(fps),
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
            str(args.out / "PhiView_fb5a96b1a2_Shooting_Demo.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
    records = []
    start = time.monotonic()
    try:
        for i in range(args.seconds * fps):
            if i in [fps * 2, fps * 5, fps * 8]:
                state = get("/api/status")
                command(
                    {
                        "op": "shoot",
                        "frame": state["frame"],
                        "x": 0.5,
                        "y": 0.5,
                        "speed": 5,
                    }
                )
            with urllib.request.urlopen(
                args.url.rstrip("/") + "/frame.jpg", timeout=15
            ) as response:
                fid = int(response.headers["X-Frame-Id"])
                jpg = response.read()
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None or frame.shape[:2] != (1080, 1920):
                raise RuntimeError("Unexpected rendered frame dimensions")
            encoder.stdin.write(frame.tobytes())
            records.append(
                {"index": i, "server_frame": fid, "elapsed_s": time.monotonic() - start}
            )
            if i in [0, fps * 2 + 3, fps * 5 + 3, fps * 8 + 3]:
                (args.out / f"frame-{i:04d}.jpg").write_bytes(jpg)
            time.sleep(max(0, start + (i + 1) / fps - time.monotonic()))
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoding failed")
        after = get("/api/status")
        report = {
            "status": "PASS"
            if after.get("projectile_contacts")
            else "FAILED_NO_CONTACT",
            "scene": before["scene"],
            "gpu": before["gpu"],
            "frames": len(records),
            "unique_server_frames": len({r["server_frame"] for r in records}),
            "fps": fps,
            "wall_duration_s": time.monotonic() - start,
            "presentation_duration_s": args.seconds,
            "source": "Actual PhiView server JPEG frames; no generated or interpolated frames",
            "commands": events,
            "frame_timing": records,
            "after": after,
        }
        (args.out / "shooting-evidence.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if not after.get("projectile_contacts"):
            raise RuntimeError(
                "No physical projectile contacts were observed; video retained for inspection"
            )
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(args.out / "PhiView_fb5a96b1a2_Shooting_Demo.mp4"),
                "-f",
                "null",
                "-",
            ],
            check=True,
        )
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in ["after", "commands", "frame_timing"]
                },
                indent=2,
            )
        )
    finally:
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
        command({"op": "reset"})
        command({"op": "view", "mode": "original"})
        command({"op": "camera", "name": "DSC03413.JPG"})
        command({"op": "resolution", "value": "native"})


if __name__ == "__main__":
    main()
