"""Fetch the RAW DROID data matching the local droid_100 RLDS episodes.

The RLDS release is 320x180 with no poses; the raw release has full-res MP4
per camera (mono + side-by-side stereo, no ZED SDK needed for RGB),
trajectory.h5 with PER-FRAME 6-DoF camera extrinsics for all three cameras
(verified: /observation/camera_extrinsics/<serial>_left ...), and the
calibration metadata json. SVO (stereo raw for depth) and trajectory_im128
are skipped by default to stay small (~40 MB/episode instead of ~150).

Episode paths come straight out of the RLDS tfrecords' episode_metadata
strings (nfs .../r2d2-data-full/<lab>/success/<date>/<ts>/...), remapped to
gs://gresearch/robotics/droid_raw/1.0.1/.

System python3 (stdlib only). Usage:
    python3 run/fetch_droid_raw.py [--include-svo] [--max-gb 20]
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

RLDS_DIR = Path("/group/worldcept/PhiRIE/data/droid/droid_100/1.0.0")
DEST = Path("/group/worldcept/PhiRIE/data/droid/raw")
BUCKET_PREFIX = "robotics/droid_raw/1.0.1"
FETCH = Path(__file__).resolve().parent / "gcs_fetch.py"


def episode_paths():
    pat = re.compile(rb"r2d2-data-full/([A-Za-z0-9_+-]+/success/"
                     rb"[0-9-]+/[A-Za-z0-9_:]+_2023)/")
    eps = set()
    for shard in sorted(RLDS_DIR.glob("*.tfrecord-*")):
        for m in pat.finditer(shard.read_bytes()):
            eps.add(m.group(1).decode())
    return sorted(eps)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-svo", action="store_true")
    ap.add_argument("--max-gb", type=float, default=20.0)
    args = ap.parse_args()

    eps = episode_paths()
    print(f"[droid_raw] {len(eps)} episodes referenced by the RLDS shards")
    skip = [] if args.include_svo else ["/SVO/", "trajectory_im128"]
    for i, ep in enumerate(eps):
        dest = DEST / ep
        if (dest / "trajectory.h5").exists() and \
                any((dest / "recordings" / "MP4").glob("*.mp4")):
            continue
        r = subprocess.run(
            [sys.executable, str(FETCH), "--bucket", "gresearch",
             "--prefix", f"{BUCKET_PREFIX}/{ep}", "--dest", str(dest),
             "--max-gb", str(args.max_gb)],
            capture_output=True, text=True)
        # gcs_fetch has no skip filter; delete the unwanted big files after
        for pat in skip:
            for f in dest.rglob("*"):
                if f.is_file() and pat.strip("/") in str(f):
                    f.unlink()
        status = "ok" if r.returncode == 0 else f"FAIL rc={r.returncode}"
        if i % 10 == 0 or status != "ok":
            print(f"[droid_raw] {i + 1}/{len(eps)} {ep}: {status}",
                  flush=True)
            if status != "ok":
                print((r.stdout + r.stderr)[-300:], flush=True)
    print("[droid_raw] DONE")


if __name__ == "__main__":
    main()
