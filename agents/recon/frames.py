"""Extract uniformly subsampled frames from a casual video (any env, CPU).

ffmpeg-only, no python video deps. Uniform TEMPORAL subsampling via the
select filter (every k-th decoded frame), not fps resampling, so slow pans
and fast sweeps keep their relative coverage.

Usage:
    python -m agents.recon.frames --video V.mp4 --out-dir D \
        [--max-frames 160] [--max-long-side 1752]
Writes D/frame_000000.jpg ... (names sort in capture order; downstream
stages key everything on these names).
"""
import argparse
import subprocess
import sys
from pathlib import Path

FFMPEG = "/usr/bin/ffmpeg"    # system binary; none of the venvs ship one
FFPROBE = "/usr/bin/ffprobe"

# 160 frames matches the DSLR trajectory density we validated the pipeline
# on (~300 frames per ScanNet++ scene, but VGGT quality saturates earlier
# and gsplat training cost is linear in views).
MAX_FRAMES = 160
# Cap the LONG side at 1752 (= ScanNet++ resized-undistorted width, whose
# frames are 1752x1168): keeping our frames in the same resolution band means
# every downstream pixel threshold (mask areas, bbox gates, crop pads)
# transfers unchanged. Capping the short side instead would shrink landscape
# clips to ~45% of the validated pixel area and silently tighten the gates.
MAX_LONG_SIDE = 1752
JPEG_Q = 2  # ffmpeg qscale, ~95% quality: frames feed both VGGT and gsplat


def count_video_frames(video: Path) -> int:
    """Packet count, not container metadata: nb_frames is often missing on
    phone captures, while counting packets needs no full decode."""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0",
         str(video)],
        check=True, capture_output=True, text=True).stdout.strip()
    n = int(out.splitlines()[0])
    if n <= 0:
        raise SystemExit(f"[frames] no video frames found in {video}")
    return n


def extract(video: Path, out_dir: Path, max_frames: int, max_long_side: int):
    n_total = count_video_frames(video)
    step = max(1, -(-n_total // max_frames))  # ceil: at most max_frames out
    out_dir.mkdir(parents=True, exist_ok=True)
    # select every step-th frame; cap the LONG side whatever the orientation
    # (expressions run on decoded frames, so rotation metadata is honoured);
    # scale never upscales, -2 keeps the other dimension even
    long_cap = (f"scale=w='if(gte(iw\\,ih)\\,min(iw\\,{max_long_side})\\,-2)':"
                f"h='if(gte(iw\\,ih)\\,-2\\,min(ih\\,{max_long_side}))'")
    vf = f"select='not(mod(n\\,{step}))',{long_cap}"
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(video), "-vf", vf, "-fps_mode", "vfr",
           "-q:v", str(JPEG_Q), "-start_number", "0",
           str(out_dir / "frame_%06d.jpg")]
    subprocess.run(cmd, check=True)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    print(f"[frames] {video.name}: {n_total} video frames, step {step} "
          f"-> {len(frames)} jpgs in {out_dir}", file=sys.stderr)
    if not frames:
        raise SystemExit("[frames] ffmpeg produced no frames")
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--max-long-side", type=int, default=MAX_LONG_SIDE)
    args = ap.parse_args()
    extract(args.video, args.out_dir, args.max_frames, args.max_long_side)


if __name__ == "__main__":
    main()
