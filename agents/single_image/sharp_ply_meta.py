"""Tiny shared helper: recover the SHARP prediction camera from a SHARP-saved
gaussian .ply's own top-level `intrinsic`/`image_size` elements.

SHARP's save_ply (sharp/utils/gaussians.py:404-430) writes:
  intrinsic  : 9 float32s, flattened row-major 3x3 K
               [f_px, 0, W/2, 0, f_px, H/2, 0, 0, 1]
  image_size : 2 uint32s, [width, height]

These are exactly the values predict_image() used internally to unproject
its NDC-space Gaussians into this .ply's own world frame (== the identity
camera pose of the input photo, OpenCV convention). Do NOT recompute focal
length from EXIF/heuristics here -- read it back out of the ply so it
matches the projection the Gaussians were actually placed under.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Fixed approximate camera-frame -> z-up-world rotation, used ONLY to bring
# obj_XX/points.ply into a frame compatible with s5_align.py's structural
# z-up assumption (rz(yaw) rotates about world-Z, and the longest-observed-
# axis scale heuristic implicitly assumes world-Z == vertical). SHARP's own
# splat frame is OpenCV camera convention (x-right, y-down, z-forward/depth)
# -- NOT z-up -- which was the root cause of most size-sanity rejections in
# the first run. DSC08561 is a roughly level, non-tilted DSLR shot, so a
# fixed axis-permutation rotation (no per-frame gravity estimation) is a
# reasonable approximation: camera "down" (y) -> world "down" (-z); camera
# "forward"/depth (z) -> a world horizontal axis; camera "right" (x) stays a
# world horizontal axis.
#   x_w = x_c ; y_w = z_c ; z_w = -y_c
# Applied as pts_zup = pts_cam @ R_ZUP.T (row-vector convention, matching
# common.py's own pts @ M.T pattern). Verified orthogonal + proper rotation
# below (not just asserted -- checked numerically before trusting it):
#   R_ZUP @ R_ZUP.T == I (rows orthonormal) and det(R_ZUP) == +1 (no
#   reflection); round-tripping pts -> pts_zup -> pts_zup @ R_ZUP recovers
#   the original camera-frame points exactly (R_ZUP is its own useful
#   "inverse-via-right-multiply" partner here since it's orthogonal).
R_ZUP = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [0.0, -1.0, 0.0]], dtype=np.float64)
assert np.allclose(R_ZUP @ R_ZUP.T, np.eye(3)), "R_ZUP is not orthogonal"
assert np.isclose(np.linalg.det(R_ZUP), 1.0), "R_ZUP is not a proper rotation"


def read_camera(ply_path) -> tuple[float, int, int]:
    """SHARP .ply -> (f_px, width, height), read from its own stored camera."""
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    names = {el.name for el in ply.elements}
    if "intrinsic" not in names or "image_size" not in names:
        raise ValueError(
            f"{ply_path}: missing 'intrinsic'/'image_size' top-level PLY "
            "elements -- not a SHARP-produced ply?")

    intrinsic = np.asarray(ply["intrinsic"]["intrinsic"], dtype=np.float64)
    if intrinsic.shape[0] != 9:
        raise ValueError(f"{ply_path}: expected 9-float 'intrinsic', got "
                         f"{intrinsic.shape}")
    K = intrinsic.reshape(3, 3)
    f_px = float(K[0, 0])

    image_size = np.asarray(ply["image_size"]["image_size"])
    width, height = int(image_size[0]), int(image_size[1])

    return f_px, width, height


if __name__ == "__main__":
    import sys

    f_px, W, H = read_camera(sys.argv[1])
    print(f"f_px={f_px} width={W} height={H}")
