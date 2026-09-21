---
name: scannetpp-gsplat-data-layout
description: "Where the ScanNet++ v2 splats live, their coordinate frame, and how to render them correctly"
metadata: 
  node_type: memory
  type: project
  originSessionId: a84b4517-f014-4a7b-99cb-664d54b4d178
---

ScanNet++ v2 3DGS splats are at `/data/ScanNetppv2_gsplat/splats/<scene>.ply` (NOT
"/data/ScanNetpp_gsplat" — /data is an autofs mount, `ls /data` shows nothing; exact
name needed). GaussianWorld MCMC 1.5M splats, SH degree 3, meta.csv has per-scene PSNR.

Splats are in the **mesh/colmap world frame** (metric, z-up). To render with gsplat:
w2c straight from `dslr/colmap/images.txt` + PINHOLE intrinsics from
`dslr/nerfstudio/transforms_undistorted.json`, GT = `dslr/resized_undistorted_images`
(1752x1168). The nerfstudio transforms are in a permuted frame:
`(x,y,z)_json = (y,x,−z)_mesh`, OpenGL c2w. Verified on 09bced689e: PSNR 33–35 dB.
Conversion helpers in [[affordancept-pipeline]] `/group/worldcept/code/affordancept/pipeline/common.py`.
