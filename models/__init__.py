"""Bridges to the external neural models: SAM3 segmentation (s1),
metric monocular depth (s2), and the TRELLIS / ReconViaGen image-to-3D
generators (s4). Each is a runnable pipeline stage; the vendored model
code itself lives in third_party/ and the weights in checkpoints/.
"""
