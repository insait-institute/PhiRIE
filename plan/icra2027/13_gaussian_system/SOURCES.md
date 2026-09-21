# Official sources checked for this campaign (2026-09-09)

Versions may change. Resolve and record source commit, model snapshot and native compatibility before running; do not replace pinned experiments with a moving latest release.

## Baselines
- SimFoundry project: https://research.nvidia.com/labs/gear/simfoundry/
- Official code and released-capability status: https://github.com/NVlabs/SimFoundry
- PolaRiS project: https://polaris-evals.github.io/
- Official code: https://github.com/arhanjain/polaris
- Actual assisted custom-environment workflow: https://github.com/arhanjain/polaris/blob/main/docs/custom_environments.md

Both systems already use video/reconstruction, and Gaussian capability must not be denied to PolaRiS. SimFoundry's released construction and policy-training capabilities differ; record what is available rather than filling missing official code with our own proxy.

## Generators and Gaussian semantics
- TRELLIS: https://github.com/microsoft/TRELLIS
- TRELLIS.2 example/API: https://github.com/microsoft/TRELLIS.2/blob/main/example.py
- ReconViaGen official main (v0.2) and v0.5 are distinct: https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen
- Chorus package/raw Gaussian inference: https://github.com/GaussianWorld/Chorus
- Original 3DGS / COLMAP converter: https://github.com/graphdeco-inria/gaussian-splatting

The actual model adapters were written against these public interfaces. TRELLIS.2 does not automatically export Gaussian objects like TRELLIS v1. ReconViaGen-v0.5 combines a newer model route; it needs its own adapter. Frozen Chorus is prior work, and input-row mapping plus matching language checkpoint are part of integration.

## Inpainting and harmonization
- Qwen-Image-Edit-2511 model card and API: https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- Gemini image models and current model names: https://ai.google.dev/gemini-api/docs/image-generation
- generateContent REST schema: https://ai.google.dev/api/generate-content
- NVIDIA DiffusionHarmonizer: https://github.com/NVIDIA/harmonizer

Nano Banana is a product family, not an immutable checkpoint. Pin gemini-2.5-flash-image and/or gemini-3.1-flash-image, save returned modelVersion and actual usage; do not assume provider determinism or a public seed guarantee. Qwen/Gemini image editing is not identical to native masked diffusion inpainting. Compare the explicitly declared mask-instruction + compositing protocol. No claim that generic temporal harmonization is new to this project.

## Dataset sources
- RoboCasa: https://robocasa.ai/
- BEHAVIOR: https://behavior.stanford.edu/
- ReplicaCAD: https://aihabitat.org/datasets/replica_cad/
- HSSD: https://3dlg-hcvc.github.io/hssd/
- Hypersim: https://github.com/apple/ml-hypersim
- ScanNet++: https://scannetpp.ml/
- Replica: https://github.com/facebookresearch/Replica-Dataset
- HM3D: https://aihabitat.org/datasets/hm3d/
- ProcTHOR: https://procthor.allenai.org/
- InteriorVerse: https://interiorverse.github.io/
- DROID: https://droid-dataset.github.io/

Read actual release licenses and access requirements before downloading/redistributing or uploading images to third-party APIs. Habitat/OmniGibson/Isaac/MuJoCo backends are not interchangeable. Static photorealistic datasets do not provide autonomous manipulation measurements by themselves.
