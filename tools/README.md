# Repository tools

- `phiview/`: the complete PhiView viewer source, bundled from
  [RunyiYang/PhysicalView](https://github.com/RunyiYang/PhysicalView) at the
  revision recorded in `phiview/SOURCE_PROVENANCE.json` (with per-file checksums
  and the upstream MIT license). `run/phiview.sh` runs its `physicalview` CLI
  from its own locked uv project; see [docs/PHIVIEW.md](../docs/PHIVIEW.md).
- `harmonizer/`: `tools.harmonizer.server`, a Unix-socket service that applies
  NVIDIA Harmonizer to rendered observations with per-stream temporal state. It
  runs inside the Harmonizer environment; `run/serve_harmonizer.sh` starts it
  (`HARMONIZER_SOURCE`, `HARMONIZER_CHECKPOINT`, optional `HARMONIZER_SOCKET`,
  `HARMONIZER_RESOLUTION`, `HARMONIZER_TIMESTEP`), and
  `robo/rendering/harmonizer_client.py` is the client. An `identity` backend
  exists for testing the protocol without the model.
- `release/verify.py`: check that every module action plans, that every module
  has a guide under `phiroom/pipelines/`, that the package and CLI versions
  agree, and that every bundled PhiView file matches `SOURCE_PROVENANCE.json`;
  runs without importing GPU backends.
- `release/archive.py`: build an exact source ZIP of a committed tree (PhiView
  included), verify ZIP CRC and required paths, and emit commit, size and
  checksum JSON.

```bash
uv run --project envs/control --locked python tools/release/verify.py
uv run --project envs/control --locked python tools/release/archive.py \
  --ref v2.0.1 --out outputs/release/PhiRIE-v2.0.1-full-source.zip
```

Archives exclude Git metadata and untracked caches by construction. The archive
tool fails if the target file exists. Publish only after validating the merged
`main` commit.
