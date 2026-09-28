# checkpoints/

Downloaded model weights. Contents are gitignored (only this README is
tracked).

Currently present:

- `dreamsim/` — DreamSim perceptual-similarity weights, downloaded
  automatically on first use.

A compatibility symlink `weights -> checkpoints` exists at the repo root
because the vendored ReconViaGen hard-codes the DreamSim download path to
`$CWD/weights/dreamsim`, and the launchers `cd` to the repo root before
running — so those downloads land here instead of creating a second weights
directory.

Most large weights do **not** live here:

- TRELLIS and the other neural-model checkpoints are resolved from the
  Hugging Face hub cache.
- pi0.5 policy checkpoints live in `$OPENPI_DATA_HOME` (default `checkpoints/openpi_cache/`)
  (see [`../run/pi05_serve.sh`](../run/pi05_serve.sh)).

See [`../docs/DATA_AND_WEIGHTS.md`](../docs/DATA_AND_WEIGHTS.md) for details.
