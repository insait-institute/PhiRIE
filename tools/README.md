# Repository tools

- `release/verify.py`: check feature entry points, version agreement and the exact
  PhiView submodule pin; runs without importing GPU backends.
- `release/archive.py`: archive a committed parent tree plus the gitlink revision's
  tracked files, verify ZIP CRC and required paths, and emit commit/size/checksum JSON.

```bash
uv run --project envs/control --locked python tools/release/verify.py
uv run --project envs/control --locked python tools/release/archive.py \
  --ref v2.0.0 --out outputs/release/PhiRoom-v2.0.0-full-source.zip
```

Archives exclude Git metadata and untracked caches by construction. Existing tracked
historical receipts remain in the exact source snapshot. The archive tool fails if
the target file exists. Publish only after validating the merged main commit.
