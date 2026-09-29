# Release infrastructure

Branch: `block/release`.

Owns packaging, repository checks, GitHub CI, release notes and contribution workflow.

```bash
uv run pytest -q
uv build
uv run python tools/validation/check_release.py
```

See the [release procedure](../../docs/RELEASE.md). Tag automation builds a draft GitHub
release with a wheel, source distribution and SHA-256 checksums. Publishing the draft
and distributing licensed datasets/weights are separate actions.
