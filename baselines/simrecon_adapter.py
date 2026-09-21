"""SimRecon adapter -- STUB. SimRecon has no public code release.

SimRecon (arXiv:2603.02133, CVPR 2026) is flagged in docs/BASELINES.md as
"the closest new [real-to-sim] competitor" and "already reports ScanNet
numbers", but that same document's own Tier-2 worklist (section 4) lists it
under "cite and contrast, do not run: ... SimRecon (until code lands)" --
i.e. no runnable public release as of the 2026-08-16 baseline audit
recorded in `baselines/release_status.yaml` (status: literature_only). A
repeated check while writing this stub found no public code repository
either.

This module contains NO reimplementation and NO numeric computation of any
kind. Every public entry point below raises `NotImplementedError`
immediately, unconditionally, regardless of arguments, so nothing can
accidentally produce a fabricated SimRecon number for the oracle table --
see plan/15_CONSTRUCTION_BASELINES.md's acceptance criterion: "Mark
unsupported cells N/A, never zero." `tests/test_baseline_release_status.py`
enforces that this stays true (it imports this module and asserts every
public callable raises).

If a public release appears: replace this stub with a real adapter, pin the
commit/weights/license in `baselines/release_status.yaml` (flip `status` to
`implemented` or `partial`) and in `docs/BASELINE_REPRODUCTION.md`, and
delete this docstring's caveats.
"""

_MESSAGE = (
    "SimRecon has no public code release as of the 2026-08-16 baseline "
    "audit (see baselines/release_status.yaml, status: literature_only, "
    "and docs/BASELINES.md section 2 / section 4 Tier-2 worklist: "
    "'SimRecon (until code lands)'). This adapter is an intentional "
    "placeholder and must not be called; it exists only so the "
    "construction-baseline table's SimRecon row is explicit N/A rather "
    "than a silently-missing or fabricated entry."
)


def run(*args, **kwargs):
    """Would run the SimRecon pipeline end-to-end. Always raises."""
    raise NotImplementedError(_MESSAGE)


def build(*args, **kwargs):
    """Would convert SimRecon output to the engine-neutral manifest. Always raises."""
    raise NotImplementedError(_MESSAGE)


def evaluate(*args, **kwargs):
    """Would score SimRecon output against common metrics. Always raises."""
    raise NotImplementedError(_MESSAGE)


def main(*args, **kwargs):
    raise NotImplementedError(_MESSAGE)


if __name__ == "__main__":
    main()
