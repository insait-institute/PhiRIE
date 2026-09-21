"""ReplicateAnyScene adapter -- STUB. No matching system was identified.

"ReplicateAnyScene" appears only in plan/15_CONSTRUCTION_BASELINES.md's
baseline list. It does not appear anywhere in docs/BASELINES.md's tiered
survey (sub-problems A-G) or in agents/baselines/README.md, and a
repo-wide, case-insensitive search for the name and close variants
("replicate any scene", "replicate.any.scene") during the 2026-08-16
baseline audit found only the plan file itself. See
`baselines/release_status.yaml` (status: unavailable) for the full note.

Rather than guess which surveyed sub-problem-A system (SimRecon, Image2Sim,
EmbodiedGen V2, SimuScene, Video2Game, MetaScenes, ...) this informal name
might refer to and silently substitute it, this module contains NO
reimplementation and NO numeric computation of any kind -- inventing a
paper/system match here would itself be the kind of fabrication
plan/15_CONSTRUCTION_BASELINES.md's acceptance criteria forbid ("Mark
unsupported cells N/A, never zero"). Every public entry point below raises
`NotImplementedError` immediately, unconditionally, regardless of
arguments. `tests/test_baseline_release_status.py` enforces that this stays
true.

If this baseline is identified in future (a specific paper + public code):
replace this stub with a real adapter, pin the commit/weights/license in
`baselines/release_status.yaml` (flip `status` to `implemented` or
`partial`) and in `docs/BASELINE_REPRODUCTION.md`, and delete this
docstring's caveats.
"""

_MESSAGE = (
    "ReplicateAnyScene has no identified paper or public code release as of "
    "the 2026-08-16 baseline audit (see baselines/release_status.yaml, "
    "status: unavailable -- the name does not appear in docs/BASELINES.md "
    "or agents/baselines/README.md). This adapter is an intentional "
    "placeholder and must not be called; it exists only so the "
    "construction-baseline table's ReplicateAnyScene row is explicit N/A "
    "rather than a silently-missing or fabricated entry."
)


def run(*args, **kwargs):
    """Would run the ReplicateAnyScene pipeline end-to-end. Always raises."""
    raise NotImplementedError(_MESSAGE)


def build(*args, **kwargs):
    """Would convert output to the engine-neutral manifest. Always raises."""
    raise NotImplementedError(_MESSAGE)


def evaluate(*args, **kwargs):
    """Would score output against common metrics. Always raises."""
    raise NotImplementedError(_MESSAGE)


def main(*args, **kwargs):
    raise NotImplementedError(_MESSAGE)


if __name__ == "__main__":
    main()
