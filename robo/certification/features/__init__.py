"""Task 12 -- feature extraction for the task-conditioned twin certificate.

Five feature-group modules (`visual.py`, `geometry.py`, `support_contact.py`,
`dynamics_probes.py`, `robot_control.py`), one per group in
`plan/12_TASK_CONDITIONED_CERTIFICATE.md`. Each group module is fully
self-contained (no shared private helper imported across files, on purpose --
see the "why not a shared _common.py" note below) and exposes:

  - `FEATURE_NAMES`: a fixed tuple of feature names this group always emits.
  - `extract_<group>_features(...) -> dict[str, float | None]`: always
    returns exactly `2 * len(FEATURE_NAMES)` keys -- `name` and
    `name + "_missing"` for every name in `FEATURE_NAMES`, on every call,
    regardless of how much evidence was available. `name`'s value is `None`
    (paired with `name_missing == 1.0`) whenever the underlying check had no
    evidence -- never a numeric stand-in, and never the value a favorable
    ("pass") verdict would have produced. See `visual.py`'s module docstring
    for the full missingness contract, shared verbatim by all five groups.

This module's job is purely compositional: `extract_all_features()` calls all
five and returns one flat dict; `ALL_FEATURE_NAMES` is their concatenated
name list (order fixed, alphabetical by group module name, for a
deterministic feature-matrix column order in `robo.certification.model`).

Why not a shared `_common.py` helper: this package's ownership boundary for
Task 12 is a fixed, enumerated file list (see the task brief) that does not
include an extra shared-helpers module, and each group's `_set`/`_objects`/
`_worst_numeric`/`_any_status` helpers are ~10 lines each -- duplicating them
per file avoids import-order fragility (this `__init__.py` importing group
submodules that in turn import back from the partially-initialized package)
for a negligible amount of repeated code.

Leakage-freedom: every group module reads only `build_audit` (Task 05),
`task_graph` (Task 11), and `alignment_report` (Task 04) dicts, and only
through named, allow-listed `.get()`/indexing calls on specific keys --
never a bare `**dict` passthrough, never a `scene_id`/`build_dir`/
`generated_at`/eval-result field. `task_graph.py`'s own module docstring
already guarantees its output contains no eval-result field by construction;
`build_audit.json` and `alignment_report.json` are QA/calibration artifacts
computed *before* any policy rollout exists, so neither can contain a real
success/reward value either. See `tests/test_certificate.py::test_feature_extraction_is_leakage_free`
for the executable proof (smuggle a bogus field into each input dict, assert
identical output with and without it -- the same pattern
`tests/test_task_graph.py::test_no_evaluation_result_leakage_in_role_refs`
uses for Task 11).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from robo.certification.features.visual import (
    FEATURE_NAMES as _VISUAL_NAMES,
    extract_visual_features,
)
from robo.certification.features.geometry import (
    FEATURE_NAMES as _GEOMETRY_NAMES,
    extract_geometry_features,
)
from robo.certification.features.support_contact import (
    FEATURE_NAMES as _SUPPORT_NAMES,
    extract_support_contact_features,
)
from robo.certification.features.dynamics_probes import (
    FEATURE_NAMES as _DYNAMICS_NAMES,
    extract_dynamics_probe_features,
)
from robo.certification.features.robot_control import (
    FEATURE_NAMES as _ROBOT_NAMES,
    extract_robot_control_features,
)

GROUP_FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    "visual": _VISUAL_NAMES,
    "geometry": _GEOMETRY_NAMES,
    "support_contact": _SUPPORT_NAMES,
    "dynamics_probes": _DYNAMICS_NAMES,
    "robot_control": _ROBOT_NAMES,
}

ALL_FEATURE_NAMES: tuple[str, ...] = tuple(
    name for group in GROUP_FEATURE_NAMES.values() for name in group
)


def graph_local_object_ids(task_graph: Mapping[str, Any] | None) -> set[str] | None:
    """The set of graph-local object-node ids from a
    `robo.certification.task_graph.build_graph()` JSON, i.e. exactly the
    objects Task 11 decided are relevant to this task
    (`eligible_object_ids`). Returns `None` (not an empty set) when no task
    graph was supplied, so downstream `extract_*_features(node_ids=None)`
    calls fall back to scene-global aggregation explicitly rather than
    silently certifying against zero objects."""
    if task_graph is None:
        return None
    return {n["node_id"] for n in (task_graph.get("nodes") or []) if n.get("kind") == "object"}


def extract_all_features(build_audit: Mapping[str, Any] | None,
                          task_graph: Mapping[str, Any] | None = None,
                          alignment_report: Mapping[str, Any] | None = None,
                          node_ids: Sequence[str] | None = "auto",
                          ) -> dict[str, float | None]:
    """Compose all five feature groups into one flat dict of
    `2 * len(ALL_FEATURE_NAMES)` keys.

    `node_ids="auto"` (the default) derives the graph-local object set from
    `task_graph` via `graph_local_object_ids` when a task graph is supplied,
    and falls back to `None` (scene-global) otherwise. Pass an explicit set
    (or `None`) to override.
    """
    if node_ids == "auto":
        node_ids = graph_local_object_ids(task_graph)

    out: dict[str, float | None] = {}
    out.update(extract_visual_features(build_audit, node_ids=node_ids,
                                        alignment_report=alignment_report))
    out.update(extract_geometry_features(build_audit, node_ids=node_ids,
                                          task_graph=task_graph,
                                          alignment_report=alignment_report))
    out.update(extract_support_contact_features(build_audit, node_ids=node_ids,
                                                 task_graph=task_graph))
    out.update(extract_dynamics_probe_features(build_audit, node_ids=node_ids))
    out.update(extract_robot_control_features(build_audit, task_graph=task_graph,
                                               alignment_report=alignment_report))

    expected = set(ALL_FEATURE_NAMES) | {f"{n}_missing" for n in ALL_FEATURE_NAMES}
    assert set(out) == expected, sorted(set(out) ^ expected)
    return out
