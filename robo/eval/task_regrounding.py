"""robo.eval.task_regrounding: re-resolve a task's object references --
built against ONE SimAny scene build's own `obj_NN` ids -- onto a
DIFFERENT, independently-reconstructed build of the SAME scene's own
`obj_NN` ids, by matching on semantic label/category evidence (with
geometry as a tie-breaker only), instead of assuming the two builds
enumerate/index objects the same way.

Why this exists (Task 09 follow-up): `robo/eval/simfoundry_condition.py`'s
own module docstring already documents this as a known, unfixed
limitation. `robo/tasks/pi05_tasks.py::generate` writes a task suite's
`target`/`receptacle` fields as raw `obj_NN` body names from the ONE build
it ran against (the SimAny factory reconstruction) -- `s3_lift.py` assigns
that `index` by enumeration order over THAT run's own detections, not by
any cross-run-stable id. A `simfoundry_repro` build of the same scene is a
genuinely independent reconstruction (its own single-view TRELLIS pass,
its own detection order/count), so its `obj_NN` names do not, in general,
name the same physical object. A real run
(`outputs/paired_runs/simfoundry_comparison_scripted/episode_ledger.jsonl`,
job 786581) measured 28/52 (54%) of that condition's episodes coming back
`Outcome.ENV_CRASH` purely from this index mismatch -- 12/12 (100%) on
scene `7b6477cb95`.

This module fixes the FALSE-NEGATIVE half of that: same physical object,
different index, wrongly treated as absent. It deliberately does NOT force
every task to resolve -- a task whose target genuinely was not detected by
the target build's (typically much coarser, single-view) reconstruction
stays unresolved, honestly, so it keeps failing exactly the way it always
did (a body-name lookup miss -> `Outcome.ENV_CRASH`, the residual bucket
`robo/eval/episode_log.py::classify_exception` funnels any unrecognized
exception into) -- now for a documented, inspectable reason instead of a
silent index coincidence.

Evidence model reuses `robo.certification.grounding`'s approach (built for
a related but distinct problem -- Task 11's task-ROLE grounding, "which
object plays the 'target' role" -- rather than this module's "which
TARGET-BUILD object is the SAME PHYSICAL OBJECT as SOURCE-BUILD object
X") for the same reason that module gives: never a silent, unlabeled
guess. `finalize_hypotheses` (ranking + resolved/ambiguous/unresolved
status via CONFIDENT_FLOOR/AMBIGUITY_MARGIN) is reused verbatim; the
evidence weights/tags below are this module's own -- identity matching is
a different question from role matching, so a different, smaller weight
table is appropriate. Category fallback is driven by
`agents.core.common.TARGET_PROMPTS`, the one concept/synonym table this
codebase already maintains for "which labels denote the same real-world
category" (its own docstring: "for eval matching ... "), rather than a
second, invented taxonomy.

KNOWN LIMITATION: the geometry tie-break ranks candidates RELATIVE to each
other (nearest of the label-matched set gets the full bonus), the same
normalization `grounding.py`'s own tie-break uses -- it does not verify
the "nearest" candidate is close in any absolute sense. If a target
build's single-view reconstruction has a spurious duplicate detection of
the same label with no real physical correspondence, and that phantom
happens to sit closer to the source object's position than the true
match, the tie-break can pick the phantom. This is a real, accepted
tradeoff (not something this pass claims to fully solve) -- geometry is
deliberately kept subordinate to label evidence and used only to
adjudicate among candidates that already cleared the label/category bar,
per the task's own instruction not to let geometry be a primary signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from robo.certification import grounding as G
from robo.tasks import pi05_tasks

# `agents.core.common` is intentionally NOT imported at module level: that
# module resolves SIMANY_SCENE/SIMANY_OUT/SIMANY_SCANNETPP_ROOT into
# module-level constants AT IMPORT TIME (see robo/eval/simfoundry_
# condition.py's module docstring, and robo/eval/reference_scene.py's
# `_decompose_similarity`, which duplicates 4 lines rather than import
# that module for the same reason). This module is imported at import
# time by robo.eval.simfoundry_condition, itself imported at import time
# by robo.eval.paired_runner -- becoming the first in-process importer of
# agents.core.common would freeze ITS constants against whatever env vars
# happen to be ambient at process start, before any scene-scoped
# SIMANY_SCENE/SIMANY_OUT is set, for the rest of the process. We only
# ever read the one STATIC dict below (TARGET_PROMPTS, independent of any
# of those env vars), so the import is deferred into `_synonym_group`
# instead.

# --------------------------------------------------------------- constants
ROLES = ("target", "receptacle")

#: Per-(task, role) `match_kind` -- the 3-way transparency field the task
#: report is built from ("direct index match, re-grounded by label, or
#: unresolved").
MATCH_DIRECT = "direct_index"          # the raw source-build id already
                                        # names the right object in the
                                        # target build -- no regrounding
                                        # was even necessary.
MATCH_REGROUNDED = "regrounded_label"  # a DIFFERENT target-build id was
                                        # confidently identified as the
                                        # same physical object.
MATCH_UNRESOLVED = "unresolved"        # no confident candidate -- either
                                        # truly absent, or only tied/weak
                                        # candidates (an "ambiguous"
                                        # `status`, folded into this same
                                        # match_kind: never a silent
                                        # guess).

EV_DIRECT_INDEX = "direct_index_match"
EV_LABEL_EXACT = "label_exact"
EV_CATEGORY_SYNONYM = "category_synonym"
EV_GEOMETRY_TIEBREAK = "geometry_nearest"

# Deliberately conservative, mirroring robo.certification.grounding's own
# comment on its weights: an exact label hit should decisively beat a
# category-only fallback; geometry alone must never be enough to nominate
# a candidate that didn't already clear a label bar -- it only breaks
# ties among candidates that did (_apply_geometry_tiebreak is never
# called with fewer than 2 hypotheses).
W_LABEL_EXACT = 0.90
W_CATEGORY_SYNONYM = 0.40
W_GEOMETRY_TIEBREAK = 0.25

_REPORT_FIELDS = ("match_kind", "status", "resolved_id", "confidence",
                  "evidence", "reason")


# ------------------------------------------------------------- label taxonomy
def _synonym_group(label: str) -> frozenset:
    """`agents.core.common.TARGET_PROMPTS` (concept -> [GT-label
    synonyms]) reused as undirected equivalence classes: "cup" and "mug"
    both belong to the "mug" entry's group, so a task built with
    target_label "cup" against one build can match a "mug"-labeled object
    in another. Falls back to a singleton group (matches only itself) for
    any label TARGET_PROMPTS doesn't know about -- this module never
    widens matching beyond that one explicit, already-maintained table.
    """
    from agents.core import common as C  # deferred -- see top-of-file note

    label = (label or "").strip().lower()
    for concept, synonyms in C.TARGET_PROMPTS.items():
        group = {concept.lower(), *(s.lower() for s in synonyms)}
        if label in group:
            return frozenset(group)
    return frozenset({label})


# ------------------------------------------------------------------ loading
def load_candidates(build_dir) -> list:
    """Structurally-valid object rows for `build_dir` -- the EXACT rows
    `robo.tasks.pi05_tasks._load_objects` (and, at rollout time,
    `env._task_rows`) already compute, so a regrounded reference is
    guaranteed to name a body that will actually exist in that build's
    exported scene.xml. Each row: {name, label, aabb, dims, mass, tier,
    bottom_z, center, drift} (see pi05_tasks._load_objects)."""
    return pi05_tasks._load_objects(Path(build_dir))


# --------------------------------------------------------------- evidence
def _add(hyps, key, weight, tag):
    h = hyps.setdefault(key, {"confidence": 0.0, "evidence": []})
    h["confidence"] += weight
    h["evidence"].append(tag)


def _apply_geometry_tiebreak(hyps, rows_by_name, source_center) -> None:
    """Tie-break ONLY -- callers never invoke this with fewer than 2
    hypotheses. Nearest candidate (world-frame centroid distance -- both
    builds anchor to the same scan/COLMAP frame; verified against real
    objects.json data, see module docstring) gets the full
    W_GEOMETRY_TIEBREAK bonus, farthest gets none, linear in between --
    the same rank-normalization `grounding.py`'s own (private)
    `_apply_geometry_bonus` uses, reimplemented locally rather than
    reaching into that underscore-prefixed helper across modules."""
    if source_center is None:
        return
    names = list(hyps)
    dists = {n: float(np.linalg.norm(np.asarray(rows_by_name[n]["center"], dtype=float)
                                      - np.asarray(source_center, dtype=float)))
             for n in names}
    dmin, dmax = min(dists.values()), max(dists.values())
    span = max(dmax - dmin, 1e-9)
    for n in names:
        closeness = 1.0 - (dists[n] - dmin) / span  # 1.0 nearest, 0.0 farthest
        bonus = W_GEOMETRY_TIEBREAK * closeness
        if bonus > 1e-9:
            hyps[n]["confidence"] += bonus
            hyps[n]["evidence"].append(EV_GEOMETRY_TIEBREAK)


# -------------------------------------------------------------- resolution
def ground_object(source_row: dict, target_rows: list) -> dict:
    """Find the object in `target_rows` (a DIFFERENT build's
    `load_candidates` output) that is the SAME PHYSICAL OBJECT as
    `source_row` (one row of the SOURCE build's own `load_candidates`
    output). Never fabricates a match: `resolved_id` is None whenever
    `status` != "resolved" (an "ambiguous" top hypothesis is still
    reported in `candidates`, for transparency, but never promoted to
    `resolved_id`).

    Returns {"match_kind", "status", "resolved_id", "confidence",
             "evidence", "reason", "candidates"}.
    """
    wanted = (source_row.get("label") or "").strip().lower()
    rows_by_name = {r["name"]: r for r in target_rows}

    # Cheapest, highest-confidence check first: maybe the raw source id
    # happens to still be correct in the target build (e.g. a small scene
    # where enumeration order coincided) -- then no label/geometry
    # reasoning is even needed.
    direct = rows_by_name.get(source_row["name"])
    if direct is not None and (direct["label"] or "").strip().lower() == wanted:
        cand = {"object_id": direct["name"], "confidence": 1.0,
                "evidence": [EV_DIRECT_INDEX]}
        return {"match_kind": MATCH_DIRECT, "status": "resolved",
                "resolved_id": direct["name"], "confidence": 1.0,
                "evidence": [EV_DIRECT_INDEX],
                "reason": f"{source_row['name']!r} already names a "
                          f"{wanted!r} in the target build -- no "
                          f"regrounding needed",
                "candidates": [cand]}

    hyps: dict = {}
    for row in target_rows:
        label = (row["label"] or "").strip().lower()
        if not label:
            continue
        if label == wanted:
            _add(hyps, row["name"], W_LABEL_EXACT, EV_LABEL_EXACT)
        elif label in _synonym_group(wanted):
            _add(hyps, row["name"], W_CATEGORY_SYNONYM, EV_CATEGORY_SYNONYM)

    if len(hyps) > 1:
        _apply_geometry_tiebreak(hyps, rows_by_name, source_row.get("center"))

    resolution = G.finalize_hypotheses("regrounded_object", hyps, multi_select=False)
    candidates = resolution["hypotheses"]

    if resolution["status"] == "unresolved":
        known_labels = sorted({(r["label"] or "").strip().lower() for r in target_rows})
        return {"match_kind": MATCH_UNRESOLVED, "status": "unresolved",
                "resolved_id": None, "confidence": 0.0, "evidence": [],
                "reason": f"no object in the target build labeled "
                          f"{wanted!r} or a category/synonym match for it "
                          f"(target build labels present: {known_labels})",
                "candidates": candidates}

    top = candidates[0]
    if resolution["status"] == "ambiguous":
        second = candidates[1] if len(candidates) > 1 else None
        detail = (f"runner-up {second['object_id']!r} conf={second['confidence']:.2f}"
                  if second else "no runner-up -- top candidate is below "
                                 "the confident floor on its own")
        return {"match_kind": MATCH_UNRESOLVED, "status": "ambiguous",
                "resolved_id": None, "confidence": top["confidence"],
                "evidence": top["evidence"],
                "reason": f"{len(candidates)} candidate(s) for {wanted!r} "
                          f"not decisively resolved (top "
                          f"{top['object_id']!r} conf={top['confidence']:.2f}, "
                          f"{detail}) -- refusing to guess",
                "candidates": candidates}

    return {"match_kind": MATCH_REGROUNDED, "status": "resolved",
            "resolved_id": top["object_id"], "confidence": top["confidence"],
            "evidence": top["evidence"],
            "reason": f"regrounded {source_row['name']!r} ({wanted!r}) -> "
                      f"{top['object_id']!r} via {sorted(top['evidence'])}",
            "candidates": candidates}


# --------------------------------------------------------------- suite-level
def ground_task_suite(tasks: list, source_build_dir, target_build_dir) -> tuple:
    """Re-resolve every `target`/`receptacle` reference in `tasks` (built
    against `source_build_dir`'s own object ids -- see
    `robo.tasks.pi05_tasks.generate`'s task schema) onto
    `target_build_dir`'s own object ids.

    Returns (regrounded_tasks, report):

      regrounded_tasks -- shallow copies of `tasks`. A role that resolves
        (match_kind DIRECT or REGROUNDED) has its id rewritten, plus
        `target_label` / `receptacle_dims` refreshed from the TARGET
        build's own measurement (needed for
        `pi05_tasks.TaskScorer`'s any_instance fallback and region
        sizing to stay correct against THIS build). A role that does NOT
        resolve is left BYTE-IDENTICAL to the input -- it fails
        downstream exactly as it already did. `task["instructions"]`
        (the language prompt handed to the policy) is NEVER touched: the
        mujoco_paired protocol requires the prompt stay frozen across
        conditions, only scene construction may vary.

      report -- one row per (task_id, role) actually referenced:
        {"task_id", "role", "source_id", "source_label", "match_kind",
         "status", "resolved_id", "confidence", "evidence", "reason"}.
    """
    source_rows = {r["name"]: r for r in load_candidates(source_build_dir)}
    target_rows = load_candidates(target_build_dir)
    rows_by_target_name = {r["name"]: r for r in target_rows}

    out_tasks, report = [], []
    for task in tasks:
        new_task = dict(task)
        for role in ROLES:
            ref = task.get(role)
            if not ref:
                continue
            source_row = source_rows.get(ref)
            if source_row is None:
                report.append({
                    "task_id": task["task_id"], "role": role,
                    "source_id": ref, "source_label": None,
                    "match_kind": MATCH_UNRESOLVED, "status": "unresolved",
                    "resolved_id": None, "confidence": 0.0, "evidence": [],
                    "reason": f"{ref!r} not found in the source build's "
                              f"own object rows (internal inconsistency) "
                              f"-- left unchanged",
                })
                continue

            result = ground_object(source_row, target_rows)
            report.append({
                "task_id": task["task_id"], "role": role,
                "source_id": ref, "source_label": source_row["label"],
                **{k: result[k] for k in _REPORT_FIELDS},
            })

            if result["resolved_id"] is None:
                continue  # leave new_task[role] untouched -- see docstring

            resolved_row = rows_by_target_name[result["resolved_id"]]
            if role == "target":
                new_task["target"] = resolved_row["name"]
                new_task["target_label"] = resolved_row["label"]
            else:
                new_task["receptacle"] = resolved_row["name"]
                new_task["receptacle_dims"] = resolved_row["dims"].tolist()
        out_tasks.append(new_task)
    return out_tasks, report


__all__ = [
    "MATCH_DIRECT", "MATCH_REGROUNDED", "MATCH_UNRESOLVED",
    "load_candidates", "ground_object", "ground_task_suite",
]
