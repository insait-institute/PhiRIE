"""Task 12 -- Dynamics probes feature group.

Plan group 4: "standardized push, lift, release, grasp closure/retention
response."

Honest status as of this task: no upstream component anywhere in this repo
runs a standardized push, lift, or grasp-closure/retention probe (checked by
grepping the tree for ik/collision-margin/push/lift/grasp/replay probes
before writing this module -- none exist outside third_party/ and unrelated
GPU-monitoring constants). The only *active* perturbation this pipeline
currently runs on a reconstructed object is `agents/eval/factory_report.py`'s
/ `robo/sim/redrop.py`'s drop test (release from ~5mm and settle) --
already consumed as `support_drop_unstable_any` in `support_contact.py`.

Rather than fabricate push/lift/grasp-closure numbers or silently reuse the
drop test a second time under a new name (which would double-count one piece
of evidence as if it were several independent probes -- exactly what the
plan's "never impute missing evidence as good" spirit rules out), this
module forward-declares the four probe feature names now, all honestly
`not_applicable` (value=None, `_missing=1.0`), the same pattern
`agents/eval/build_audit.py` itself uses for its own not-yet-built Task 04/06
checks (see docs/BUILD_AUDIT_SCHEMA.md sections 7-8: "so a reader ... knows
this is a missing upstream component, not a build defect"). `release` is the
one exception: it is a real reuse of the drop test's own stability verdict
(the drop test *is* a release probe), reported here as `dyn_release_stability`
so the certificate has at least one populated feature in this group, with a
docstring note that it is the *same* underlying evidence as
`support_contact.support_drop_unstable_any`, not a new measurement -- a
regularized model is free to learn that the two are redundant; that is a
much safer failure mode than a feature module inventing evidence.

When a real push/lift/grasp-closure probe (`robo/sim/dynamics_probe.py` or
similar) lands, this module should stop forward-declaring
`dyn_push_response_score` / `dyn_lift_response_score` /
`dyn_grasp_closure_retention_score` and read its output the same way
`support_contact.py` reads the drop test.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

FEATURE_NAMES: tuple[str, ...] = (
    "dyn_push_response_score",
    "dyn_lift_response_score",
    "dyn_release_stability",
    "dyn_grasp_closure_retention_score",
)


def _set(out: dict, name: str, value) -> None:
    out[name] = None if value is None else float(value)
    out[f"{name}_missing"] = 1.0 if value is None else 0.0


def _objects(build_audit: Mapping[str, Any] | None, node_ids: Sequence[str] | None):
    objs = (build_audit or {}).get("objects") or []
    if node_ids is None:
        return objs
    node_ids = set(node_ids)
    return [o for o in objs if o.get("object_id") in node_ids]


def _obj_check(obj: Mapping[str, Any], name: str) -> dict:
    return ((obj.get("checks") or {}).get(name)) or {}


def _release_stability(objs) -> float | None:
    """1.0 if any in-scope object's drop test reports `stable: false`
    (unstable release), 0.0 if every in-scope object with drop-test evidence
    was stable, `None` if no drop-test evidence exists at all in scope. This
    is literally `agents/eval/build_audit.py`'s `drop_response` check --
    reused here under this group's own name, not recomputed."""
    saw_evidence = False
    for o in objs:
        c = _obj_check(o, "drop_response")
        if c.get("status") == "not_applicable":
            continue
        saw_evidence = True
        if c.get("status") == "fail":
            return 1.0
    return 0.0 if saw_evidence else None


def extract_dynamics_probe_features(build_audit: Mapping[str, Any] | None,
                                     node_ids: Sequence[str] | None = None
                                     ) -> dict[str, float | None]:
    build_audit = build_audit or {}
    objs = _objects(build_audit, node_ids)
    out: dict[str, float | None] = {}

    # Forward-declared, no upstream probe component exists yet -- see
    # module docstring.
    _set(out, "dyn_push_response_score", None)
    _set(out, "dyn_lift_response_score", None)
    _set(out, "dyn_grasp_closure_retention_score", None)

    _set(out, "dyn_release_stability", _release_stability(objs))

    assert set(out) == set(FEATURE_NAMES) | {f"{n}_missing" for n in FEATURE_NAMES}
    return out
