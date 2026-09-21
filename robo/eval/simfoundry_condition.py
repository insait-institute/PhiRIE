"""robo.eval.simfoundry_condition: turns a finished `baselines/
simfoundry_repro.py` output directory into the exact same MJCF export shape
`robo/sim/export_mjcf.py` produces for the "simany" condition, so
`robo.eval.paired_runner` can run it through the IDENTICAL episode/manifest/
rollout machinery it already uses for "simany" and "reference" -- the third
`mujoco_paired` condition (`configs/experiments/icra_contract_v1.yaml`'s
`mujoco_paired.admissible_baselines` now includes `simfoundry_repro`; design
context: `docs/MUJOCO_PAIRED_PROTOCOL.md`).

Why this is thin: `baselines/simfoundry_repro.py` composes
`agents.discover.s0_select_frame` through `robo.sim.s7_sim` -- the SAME
stage sequence, same modules, in the same order, as `run/run_simfoundry.sh`
-- so a completed build has the SAME `objects/objects.json` +
`objects/obj_*/{aligned,physics}.json` + `object.urdf` + `collision/part_*
.obj` layout any SimAny factory build has. `robo/sim/export_mjcf.py` reads
ONLY that layout (plus GT instances for its default room-collision carving,
available here because this baseline is only admissible on real ScanNet++
scenes -- see the scope note in `baselines/simfoundry_repro.py`); it has no
idea which pipeline configuration produced its inputs. So turning a
completed simfoundry_repro build into a `scene.xml` is exactly the SAME
`export_mjcf.py` invocation the full pipeline already runs for "simany",
just pointed (via `SIMANY_OUT`/`SIMANY_SCENE`) at the baseline's own output
directory -- no new exporter, no new MJCF-writing logic lives here.

Run as a SUBPROCESS, not an in-process import: `agents.core.common` (which
`export_mjcf.py` imports) resolves `SIMANY_OUT`/`SIMANY_SCENE`/
`SIMANY_SCANNETPP_ROOT` into module-level constants AT IMPORT TIME. A
second in-process call with different env vars would silently reuse
whichever scene the FIRST import in this interpreter resolved -- exactly
the "string-path regression" class of bug this codebase has already been
bitten by once (see the PhiRoom reorg notes). `baselines/simfoundry_repro.py`
sidesteps the same hazard the same way, for the same reason.

FORMERLY a known, unfixed limitation, now MITIGATED (not fully solved) by
`robo.eval.task_regrounding`: simfoundry_repro is a genuinely independent
reconstruction (its own single-view TRELLIS pass, its own
`agents.discover.s3_lift` detection order) -- its `obj_NN` body names are
NOT guaranteed to name the same physical object as the SimAny factory
build's `obj_NN` (`s3_lift.py` assigns `index` by enumeration over that
run's OWN detections, not by any cross-run-stable ID; only `gt_object_id`,
which the regrounding module does not consult either -- it is a debug/
eval-only field, not something a real independent reconstruction could
rely on -- would be cross-run-stable). A real run measured 28/52 (54%) of
this condition's episodes coming back `Outcome.ENV_CRASH` purely from this
index mismatch (12/12 on scene `7b6477cb95`).

`resolve_condition`, when passed the shared task suite via its `suite`
kwarg, now re-resolves each task's `target`/`receptacle` reference onto
THIS build's own object ids by semantic label/category match (with
geometry as a tie-breaker only) via `robo.eval.task_regrounding` --
`robo.eval.paired_runner.build_env` passes `suite=suite` for exactly this
reason. `robo.eval.paired_runner` still reuses the SAME frozen task
suite's robot/camera/rubric/table/language across all three conditions per
the mujoco_paired contract (`ground_task_suite` never touches those
fields, or `task["instructions"]` -- only the target/receptacle
object-id bookkeeping). A task whose reference genuinely cannot be
regrounded confidently (the object was never detected in this
reconstruction, or only ambiguous/tied candidates exist) is left pointing
at the SOURCE build's id, unchanged, so it surfaces as `Outcome.ENV_CRASH`
exactly as it always did when the scorer looks it up -- now for a
documented, inspectable reason (see
`<baseline_dir>/task_regrounding_report.json`), not a silent index
coincidence.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from robo.eval import task_regrounding as tregr

ROOT = Path(__file__).resolve().parents[2]

MANIFEST_FILENAME = "simfoundry_repro_manifest.json"
TASK_REGROUNDING_REPORT_FILENAME = "task_regrounding_report.json"
DEFAULT_INTERPRETER = str(ROOT / ".venv" / "bin" / "python")
DEFAULT_EXPORT_TIMEOUT_S = 1800
_FACTORY_SUFFIX = "_factory"


class SimFoundryReproNotReady(RuntimeError):
    """A simfoundry_repro output dir has no completed (`build_success:
    true`) manifest yet -- either the construction job hasn't finished, has
    not started, or finished with a failure. `robo.eval.paired_runner.
    build_env` lets this funnel into the SAME `elog.BuildFailureError` path
    every other build problem uses (Task 09 step 2: "coverage failure, not
    a skip"), so this exception's message is written verbatim into that
    failure's ledger row -- keep it self-explanatory."""


def _bare_scene_id(scene_id: str) -> str:
    """"c50d2d1d42_factory" -> "c50d2d1d42". A no-op on an id that doesn't
    carry the SimAny factory-build suffix."""
    return (scene_id[: -len(_FACTORY_SUFFIX)]
            if scene_id.endswith(_FACTORY_SUFFIX) else scene_id)


def default_baseline_dir(scene_id: str) -> Path:
    """`outputs/<bare_scene>_baselines/simfoundry_repro/` -- the exact
    layout `run/slurm/simfoundry_baseline.sbatch` writes to (and where jobs
    784635-784640 are writing today). `scene_id` may carry paired_runner's
    own `scene_cfg["id"]` convention (e.g. "c50d2d1d42_factory"); the
    baseline dir is keyed by the bare scene id regardless.
    """
    return ROOT / "outputs" / f"{_bare_scene_id(scene_id)}_baselines" / "simfoundry_repro"


def load_manifest(baseline_dir) -> "dict | None":
    """None if the manifest file doesn't exist yet, or can't be parsed (a
    build still mid-write) -- never an exception. The caller
    (`require_complete_build`) is what turns "no manifest yet" into a
    loud, classified failure; this helper alone stays a plain lookup."""
    p = Path(baseline_dir) / MANIFEST_FILENAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def is_build_complete(manifest: "dict | None") -> bool:
    return bool(manifest) and manifest.get("build_success") is True


def require_complete_build(baseline_dir) -> dict:
    """Return the parsed manifest, or raise `SimFoundryReproNotReady` with
    a message that names exactly what's missing/unfinished -- never a bare
    `FileNotFoundError`/`KeyError` surfacing from deeper in the export
    pipeline with no context about WHICH scene/condition it was for."""
    baseline_dir = Path(baseline_dir)
    manifest = load_manifest(baseline_dir)
    if manifest is None:
        raise SimFoundryReproNotReady(
            f"no {MANIFEST_FILENAME} under {baseline_dir} yet -- the "
            f"simfoundry_repro construction job for this scene has not "
            f"finished (or has not started)")
    if not is_build_complete(manifest):
        failed = next((s for s in manifest.get("stage_log", [])
                       if s.get("returncode") not in (0, None)), None)
        reason = (f"stage {failed['stage']!r} ({failed['module']}) failed "
                  f"with returncode {failed['returncode']}" if failed
                  else f"build_success={manifest.get('build_success')!r}")
        raise SimFoundryReproNotReady(
            f"{baseline_dir / MANIFEST_FILENAME} exists but is not a "
            f"completed build ({reason})")
    return manifest


def scene_export_paths(baseline_dir) -> dict:
    exp = Path(baseline_dir) / "sim_export"
    return {"scene_xml": exp / "scene.xml", "settle_json": exp / "mujoco_settle.json"}


def build_scene_export(baseline_dir, scene_id, *, scannetpp_root=None,
                        collision_mode="room", force=False,
                        interpreter=None,
                        timeout_s=DEFAULT_EXPORT_TIMEOUT_S) -> dict:
    """Ensure `<baseline_dir>/sim_export/scene.xml` (+ `mujoco_settle.json`)
    exists, by invoking `robo.sim.export_mjcf --test` -- the SAME module the
    "simany" condition's own factory build already ran -- as a subprocess
    (see module docstring for why not an in-process import).

    Idempotent: a no-op (returns immediately, `skipped=True`) if
    `scene.xml` already exists and `force` is False -- mirrors `run/env.sh`
    's own `done_skip()` convention already used by every stage in this
    pipeline, so a re-invocation against an already-exported baseline is
    cheap.
    """
    baseline_dir = Path(baseline_dir).resolve()
    paths = scene_export_paths(baseline_dir)
    if paths["scene_xml"].exists() and not force:
        return {**paths, "skipped": True, "returncode": 0, "stderr_tail": None}

    env = dict(os.environ)
    env["SIMANY_OUT"] = str(baseline_dir)
    env["SIMANY_SCENE"] = _bare_scene_id(scene_id)
    if scannetpp_root:
        env["SIMANY_SCANNETPP_ROOT"] = str(scannetpp_root)
    interp = interpreter or env.get("SIMANY_PY", DEFAULT_INTERPRETER)

    proc = subprocess.run(
        [interp, "-m", "robo.sim.export_mjcf", "--test",
         "--collision-mode", collision_mode],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout_s)
    return {**paths, "skipped": False, "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-4000:] if proc.returncode != 0 else None}


def instance_inventory(baseline_dir):
    """[{object_id, label, tier, asset_hash}], the SAME shape
    `robo.eval.reference_scene.instance_inventory` returns for the
    "reference" condition, over THIS condition's own reconstruction. Used
    by `robo.eval.paired_runner._scene_manifest_hash` so a
    "simfoundry_repro" scene_manifest_hash is built the same way the other
    two conditions' are: real declared scene-construction facts, not a
    placeholder. `asset_hash` is always None here -- no per-object mesh
    hash is computed by this glue layer (export_mjcf.py's own
    isaac_manifest.json, written alongside scene.xml, is the authoritative
    per-object record for this build if a caller ever needs more).
    """
    baseline_dir = Path(baseline_dir)
    objects_json = baseline_dir / "objects" / "objects.json"
    if not objects_json.exists():
        return []
    objects = json.loads(objects_json.read_text())
    rows = []
    for m in objects:
        name = f"obj_{m['index']:02d}"
        odir = baseline_dir / "objects" / name
        af = odir / "aligned.json"
        if not af.exists():
            continue
        al = json.loads(af.read_text())
        if al.get("rejected") or not (odir / "object.urdf").exists():
            continue
        rows.append({"object_id": name, "label": m.get("label", "object"),
                     "tier": al.get("tier", "A"), "asset_hash": None})
    return rows


def _source_build_dir(scene_cfg: dict, suite: dict) -> Path:
    """The SOURCE (simany factory) build a task suite's target/receptacle
    ids were originally minted against -- `scene_cfg["factory_dir"]` when
    declared (the SAME default `robo.eval.paired_runner.build_env` itself
    applies), else derived from `suite["scene_xml"]` (`<factory_dir>/
    sim_export/scene.xml`, `robo.tasks.pi05_tasks.generate`'s own
    convention -- `_bare_scene_id`-style, no need to duplicate the
    tasks_json-path arithmetic build_env does)."""
    fd = scene_cfg.get("factory_dir")
    if fd:
        return Path(fd)
    return Path(suite["scene_xml"]).resolve().parents[1]


def ground_task_suite_for_condition(scene_cfg: dict, suite: dict, baseline_dir) -> tuple:
    """Re-resolve `suite["tasks"]` (built against the SOURCE simany
    build's own object ids) onto THIS simfoundry_repro build's own object
    ids (`robo.eval.task_regrounding.ground_task_suite`), persist the
    per-task transparency report to
    `<baseline_dir>/TASK_REGROUNDING_REPORT_FILENAME`, and return
    `(regrounded_suite, report)`. A task whose reference cannot be
    confidently regrounded is left pointing at the SOURCE build's id, so
    it fails exactly the way it always did (`Outcome.ENV_CRASH` via a
    body-name lookup miss) -- see `robo.eval.task_regrounding`'s module
    docstring.
    """
    source_dir = _source_build_dir(scene_cfg, suite)
    tasks, report = tregr.ground_task_suite(suite["tasks"], source_dir, baseline_dir)
    regrounded_suite = {**suite, "tasks": tasks}
    report_path = Path(baseline_dir) / TASK_REGROUNDING_REPORT_FILENAME
    report_path.write_text(json.dumps(report, indent=1))
    return regrounded_suite, report


def resolve_condition(scene_cfg: dict, *, suite: "dict | None" = None) -> dict:
    """The single entry point `robo.eval.paired_runner.build_env` calls for
    `condition == "simfoundry_repro"`. `scene_cfg` may declare an explicit
    `simfoundry_repro_dir` (and `simfoundry_scene_id`/`scannetpp_root`/
    `collision_mode` overrides); otherwise `default_baseline_dir(scene_cfg
    ["id"])` is used. Raises `SimFoundryReproNotReady` (never a bare crash)
    if the build isn't finished; otherwise ensures the MJCF export exists
    (building it on first use, cached after) and returns
    `{"baseline_dir", "scene_xml", "settle_json", "manifest"}`.

    `suite`, when given, is the ORIGINAL (shared, simany-build-object-id)
    task suite `robo.eval.paired_runner.build_env` loaded before
    dispatching on condition. When present, this call ALSO regrounds every
    task's target/receptacle reference onto THIS condition's own object
    ids (`ground_task_suite_for_condition`, above), adding
    `"suite"` (the regrounded suite) and `"task_regrounding_report"` to
    the returned dict. Omitting `suite` (the default) skips regrounding
    entirely -- kept optional so any other caller's existing
    `resolve_condition(scene_cfg)` usage is unaffected.
    """
    baseline_dir = Path(scene_cfg.get("simfoundry_repro_dir")
                        or default_baseline_dir(scene_cfg["id"]))
    manifest = require_complete_build(baseline_dir)
    scene_id = scene_cfg.get("simfoundry_scene_id", scene_cfg["id"])
    build = build_scene_export(
        baseline_dir, scene_id,
        scannetpp_root=scene_cfg.get("scannetpp_root"),
        collision_mode=scene_cfg.get("collision_mode", "room"))
    if build["returncode"] != 0:
        raise SimFoundryReproNotReady(
            f"export_mjcf.py failed for {baseline_dir} "
            f"(scene_id={scene_id!r}): {build['stderr_tail']}")
    if not build["scene_xml"].exists():
        raise SimFoundryReproNotReady(
            f"export_mjcf.py reported success but {build['scene_xml']} "
            f"still does not exist for {baseline_dir}")
    out = {"baseline_dir": baseline_dir, "scene_xml": build["scene_xml"],
           "settle_json": build["settle_json"], "manifest": manifest}
    if suite is not None:
        out["suite"], out["task_regrounding_report"] = ground_task_suite_for_condition(
            scene_cfg, suite, baseline_dir)
    return out


__all__ = [
    "SimFoundryReproNotReady",
    "MANIFEST_FILENAME",
    "TASK_REGROUNDING_REPORT_FILENAME",
    "default_baseline_dir",
    "load_manifest",
    "is_build_complete",
    "require_complete_build",
    "scene_export_paths",
    "build_scene_export",
    "instance_inventory",
    "ground_task_suite_for_condition",
    "resolve_condition",
]
