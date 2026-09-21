"""Read-only final-state checks for native target-only policy records.

Official success is NEVER replaced. Surface-centroid distances and containment
inside declared native goal parallelepipeds are supplementary geometric probes,
not a new task rubric or proof of collision-free whole-mesh containment.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def surface_centroid(vertices, faces):
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, int)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("finite world vertices required")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("triangle faces required")
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) * .5
    if area.sum() <= 0 or not np.isfinite(area).all():
        raise ValueError("nondegenerate visual surface required")
    return np.average(triangles.mean(axis=1), axis=0, weights=area)


def points_in_regions(vertices, regions, tolerance=1e-9):
    """Membership in a UNION of parallelepipeds, with normalized coordinates.

    This deliberately does not duplicate native obj_inside_of's unnormalized
    threshold. For a nonconvex union, vertices inside do not certify every face.
    """
    vertices = np.asarray(vertices, float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all() or not len(vertices):
        raise ValueError("nonempty finite Nx3 vertices required")
    if not regions:
        return None
    inside = np.zeros(len(vertices), dtype=bool)
    for region in regions:
        p = np.asarray(region, float)
        if p.shape != (4, 3) or not np.isfinite(p).all():
            raise ValueError("goal region needs origin and three axis endpoints")
        basis = (p[1:] - p[0]).T
        if abs(np.linalg.det(basis)) < 1e-12:
            raise ValueError("degenerate native goal region")
        coords = np.linalg.solve(basis, (vertices - p[0]).T).T
        inside |= np.all((coords >= -tolerance) & (coords <= 1 + tolerance), axis=1)
    return inside


def geometric_diagnostics(vertices, faces, gripper, native_origin, regions=()):
    center = surface_centroid(vertices, faces)
    gripper, native_origin = np.asarray(gripper, float), np.asarray(native_origin, float)
    if gripper.shape != (3,) or native_origin.shape != (3,) or not np.isfinite([gripper, native_origin]).all():
        raise ValueError("finite world gripper and native origin required")
    membership = points_in_regions(vertices, regions)
    dc = float(np.linalg.norm(center - gripper)); do = float(np.linalg.norm(native_origin - gripper))
    return {
        "visual_surface_centroid_world_m": center.tolist(),
        "gripper_to_visual_centroid_m": dc, "gripper_to_body_origin_m": do,
        "body_origin_to_visual_centroid_m": float(np.linalg.norm(center - native_origin)),
        "retreat_threshold_m": .25,
        "centroid_retreat_pass": dc > .25, "origin_retreat_pass": do > .25,
        "retreat_reference_point_disagrees": (dc > .25) != (do > .25),
        "goal_region_available": membership is not None,
        "visual_vertices_inside_goal_fraction": None if membership is None else float(membership.mean()),
        "all_visual_vertices_in_goal_regions": None if membership is None else bool(membership.all()),
        "goal_region_scope": "native declared parallelepiped union; not cavity or collision proof",
        "official_success_modified": False,
    }


def load_trace(path):
    p = Path(path)
    text = gzip.open(p, "rt").read() if p.suffix == ".gz" else p.read_text()
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(rows, dict):
        rows = rows.get("ticks", rows.get("records", rows.get("timeseries")))
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty canonical tick trace required")
    if any(row.get("tick") != i for i, row in enumerate(rows)):
        raise ValueError("trace must contain the original contiguous ticks")
    return rows


def object_directory(unit, episode):
    if unit["controller_method"] == "REF_NATIVE":
        return None
    imported = json.loads((episode.parent / "import_receipt.json").read_text())
    sources = imported.get("source_hashes", {})
    aligned = [Path(p) for p in sources if Path(p).name == "aligned.json"]
    if len(aligned) != 1:
        raise ValueError("target source not uniquely bound by original import receipt")
    for path, digest in sources.items():
        if sha(path) != digest:
            raise ValueError("original imported asset changed: " + path)
    return aligned[0].parent


def native_goal_regions(env, task):
    if task == "PickPlaceCounterToSink":
        fixture = env.sink
    elif task == "PickPlaceCounterToCabinet":
        fixture = env.cab
    else:
        return [], "UNAVAILABLE: no declared native fixture parallelepiped for this task"
    fixture = env.get_fixture(fixture)
    sites = fixture.get_int_sites(relative=False)
    regions = [np.asarray(value, float).tolist() for value in sites.values()
               if np.asarray(value).shape == (4, 3)]
    return regions, "AVAILABLE" if regions else "UNAVAILABLE: no four-site region"


def evaluate_unit(unit):
    """Restore in a NEW evaluator process; set final qpos/qvel, never step physics."""
    from robo.roundtrip.adapters.robocasa import RoboCasaAdapter
    from robo.roundtrip.paired import prepare_paired_adapter
    from robo.roundtrip.scorer import predicate_components
    from robo.roundtrip.fidelity_native import visual_surface
    from robo.manifest.hash import canonical_hash

    if unit["scope"] != "L0_target_only":
        raise ValueError("this evaluator supports L0 only; L1 needs its frozen fixture adapter")
    episode = Path(unit["result_path"]).resolve(strict=True).parent
    result = json.loads((episode / "result.json").read_text())
    config = json.loads(Path(unit["config_path"]).read_text())
    if canonical_hash(config) != result["config_sha256"]:
        raise ValueError("original config/result mismatch")
    trace_path = episode / "trace.json.gz"; trace = load_trace(trace_path)
    if len(trace) != result["ticks"] or not result["executed"] or result["error"] is not None:
        raise ValueError("complete executed policy trace required for final-state diagnostics")
    final = trace[-1]
    root = Path(unit["bundle_dir"])
    source = {"state": json.loads((root / "canonical_state.json").read_text()),
              "xml": (root / "scene.xml").read_text(),
              "provenance": {"reset_seed": config["reset_seeds"][0]}}
    adapter = RoboCasaAdapter(config)
    original_sources = {str(p): sha(p) for p in (episode / "result.json", trace_path,
                        episode.parent / "import_receipt.json", Path(unit["config_path"]))}
    try:
        prepare_paired_adapter(adapter, source, object_dir=object_directory(unit, episode))
        if hashlib.sha256(adapter.source_xml().encode()).hexdigest() != result["imported_xml_sha256"]:
            raise ValueError("reconstructed evaluator XML differs from original episode")
        sim = adapter.native.sim
        qpos, qvel = np.asarray(final["qpos"], float), np.asarray(final["qvel"], float)
        if qpos.shape != sim.data.qpos.shape or qvel.shape != sim.data.qvel.shape:
            raise ValueError("final state topology differs")
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("final state is nonfinite")
        sim.data.qpos[:] = qpos; sim.data.qvel[:] = qvel
        sim.data.time = float(final["simulation_time_s"])
        sim.forward()  # Derived contacts/kinematics only; not integration or policy execution.
        tracked = adapter.tracked_objects()
        for role, pose in final["objects"].items():
            if role not in tracked or not np.allclose(tracked[role], pose, atol=1e-8, rtol=0):
                raise ValueError("restored object frame differs from logged final state")
        native = bool(adapter.native_success())
        if native != result["success"]:
            raise ValueError("restored native success disagrees; no supplementary score promoted")
        components = predicate_components(adapter.native, result["task_id"])
        body = adapter.native.obj_body_id["obj"]
        mesh, names = visual_surface(sim.model._model, sim.data._data, body)
        gripper = sim.data.site_xpos[adapter.native.robots[0].eef_site_id["right"]]
        regions, region_status = native_goal_regions(adapter.native, result["task_id"])
        diagnostics = geometric_diagnostics(mesh.vertices, mesh.faces, gripper,
                                           sim.data.body_xpos[body], regions)
        return dict(unit_id=unit["unit_id"], canonical_instance_id=unit["canonical_instance_id"],
                    reset_id=unit["reset_id"], method=unit["controller_method"], task_id=result["task_id"],
                    status="MEASURED", native_success=native, native_components=components,
                    diagnostics=diagnostics, region_status=region_status,
                    native_visual_geoms=names, source_hashes=original_sources,
                    policy_calls=0, physics_steps=0, analysis_kind="posthoc_sensitivity_not_corrected_rubric")
    finally:
        adapter.close()


def run(planned, ledger, out):
    from robo.eval.native_scale_tables import load_units
    from robo.roundtrip.matrix import save_new
    units = load_units(planned, ledger)
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    save_new(out / "analysis_manifest.json", {"planned": str(Path(planned).resolve()), "planned_sha256": sha(planned),
        "ledger": str(Path(ledger).resolve()), "ledger_sha256": sha(ledger),
        "method": "posthoc final-state sensitivity; no outcome filtering", "planned_units": len(units),
        "source_sha256": sha(__file__), "official_results_unchanged": True})
    results = []
    for unit in units:
        path = out / "units" / (unit["unit_id"] + ".json")
        if unit.get("executed") is not True:
            row = dict(unit_id=unit["unit_id"], method=unit["controller_method"], status="NOT_EXECUTED",
                       original_status=unit["terminal_status"], native_success=None)
        else:
            try:
                row = evaluate_unit(unit)
            except Exception as exc:
                row = dict(unit_id=unit["unit_id"], method=unit["controller_method"], status="DIAGNOSTIC_UNAVAILABLE",
                           native_success=unit.get("success"), reason=f"{type(exc).__name__}: {exc}")
        save_new(path, row); results.append(row)
    summary = []
    for method in sorted({u["controller_method"] for u in units}):
        arm = [r for r in results if r["method"] == method]
        measured = [r for r in arm if r["status"] == "MEASURED"]
        summary.append(dict(method=method, planned=len(arm), measured=len(measured),
            unavailable=sum(r["status"] == "DIAGNOSTIC_UNAVAILABLE" for r in arm),
            origin_centroid_retreat_disagreements=sum(r["diagnostics"]["retreat_reference_point_disagrees"] for r in measured),
            native_successes_with_measured_checks=sum(r["native_success"] for r in measured),
            native_successes_with_centroid_retreat_disagreement=sum(r["native_success"] and r["diagnostics"]["retreat_reference_point_disagrees"] for r in measured),
            native_successes_with_available_region_check=sum(r["native_success"] and r["diagnostics"]["goal_region_available"] for r in measured),
            native_successes_with_vertices_outside_goal=sum(r["native_success"] and r["diagnostics"]["all_visual_vertices_in_goal_regions"] is False for r in measured)))
    save_new(out / "sensitivity_summary.json", summary)
    save_new(out / "sensitivity_records.jsonl", results, jsonl=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planned", required=True); p.add_argument("--ledger", required=True); p.add_argument("--out", required=True)
    a = p.parse_args(argv); print(json.dumps(run(a.planned, a.ledger, a.out), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
