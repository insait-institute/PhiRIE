"""Route TRELLIS.2 meshes into the existing registration/physics producer.

This is a separate engineering probe, not an A0--A4 treatment or a complete
photoreal twin. Its only targets are sealed construction-time TRAIN depth.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from robo.manifest.hash import canonical_hash
from run.icra2027.e3_auto_discovery_pilot import PilotError, identity, sha, write_new
from run.icra2027.e3_fresh_generation_contract import checked_identity, FRESH


def validate_observations(config, generation_manifest):
    """Read-only authentication of the entire fixed observation population."""
    source = config["observation_source"]
    manifest_path = checked_identity(source["manifest"])
    seal_path = checked_identity(source["seal"])
    e0_path = checked_identity(source["e0"])
    observation = json.loads(manifest_path.read_text())
    seal = json.loads(seal_path.read_text())
    e0 = json.loads(e0_path.read_text())
    payload = {k: v for k, v in e0.items() if k not in {"created_utc", "environment", "contract_sha256"}}
    scene = config["output_scene_id"]
    freeze = source["source_freeze_id"]
    expected_dir = e0_path.parent.parent / "agentic/observations" / scene
    if (e0_path.name != "freeze_manifest.json" or e0_path.parent.name != "contract"
            or manifest_path != expected_dir / "manifest.json" or seal_path != expected_dir / "seal.json"
            or manifest_path.absolute() != manifest_path.resolve()
            or e0.get("contract_sha256") != canonical_hash(payload)
            or e0.get("freeze_id") != freeze or e0["code"].get("dirty") is not False
            or e0["code"].get("dirty_override_for_smoke", False)
            or e0["code"].get("commit") != source["source_code_commit"]
            or observation.get("code_commit") != source["source_code_commit"]
            or observation.get("freeze_id") != freeze or observation.get("scene_id") != scene):
        raise PilotError("observation source E0/code/path closure differs")
    if (observation.get("geometry_source") != "source_scene_gaussian_expected_depth"
            or observation.get("evaluation_geometry_read") is not False
            or generation_manifest.get("source_gaussian_training_provenance") != FRESH):
        raise PilotError("only construction-time TRAIN Gaussian depth is admissible")
    protocol = observation.get("observation_protocol", {})
    if (protocol.get("output_coordinate_frame") != "world" or protocol.get("output_unit") != "metre"
            or protocol.get("crop_channel_used") != "rgba_alpha_only"
            or protocol.get("rendered_rgb") != "discarded"
            or protocol.get("rendered_channels_used") != ["expected_depth", "gaussian_alpha"]):
        raise PilotError("observation protocol differs from canonical TRAIN depth")
    member = {"manifest.json": source["manifest"]["sha256"]}
    if seal.get("members") != member or seal.get("controller_members") != member:
        raise PilotError("observation manifest seal differs")
    discovery_path = Path(config["source_pilot"]) / "input_manifest.json"
    if sha(discovery_path) != config["source_discovery_hashes"]["input_manifest.json"]:
        raise PilotError("TRAIN discovery manifest hash differs")
    discovery = json.loads(discovery_path.read_text())
    if (discovery.get("scene_id") != scene or discovery.get("source_gaussian_training_provenance") != FRESH
            or discovery.get("gaussian_provenance") != generation_manifest.get("gaussian_provenance")):
        raise PilotError("observation and generator discovery provenance differ")
    source_gs = observation["source_scene_gaussian"]
    discovery_gs = discovery["gaussian"]
    if (source_gs["sha256"] != discovery_gs["sha256"]
            or source_gs["path"] != discovery_gs["path"]
            or source_gs["size_bytes"] != discovery_gs["bytes"]):
        raise PilotError("observation and discovery source Gaussian differ")
    training = set(discovery["boundary"]["training_frames"])
    jobs = generation_manifest["jobs"]
    expected_ids = {f"{scene}/obj_{job['automatic_instance_id']}" for job in jobs}
    records = observation["records"]
    if len({r["job_id"] for r in records}) != len(records) or {r["job_id"] for r in records} != expected_ids:
        raise PilotError("observation population differs from complete planned jobs")
    by_job = {r["job_id"]: r for r in records}
    expected_members = {f"{r['object_slot']}.npz": r["observation_sha256"] for r in records}
    if seal.get("observation_members") != expected_members:
        raise PilotError("observation point-file seal differs")
    checked = {}
    for job in jobs:
        instance = job["automatic_instance_id"]
        if type(instance) is not int or instance < 0 or job["job_id"] != f"{scene}:auto:{instance}":
            raise PilotError("automatic instance/job identity differs")
        slot = f"obj_{instance}"
        record = by_job[f"{scene}/{slot}"]
        relative = f"outputs/icra2027/{freeze}/agentic/observations/{scene}/points/{slot}.npz"
        if (record["object_slot"] != slot or record["scene_id"] != scene or record["freeze_id"] != freeze
                or record["observation_path"] != relative
                or record["registration_surface_hash"] != record["observation_sha256"]
                or record["source_scene_gaussian_sha256"] != source_gs["sha256"]
                or record["source_frame"] not in training):
            raise PilotError("observation point identity or TRAIN frame differs")
        if job["prepared"] and (record["rgba_sha256"] != job["input"]["sha256"]
                                or record["source_frame"] != job["frame"]):
            raise PilotError("observation crop or source-frame identity differs")
        path = manifest_path.parent / "points" / f"{slot}.npz"
        if path.absolute() != path.resolve() or sha(path) != record["observation_sha256"]:
            raise PilotError("observation point bytes changed")
        with np.load(path, allow_pickle=False) as archive:
            points = archive["points"]
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                or len(points) != record["observation_point_count"]
                or not np.isfinite(record["observation_mask_fraction"])
                or not 0 <= record["observation_mask_fraction"] <= 1):
            raise PilotError("invalid observation points or visibility evidence")
        checked[job["job_id"]] = dict(record=record, path=path, points=points)
    return checked


def _run_probe(*args, **kwargs):
    # The existing registration/physics producer owns every threshold and metric.
    from agents.orchestrator.runtime import align_and_probe
    return align_and_probe(*args, **kwargs)


def _validate_runtime(result, directory, inputs, commit):
    # Reuse the canonical artifact/schema definitions; importing them does not
    # open any evaluation data or run an evaluation producer.
    from robo.eval.agentic_ablation import RUNTIME_ARTIFACT_ROLES, EVIDENCE_MANIFEST_FIELDS
    if set(result.get("artifact_files", [])) != set(RUNTIME_ARTIFACT_ROLES):
        raise PilotError("runtime artifact closure differs from canonical producer")
    for name in RUNTIME_ARTIFACT_ROLES:
        path = directory / name
        if not path.is_file() or path.is_symlink() or not path.stat().st_size:
            raise PilotError("missing canonical runtime artifact: " + name)
    evidence = json.loads((directory / "evidence.json").read_text())
    if (set(evidence) != set(EVIDENCE_MANIFEST_FIELDS)
            or evidence["producer"] != "agents.orchestrator.runtime.align_and_probe"
            or evidence["producer_commit"] != commit or evidence["input_hashes"] != inputs
            or evidence["raw_values"] != result["evidence"]
            or evidence["missing_flags"] != result["evidence"]["missing_evidence"]
            or evidence["wall_s"] != result["wall_s"]):
        raise PilotError("runtime evidence binding differs")
    transform = np.asarray(result["alignment"]["T"])
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise PilotError("runtime registration transform is invalid")


def probe(config_path, freeze_root):
    from run.icra2027 import e3_trellis_generation_pilot as runner
    from run.icra2027.e3_generator_backend import audit_trellis2, generator
    config, commit = runner.context(config_path, freeze_root)
    if generator(config) != "trellis2":
        raise PilotError("mesh probe requires an explicit TRELLIS.2 engineering study")
    out = runner.output_directory(config, freeze_root)
    destination = out / "mesh_probe"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing existing mesh probe: {destination}")
    expected_audit = audit_trellis2(config_path, freeze_root, write_report=False)
    audit_path = out / "postrun_audit.json"
    if json.loads(audit_path.read_text()) != expected_audit:
        raise PilotError("TRELLIS.2 generation audit changed")
    manifest = json.loads((out / "input_manifest.json").read_text())
    pool = json.loads((out / "proposal_pool.json").read_text())
    observations = validate_observations(config, manifest)
    by_job = {row["job_id"]: row for row in pool["rows"]}
    if len(by_job) != len(pool["rows"]) or set(by_job) != {j["job_id"] for j in manifest["jobs"]}:
        raise PilotError("generation pool differs from complete planned population")
    destination.mkdir()
    anchors = {"config": identity(config_path), "new_e0": identity(Path(freeze_root) / "contract/freeze_manifest.json"),
               "input_manifest": identity(out / "input_manifest.json"), "proposal_pool": identity(out / "proposal_pool.json"),
               "postrun_audit": identity(audit_path), "observation_source": config["observation_source"]}
    write_new(destination / "inputs.json", anchors)
    rows = []
    for job in manifest["jobs"]:
        started = time.monotonic()
        proposal = by_job[job["job_id"]]
        row = dict(job_id=job["job_id"], automatic_instance_id=job["automatic_instance_id"],
                   proposal_id=proposal["proposal_id"], generator="trellis2", code_commit=commit,
                   freeze_id=config["freeze_id"], source_status=proposal["status"],
                   native_gaussian=False, full_twin_ready=False, paper_ready=False,
                   policy_acceptance="NOT_RUN", evaluation_ground_truth_read=False,
                   study_scope="trellis2_mesh_engineering", artifacts={})
        object_dir = destination / f"obj_{job['automatic_instance_id']}"
        object_dir.mkdir()
        if proposal["status"] != "available":
            row.update(status="NOT_RUN", failure_class="upstream_proposal_unavailable", reason=proposal.get("reason"))
        else:
            item = observations[job["job_id"]]
            observation = item["record"]
            target = item["points"][::max(len(item["points"]) // 6000, 1)]
            inputs = {"raw_mesh": proposal["artifacts"]["trellis2_mesh.ply"]["sha256"],
                      "visible_observation": observation["observation_sha256"]}
            row.update(input_hashes=inputs, observation_point_count=len(item["points"]),
                       registration_point_count=len(target), signed_source_up=False, label="object")
            try:
                if len(target) < 50:
                    raise ValueError("visible observation contains fewer than 50 points")
                result = _run_probe(proposal["artifacts"]["trellis2_mesh.ply"]["path"], target,
                                    label="object", visible_fraction=float(observation["observation_mask_fraction"]),
                                    out_dir=object_dir / "runtime", signed_source_up=False,
                                    producer_commit=commit, input_hashes=inputs)
                _validate_runtime(result, object_dir / "runtime", inputs, commit)
                row.update(status="PROBED", runtime=result,
                           collision_valid=result["evidence"].get("collision_valid"),
                           settle_stable=result["evidence"].get("settle_stable"),
                           missing_evidence=result["evidence"].get("missing_evidence", []))
            except Exception as exc:
                row.update(status="PROBE_FAILED", failure_class="registration_or_physics_execution_failure",
                           error_type=type(exc).__name__, reason=f"{type(exc).__name__}: {exc}")
            for path in sorted(object_dir.rglob("*")):
                if path.is_symlink():
                    raise PilotError("probe artifact cannot be a symlink")
                if path.is_file():
                    row["artifacts"][str(path.relative_to(object_dir))] = identity(path)
        row["wall_s"] = time.monotonic() - started
        write_new(object_dir / "record.json", row)
        rows.append(row)
    summary = dict(schema_version=1, study_scope="trellis2_mesh_engineering", freeze_id=config["freeze_id"],
                   code_commit=commit, inputs_sha256=sha(destination / "inputs.json"),
                   planned_jobs=len(manifest["jobs"]), rows=rows,
                   probed_jobs=sum(r["status"] == "PROBED" for r in rows),
                   failed_jobs=sum(r["status"] == "PROBE_FAILED" for r in rows),
                   unavailable_jobs=sum(r["status"] == "NOT_RUN" for r in rows),
                   native_gaussian=False, full_twin_ready=False, paper_ready=False,
                   policy_acceptance="NOT_RUN", evaluation_ground_truth_read=False)
    write_new(destination / "summary.json", summary)
    members = {str(path.relative_to(destination)): identity(path)
               for path in sorted(destination.rglob("*")) if path.is_file()}
    write_new(destination / "seal.json", dict(schema_version=1, members=members))
    return summary


# Evaluation deliberately lives beside this existing producer, outside controller
# execution. The original meshes/transforms are never rewritten or re-registered.
GEOMETRY_SCOPE = "trellis2_mesh_only_independent_geometry_engineering_pilot"


def _geometry_identity(spec):
    normalized = dict(spec, bytes=spec.get("bytes", spec.get("size_bytes")))
    path = checked_identity(normalized)
    if path.absolute() != path.resolve():
        raise PilotError("geometry source identity contains a symlink")
    size = spec.get("bytes", spec.get("size_bytes"))
    if path.stat().st_size != size:
        raise PilotError("geometry source byte count differs")
    return path


def _geometry_source_process(source, script, arguments):
    """Use the historical public validator in its actual, unchanged checkout."""
    import os
    import subprocess
    import sys
    root = Path(source["code_root"])
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if (root.absolute() != root.resolve() or Path(git("rev-parse", "--show-toplevel")) != root
            or git("rev-parse", "HEAD") != source["code_commit"]
            or git("status", "--porcelain")):
        raise PilotError("historical geometry validator requires its exact clean source")
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    env.pop("PYTHONHOME", None)
    command = [sys.executable, "-c", script, *map(str, arguments)]
    result = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)
    if result.returncode:
        raise PilotError("historical geometry source validation failed: " + result.stderr[-8000:])
    return json.loads(result.stdout)


_T2_GEOMETRY_VALIDATOR = r'''
import contextlib,json,sys
from pathlib import Path
from run.icra2027 import e3_trellis_generation_pilot as runner
from run.icra2027.e3_generator_backend import audit_trellis2
from run.icra2027.e3_trellis2_mesh_probe import validate_observations
with contextlib.redirect_stdout(sys.stderr):
    config,commit=runner.context(sys.argv[1],sys.argv[2])
    audit=audit_trellis2(sys.argv[1],sys.argv[2],write_report=False)
    out=runner.output_directory(config,sys.argv[2])
    manifest=json.loads((out/'input_manifest.json').read_text())
    validate_observations(config,manifest)
print(json.dumps(dict(config=config,generation_audit=audit,input_manifest=manifest)))
'''

_REFERENCE_GEOMETRY_VALIDATOR = r'''
import contextlib,json,sys
from pathlib import Path
from agents.eval.automatic_matching_manifest import load_references
from robo.eval.agentic_ablation import _verify_sealed_directory
path=Path(sys.argv[1]);payload=json.loads(path.read_text());scene=payload['scenes'][0]
with contextlib.redirect_stdout(sys.stderr):
    control=Path(scene['construction_root'])/'control'/scene['scene_id']
    _verify_sealed_directory(control)
    rows,validated=load_references(path,sys.argv[2],Path(scene['construction_root']),scene['scene_id'],scene['controller_shard_sha256'],json.loads(sys.argv[3]))
print(json.dumps(validated))
'''


def _validate_probe_for_geometry(source):
    """Close all construction artifacts before any evaluation reference is read."""
    config_path = _geometry_identity(source["config"])
    e0_path = _geometry_identity(source["e0"])
    seal_path = _geometry_identity(source["probe_seal"])
    audit_path = _geometry_identity(source["completion_audit"])
    root = e0_path.parent.parent
    original = _geometry_source_process(source, _T2_GEOMETRY_VALIDATOR, [config_path, root])
    config = original["config"]
    e0 = json.loads(e0_path.read_text())
    if e0.get("contract_sha256") != canonical_hash({k:v for k,v in e0.items() if k not in {"created_utc", "environment", "contract_sha256"}}):
        raise PilotError("original TRELLIS.2 E0 digest differs")
    scene = config["output_scene_id"]
    probe_root = root / "trellis2_initial" / scene / "mesh_probe"
    if (scene != "38d58a7a31" or root.name != config["freeze_id"]
            or e0_path != root / "contract/freeze_manifest.json"
            or seal_path != probe_root / "seal.json"
            or audit_path != root / "trellis2_pilot_completion_audit.json"):
        raise PilotError("TRELLIS.2 source paths or fixed scene differ")
    seal = json.loads(seal_path.read_text())
    members = seal.get("members", {})
    actual = {p.relative_to(probe_root).as_posix() for p in probe_root.rglob("*") if p.is_file() and p != seal_path}
    if set(members) != actual or any(p.is_symlink() for p in probe_root.rglob("*")):
        raise PilotError("TRELLIS.2 probe seal population differs")
    for name, spec in members.items():
        if _geometry_identity(spec) != probe_root / name:
            raise PilotError("TRELLIS.2 sealed artifact path differs")
    inputs = json.loads((probe_root / "inputs.json").read_text())
    if (_geometry_identity(inputs["config"]) != config_path
            or _geometry_identity(inputs["new_e0"]) != e0_path
            or inputs["observation_source"] != config["observation_source"]):
        raise PilotError("probe construction anchors differ")
    for name in ("input_manifest", "proposal_pool", "postrun_audit"):
        if _geometry_identity(inputs[name]) != probe_root.parent / (name + ".json"):
            raise PilotError("probe generation anchor path differs")
    summary = json.loads((probe_root / "summary.json").read_text())
    pool = json.loads((probe_root.parent / "proposal_pool.json").read_text())
    audit = json.loads(audit_path.read_text())
    if (audit.get("status") != "PASS" or audit.get("source_commit") != source["code_commit"]
            or audit.get("generation_audit") != original["generation_audit"]
            or audit.get("seal_sha256") != sha(seal_path)
            or audit.get("summary_sha256") != sha(probe_root / "summary.json")
            or summary.get("code_commit") != source["code_commit"]
            or summary.get("freeze_id") != config["freeze_id"]
            or summary.get("inputs_sha256") != sha(probe_root / "inputs.json")
            or summary.get("planned_jobs") != 15 or summary.get("probed_jobs") != 15
            or summary.get("evaluation_ground_truth_read") is not False):
        raise PilotError("TRELLIS.2 completion/source closure differs")
    rows = summary["rows"]
    expected = {f"{scene}:auto:{i}" for i in range(1000, 1015)}
    if len(rows) != 15 or {r["job_id"] for r in rows} != expected:
        raise PilotError("TRELLIS.2 planned population differs")
    proposals = {r["job_id"]: r for r in pool["rows"]}
    observation_manifest = json.loads(_geometry_identity(config["observation_source"]["manifest"]).read_text())
    observations = {r["job_id"]: r for r in observation_manifest["records"]}
    for row in rows:
        directory = probe_root / f"obj_{row['automatic_instance_id']}"
        proposal = proposals[row["job_id"]]
        if (row != json.loads((directory / "record.json").read_text())
                or row["status"] != "PROBED" or row["proposal_id"] != proposal["proposal_id"]
                or row["code_commit"] != source["code_commit"]
                or row["evaluation_ground_truth_read"] is not False
                or row["input_hashes"]["visible_observation"] != observations[row["job_id"].replace(":auto:", "/obj_")]["observation_sha256"]
                or row["input_hashes"]["raw_mesh"] != proposal["artifacts"]["trellis2_mesh.ply"]["sha256"]
                or row["runtime"]["alignment"] != json.loads((directory / "runtime/registration.json").read_text())):
            raise PilotError("TRELLIS.2 sealed registration/proposal differs")
        _validate_runtime(row["runtime"], directory / "runtime", row["input_hashes"], source["code_commit"])
    return dict(config=config, summary=summary, pool=pool, input_manifest=original["input_manifest"])


def _validate_geometry_references(source, construction):
    path = _geometry_identity(source["manifest"])
    audit_path = _geometry_identity(source["completion_audit"])
    ids = [r["job_id"].replace(":auto:", "/obj_") for r in construction["summary"]["rows"]]
    references = _geometry_source_process(source, _REFERENCE_GEOMETRY_VALIDATOR,
                                         [path, source["manifest"]["sha256"], json.dumps(ids)])
    audit = json.loads(audit_path.read_text())
    required_checks = {"construction_train_only_discovery_binding", "control_frozen_before_evaluation",
                       "independent_GT_input_hashes", "matching_and_surface_replay_exact",
                       "matching_manifest_seal_E0_source_config_binding", "unmatched_geometry_null", "nonheadline_gates"}
    if (references["code_commit"] != source["code_commit"] or references["planned_jobs"] != 15
            or references["planned_scenes"] != 1 or references["matched_jobs"] != 3
            or audit_path != path.parent.parent / "independent_evaluation_completion_audit.json"
            or audit.get("paper_ready") is not False or audit.get("headline_eligible") is not False
            or audit.get("claim_gate") != "NOT_RUN"
            or not required_checks.issubset(audit.get("checks", {}))
            or any(v != "PASS" for v in audit.get("checks", {}).values())
            or audit.get("evidence_hashes", {}).get(str(path), {}).get("sha256") != sha(path)):
        raise PilotError("independent geometry reference audit differs")
    binding = references["scenes"][0]["discovery_binding"]
    if (set(binding) != {"source_discovery_hashes", "source_gaussian_training_provenance", "gaussian_provenance"}
            or binding["source_gaussian_training_provenance"] != FRESH
            or any(construction["input_manifest"].get(k) != v for k, v in binding.items())):
        raise PilotError("T2 and evaluation references use different discovery inputs")
    if (len(references["rows"]) != 15 or sum(r["status"] == "matched" for r in references["rows"]) != 3
            or {r["job_id"] for r in references["rows"]} != set(ids)):
        raise PilotError("independent geometry matched/planned denominator differs")
    return references


def _geometry_rows(construction, references):
    from agents.assets.s5_align import apply_T
    from robo.eval.agentic_ablation import _load_mesh, _sample_mesh, _stable_seed, MESH_SAMPLE_COUNT
    from robo.eval.fidelity_metrics import geometry_metrics, load_surface
    refs = {r["job_id"]: r for r in references["rows"]}
    proposals = {r["job_id"]: r for r in construction["pool"]["rows"]}
    rows = []
    for probe_row in construction["summary"]["rows"]:
        job_id = probe_row["job_id"].replace(":auto:", "/obj_")
        reference = refs[job_id]
        row = dict(job_id=job_id, proposal_id=probe_row["proposal_id"],
                   generation_status=probe_row["source_status"], registration_status=probe_row["status"],
                   geometry_status=reference["status"], matched_gt_id=reference["matched_gt_id"],
                   cd_cm=None, f1_20=None, precision20=None, recall20=None, collapse=None, psnr=None, ssim=None, lpips=None,
                   construction_settle_stable=probe_row["settle_stable"])
        if reference["status"] == "matched":
            target_path = _geometry_identity(reference["evaluation_surface"])
            if sha(target_path) == probe_row["input_hashes"]["visible_observation"]:
                raise PilotError("registration and evaluation surfaces coincide")
            mesh_spec = proposals[probe_row["job_id"]]["artifacts"]["trellis2_mesh.ply"]
            mesh_path = _geometry_identity(mesh_spec)
            predicted = _sample_mesh(_load_mesh(mesh_path), MESH_SAMPLE_COUNT,
                                     _stable_seed(probe_row["proposal_id"], "evaluation"))
            target = load_surface(target_path)
            transform = np.asarray(probe_row["runtime"]["alignment"]["T"])
            row.update(geometry_metrics(apply_T(transform, predicted), target, threshold_m=0.02))
        rows.append(row)
    return rows


def evaluate_geometry(config_path, contract_path, destination):
    """Fixed 15-job mesh-only pilot using the canonical geometry estimator."""
    from robo.eval.agentic_ablation import _validate_cli_execution
    from robo.eval.agentic_ablation import _atomic_directory, _write_json_inside, _write_bytes_inside
    import csv
    import io
    config = json.loads(Path(config_path).read_text())
    if (config.get("schema_version") != 1 or config.get("scope") != GEOMETRY_SCOPE
            or config.get("planned_jobs") != 15 or config.get("paper_ready") is not False):
        raise PilotError("explicit fixed TRELLIS.2 geometry pilot config required")
    destination = Path(destination)
    contract = _validate_cli_execution(contract_path, config["freeze_id"], destination,
                                       config_paths=[config_path])
    if destination != Path(contract_path).parent.parent / "trellis2_geometry":
        raise PilotError("geometry output must belong to its new E0 freeze")
    if destination.exists():
        raise FileExistsError("refusing to overwrite TRELLIS.2 geometry output")
    construction = _validate_probe_for_geometry(config["construction_source"])
    references = _validate_geometry_references(config["reference_source"], construction)
    rows = _geometry_rows(construction, references)
    measured = [r for r in rows if r["geometry_status"] == "matched"]
    payload = dict(schema_version=1, scope=GEOMETRY_SCOPE, freeze_id=config["freeze_id"],
                   source_commit=contract["code"]["commit"], sources=config,
                   planned_jobs=len(rows), generated_jobs=construction["summary"]["probed_jobs"],
                   geometry_evaluated_jobs=len(measured), unmatched_jobs=len(rows)-len(measured),
                   cd_cm=float(np.mean([r["cd_cm"] for r in measured])),
                   f1_20=float(np.mean([r["f1_20"] for r in measured])),
                   catastrophic_collapses=sum(r["collapse"] for r in measured), rows=rows,
                   native_gaussian=False, full_twin_ready=False, paper_ready=False,
                   headline_eligible=False, claim_gate="NOT_RUN", policy_acceptance="NOT_RUN",
                   sampling="canonical agentic mesh sampler/proposal-ID evaluation seed; 20000 samples",
                   limitations="Geometry conditional on three frozen matched jobs; twelve unmatched jobs retain null geometry. Construction physics is not manipulation validation. No appearance evaluation or new registration.")
    csv_out = io.StringIO()
    writer = csv.DictWriter(csv_out, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    with _atomic_directory(destination) as staging:
        _write_json_inside(staging / "geometry.json", payload)
        _write_bytes_inside(staging / "geometry.csv", csv_out.getvalue().encode("utf-8"))
        _write_json_inside(staging / "seal.json", {"schema_version": 1, "members": {
            name: sha(staging / name) for name in ("geometry.json", "geometry.csv")}})
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate sealed TRELLIS.2 mesh-only geometry; never construct with GT")
    parser.add_argument("--geometry-config", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = evaluate_geometry(args.geometry_config, args.contract, args.out)
    print(json.dumps({k: result[k] for k in ("planned_jobs", "geometry_evaluated_jobs", "unmatched_jobs", "claim_gate")}))
