import copy
import json

import numpy as np
import pytest

from robo.manifest.hash import canonical_hash
from run.icra2027 import e3_trellis2_mesh_probe as probe
from run.icra2027.e3_auto_discovery_pilot import PilotError, identity


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def case(tmp_path):
    scene, freeze = "aaaaaaaaaa", "old-freeze"
    old = tmp_path / freeze
    odir = old / "agentic/observations" / scene
    e0_path = old / "contract/freeze_manifest.json"
    e0 = dict(freeze_id=freeze, code=dict(commit="old-code", dirty=False))
    e0["contract_sha256"] = canonical_hash(e0)
    write(e0_path, e0)
    gs = dict(path=str(tmp_path / "train.ply"), bytes=12, sha256="source-gs")
    discovery = dict(scene_id=scene, source_gaussian_training_provenance=probe.FRESH,
                     gaussian_provenance={"status": probe.FRESH}, gaussian=gs,
                     boundary={"training_frames": ["train0.JPG", "train1.JPG", "train2.JPG"]})
    discovery_path = tmp_path / "discovery/input_manifest.json"
    write(discovery_path, discovery)
    jobs, records, members = [], [], {}
    for i, count in enumerate((12001, 90, 60)):
        instance, slot = 1000 + i, f"obj_{1000+i}"
        points = np.arange(count * 3, dtype=np.float32).reshape(count, 3) / 10000
        path = odir / "points" / f"{slot}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, points=points)
        digest = identity(path)["sha256"]
        jobs.append(dict(job_id=f"{scene}:auto:{instance}", automatic_instance_id=instance,
                         prepared=True, input={"sha256": f"crop-{i}"}, frame=f"train{i}.JPG", label="GT_LABEL_FORBIDDEN"))
        records.append(dict(job_id=f"{scene}/{slot}", object_slot=slot, scene_id=scene, freeze_id=freeze,
                            observation_path=f"outputs/icra2027/{freeze}/agentic/observations/{scene}/points/{slot}.npz",
                            observation_sha256=digest, registration_surface_hash=digest,
                            source_scene_gaussian_sha256=gs["sha256"], source_frame=f"train{i}.JPG",
                            rgba_sha256=f"crop-{i}", observation_point_count=count, observation_mask_fraction=.8))
        members[f"{slot}.npz"] = digest
    obs = dict(code_commit="old-code", freeze_id=freeze, scene_id=scene,
               geometry_source="source_scene_gaussian_expected_depth", evaluation_geometry_read=False,
               observation_protocol=dict(output_coordinate_frame="world", output_unit="metre",
               crop_channel_used="rgba_alpha_only", rendered_rgb="discarded",
               rendered_channels_used=["expected_depth", "gaussian_alpha"]), records=records,
               source_scene_gaussian=dict(path=gs["path"], size_bytes=gs["bytes"], sha256=gs["sha256"]))
    write(odir / "manifest.json", obs)
    digest = identity(odir / "manifest.json")["sha256"]
    seal = dict(members={"manifest.json": digest}, controller_members={"manifest.json": digest}, observation_members=members)
    write(odir / "seal.json", seal)
    config = dict(generator="trellis2", freeze_id="new-freeze", output_scene_id=scene,
                  source_pilot=str(discovery_path.parent), source_discovery_hashes={"input_manifest.json": identity(discovery_path)["sha256"]},
                  observation_source=dict(manifest=identity(odir / "manifest.json"), seal=identity(odir / "seal.json"),
                  e0=identity(e0_path), source_freeze_id=freeze, source_code_commit="old-code"))
    generation = dict(jobs=jobs, source_gaussian_training_provenance=probe.FRESH,
                      gaussian_provenance=discovery["gaussian_provenance"])
    return config, generation, odir


def test_authenticates_complete_points_and_automatic_ids(case):
    config, generation, _ = case
    checked = probe.validate_observations(config, generation)
    assert list(checked) == [j["job_id"] for j in generation["jobs"]]
    assert [len(v["points"]) for v in checked.values()] == [12001, 90, 60]


@pytest.mark.parametrize("change", ["crop", "frame", "job_id", "population", "gaussian"])
def test_observation_binding_rejects_changed_generator(case, change):
    config, generation, _ = case
    if change == "crop":
        generation["jobs"][0]["input"]["sha256"] = "changed"
    elif change == "frame":
        generation["jobs"][0]["frame"] = "eval.JPG"
    elif change == "job_id":
        generation["jobs"][0]["job_id"] = "another:auto:1000"
    elif change == "population":
        generation["jobs"].pop()
    else:
        generation["gaussian_provenance"] = {"status": "UNKNOWN"}
    with pytest.raises(PilotError):
        probe.validate_observations(config, generation)


def test_changed_observation_points_rejected(case):
    config, generation, odir = case
    np.savez(odir / "points/obj_1000.npz", points=np.zeros((12001, 3)))
    with pytest.raises(PilotError, match="bytes changed"):
        probe.validate_observations(config, generation)


def test_evaluation_geometry_even_resealed_is_forbidden(case):
    config, generation, odir = case
    path = odir / "manifest.json"
    obs = json.loads(path.read_text())
    obs["evaluation_geometry_read"] = True
    write(path, obs)
    config["observation_source"]["manifest"] = identity(path)
    seal = json.loads((odir / "seal.json").read_text())
    seal["members"]["manifest.json"] = identity(path)["sha256"]
    seal["controller_members"]["manifest.json"] = identity(path)["sha256"]
    write(odir / "seal.json", seal)
    config["observation_source"]["seal"] = identity(odir / "seal.json")
    with pytest.raises(PilotError, match="construction-time TRAIN"):
        probe.validate_observations(config, generation)


def test_dirty_source_e0_rejected(case):
    config, generation, _ = case
    from pathlib import Path
    path = Path(config["observation_source"]["e0"]["path"])
    e0 = json.loads(path.read_text())
    e0["code"]["dirty"] = True
    e0["contract_sha256"] = canonical_hash({k: v for k, v in e0.items() if k != "contract_sha256"})
    write(path, e0)
    config["observation_source"]["e0"] = identity(path)
    with pytest.raises(PilotError, match="E0"):
        probe.validate_observations(config, generation)


@pytest.fixture
def execution(case, tmp_path, monkeypatch):
    from run.icra2027 import e3_trellis_generation_pilot as runner
    from run.icra2027 import e3_generator_backend as backend
    config, generation, odir = case
    root = tmp_path / config["freeze_id"]
    out = root / "trellis2_initial" / config["output_scene_id"]
    write(root / "contract/freeze_manifest.json", {"frozen": True})
    write(out / "input_manifest.json", generation)
    rows = []
    for i, job in enumerate(generation["jobs"]):
        path = out / f"mesh-{i}.ply"
        path.write_text("mock mesh bytes")
        rows.append(dict(job_id=job["job_id"], proposal_id=f"proposal-{i}",
                         status="available" if i < 2 else "generation_failed",
                         reason=None if i < 2 else "empty generation", artifacts={"trellis2_mesh.ply": identity(path)}))
    write(out / "proposal_pool.json", {"rows": rows})
    write(out / "postrun_audit.json", {"artifact_checks": "PASS"})
    config_path = tmp_path / "config.json"
    write(config_path, config)
    monkeypatch.setattr(runner, "context", lambda *args: (config, "new-code"))
    monkeypatch.setattr(runner, "output_directory", lambda *args: out)
    monkeypatch.setattr(backend, "audit_trellis2", lambda *args, **kw: {"artifact_checks": "PASS"})
    calls = []

    def runtime(mesh, target, **kwargs):
        from robo.eval.agentic_ablation import RUNTIME_ARTIFACT_ROLES
        calls.append((mesh, target.copy(), kwargs))
        kwargs["out_dir"].mkdir()
        write(kwargs["out_dir"] / "evidence.json", {"evidence": "unchanged existing producer"})
        if len(calls) == 2:
            raise RuntimeError("registration failed")
        result = dict(evidence=dict(collision_valid=False, settle_stable=False,
                      missing_evidence=["support_gap_m"]), artifact_files=list(RUNTIME_ARTIFACT_ROLES),
                      alignment={"T": np.eye(4).tolist()}, wall_s=.5)
        for name in RUNTIME_ARTIFACT_ROLES:
            write(kwargs["out_dir"] / name, {"artifact": "mock"})
        write(kwargs["out_dir"] / "evidence.json", dict(producer="agents.orchestrator.runtime.align_and_probe",
              producer_commit=kwargs["producer_commit"], input_hashes=kwargs["input_hashes"],
              raw_values=result["evidence"], missing_flags=result["evidence"]["missing_evidence"],
              started_utc="start", finished_utc="end", wall_s=.5))
        return result

    monkeypatch.setattr(probe, "_run_probe", runtime)
    return config_path, root, out, calls


def test_probe_keeps_full_denominator_negative_result_and_failures(execution):
    config_path, root, out, calls = execution
    summary = probe.probe(config_path, root)
    assert summary["planned_jobs"] == 3
    assert (summary["probed_jobs"], summary["failed_jobs"], summary["unavailable_jobs"]) == (1, 1, 1)
    assert [r["status"] for r in summary["rows"]] == ["PROBED", "PROBE_FAILED", "NOT_RUN"]
    assert summary["rows"][0]["collision_valid"] is False
    assert summary["paper_ready"] is summary["full_twin_ready"] is summary["native_gaussian"] is False
    assert summary["policy_acceptance"] == "NOT_RUN"
    assert len(calls) == 2 and len(calls[0][1]) == 6001
    assert all(call[2]["label"] == "object" and call[2]["signed_source_up"] is False for call in calls)
    assert all(set(call[2]["input_hashes"]) == {"raw_mesh", "visible_observation"} for call in calls)
    seal = json.loads((out / "mesh_probe/seal.json").read_text())
    for spec in seal["members"].values():
        assert identity(spec["path"])["sha256"] == spec["sha256"]
    failed = summary["rows"][1]
    assert failed["artifacts"]["runtime/evidence.json"]


def test_probe_refuses_existing_output(execution):
    config_path, root, out, calls = execution
    probe.probe(config_path, root)
    previous = (out / "mesh_probe/summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        probe.probe(config_path, root)
    assert (out / "mesh_probe/summary.json").read_bytes() == previous
    assert len(calls) == 2


def test_probe_rejects_changed_generation_audit(execution):
    config_path, root, out, calls = execution
    write(out / "postrun_audit.json", {"artifact_checks": "FAIL"})
    with pytest.raises(PilotError, match="audit changed"):
        probe.probe(config_path, root)
    assert not calls and not (out / "mesh_probe").exists()


def test_too_few_points_is_typed_failure_and_denominator_retained(execution, monkeypatch):
    config_path, root, _, calls = execution
    original = probe.validate_observations

    def too_few(*args):
        checked = original(*args)
        first = next(iter(checked.values()))
        first["points"] = first["points"][:49]
        return checked

    monkeypatch.setattr(probe, "validate_observations", too_few)
    summary = probe.probe(config_path, root)
    assert summary["planned_jobs"] == 3 and summary["rows"][0]["status"] == "PROBE_FAILED"
    assert "fewer than 50" in summary["rows"][0]["reason"]
    assert len(calls) == 1


def test_missing_runtime_artifact_cannot_be_probed_success(execution, monkeypatch):
    config_path, root, _, _ = execution
    original = probe._run_probe

    def missing(*args, **kwargs):
        result = original(*args, **kwargs)
        (kwargs["out_dir"] / "physical/mesh_sim.obj").unlink()
        return result

    monkeypatch.setattr(probe, "_run_probe", missing)
    summary = probe.probe(config_path, root)
    assert summary["rows"][0]["status"] == "PROBE_FAILED"
    assert "missing canonical runtime" in summary["rows"][0]["reason"]
