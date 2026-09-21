"""Wrapper boundaries; real materializer/GT rejection covered by its own suite."""
import json
from pathlib import Path

import pytest

from run.icra2027 import e4_compact_materialization as compact


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(compact.e3, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(compact.materializer, "REPOSITORY_ROOT", tmp_path)
    root = tmp_path / compact.SOURCE_FREEZE / "agentic"
    root.mkdir(parents=True)
    protocol = dict(study_scope="e4_compact_canonical_engineering", paper_ready=False,
        population=dict(scene_ids=list(compact.SCENES), planned_objects=17, planned_policy_object_rows=85),
        fixed_manipulation_tasks=[dict(task_id=t) for t in compact.FIXED_TASKS],
        no_substitution_after_qualification=True, no_policy_outcome_selection=True,
        planned_manipulation_episodes=40, episodes_per_task_arm=5, construction_arms=["A0", "A4"])
    protocol_path = write(tmp_path / "protocol.json", protocol)
    config = dict(automatic_sources=[dict(scene_id=s, discovery_directory=str(tmp_path / s),
                                        discovery_hashes={}) for s in compact.SCENES])
    config_path = write(tmp_path / "jobs.json", config)
    resources = [dict(id=label, resolved_path=str(path), **{k: v for k, v in compact.identity(path).items() if k != "path"})
                 for label, path in [("canonical_scene_roster", protocol_path), ("agentic_fresh_jobs", config_path)]]
    contract = dict(code=dict(commit=compact.SOURCE_COMMIT, dirty=False), freeze_id=compact.SOURCE_FREEZE,
                    resource_inventory=resources)
    contract["contract_sha256"] = compact.canonical_hash(contract)
    monkeypatch.setattr(compact, "SOURCE_CONTRACT", contract["contract_sha256"])
    write(root.parent / "contract/freeze_manifest.json", contract)
    jobs = dict(freeze_id=compact.SOURCE_FREEZE,
        source_contract=dict(code_commit=compact.SOURCE_COMMIT, contract_sha256=compact.SOURCE_CONTRACT,
                             jobs_sha256=compact.canonical_hash(config)),
        counts=dict(scenes=2, jobs=17, policy_object_rows=85), observation_protocol={"TRAIN_only": True},
        scenes=[dict(scene_id=s, jobs=[dict(job_id=f"{s}/obj_{1000+i}") for i in range(count)],
                     source_scene_gaussian={}, camera_artifacts={}) for s, count in compact.SCENES.items()])
    seal = write(root / "input_inventory/seal.json", {})
    monkeypatch.setattr(compact.materializer, "_verify_inventory", lambda path: (jobs, {}, {}, {"seal": compact.identity(seal)}))
    monkeypatch.setattr(compact.materializer, "_automatic_scene_audit", lambda *args: {})
    for scene in jobs["scenes"]:
        sid = scene["scene_id"]
        observation = dict(freeze_id=compact.SOURCE_FREEZE, code_commit=compact.SOURCE_COMMIT,
            evaluation_geometry_read=False, source_scene_gaussian={}, camera_artifacts={},
            observation_protocol=jobs["observation_protocol"])
        obs = write(root / "observations" / sid / "manifest.json", observation)
        write(obs.parent / "seal.json", {})
        write(root / "control" / sid / "seal.json", {})
        for phase in ("observe", "control"):
            receipts = root.parent / "execution_receipts" / sid
            write(receipts / (phase + "_execution_start.json"), dict(phase=phase, scene_id=sid,
                code_commit=compact.SOURCE_COMMIT, contract_sha256=compact.SOURCE_CONTRACT,
                planned_jobs=compact.SCENES[sid], planned_policy_object_rows=5*compact.SCENES[sid]))
            write(receipts / (phase + "_execution_result.json"), dict(status="PASS", exit_code=0))
    def control(path, scene):
        compact.identity(path / "control" / scene / "seal.json")
        return path / "control" / scene, dict(freeze_id=compact.SOURCE_FREEZE,
            job_count=compact.SCENES[scene], ledger_row_count=5*compact.SCENES[scene],
            observation_manifest_sha256=compact.e3.sha256_file(path / "observations" / scene / "manifest.json")), {}
    def observation(path, scene):
        p = path / "observations" / scene / "manifest.json"
        return p.parent, compact.read(p), {f"{scene}/obj_{1000+i}": {} for i in range(compact.SCENES[scene])}
    monkeypatch.setattr(compact.e3, "_load_control_scene", control)
    monkeypatch.setattr(compact.e3, "_load_observation_scene", observation)
    return root, jobs, protocol_path


def test_ready_binds_complete_population_and_control_sources(source):
    root, _, _ = source
    result = compact.inspect_source(root)
    assert result["status"] == "READY" and result["planned_objects"] == 17
    assert list(result["source"]["scenes"]) == list(compact.SCENES)
    assert result["fixed_task_ids"] == compact.FIXED_TASKS


def test_unfinished_controls_wait_without_artifact_failure_labels(source):
    root, _, _ = source
    first = next(iter(compact.SCENES))
    (root / "control" / first / "seal.json").unlink()
    (root / "control" / first).rmdir()
    result = compact.inspect_source(root)
    assert result["status"] == "WAITING" and result["planned_manipulation_episodes"] == 40
    assert "source" not in result and "failed_objects" not in result


def test_published_incomplete_control_is_error_not_waiting(source):
    root, _, _ = source
    (root / "control" / next(iter(compact.SCENES)) / "seal.json").unlink()
    with pytest.raises(FileNotFoundError):
        compact.inspect_source(root)


@pytest.mark.parametrize("mutation", ["count", "scene", "source_commit", "jobs_config"])
def test_source_inventory_identity_and_population_tampering_rejected(source, mutation):
    root, jobs, _ = source
    if mutation == "count": jobs["counts"]["jobs"] = 16
    elif mutation == "scene": jobs["scenes"].reverse()
    elif mutation == "source_commit": jobs["source_contract"]["code_commit"] = "0"*40
    else: jobs["source_contract"]["jobs_sha256"] = "0"*64
    with pytest.raises(ValueError, match="inventory"):
        compact.inspect_source(root)


def test_protocol_byte_tampering_rejected(source):
    root, _, protocol = source
    protocol.write_text(protocol.read_text() + " ")
    with pytest.raises(ValueError, match="resource changed"):
        compact.inspect_source(root)


def test_control_receipt_cannot_claim_another_producer(source):
    root, _, _ = source
    path = root.parent / "execution_receipts" / next(iter(compact.SCENES)) / "control_execution_start.json"
    receipt = compact.read(path)
    receipt["code_commit"] = "c"*40
    write(path, receipt)
    with pytest.raises(ValueError, match="execution receipt"):
        compact.inspect_source(root)


@pytest.fixture
def stage(source, tmp_path, monkeypatch):
    root, _, _ = source
    config = dict(schema_version=1, scope="e4_compact_materialization_engineering",
        paper_ready=False, freeze_id="new-stage", python=str(Path(compact.sys.executable).resolve()),
        e3_root=str(root), source=compact.inspect_source(root)["source"])
    config_path = write(tmp_path / "stage_config.json", config)
    snapshot = dict(commit="b"*40, dirty=False, status=[], code_root=str(compact.CODE))
    monkeypatch.setattr(compact.materializer, "_require_clean_code_snapshot", lambda: snapshot)
    contract = dict(freeze_id="new-stage", code=snapshot, resource_inventory=[])
    for name, path in [("e4_compact_materialization_config", config_path), ("materialization_python", config["python"])]:
        row = compact.identity(path)
        contract["resource_inventory"].append(dict(id=name, resolved_path=row.pop("path"), **row))
    contract["contract_sha256"] = compact.canonical_hash(contract)
    stage_root = tmp_path / "new-stage"
    write(stage_root / "contract/freeze_manifest.json", contract)
    return config_path, stage_root, root


def test_new_stage_requires_exact_e0_and_source_bundle(stage):
    config, root, _ = stage
    _, reviewed, snapshot = compact.validate_stage(config, root)
    assert reviewed["status"] == "READY" and snapshot["commit"] == "b"*40


@pytest.mark.parametrize("mutation", ["stage_config", "contract", "control_source"])
def test_bound_stage_tampering_rejected(stage, mutation):
    config, root, source_root = stage
    if mutation == "stage_config": config.write_text(config.read_text() + " ")
    elif mutation == "contract":
        path = root / "contract/freeze_manifest.json"
        contract = compact.read(path)
        contract["code"]["commit"] = "c"*40
        write(path, contract)
    else:
        path = source_root / "control" / next(iter(compact.SCENES)) / "seal.json"
        path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        compact.validate_stage(config, root)


def test_task_substitution_rejected_even_with_recomputed_input_hash(source):
    _, _, path = source
    protocol = compact.read(path)
    protocol["fixed_manipulation_tasks"][0]["task_id"] = "other"
    with pytest.raises(ValueError, match="protocol"):
        compact.validate_protocol(protocol)


def test_stage_waiting_creates_no_output_directory(tmp_path, monkeypatch):
    root = tmp_path / "new-stage"
    monkeypatch.setattr(compact, "validate_stage", lambda *a: ({}, {"status": "WAITING", "paper_ready": False}, {}))
    assert compact.run("config", root, scene_id=next(iter(compact.SCENES)))["status"] == "WAITING"
    assert not root.exists()


@pytest.mark.parametrize("kwargs", [{}, dict(scene_id="undeclared"), dict(scene_id="27dd4da69e", handoff=True)])
def test_undeclared_or_ambiguous_phase_rejected(kwargs):
    with pytest.raises(ValueError, match="choose"):
        compact.run("unused", "unused", **kwargs)


def test_array_rejected_before_source_reads(monkeypatch):
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "1")
    with pytest.raises(ValueError, match="ordinary"):
        compact.run("unused", "unused", scene_id="27dd4da69e")


def test_gpu_allocation_rejected_before_source_reads(monkeypatch):
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "1")
    with pytest.raises(ValueError, match="CPU"):
        compact.run("unused", "unused", scene_id="27dd4da69e")


def test_existing_materializer_delegation_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(compact.e3, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(compact.materializer, "REPOSITORY_ROOT", tmp_path)
    scene = next(iter(compact.SCENES))
    config = {"e3_root": str(tmp_path / "source"), "source": {"scenes": {scene: {"descriptor": {"scene_id": scene}}}}}
    snapshot = {"commit": "b"*40}
    monkeypatch.setattr(compact, "validate_stage", lambda *a: (config, {"status": "READY"}, snapshot))
    calls = []
    def generate(**kwargs):
        calls.append(kwargs["policy_id"])
        write(kwargs["out"] / "materialization_manifest.json", dict(e3_freeze_id=compact.SOURCE_FREEZE,
              e3_code_commit=compact.SOURCE_COMMIT, source_scene={"automatic_scene_descriptor": compact.materializer._identity(kwargs["automatic_scene_contract"])}))
    monkeypatch.setattr(compact.materializer, "materialize_factory_variant", generate)
    monkeypatch.setattr(compact.materializer, "validate_materialized_factory", lambda *a, **kw: {"roster": {"job_count": 7}})
    config_path = write(tmp_path / "config.json", config)
    result = compact.run(config_path, tmp_path / "stage", scene_id=scene)
    assert calls == ["A0", "A4"] and result["planned_objects"] == 7
    assert result["planned_manipulation_episodes"] == 20
    with pytest.raises(FileExistsError, match="overwritten"):
        compact.run(config_path, tmp_path / "stage", scene_id=scene)
    assert calls == ["A0", "A4"]
