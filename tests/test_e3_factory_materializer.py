from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agents.orchestrator.artifact import sha256_file
from robo.eval import agentic_ablation as e3
from robo.eval import e3_factory_materializer as materializer
from robo.tasks.pi05_tasks import _load_objects


SCENE = "0123456789"
FREEZE = "test-e3-freeze"
VALIDATOR_COMMIT = "1" * 40


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _identity(root: Path, path: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _external(name: str) -> dict:
    return {
        "path": f"/frozen/source/{name}",
        "sha256": hashlib.sha256(name.encode()).hexdigest(),
        "size_bytes": 123,
    }


def _reseal_inventory(e3_root: Path) -> None:
    inventory = e3_root / "input_inventory"
    members = {
        name: sha256_file(inventory / name) for name in materializer.INVENTORY_MEMBERS
    }
    _write_json(
        inventory / "seal.json",
        {
            "controller_members": {
                name: members[name]
                for name in ("resolved_jobs.json", "resolved_proposals.json")
            },
            "members": members,
            "schema_version": 1,
        },
    )


def _reseal_materialization(factory: Path, manifest: dict) -> None:
    _write_json(factory / "materialization_manifest.json", manifest)
    path = factory / "materialization_manifest.json"
    _write_json(
        factory / "seal.json",
        {
            "manifest_kind": "e4_construction_variant_materialization",
            "members": {
                "materialization_manifest.json": {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            },
            "schema_version": 1,
        },
    )


def _reseal_aggregate(e3_root: Path, policy: str) -> None:
    selected_root = e3_root / "selected_assets"
    selected_dir = selected_root / policy / SCENE
    selected_members = {
        f"{policy}/{SCENE}/{path.name}": {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(selected_dir.glob("*.json"))
    }
    _write_json(
        e3_root / "aggregate_seal.json",
        {
            "members": {
                name: sha256_file(e3_root / name)
                for name in materializer.AGGREGATE_MEMBERS
            },
            "schema_version": 1,
            "selected_asset_directories": [policy, f"{policy}/{SCENE}"],
            "selected_asset_members": selected_members,
        },
    )


def _registration(index: int) -> dict:
    return {
        "T": [
            [1.0, 0.0, 0.0, 1.0 + index],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 0.75],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "mesh_sample_count": 20_000,
        "mesh_sample_seed": 42,
        "observation_dims_m": [0.1, 0.2, 0.3],
        "registration_initial_objective_m": 0.02,
        "registration_median_m": 0.003 + index * 0.001,
        "residual_tilt_deg": 1.25,
        "scale": 1.0,
        "scale_ratio_vs_observation": 1.1,
        "source_up_hypothesis": "+z",
        "symmetric_clipped_registration_residual_m": 0.01,
        "world_dims_m": [0.10, 0.20, 0.30],
    }


def _accepted_record(
    root: Path, e3_root: Path, source_factory: Path, policy: str, index: int
) -> dict:
    slot = f"obj_{index:02d}"
    job_id = f"{SCENE}/{slot}"
    proposal_id = f"{job_id}:trellis"
    raw = source_factory / "objects" / slot / "trellis"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "trellis_mesh.ply").write_bytes(f"mesh-{index}".encode())
    (raw / "trellis_gs.ply").write_bytes(f"gaussian-{index}".encode())

    runtime = (
        e3_root
        / "control"
        / SCENE
        / "proposals"
        / proposal_id.replace("/", "_").replace(":", "_")
    )
    physical = runtime / "physical"
    (physical / "collision").mkdir(parents=True, exist_ok=True)
    (runtime / "registration.json").write_text(
        json.dumps(_registration(index), sort_keys=True) + "\n", encoding="utf-8"
    )
    (runtime / "evidence.json").write_text("{\"producer\": \"fixture\"}\n")
    (runtime / "transform.npy").write_bytes(b"fixture-npy")
    (physical / "mesh_sim.obj").write_bytes(b"v 0 0 0\n")
    (physical / "collision" / "part_00.obj").write_bytes(b"v 0 0 0\n")
    _write_json(
        physical / "physics.json",
        {"friction": 0.5, "mass_kg": 0.3, "restitution": 0.1, "source": "fixture"},
    )
    _write_json(
        physical / "probe.json",
        {
            "collision_backend": "fixture",
            "collision_valid": True,
            "contact_dynamics": {},
            "final_aabb_min_z_m": 0.0,
            "initial_penetration_m": 0.0,
            "settle_drift_m": 0.0,
            "settle_stable": True,
            "settle_sunk": False,
            "support_gap_m": 0.005,
            "support_overlap_fraction": 1.0,
            "usable_convex_parts": 1,
        },
    )
    (physical / "object.urdf").write_text(
        """<?xml version="1.0"?>
<robot name="fixture"><link name="base">
<visual><geometry><mesh filename="mesh_sim.obj"/></geometry></visual>
<collision><geometry><mesh filename="collision/part_00.obj"/></geometry></collision>
</link></robot>
""",
        encoding="utf-8",
    )
    role_paths = {
        "collision": physical / "collision" / "part_00.obj",
        "evidence": runtime / "evidence.json",
        "mesh_sim": physical / "mesh_sim.obj",
        "physics": physical / "physics.json",
        "probe": physical / "probe.json",
        "raw_gaussian": raw / "trellis_gs.ply",
        "raw_mesh": raw / "trellis_mesh.ply",
        "registration": runtime / "registration.json",
        "transform": runtime / "transform.npy",
        "urdf": physical / "object.urdf",
    }
    paths = {role: path.relative_to(root).as_posix() for role, path in role_paths.items()}
    hashes = {role: sha256_file(path) for role, path in role_paths.items()}
    sizes = {role: path.stat().st_size for role, path in role_paths.items()}
    return {
        "freeze_id": FREEZE,
        "job_id": job_id,
        "object_slot": slot,
        "policy_id": policy,
        "proposal_id": proposal_id,
        "reason_codes": ["frozen_gate_pass"],
        "retry_attempted": False,
        "retry_invoked": False,
        "retry_produced": False,
        "retry_reason_codes": [],
        "retry_required": False,
        "scene_id": SCENE,
        "schema_version": 1,
        "selected_asset": {
            "artifact_hashes": hashes,
            "artifact_paths": paths,
            "artifact_sizes": sizes,
            "proposal_digest": "a" * 64,
            "registration_surface_hash": "b" * 64,
            "tool": "trellis",
        },
        "selected_proposal_id": proposal_id,
        "support_label": "not_evaluated" if policy == "A0" else "supported",
        "terminal_action": "accept",
    }


def _abstained_record(policy: str, index: int) -> dict:
    slot = f"obj_{index:02d}"
    job_id = f"{SCENE}/{slot}"
    return {
        "freeze_id": FREEZE,
        "job_id": job_id,
        "object_slot": slot,
        "policy_id": policy,
        "proposal_id": f"{job_id}:terminal:A4:abstain",
        "reason_codes": ["no_proposal_passed_frozen_gate"],
        "retry_attempted": True,
        "retry_invoked": True,
        "retry_produced": True,
        "retry_reason_codes": ["unstable_settle"],
        "retry_required": True,
        "scene_id": SCENE,
        "schema_version": 1,
        "selected_asset": None,
        "selected_proposal_id": None,
        "support_label": "unsupported",
        "terminal_action": "abstain",
    }


def _proposal(job: dict, tool: str, selected: dict | None = None) -> dict:
    available = (
        selected is not None
        and selected.get("selected_asset") is not None
        and selected["selected_asset"]["tool"] == tool
    )
    row = {
        "artifact_hashes": {},
        "artifact_paths": {},
        "artifact_sizes": {},
        "availability": "typed_unavailable",
        "construction_evidence": {
            "missing_artifacts": ["raw_mesh", "raw_gaussian"],
            "typed_failure": "fixture",
        },
        "freeze_id": FREEZE,
        "job_id": job["job_id"],
        "object_slot": job["object_slot"],
        "proposal_id": f"{job['job_id']}:{tool}",
        "raw_generator_provenance": {
            "artifact_bytes_hash_frozen": True,
            "checkpoint_identity_recorded": False,
            "claim_status": "fixture",
            "generator_commit": None,
            "runtime_manifest_recorded": False,
        },
        "scene_id": SCENE,
        "tool_id": tool,
    }
    if available:
        asset = selected["selected_asset"]
        for field in ("artifact_hashes", "artifact_paths", "artifact_sizes"):
            row[field] = {
                role: asset[field][role] for role in ("raw_gaussian", "raw_mesh")
            }
        row["availability"] = "available"
        row["construction_evidence"] = {}
    return row


def _build_fixture(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, policy: str = "A4"
) -> dict[str, Path]:
    monkeypatch.setattr(materializer, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(e3, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(
        materializer,
        "_git_snapshot",
        lambda: {
            "code_root": str(materializer.CODE_ROOT),
            "commit": VALIDATOR_COMMIT,
            "dirty": False,
            "status": [],
        },
    )
    e3_root = root / "aggregates" / "e3" / "agentic"
    inventory = e3_root / "input_inventory"
    inventory.mkdir(parents=True)
    source_factory = root / "outputs" / f"{SCENE}_factory"
    objects_dir = source_factory / "objects"
    objects_dir.mkdir(parents=True)

    rows = []
    jobs = []
    tiers = ["A", "B"]
    for index, (label, tier) in enumerate(zip(["cup", "box"], tiers, strict=True)):
        slot = f"obj_{index:02d}"
        frame = f"FRAME{index}.JPG"
        bbox = [index, index + 1, index + 10, index + 12]
        row = {
            "aabb": [[index, 0.0, 0.75], [index + 0.1, 0.2, 1.05]],
            "bbox_px": bbox,
            "frame": frame,
            "gt_object_id": 100 + index,
            "index": index,
            "label": label,
        }
        rows.append(row)
        object_dir = objects_dir / slot
        object_dir.mkdir()
        _write_json(object_dir / "meta.json", {"bbox_px": bbox, "frame": frame})
        (object_dir / "rgba.png").write_bytes(f"rgba-{index}".encode())
        jobs.append(
            {
                "artifact_hashes": {"rgba": sha256_file(object_dir / "rgba.png")},
                "artifact_paths": {
                    "rgba": (object_dir / "rgba.png").relative_to(root).as_posix()
                },
                "construction_evidence": {
                    "bbox_px": bbox,
                    "legacy_mask_provenance": "opaque_fixed_input",
                    "source_frame": frame,
                },
                "freeze_id": FREEZE,
                "job_id": f"{SCENE}/{slot}",
                "object_slot": slot,
                "scene_id": SCENE,
            }
        )
    _write_json(objects_dir / "objects.json", rows)
    report_path = source_factory / "report.json"
    _write_json(
        report_path,
        {
            "objects": [
                {
                    "gt_object_id": row["gt_object_id"],
                    "index": row["index"],
                    "label": row["label"],
                    "rejected": None,
                    "tier": tiers[row["index"]],
                }
                for row in rows
            ]
        },
    )
    prehybrid_path = objects_dir / "aligned_all_prehybrid.json"
    _write_json(
        prehybrid_path,
        [
            {
                "index": row["index"],
                "label": row["label"],
                "rejected": None,
                "tier": tiers[row["index"]],
            }
            for row in rows
        ],
    )

    gaussian = _external("scene.ply")
    cameras = {"intrinsics": _external("intrinsics.json"), "poses": _external("poses.txt")}
    e2_manifest_path = root / "fidelity" / SCENE / "manifest.json"
    source_artifacts = [_identity(root, objects_dir / "objects.json")]
    source_artifacts.extend(
        _identity(root, objects_dir / f"obj_{index:02d}" / "meta.json")
        for index in range(2)
    )
    _write_json(
        e2_manifest_path,
        {
            "camera_artifacts": cameras,
            "construction_sources": {
                "factorized_gt_discovery": {
                    "accepted_object_ids": [0, 1],
                    "artifacts": source_artifacts,
                    "source": f"outputs/{SCENE}_factory",
                }
            },
            "scene_id": SCENE,
            "source_scene_gaussian": gaussian,
        },
    )

    resolved_jobs = {
        "counts": {"jobs": 2, "policy_object_rows": 10, "scenes": 1},
        "freeze_id": FREEZE,
        "observation_protocol": e3.OBSERVATION_PROTOCOL,
        "scenes": [
            {
                "camera_artifacts": cameras,
                "jobs": jobs,
                "scene_id": SCENE,
                "source_scene_gaussian": gaussian,
            }
        ],
        "schema_version": 1,
        "source_contract": {
            "code_commit": "c" * 40,
            "contract_freeze_id": "contract",
            "contract_hash_method": "canonical_json_sha256",
            "contract_sha256": "d" * 64,
            "jobs_hash_method": "canonical_structured_sha256",
            "jobs_sha256": "e" * 64,
        },
        "study_scope": "fixture",
    }
    selected_dir = e3_root / "selected_assets" / policy / SCENE
    selected_dir.mkdir(parents=True)
    first = _accepted_record(root, e3_root, source_factory, policy, 0)
    second = (
        _abstained_record(policy, 1)
        if policy == "A4"
        else _accepted_record(root, e3_root, source_factory, policy, 1)
    )
    selected_by_job = {first["job_id"]: first, second["job_id"]: second}
    proposals = [
        _proposal(job, tool, selected_by_job[job["job_id"]])
        for job in jobs
        for tool in ("trellis", "reconviagen")
    ]
    resolved_proposals = {
        "counts": {
            "jobs": 2,
            "reconviagen_available": 0,
            "reconviagen_typed_unavailable": 2,
            "trellis_available": sum(
                row.get("selected_asset") is not None for row in (first, second)
            ),
        },
        "freeze_id": FREEZE,
        "proposals": proposals,
        "schema_version": 1,
    }
    audit = {
        "counts": {},
        "created_utc": "2026-09-03T00:00:00Z",
        "evaluation_references": [],
        "freeze_id": FREEZE,
        "legacy_hybrid_winner_used": False,
        "membership_authority": "factory_report_json",
        "resolved_roster_sha256": "f" * 64,
        "scenes": [
            {
                "accepted_jobs": 2,
                "camera_artifacts": cameras,
                "job_ids": [job["job_id"] for job in jobs],
                "prehybrid_identity": _identity(root, prehybrid_path),
                "reconviagen_available": 0,
                "report_identity": _identity(root, report_path),
                "scene_id": SCENE,
                "source_identity_manifest": _identity(root, e2_manifest_path),
                "source_scene_gaussian": gaussian,
            }
        ],
        "schema_version": 1,
        "source_identity_manifest_sha256": "0" * 64,
    }
    _write_json(inventory / "resolved_jobs.json", resolved_jobs)
    _write_json(inventory / "resolved_proposals.json", resolved_proposals)
    _write_json(inventory / "inventory_audit.json", audit)
    (inventory / "artifact_hashes.sha256").write_text("fixture\n", encoding="utf-8")
    inventory_members = {
        name: sha256_file(inventory / name)
        for name in materializer.INVENTORY_MEMBERS
    }
    _write_json(
        inventory / "seal.json",
        {
            "controller_members": {
                name: inventory_members[name]
                for name in ("resolved_jobs.json", "resolved_proposals.json")
            },
            "members": inventory_members,
            "schema_version": 1,
        },
    )

    for name in materializer.AGGREGATE_MEMBERS:
        if name == "agentic_ablation.json":
            _write_json(
                e3_root / name,
                {
                    "freeze_id": FREEZE,
                    "headline_eligible": False,
                    "paper_ready": False,
                    "retry_claim_status": "fixture_preliminary_only",
                    "study_scope": "fixture",
                },
            )
        else:
            (e3_root / name).write_text(f"{name}\n", encoding="utf-8")
    _write_json(selected_dir / "obj_00.json", first)
    _write_json(selected_dir / "obj_01.json", second)
    selected_members = {
        f"{policy}/{SCENE}/{path.name}": {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(selected_dir.glob("*.json"))
    }
    _write_json(
        e3_root / "aggregate_seal.json",
        {
            "members": {
                name: sha256_file(e3_root / name)
                for name in materializer.AGGREGATE_MEMBERS
            },
            "schema_version": 1,
            "selected_asset_directories": [policy, f"{policy}/{SCENE}"],
            "selected_asset_members": selected_members,
        },
    )
    return {
        "e3_root": e3_root,
        "out": root / "materialized" / f"{SCENE}_factory",
        "selected": selected_dir,
        "source_factory": source_factory,
    }


def test_a4_materializes_exact_roster_and_keeps_abstention_assetless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch, policy="A4")
    manifest = materializer.materialize_factory_variant(
        e3_root=fixture["e3_root"], scene_id=SCENE, policy_id="A4", out=fixture["out"]
    )
    out = fixture["out"]
    assert manifest["roster"]["accepted_slots"] == ["obj_00"]
    assert manifest["roster"]["abstained_slots"] == ["obj_01"]
    assert [row["index"] for row in json.loads((out / "objects/objects.json").read_text())] == [0, 1]

    accepted = out / "objects/obj_00"
    aligned = json.loads((accepted / "aligned.json").read_text())
    assert aligned["T"] == _registration(0)["T"]
    assert aligned["world_dims"] == _registration(0)["world_dims_m"]
    assert aligned["registration_median_m"] == _registration(0)["registration_median_m"]
    assert aligned["symmetric_clipped_registration_residual_m"] == 0.01
    assert aligned["tier"] == "A"
    assert aligned["rejected"] is None
    assert {
        "collision/part_00.obj",
        "evidence.json",
        "mesh_sim.obj",
        "object.urdf",
        "physics.json",
        "probe.json",
        "registration.json",
        "selected_asset.json",
        "transform.npy",
        "trellis_gs.ply",
        "trellis_mesh.ply",
    }.issubset(
        {
            path.relative_to(accepted).as_posix()
            for path in accepted.rglob("*")
            if path.is_file()
        }
    )
    assert len(_load_objects(out)) == 1

    abstained = out / "objects/obj_01"
    assert sorted(path.name for path in abstained.iterdir()) == [
        "aligned.json",
        "selected_asset.json",
    ]
    abstained_aligned = json.loads((abstained / "aligned.json").read_text())
    assert abstained_aligned["rejected"] == "e3_policy_abstention"
    assert not (abstained / "object.urdf").exists()
    seal = json.loads((out / "seal.json").read_text())
    assert seal["members"]["materialization_manifest.json"]["sha256"] == sha256_file(
        out / "materialization_manifest.json"
    )


def test_a0_materializes_all_assets_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch, policy="A0")
    manifest = materializer.materialize_factory_variant(
        e3_root=fixture["e3_root"], scene_id=SCENE, policy_id="A0", out=fixture["out"]
    )
    assert manifest["roster"]["accepted_count"] == 2
    assert manifest["roster"]["abstained_count"] == 0
    assert (fixture["out"] / "objects/obj_01/object.urdf").is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        materializer.materialize_factory_variant(
            e3_root=fixture["e3_root"],
            scene_id=SCENE,
            policy_id="A0",
            out=fixture["out"],
        )


def test_split_code_and_evidence_roots_are_sealed_and_revalidated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch, policy="A0")
    manifest = materializer.materialize_factory_variant(
        e3_root=fixture["e3_root"],
        scene_id=SCENE,
        policy_id="A0",
        out=fixture["out"],
    )
    provenance = manifest["provenance"]
    assert provenance == {
        "code_root": str(materializer.CODE_ROOT),
        "e3_producer_commit": "c" * 40,
        "evidence_root": str(tmp_path),
        "materializer_commit": VALIDATOR_COMMIT,
        "materializer_dirty": False,
        "validator_commit": VALIDATOR_COMMIT,
        "validator_dirty": False,
    }
    assert provenance["code_root"] != provenance["evidence_root"]

    report = materializer.validate_materialized_factory(
        fixture["out"],
        expected_scene_id=SCENE,
        expected_policy_id="A0",
        repository_root=tmp_path,
    )
    assert report["e3_producer_commit"] == "c" * 40
    assert report["materializer_commit"] == VALIDATOR_COMMIT
    assert report["validator_commit"] == VALIDATOR_COMMIT
    assert report["code_root"] == str(materializer.CODE_ROOT)
    assert report["evidence_root"] == str(tmp_path)


def test_dirty_code_snapshot_fails_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch, policy="A0")
    monkeypatch.setattr(
        materializer,
        "_git_snapshot",
        lambda: {
            "code_root": str(materializer.CODE_ROOT),
            "commit": VALIDATOR_COMMIT,
            "dirty": True,
            "status": [" M robo/eval/e3_factory_materializer.py"],
        },
    )
    with pytest.raises(ValueError, match="clean code worktree"):
        materializer.materialize_factory_variant(
            e3_root=fixture["e3_root"],
            scene_id=SCENE,
            policy_id="A0",
            out=fixture["out"],
        )
    assert not fixture["out"].exists()


def test_revalidation_rejects_evidence_root_provenance_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch, policy="A0")
    materializer.materialize_factory_variant(
        e3_root=fixture["e3_root"],
        scene_id=SCENE,
        policy_id="A0",
        out=fixture["out"],
    )
    manifest_path = fixture["out"] / "materialization_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["evidence_root"] = str(materializer.CODE_ROOT)
    _reseal_materialization(fixture["out"], manifest)
    with pytest.raises(ValueError, match="evidence root differs"):
        materializer.validate_materialized_factory(
            fixture["out"],
            expected_scene_id=SCENE,
            expected_policy_id="A0",
            repository_root=tmp_path,
        )


def test_published_factory_revalidation_binds_policy_and_detects_object_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch, policy="A0")
    materializer.materialize_factory_variant(
        e3_root=fixture["e3_root"], scene_id=SCENE, policy_id="A0", out=fixture["out"]
    )
    report = materializer.validate_materialized_factory(
        fixture["out"],
        expected_scene_id=SCENE,
        expected_policy_id="A0",
        repository_root=tmp_path,
    )
    assert report["scene_id"] == SCENE
    assert report["policy_id"] == "A0"
    with pytest.raises(ValueError, match="policy binding differs"):
        materializer.validate_materialized_factory(
            fixture["out"],
            expected_scene_id=SCENE,
            expected_policy_id="A4",
            repository_root=tmp_path,
        )
    with (fixture["out"] / "objects/objects.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="output drift"):
        materializer.validate_materialized_factory(
            fixture["out"],
            expected_scene_id=SCENE,
            expected_policy_id="A0",
            repository_root=tmp_path,
        )


def test_tampered_selected_record_fails_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch)
    with (fixture["selected"] / "obj_00.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="selected record (size|hash) drift"):
        materializer.materialize_factory_variant(
            e3_root=fixture["e3_root"], scene_id=SCENE, policy_id="A4", out=fixture["out"]
        )
    assert not fixture["out"].exists()


def test_artifact_hash_drift_rolls_back_atomic_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch)
    selected = json.loads((fixture["selected"] / "obj_00.json").read_text())
    raw_mesh = tmp_path / selected["selected_asset"]["artifact_paths"]["raw_mesh"]
    raw_mesh.write_bytes(b"drift!")
    with pytest.raises(ValueError, match="drifted during copy"):
        materializer.materialize_factory_variant(
            e3_root=fixture["e3_root"], scene_id=SCENE, policy_id="A4", out=fixture["out"]
        )
    assert not fixture["out"].exists()
    assert not list(fixture["out"].parent.glob(f".{fixture['out'].name}.staging-*"))


def test_symlinked_selected_artifact_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch)
    selected_path = fixture["selected"] / "obj_00.json"
    selected = json.loads(selected_path.read_text())
    raw_role = "raw_mesh"
    original = tmp_path / selected["selected_asset"]["artifact_paths"][raw_role]
    target = original.with_name("target.ply")
    original.rename(target)
    original.symlink_to(target.name)
    with pytest.raises(ValueError, match="symlink"):
        materializer.materialize_factory_variant(
            e3_root=fixture["e3_root"], scene_id=SCENE, policy_id="A4", out=fixture["out"]
        )
    assert not fixture["out"].exists()


def test_revalidation_rejects_dot_segment_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch, policy="A0")
    materializer.materialize_factory_variant(
        e3_root=fixture["e3_root"], scene_id=SCENE, policy_id="A0", out=fixture["out"]
    )
    alias_parent = fixture["out"].parent / "alias"
    alias_parent.mkdir()
    aliased = alias_parent / ".." / fixture["out"].name
    with pytest.raises(ValueError, match="dot segments"):
        materializer.validate_materialized_factory(
            aliased,
            expected_scene_id=SCENE,
            expected_policy_id="A0",
            repository_root=tmp_path,
        )


def test_revalidation_rejects_self_resealed_fake_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch, policy="A0")
    materializer.materialize_factory_variant(
        e3_root=fixture["e3_root"], scene_id=SCENE, policy_id="A0", out=fixture["out"]
    )
    path = fixture["out"] / "materialization_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["input_identities"]["resolved_jobs"]["sha256"] = "0" * 64
    _reseal_materialization(fixture["out"], manifest)
    with pytest.raises(ValueError, match="input identities differ"):
        materializer.validate_materialized_factory(
            fixture["out"],
            expected_scene_id=SCENE,
            expected_policy_id="A0",
            repository_root=tmp_path,
        )


def test_noncanonical_source_order_is_materialized_in_numeric_slot_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch, policy="A0")
    inventory = fixture["e3_root"] / "input_inventory"
    jobs_path = inventory / "resolved_jobs.json"
    jobs = json.loads(jobs_path.read_text())
    jobs["scenes"][0]["jobs"].reverse()
    _write_json(jobs_path, jobs)

    source_manifest_path = tmp_path / "fidelity" / SCENE / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_manifest["construction_sources"]["factorized_gt_discovery"][
        "accepted_object_ids"
    ] = [1, 0]
    _write_json(source_manifest_path, source_manifest)

    audit_path = inventory / "inventory_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["scenes"][0]["job_ids"].reverse()
    audit["scenes"][0]["source_identity_manifest"] = _identity(
        tmp_path, source_manifest_path
    )
    _write_json(audit_path, audit)
    _reseal_inventory(fixture["e3_root"])

    manifest = materializer.materialize_factory_variant(
        e3_root=fixture["e3_root"], scene_id=SCENE, policy_id="A0", out=fixture["out"]
    )
    assert manifest["roster"]["object_slots"] == ["obj_00", "obj_01"]
    materializer.validate_materialized_factory(
        fixture["out"],
        expected_scene_id=SCENE,
        expected_policy_id="A0",
        repository_root=tmp_path,
    )


def test_short_source_commit_fails_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch, policy="A0")
    jobs_path = fixture["e3_root"] / "input_inventory" / "resolved_jobs.json"
    jobs = json.loads(jobs_path.read_text())
    jobs["source_contract"]["code_commit"] = "abc1234"
    _write_json(jobs_path, jobs)
    _reseal_inventory(fixture["e3_root"])
    with pytest.raises(ValueError, match="full Git SHA"):
        materializer.materialize_factory_variant(
            e3_root=fixture["e3_root"],
            scene_id=SCENE,
            policy_id="A0",
            out=fixture["out"],
        )
    assert not fixture["out"].exists()


def test_selected_tool_must_match_proposal_and_raw_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch, policy="A0")
    selected_path = fixture["selected"] / "obj_00.json"
    selected = json.loads(selected_path.read_text())
    selected["selected_asset"]["tool"] = "reconviagen"
    _write_json(selected_path, selected)
    _reseal_aggregate(fixture["e3_root"], "A0")
    with pytest.raises(ValueError, match="tool/proposal/raw variant differ"):
        materializer.materialize_factory_variant(
            e3_root=fixture["e3_root"],
            scene_id=SCENE,
            policy_id="A0",
            out=fixture["out"],
        )
    assert not fixture["out"].exists()


def test_aggregate_claim_freeze_drift_fails_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch, policy="A0")
    result_path = fixture["e3_root"] / "agentic_ablation.json"
    result = json.loads(result_path.read_text())
    result["freeze_id"] = "different-freeze"
    _write_json(result_path, result)
    _reseal_aggregate(fixture["e3_root"], "A0")
    with pytest.raises(ValueError, match="claim status differs"):
        materializer.materialize_factory_variant(
            e3_root=fixture["e3_root"],
            scene_id=SCENE,
            policy_id="A0",
            out=fixture["out"],
        )
    assert not fixture["out"].exists()
