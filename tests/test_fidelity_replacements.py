from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from agents.discover.factory_prepare import (
    GT_SURFACE_SAMPLE_SEED,
    load_frame_allowlist,
    load_output_index_map,
)
from agents.assets.factory_align import (ALIGN_MESH_SAMPLE_SEED,
                                         SIZE_RATIO_RANGE, TIER_A_F1_20,
                                         TIER_B_F1_40)
from agents.assets.s5_align import (ICP_DIST, SIGNED_SOURCE_UP_HYPOTHESES,
                                    SIGNED_SOURCE_UP_TIE_EPS_M,
                                    TILT_SNAP_DEG, YAW_STEP_DEG)
from agents.core.common import resolve_sam3_ckpt
from robo.eval.fidelity_replacements import (
    EXPECTED_BUNDLES,
    EXPECTED_STAGE_PARAMETERS,
    ReplacementError,
    _canonical_hash,
    _load_config,
    _tree_inventory,
    build_replacement_bundle,
    validate_replacement_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = REPO_ROOT / "outputs/icra2027/.fidelity-replacements-tests"
CONFIG_SOURCE = REPO_ROOT / "configs/experiments/icra2027/fidelity_manifest.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@contextmanager
def _workspace():
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    root = WORK_ROOT / uuid.uuid4().hex
    root.mkdir()
    try:
        yield root
    finally:
        shutil.rmtree(root)
        try:
            WORK_ROOT.rmdir()
        except OSError:
            pass


def _contract(path: Path, config_path: Path, freeze_id: str) -> None:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "freeze_id": freeze_id,
        "mode": "smoke",
        "created_utc": "ignored by contract hash",
        "environment": {"ignored": True},
        "code": {"commit": commit, "dirty": False},
        "configs": [{
            "field": "fidelity_manifest",
            "source_content_sha256": _sha(config_path),
            "sha256": _canonical_hash(config),
        }],
    }
    payload["contract_sha256"] = _canonical_hash({
        key: value for key, value in payload.items()
        if key not in {"created_utc", "environment", "contract_sha256"}
    })
    _write_json(path, payload)


def _legacy_bundle(root: Path, bundle_id: str) -> Path:
    spec = EXPECTED_BUNDLES[bundle_id]
    source = root / spec["legacy_source"]
    objects = []
    for index, object_id, label, old_frame in spec["targets"]:
        meta = {
            "index": index,
            "gt_object_id": object_id,
            "label": label,
            "frame": old_frame,
        }
        objects.append(meta)
        odir = source / "objects" / f"obj_{index:02d}"
        _write_json(odir / "meta.json", meta)
        _write_json(odir / "aligned.json", {
            "index": index, "label": label, "tier": "A", "rejected": None,
        })
        (odir / "rgba.png").write_bytes(f"legacy-rgba-{index}".encode())
    _write_json(source / "objects" / "objects.json", objects)
    if spec["mode"] == "auto":
        (source / "auto_instances.npz").write_bytes(b"fake-auto-instance-source")
    return source


def _install_fake_model_inputs(root: Path, config: dict) -> dict[str, Path]:
    """Replace the production multi-GB pins with tiny content-addressed fixtures."""
    cache = root / ".cache/icra2027/e2-replacements"
    trellis_source = cache / "trellis-source-fixture"
    trellis = cache / "trellis-fixture"
    dino = cache / "torch/hub/dinov2-fixture"
    dino_checkpoint = cache / "torch/hub/checkpoints/dinov2-fixture.pth"
    sam3_source = cache / "sam3-source-fixture"
    sam3 = cache / "huggingface/hub/sam3-fixture/sam3.pt"
    (trellis_source / "trellis/__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (trellis_source / "trellis/__init__.py").write_bytes(b"# source fixture\n")
    (trellis_source / "third_party/FlexiCubes/README.md").parent.mkdir(parents=True)
    (trellis_source / "third_party/FlexiCubes/README.md").write_bytes(
        b"flexicubes source fixture\n"
    )
    (trellis / "pipeline.json").parent.mkdir(parents=True, exist_ok=True)
    (trellis / "pipeline.json").write_bytes(b"trellis-pipeline-fixture")
    (trellis / "weights/model.bin").parent.mkdir(parents=True)
    (trellis / "weights/model.bin").write_bytes(b"trellis-weights-fixture")
    (dino / "hubconf.py").parent.mkdir(parents=True, exist_ok=True)
    (dino / "hubconf.py").write_bytes(b"def dinov2_vitl14_reg(): pass\n")
    (dino / "dinov2/layers.py").parent.mkdir(parents=True)
    (dino / "dinov2/layers.py").write_bytes(b"# fixture\n")
    dino_checkpoint.parent.mkdir(parents=True)
    dino_checkpoint.write_bytes(b"dinov2-checkpoint-fixture")
    (sam3_source / "sam3/__init__.py").parent.mkdir(parents=True)
    (sam3_source / "sam3/__init__.py").write_bytes(b'__version__ = "0.1.0"\n')
    (sam3_source / "README.md").write_bytes(b"sam3 source fixture\n")
    sam3.parent.mkdir(parents=True)
    sam3.write_bytes(b"sam3-checkpoint-fixture")

    trellis_source_tree = _tree_inventory(
        trellis_source, label="fixture TRELLIS source"
    )
    trellis_tree = _tree_inventory(trellis, label="fixture TRELLIS")
    dino_tree = _tree_inventory(dino, label="fixture DINOv2")
    sam3_tree = _tree_inventory(sam3_source, label="fixture SAM3 source")
    model_inputs = {
        "trellis_source": {
            "path": str(trellis_source.relative_to(root)),
            "upstream_repository": "fixture/TRELLIS",
            "upstream_commit": "3" * 40,
            "flexicubes_commit": "4" * 40,
            "file_count": trellis_source_tree["file_count"],
            "tree_sha256": trellis_source_tree["tree_sha256"],
        },
        "trellis_snapshot": {
            "path": str(trellis.relative_to(root)),
            "upstream_revision": "1" * 40,
            "file_count": trellis_tree["file_count"],
            "tree_sha256": trellis_tree["tree_sha256"],
        },
        "dinov2_source": {
            "path": str(dino.relative_to(root)),
            "upstream_repository": "fixture/dinov2",
            "upstream_commit": None,
            "upstream_commit_reason": "fixture tree is content addressed",
            "file_count": dino_tree["file_count"],
            "tree_sha256": dino_tree["tree_sha256"],
        },
        "dinov2_checkpoint": {
            "path": str(dino_checkpoint.relative_to(root)),
            "size_bytes": dino_checkpoint.stat().st_size,
            "sha256": _sha(dino_checkpoint),
        },
        "sam3_source": {
            "path": str(sam3_source.relative_to(root)),
            "upstream_repository": "fixture/sam3",
            "upstream_commit": "5" * 40,
            "package_version": "0.1.0",
            "file_count": sam3_tree["file_count"],
            "tree_sha256": sam3_tree["tree_sha256"],
        },
        "sam3_checkpoint": {
            "path": str(sam3.relative_to(root)),
            "upstream_revision": "2" * 40,
            "size_bytes": sam3.stat().st_size,
            "sha256": _sha(sam3),
        },
    }
    config["leakage_remediation"]["model_inputs"] = model_inputs
    return {
        "trellis_source": trellis_source,
        "trellis_snapshot": trellis,
        "dinov2_source": dino,
        "dinov2_checkpoint": dino_checkpoint,
        "sam3_source": sam3_source,
        "sam3_checkpoint": sam3,
    }


def _fixture(
    root: Path,
    *,
    bundle_id: str = "0d2ee665be-auto",
    split_overlap: bool = False,
):
    config = json.loads(CONFIG_SOURCE.read_text(encoding="utf-8"))
    model_paths = _install_fake_model_inputs(root, config)
    config_path = root / "configs/experiments/icra2027/fidelity_manifest.json"
    _write_json(config_path, config)
    freeze_id = f"unit-{uuid.uuid4().hex}"
    contract_path = root / "outputs/e0/freeze_manifest.json"
    _contract(contract_path, config_path, freeze_id)
    source = _legacy_bundle(root, bundle_id)
    spec = EXPECTED_BUNDLES[bundle_id]
    scene = root / "dataset/data" / spec["scene_id"]
    images = scene / "dslr/resized_undistorted_images"
    images.mkdir(parents=True)
    train = ["TRAIN000.JPG", "TRAIN001.JPG"]
    test = [target[3] for target in spec["targets"]] + ["HELDOUT.JPG"]
    if split_overlap:
        test.append(train[0])
    _write_json(
        scene / "dslr/train_test_lists.json", {"train": train, "test": test}
    )
    for frame in train:
        (images / frame).write_bytes(f"image-{frame}".encode())
    _write_json(
        scene / "dslr/nerfstudio/transforms_undistorted.json",
        {"camera_model": "PINHOLE"},
    )
    (scene / "dslr/colmap/images.txt").parent.mkdir(parents=True)
    (scene / "dslr/colmap/images.txt").write_bytes(b"fixture poses\n")
    (scene / "scans/mesh_aligned_0.05.ply").parent.mkdir(parents=True)
    (scene / "scans/mesh_aligned_0.05.ply").write_bytes(b"fixture mesh\n")
    _write_json(scene / "scans/segments.json", {"segIndices": []})
    _write_json(scene / "scans/segments_anno.json", {"segGroups": []})
    out = root / (
        f"outputs/icra2027/{freeze_id}/fidelity/replacements/{bundle_id}"
    )
    return {
        "root": root,
        "config": config_path,
        "contract": contract_path,
        "freeze_id": freeze_id,
        "bundle_id": bundle_id,
        "spec": spec,
        "source": source,
        "dataset": root / "dataset",
        "train": train,
        "test": test,
        "out": out,
        "model_paths": model_paths,
    }


def _fake_runner(
    spec: dict,
    *,
    chosen_frame: str = "TRAIN000.JPG",
    omit_object_id: int | None = None,
    reject_alignment: bool = False,
    fail_stage: str | None = None,
):
    def run(stage, command, environment, staging, repo_root):
        assert Path(environment["SIMANY_OUT"]) == staging
        assert environment["SIMANY_SCENE"] == spec["scene_id"]
        assert Path(repo_root).is_dir()
        assert command[1:3] == ["-m", command[2]]
        assert Path(environment["SIMANY_TRELLIS_DIR"]).is_dir()
        assert Path(environment["SIMANY_TRELLIS_MODEL"]).is_dir()
        assert Path(environment["SIMANY_DINOV2_REPO"]).is_dir()
        assert Path(environment["SIMANY_SAM3_CKPT"]).is_file()
        if stage == "factory_refine_masks":
            assert environment["PYTHONPATH"].endswith("sam3-source-fixture")
        else:
            assert "PYTHONPATH" not in environment
        assert "PYTHONOPTIMIZE" not in environment
        assert Path(environment["TORCH_HOME"]).is_dir()
        assert "SIMANY_MESH_SRC" not in environment
        assert "SIMF_MESH_SRC" not in environment
        assert "SIMANY_FULL" not in environment
        assert "SIMF_FULL" not in environment
        assert environment["SIMANY_MIN_BBOX_PX"] == "48"
        assert environment["SIMANY_VIS_TOL_M"] == "0.02"
        assert environment["SIMANY_MIN_MASK_PX"] == "400"
        assert environment["ATTN_BACKEND"] == "xformers"
        assert environment["SPCONV_ALGO"] == "native"
        if fail_stage == stage:
            raise RuntimeError(f"injected {stage} failure")
        if stage == "factory_prepare":
            objects = []
            for index, object_id, label, _old_frame in spec["targets"]:
                if object_id == omit_object_id:
                    continue
                meta = {
                    "index": index,
                    "gt_object_id": object_id,
                    "label": label,
                    "frame": chosen_frame,
                    "gt_surface_sample_seed": GT_SURFACE_SAMPLE_SEED,
                }
                objects.append(meta)
                odir = staging / "objects" / f"obj_{index:02d}"
                _write_json(odir / "meta.json", meta)
                (odir / "rgba.png").write_bytes(f"new-rgba-{index}".encode())
                (odir / "gt_points.ply").write_bytes(f"points-{index}".encode())
            _write_json(staging / "objects/objects.json", objects)
            _write_json(
                staging / "provenance/factory_read_frames.json",
                {"frames": [chosen_frame]},
            )
        elif stage == "factory_refine_masks":
            for index, *_ in spec["targets"]:
                path = staging / "objects" / f"obj_{index:02d}/rgba.png"
                path.write_bytes(path.read_bytes() + b"-refined")
        elif stage == "trellis":
            for index, *_ in spec["targets"]:
                odir = staging / "objects" / f"obj_{index:02d}"
                for name in (
                    "trellis_mesh.ply", "mesh_sim.ply", "mesh_sim.obj",
                    "trellis_gs.ply",
                ):
                    (odir / name).write_bytes(f"{name}-{index}".encode())
        elif stage == "factory_align":
            rows = []
            for index, _object_id, label, _old_frame in spec["targets"]:
                rejected = "injected rejection" if reject_alignment else None
                row = {
                    "index": index,
                    "label": label,
                    "tier": "C" if reject_alignment else "A",
                    "rejected": rejected,
                    "source_up_hypothesis": "+z",
                    "alignment_mesh_sample_seed": 42,
                    "T": [[1, 0, 0, 0], [0, 1, 0, 0],
                          [0, 0, 1, 0], [0, 0, 0, 1]],
                }
                _write_json(
                    staging / "objects" / f"obj_{index:02d}/aligned.json", row
                )
                rows.append(row)
            _write_json(staging / "objects/aligned_all.json", rows)
        else:  # pragma: no cover - protects the test harness from stage drift
            raise AssertionError(stage)
    return run


def _build(fixture, runner):
    def sam3_probe(_python, environment, source):
        assert environment["PYTHONPATH"] == str(source)
        assert "PYTHONOPTIMIZE" not in environment
        return {
            "module_file": str(source / "sam3/__init__.py"),
            "package_version": "0.1.0",
        }

    return build_replacement_bundle(
        config_path=fixture["config"],
        contract_manifest=fixture["contract"],
        bundle_id=fixture["bundle_id"],
        freeze_id=fixture["freeze_id"],
        out=fixture["out"],
        dataset_root=fixture["dataset"],
        python="python-main",
        sam3_python="python-sam3",
        repo_root=fixture["root"],
        allow_dirty=True,
        _stage_runner=runner,
        _sam3_runtime_probe=sam3_probe,
    )


def _validate(fixture):
    return validate_replacement_bundle(
        config_path=fixture["config"],
        contract_manifest=fixture["contract"],
        bundle_id=fixture["bundle_id"],
        freeze_id=fixture["freeze_id"],
        out=fixture["out"],
        dataset_root=fixture["dataset"],
        repo_root=fixture["root"],
        allow_dirty=True,
    )


def test_config_pins_exact_five_bundles_and_eight_targets():
    config, bundles = _load_config(CONFIG_SOURCE)
    assert config["leakage_remediation"]["require_train_only_generation"] is True
    assert set(bundles) == set(EXPECTED_BUNDLES)
    assert sum(len(bundle["targets"]) for bundle in bundles.values()) == 8
    assert {
        (bundle["scene_id"], bundle["mode"], target["legacy_index"],
         target["gt_object_id"], target["old_frame"])
        for bundle in bundles.values() for target in bundle["targets"]
    } == {
        (spec["scene_id"], spec["mode"], index, object_id, old_frame)
        for spec in EXPECTED_BUNDLES.values()
        for index, object_id, _label, old_frame in spec["targets"]
    }


def test_factory_prepare_control_parsers_are_strict():
    with _workspace() as root:
        allowlist = root / "allow.json"
        index_map = root / "map.json"
        _write_json(allowlist, {"frames": ["TRAIN000.JPG", "TRAIN001.JPG"]})
        _write_json(index_map, {"1007": 6, "1028": 26})
        assert load_frame_allowlist(allowlist) == {"TRAIN000.JPG", "TRAIN001.JPG"}
        assert load_output_index_map(index_map) == {1007: 6, 1028: 26}
        _write_json(root / "nested.json", ["nested/TRAIN.JPG"])
        with pytest.raises(ValueError, match="plain"):
            load_frame_allowlist(root / "nested.json")
        _write_json(root / "collision.json", {"1007": 6, "1028": 6})
        with pytest.raises(ValueError, match="unique"):
            load_output_index_map(root / "collision.json")


def test_manifest_alignment_policy_matches_production_constants():
    assert EXPECTED_STAGE_PARAMETERS["factory_prepare"][
        "gt_surface_sample_seed"
    ] == GT_SURFACE_SAMPLE_SEED
    policy = EXPECTED_STAGE_PARAMETERS["factory_align"]
    assert policy["source_up_hypotheses_order"] == [
        name for name, _ in SIGNED_SOURCE_UP_HYPOTHESES
    ]
    assert policy["yaw_step_deg"] == YAW_STEP_DEG
    assert policy["cross_hypothesis_tie_epsilon_m"] == \
        SIGNED_SOURCE_UP_TIE_EPS_M
    assert policy["icp_distance_m"] == ICP_DIST
    assert policy["tilt_snap_deg"] == TILT_SNAP_DEG
    assert policy["alignment_mesh_sample_seed"] == ALIGN_MESH_SAMPLE_SEED
    assert policy["size_ratio_range"] == list(SIZE_RATIO_RANGE)
    assert policy["tier_a_f1_20"] == TIER_A_F1_20
    assert policy["tier_b_f1_40"] == TIER_B_F1_40


def test_happy_path_preserves_sparse_indices_hashes_and_legacy_sources():
    with _workspace() as root:
        fixture = _fixture(root)
        source_snapshot = {
            str(path.relative_to(fixture["source"])): path.read_bytes()
            for path in fixture["source"].rglob("*") if path.is_file()
        }
        manifest = _build(fixture, _fake_runner(fixture["spec"]))
        assert fixture["out"].is_dir()
        assert manifest["optimization_input_frames"] == ["TRAIN000.JPG"]
        assert [row["legacy_index"] for row in manifest["targets"]] == [6, 26]
        assert [row["gt_object_id"] for row in manifest["targets"]] == [1007, 1028]
        assert all(row["new_train_frame"] in fixture["train"]
                   for row in manifest["targets"])
        assert all(row["old_test_frame"] in fixture["test"]
                   for row in manifest["targets"])
        assert not (fixture["out"] / "objects/obj_00").exists()
        assert (fixture["out"] / "objects/obj_06/trellis_gs.ply").is_file()
        assert (fixture["out"] / "objects/obj_26/aligned.json").is_file()
        assert (fixture["out"] / "auto_instances.npz").read_bytes() \
            == (fixture["source"] / "auto_instances.npz").read_bytes()
        assert all(
            len(artifact["sha256"]) == 64 and artifact["size_bytes"] > 0
            for row in manifest["targets"] for artifact in row["generated_artifacts"]
        )
        assert manifest["model_inputs"]["declared"] == json.loads(
            fixture["config"].read_text()
        )["leakage_remediation"]["model_inputs"]
        assert manifest["model_inputs"]["observed"]["trellis_snapshot"][
            "file_count"
        ] == 2
        assert manifest["model_inputs"]["observed"]["trellis_source"][
            "flexicubes_commit"
        ] == "4" * 40
        assert manifest["model_inputs"]["observed"]["dinov2_source"][
            "upstream_commit_provenance"
        ] == "unavailable_from_gitless_torch_hub_snapshot"
        assert manifest["model_inputs"]["runtime"]["sam3"][
            "module_file"
        ].endswith("sam3-source-fixture/sam3/__init__.py")
        assert manifest["stage_parameters"]["factory_prepare"] == {
            "min_bbox_px": 48,
            "visibility_tolerance_m": 0.02,
            "min_mask_px": 400,
            "full_vocabulary": False,
            "gt_surface_sample_seed": 42,
        }
        assert manifest["stage_parameters"]["factory_align"] == {
            "policy": "signed_source_up_v1",
            "source_up_hypotheses_order": [
                "+z", "-z", "+x", "-x", "+y", "-y",
            ],
            "yaw_step_deg": 10,
            "cross_hypothesis_tie_epsilon_m": 1e-9,
            "icp_distance_m": 0.03,
            "tilt_snap_deg": 15.0,
            "alignment_mesh_sample_seed": 42,
            "size_ratio_range": [0.4, 2.5],
            "tier_a_f1_20": 0.40,
            "tier_b_f1_40": 0.20,
        }
        assert manifest["dataset_inputs"]["read_images"][0]["frame"] == (
            "TRAIN000.JPG"
        )
        assert manifest["dataset_inputs"]["read_frames_control"]["path"] == (
            "provenance/factory_read_frames.json"
        )
        assert json.loads(
            (fixture["out"] / "replacement_manifest.json").read_text()
        )["validation"]["generation_eval_overlap"] == []
        assert source_snapshot == {
            str(path.relative_to(fixture["source"])): path.read_bytes()
            for path in fixture["source"].rglob("*") if path.is_file()
        }
        assert not list(fixture["out"].parent.glob(
            f".{fixture['bundle_id']}.staging-*"
        ))


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        ("missing", "exact target population"),
        ("leakage", "not train-only"),
        ("rejected", "alignment rejected"),
        ("stage", "injected trellis failure"),
    ],
)
def test_fail_closed_cases_leave_no_published_or_staging_bundle(failure, match):
    with _workspace() as root:
        fixture = _fixture(root)
        kwargs = {}
        if failure == "missing":
            kwargs["omit_object_id"] = fixture["spec"]["targets"][-1][1]
        elif failure == "leakage":
            kwargs["chosen_frame"] = fixture["spec"]["targets"][0][3]
        elif failure == "rejected":
            kwargs["reject_alignment"] = True
        elif failure == "stage":
            kwargs["fail_stage"] = "trellis"
        with pytest.raises((ReplacementError, RuntimeError), match=match):
            _build(fixture, _fake_runner(fixture["spec"], **kwargs))
        assert not fixture["out"].exists()
        assert not list(fixture["out"].parent.glob(
            f".{fixture['bundle_id']}.staging-*"
        ))


def test_official_split_overlap_and_contract_drift_fail_before_publish():
    with _workspace() as root:
        fixture = _fixture(root, split_overlap=True)
        with pytest.raises(ReplacementError, match="train/test frames overlap"):
            _build(fixture, _fake_runner(fixture["spec"]))
        assert not fixture["out"].exists()

    with _workspace() as root:
        fixture = _fixture(root)
        config = json.loads(fixture["config"].read_text())
        config["paper_table"] = "drifted-after-freeze"
        _write_json(fixture["config"], config)
        with pytest.raises(ReplacementError, match="config bytes differ"):
            _build(fixture, _fake_runner(fixture["spec"]))
        assert not fixture["out"].exists()


def test_model_input_hash_drift_fails_before_publish():
    with _workspace() as root:
        fixture = _fixture(root)
        model_file = fixture["model_paths"]["trellis_snapshot"] / "pipeline.json"
        model_file.write_bytes(model_file.read_bytes() + b"-drift")
        with pytest.raises(ReplacementError, match="tree hash differs"):
            _build(fixture, _fake_runner(fixture["spec"]))
        assert not fixture["out"].exists()


def test_sam3_checkpoint_override_is_strict(monkeypatch):
    with _workspace() as root:
        checkpoint = root / "sam3.pt"
        checkpoint.write_bytes(b"sam3")
        monkeypatch.setenv("SIMANY_SAM3_CKPT", str(checkpoint))
        assert resolve_sam3_ckpt() == str(checkpoint)
        monkeypatch.setenv("SIMANY_SAM3_CKPT", "relative/sam3.pt")
        with pytest.raises(ValueError, match="absolute"):
            resolve_sam3_ckpt()


def test_refuses_to_overwrite_a_completed_bundle():
    with _workspace() as root:
        fixture = _fixture(root, bundle_id="286b55a2bf-factory")
        runner = _fake_runner(fixture["spec"])
        _build(fixture, runner)
        assert _validate(fixture)["bundle_id"] == fixture["bundle_id"]
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            _build(fixture, runner)


def test_resume_validator_rejects_alignment_policy_drift():
    with _workspace() as root:
        fixture = _fixture(root, bundle_id="3f15a9266d-factory")
        _build(fixture, _fake_runner(fixture["spec"]))
        manifest_path = fixture["out"] / "replacement_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["stage_parameters"]["factory_align"][
            "source_up_hypotheses_order"
        ] = ["+x", "-x", "+y", "-y", "+z", "-z"]
        _write_json(manifest_path, manifest)

        with pytest.raises(ReplacementError, match="stage parameters mismatch"):
            _validate(fixture)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("source_up_hypothesis", "diagonal"),
        ("alignment_mesh_sample_seed", 7),
    ],
)
def test_resume_validator_rejects_alignment_record_policy_drift(field, bad_value):
    with _workspace() as root:
        fixture = _fixture(root, bundle_id="3f15a9266d-factory")
        _build(fixture, _fake_runner(fixture["spec"]))
        aligned_path = fixture["out"] / "objects/obj_193/aligned.json"
        aligned = json.loads(aligned_path.read_text(encoding="utf-8"))
        aligned[field] = bad_value
        _write_json(aligned_path, aligned)

        with pytest.raises(ReplacementError, match="exact policy provenance"):
            _validate(fixture)


def test_resume_validator_rejects_gt_surface_sample_seed_drift():
    with _workspace() as root:
        fixture = _fixture(root, bundle_id="3f15a9266d-factory")
        _build(fixture, _fake_runner(fixture["spec"]))
        objects_path = fixture["out"] / "objects/objects.json"
        objects = json.loads(objects_path.read_text(encoding="utf-8"))
        objects[0]["gt_surface_sample_seed"] = 7
        _write_json(objects_path, objects)

        with pytest.raises(ReplacementError, match="new metadata mismatch"):
            _validate(fixture)


def test_resume_validator_rejects_hash_drift_and_undeclared_files():
    with _workspace() as root:
        fixture = _fixture(root, bundle_id="3f15a9266d-factory")
        _build(fixture, _fake_runner(fixture["spec"]))
        mesh = fixture["out"] / "objects/obj_193/trellis_mesh.ply"
        mesh.write_bytes(mesh.read_bytes() + b"tampered")
        with pytest.raises(ReplacementError, match="size mismatch|hash mismatch"):
            _validate(fixture)

    with _workspace() as root:
        fixture = _fixture(root, bundle_id="286b55a2bf-factory")
        _build(fixture, _fake_runner(fixture["spec"]))
        (fixture["out"] / "undeclared.bin").write_bytes(b"not in manifest")
        with pytest.raises(ReplacementError, match="closure mismatch"):
            _validate(fixture)

    with _workspace() as root:
        fixture = _fixture(root, bundle_id="0d2ee665be-factory")
        _build(fixture, _fake_runner(fixture["spec"]))
        image = fixture["dataset"] / (
            "data/0d2ee665be/dslr/resized_undistorted_images/TRAIN000.JPG"
        )
        image.write_bytes(image.read_bytes() + b"tampered")
        with pytest.raises(ReplacementError, match="size mismatch|hash mismatch"):
            _validate(fixture)


def test_config_target_drift_is_rejected_even_when_json_is_well_formed():
    payload = json.loads(CONFIG_SOURCE.read_text())
    payload = copy.deepcopy(payload)
    payload["leakage_remediation"]["bundles"][0]["targets"][0][
        "gt_object_id"
    ] = 9999
    with _workspace() as root:
        path = root / "drift.json"
        _write_json(path, payload)
        with pytest.raises(ReplacementError, match="target contract drifted"):
            _load_config(path)


def test_launcher_is_non_array_with_pinned_bundle_and_strict_resume_validation():
    launcher = (
        REPO_ROOT / "run/slurm/icra2027_e2_replacements.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --array" not in launcher
    assert "e2-replace-%j.out" in launcher
    assert "e2-replace-%j.err" in launcher
    assert ': "${E2_BUNDLE_ID:?' in launcher
    assert "replacement launcher rejects Slurm array jobs" in launcher
    assert "BUNDLE_ID=$E2_BUNDLE_ID" in launcher
    for bundle_id in EXPECTED_BUNDLES:
        assert bundle_id in launcher
    assert "--validate-existing" in launcher
    assert "E2_REPLACEMENT_JOB=REUSED" in launcher
    assert '"${VALIDATE_COMMAND[@]}"' in launcher
    assert "map(Path" not in launcher
    assert "fidelity/replacements/$BUNDLE_ID" in launcher
    assert 'export TMPDIR="$RUNTIME/tmp"' in launcher
    assert "#SBATCH --qos=" not in launcher
    assert "#SBATCH --partition=" not in launcher
    assert "#SBATCH --nodelist=" not in launcher
    assert "#SBATCH --gpus=" not in launcher
    assert 'export HF_HOME="$RUNTIME/huggingface"' in launcher
    assert "SIMANY_TRELLIS_MODEL" in launcher
    assert "SIMANY_TRELLIS_DIR" in launcher
    assert "SIMANY_DINOV2_REPO" in launcher
    assert "SIMANY_SAM3_CKPT" in launcher
    assert "SLURM_ARRAY_TASK_ID:?" not in launcher
    assert "SLURM_ARRAY_TASK_COUNT:?" not in launcher
    assert "SLURM_ARRAY_TASK_MIN:?" not in launcher
    assert "SLURM_ARRAY_TASK_MAX:?" not in launcher
    assert "SLURM_ARRAY_TASK_STEP:?" not in launcher
    assert "torch.cuda.device_count() != 1" in launcher
    assert "DATASET_ROOT=/data/ScanNetpp" in launcher
    assert "_validate_model_inputs" in launcher
    assert "SLURMD_NODENAME" in launcher
    assert "SLURM_JOB_PARTITION" in launcher
    assert "SLURM_JOB_QOS" in launcher
    assert "unset PYTHONPATH PYTHONOPTIMIZE" in launcher
    assert "SIMANY_SAM3_PY:-" not in launcher
    assert 'PYTHON_ENV="$ROOT/.envs/sam3d-objects"' in launcher
    assert 'PYTHON="$PYTHON_ENV/bin/python"' in launcher
    assert 'SAM3PY="$PYTHON"' in launcher
    assert "${CONDA_ROOT:-$HOME/miniconda3}" not in launcher
    assert 'SAM3_SOURCE="${MODEL_INPUTS[4]}"' in launcher
    assert 'SAM3_VERSION="${MODEL_INPUTS[7]}"' in launcher
    assert 'export PATH="$PYTHON_ENV/bin:$PATH"' in launcher
    assert "export ATTN_BACKEND=xformers" in launcher
    assert "export SPCONV_ALGO=native" in launcher
    assert '"spconv", "spconv.pytorch", "ninja", "gsplat"' in launcher
    assert '"torch": "2.5.1+cu121"' in launcher
    assert '"spconv": "2.3.8"' in launcher
    assert '"ninja": "1.13.0"' in launcher
    assert "TrellisImageTo3DPipeline" in launcher
    assert 'PYTHONPATH="$SAM3_SOURCE" "$SAM3PY"' in launcher
    assert 'importlib.import_module("sam3.model_builder")' in launcher
    assert 'importlib.import_module("sam3.model.sam3_image_processor")' in launcher
    assert "replacement_main_runtime=" in launcher
    assert "replacement_sam3_runtime=" in launcher
    assert "replacement_runtime_probe=PASS" in launcher
    assert "-m robo.eval.fidelity_replacements --help" in launcher
    assert "-m agents.discover.factory_prepare --help" in launcher
    assert "-m agents.discover.factory_refine_masks --help" in launcher
    assert "${HF_HOME:-$HOME/.cache/huggingface}" not in launcher
    assert "${TORCH_HOME:-$HOME/.cache/torch}" not in launcher


def test_replacement_launcher_requires_exact_gpu_profile_and_sm80_caches():
    launcher = (
        REPO_ROOT / "run/slurm/icra2027_e2_replacements.sbatch"
    ).read_text(encoding="utf-8")

    assert ': "${E2_GPU_PROFILE:?replacement jobs require E2_GPU_PROFILE=' in launcher
    assert "hala-a6000)" in launcher
    assert "E2_EXPECTED_NODE=hala" in launcher
    assert "E2_EXPECTED_PARTITION=debug" in launcher
    assert "E2_EXPECTED_QOS=debug" in launcher
    assert 'E2_EXPECTED_GPU_NAME="NVIDIA RTX A6000"' in launcher
    assert "E2_EXPECTED_GPU_CC=8.6" in launcher
    assert "gcp-a100)" in launcher
    assert "E2_EXPECTED_NODE=gcp-eu1-a100-80g-qrfh" in launcher
    assert "E2_EXPECTED_PARTITION=batch" in launcher
    assert "E2_EXPECTED_QOS=normal" in launcher
    assert 'E2_EXPECTED_GPU_NAME="NVIDIA A100-SXM4-80GB"' in launcher
    assert "E2_EXPECTED_GPU_CC=8.0" in launcher
    assert "E2_TORCH_CUDA_ARCH_LIST=8.0" in launcher
    assert "E2_CUMM_CUDA_ARCH_LIST=8.0" in launcher
    assert 'TORCH_EXTENSIONS_DIR="$RUNTIME/gcp-a100/torch-extensions-sm80"' in launcher
    assert 'CUDA_CACHE_PATH="$RUNTIME/gcp-a100/cuda-sm80"' in launcher
    assert 'TRITON_CACHE_DIR="$RUNTIME/gcp-a100/triton-sm80"' in launcher
    assert 'export TORCH_CUDA_ARCH_LIST="$E2_TORCH_CUDA_ARCH_LIST"' in launcher
    assert 'export CUMM_CUDA_ARCH_LIST="$E2_CUMM_CUDA_ARCH_LIST"' in launcher
    assert "CUDA_VISIBLE_DEVICES" in launcher
    assert "torch.cuda.is_available()" in launcher
    assert "torch.cuda.get_device_name(0)" in launcher
    assert "torch.cuda.get_device_capability(0)" in launcher
    assert "observed_name != expected_name" in launcher
    assert "observed_cc != expected_cc" in launcher
