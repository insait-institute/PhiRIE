import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import yaml
from PIL import Image

from agents.eval.fidelity_room_export import (
    METHOD_AUTO,
    METHOD_GT,
    METHOD_INPUT,
    _inside_repo,
    _load_replacement_overlay,
    export_scene,
    select_official_frames,
)


def _sha256(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path, relative_to):
    return {
        "path": str(path.relative_to(relative_to)),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _split(n_test=12):
    return {
        "train": [f"train_{index:03d}.JPG" for index in range(3)],
        "test": [f"test_{index:03d}.JPG" for index in range(n_test)],
    }


def test_selects_eight_unique_official_test_frames_deterministically():
    payload = _split()
    registered = set(payload["train"] + payload["test"])
    selected = select_official_frames(payload, registered)
    assert len(selected) == 8
    assert len(set(selected)) == 8
    assert selected[0] == "test_000.JPG"
    assert selected[-1] == "test_011.JPG"
    assert not (set(selected) & set(payload["train"]))


def test_rejects_train_test_leakage():
    payload = _split()
    payload["train"].append(payload["test"][4])
    with pytest.raises(ValueError, match="overlaps"):
        select_official_frames(payload, set(payload["train"] + payload["test"]))


def test_selects_only_views_disjoint_from_generation_inputs():
    payload = _split(n_test=12)
    registered = set(payload["train"] + payload["test"])
    excluded = set(payload["test"][:4])
    selected = select_official_frames(
        payload, registered, excluded_names=excluded
    )
    assert len(selected) == 8
    assert not (set(selected) & excluded)


def test_rejects_when_optimization_inputs_leave_fewer_than_eight_views():
    payload = _split(n_test=10)
    registered = set(payload["train"] + payload["test"])
    with pytest.raises(ValueError, match="disjoint from optimization inputs"):
        select_official_frames(
            payload,
            registered,
            excluded_names=set(payload["test"][:3]),
        )


def test_rejects_legacy_or_incomplete_frame_rosters():
    with pytest.raises(ValueError, match="'train'"):
        select_official_frames({"test": ["a.JPG"] * 8}, {"a.JPG"})
    payload = _split(n_test=7)
    with pytest.raises(ValueError, match="at least 8"):
        select_official_frames(payload, set(payload["train"] + payload["test"]))


def test_rejects_nested_or_colliding_frame_names():
    payload = _split()
    payload["test"][0] = "nested/test_000.JPG"
    with pytest.raises(ValueError, match="plain filename"):
        select_official_frames(payload, set(payload["train"] + payload["test"]))


def test_output_method_ids_are_the_three_declared_non_harmonized_rows():
    assert [METHOD_INPUT, METHOD_GT, METHOD_AUTO] == [
        "input_scene_gaussian",
        "factorized_gt_discovery",
        "factorized_auto_discovery",
    ]


def test_path_guard_rejects_outside_checkout():
    with pytest.raises(ValueError, match="must stay inside"):
        _inside_repo("/etc/passwd", must_exist=True)


def test_module_contains_no_legacy_fallback_contract():
    source = Path(__file__).resolve().parents[1] / "agents/eval/fidelity_room_export.py"
    text = source.read_text(encoding="utf-8")
    assert "legacy-linspace" not in text
    assert "train_test_lists.json" in text


def test_launcher_normalizes_yaml_numeric_scene_ids():
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/experiments/icra2027/fidelity_manifest.json").read_text()
    )
    roster = yaml.safe_load(
        (root / "configs/experiments/icra2027/construction_regimes.yaml").read_text()
    )
    raw_ids = roster["population"]["scene_ids"]
    # PyYAML decodes unquoted all-digit identifiers as integers. The launcher
    # must normalize them before comparing against the JSON string roster.
    assert any(isinstance(value, int) for value in raw_ids)
    assert [str(value).strip() for value in raw_ids] == config["population"]["scene_ids"]
    launcher = (root / "run/slurm/icra2027_e2_fidelity.sbatch").read_text()
    assert "roster_scene_ids = [str(value).strip()" in launcher
    assert "E2_RENDER_ROOM=REUSED" in launcher
    assert 'postcheck_room_scene "$SCENE_OUT" "$SCENE_ID"' in launcher


def test_launcher_requires_individual_scene_jobs_and_rejects_array_context():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "run/slurm/icra2027_e2_fidelity.sbatch").read_text()

    assert (
        ': "${E2_SCENE_ID:?render-room requires one exact E2_SCENE_ID per '
        'individual job}"'
    ) in launcher
    assert '[[ "$E2_SCENE_ID" =~ ^[0-9a-f]{10}$ ]]' in launcher
    assert 'TASK_TAG="scene-$E2_SCENE_ID"' in launcher
    assert 'SCENE_ID="$E2_SCENE_ID"' in launcher
    assert 'for EXPECTED_SCENE_ID in "${SCENES[@]}"' in launcher
    assert 'not in the exact $E2_MODE scene roster' in launcher

    for variable in (
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_ARRAY_TASK_COUNT",
        "SLURM_ARRAY_TASK_MIN",
        "SLURM_ARRAY_TASK_MAX",
        "SLURM_ARRAY_TASK_STEP",
    ):
        assert variable in launcher
    assert 'if [[ -v "$ARRAY_VARIABLE" ]]' in launcher
    assert "E2 must not run with Slurm array context" in launcher
    assert 'SCENES[$SLURM_ARRAY_TASK_ID]' not in launcher
    assert "render-room must be submitted as an array" not in launcher


def test_launcher_requires_explicit_validated_gpu_profile_with_sm80_caches():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "run/slurm/icra2027_e2_fidelity.sbatch").read_text()

    assert ': "${E2_GPU_PROFILE:?GPU phases require E2_GPU_PROFILE=' in launcher
    assert "hala-a6000" in launcher
    assert "gcp-a100" in launcher
    assert "E2_EXPECTED_NODE=hala" in launcher
    assert "E2_EXPECTED_PARTITION=debug" in launcher
    assert "E2_EXPECTED_QOS=debug" in launcher
    assert 'E2_EXPECTED_GPU_NAME="NVIDIA RTX A6000"' in launcher
    assert "E2_EXPECTED_GPU_CC=8.6" in launcher
    assert "E2_EXPECTED_NODE=gcp-eu1-a100-80g-qrfh" in launcher
    assert "E2_EXPECTED_PARTITION=batch" in launcher
    assert "E2_EXPECTED_QOS=normal" in launcher
    assert 'E2_EXPECTED_GPU_NAME="NVIDIA A100-SXM4-80GB"' in launcher
    assert "E2_EXPECTED_GPU_CC=8.0" in launcher
    assert "gcp-a100/torch-extensions-sm80" in launcher
    assert "gcp-a100/cuda-sm80" in launcher
    assert "torch.cuda.device_count() != 1" in launcher
    assert "torch.cuda.get_device_name(0)" in launcher
    assert "torch.cuda.get_device_capability(0)" in launcher
    assert "observed_name != expected_name" in launcher
    assert "observed_cc != expected_cc" in launcher
    assert "inventory is CPU-only and must not set E2_GPU_PROFILE" in launcher
    assert "validate_gpu_runtime render-room" in launcher
    assert "validate_gpu_runtime metrics" in launcher


def test_cpu_fake_export_writes_exact_closed_scene_bundle(tmp_path, monkeypatch):
    scene_id = "0123456789"
    factory = tmp_path / f"{scene_id}_factory"
    automatic = tmp_path / f"{scene_id}_auto"
    for source in (factory, automatic):
        (source / "objects").mkdir(parents=True)
        (source / "objects" / "objects.json").write_text("[]\n")

    scene_dir = tmp_path / "dataset" / scene_id
    dslr_dir = scene_dir / "dslr"
    images_dir = dslr_dir / "images"
    images_dir.mkdir(parents=True)
    train = ["train_000.JPG"]
    test = [f"test_{index:03d}.JPG" for index in range(8)]
    for index, name in enumerate(test):
        Image.fromarray(np.full((4, 6, 3), index, dtype=np.uint8)).save(
            images_dir / name
        )
    split_path = dslr_dir / "train_test_lists.json"
    split_path.write_text(json.dumps({"train": train, "test": test}))
    transforms = scene_dir / "transforms.json"
    poses = scene_dir / "images.txt"
    splat = scene_dir / "scene.ply"
    transforms.write_text("{}\n")
    poses.write_text("poses\n")
    splat.write_bytes(b"fake-splat")

    background = {"sh_degree": 0}
    fake_common = SimpleNamespace(
        SCENE_ID=scene_id,
        SCENE_DIR=scene_dir,
        IMAGES_DIR=images_dir,
        SPLAT_PLY=splat,
        TRANSFORMS_JSON=transforms,
        COLMAP_IMAGES_TXT=poses,
        load_colmap_w2c=lambda: {name: np.eye(4) for name in test},
        load_intrinsics=lambda: (np.eye(3), 6, 4, {}),
        load_gaussians=lambda _path: background,
        render_view=lambda *_args, **_kwargs: (
            np.zeros((4, 6, 3), dtype=np.float32), None, np.ones((4, 6))
        ),
    )
    import agents.core
    monkeypatch.setattr(agents.core, "common", fake_common, raising=False)
    monkeypatch.setitem(sys.modules, "agents.core.common", fake_common)
    monkeypatch.setenv("SIMANY_ROOT", str(Path(__file__).resolve().parents[1]))
    monkeypatch.setenv("SIMANY_SCENE", scene_id)
    monkeypatch.setenv("SIMANY_OUT", str(factory))

    out = tmp_path / "room_runs" / scene_id
    manifest = export_scene(
        freeze_id="test-freeze",
        scene_id=scene_id,
        factory_source=factory,
        auto_source=automatic,
        out=out,
        allow_dirty=True,
    )

    assert out.is_dir()
    assert len(list((out / "gt").glob("*.png"))) == 8
    for method in (METHOD_INPUT, METHOD_GT, METHOD_AUTO):
        assert len(list((out / method).glob("*.png"))) == 8
    assert len(manifest["output_artifacts"]) == 32
    assert manifest["optimization_input_frames"] == train
    assert len({row["relative_path"] for row in manifest["output_artifacts"]}) == 32
    assert json.loads((out / "manifest.json").read_text())["freeze_id"] == "test-freeze"
    assert not list(out.parent.glob(f".{scene_id}.partial-*"))


def test_failed_export_removes_only_its_staging_directory(tmp_path, monkeypatch):
    import agents.eval.fidelity_room_export as exporter

    scene_id = "0123456789"
    out = tmp_path / "room_runs" / scene_id

    def fail_after_staging(**kwargs):
        output = Path(kwargs["out"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_name(f".{output.name}.partial-{__import__('os').getpid()}").mkdir()
        raise RuntimeError("injected failure after staging")

    monkeypatch.setattr(exporter, "_export_scene_impl", fail_after_staging)
    with pytest.raises(RuntimeError, match="injected failure"):
        export_scene(
            freeze_id="test-freeze",
            scene_id=scene_id,
            factory_source=tmp_path / "factory",
            auto_source=tmp_path / "auto",
            out=out,
        )
    assert not out.exists()
    assert not list(out.parent.glob(f".{scene_id}.partial-*"))


def test_loads_and_rehashes_frozen_replacement_overlay(tmp_path, monkeypatch):
    import agents.eval.fidelity_room_export as exporter

    monkeypatch.setattr(exporter, "REPO_ROOT", tmp_path)
    freeze_id = "e2-test"
    scene_id = "0d2ee665be"
    bundle_id = f"{scene_id}-factory"
    config = {
        "schema_version": 2,
        "leakage_remediation": {
            "schema_version": 1,
            "bundles": [{
                "bundle_id": bundle_id,
                "scene_id": scene_id,
                "mode": "factory",
                "legacy_source": f"outputs/{scene_id}_factory",
                "targets": [{
                    "legacy_index": 4,
                    "gt_object_id": 16,
                    "label": "mug",
                    "old_frame": "test.JPG",
                }],
            }],
        },
    }
    config_path = tmp_path / "configs/fidelity.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config))
    root = tmp_path / f"outputs/icra2027/{freeze_id}/fidelity/replacements"
    bundle = root / bundle_id
    object_dir = bundle / "objects/obj_04"
    object_dir.mkdir(parents=True)
    (object_dir / "meta.json").write_text(json.dumps({
        "index": 4, "frame": "train.JPG", "gt_object_id": 16, "label": "mug",
    }))
    (object_dir / "aligned.json").write_text(json.dumps({
        "index": 4, "tier": "A", "rejected": False, "T": np.eye(4).tolist(),
    }))
    (object_dir / "trellis_gs.ply").write_bytes(b"replacement-gaussian")
    generated = [
        _artifact(path, bundle)
        for path in sorted(object_dir.iterdir())
    ]
    legacy_meta = tmp_path / f"outputs/{scene_id}_factory/objects/obj_04/meta.json"
    legacy_meta.parent.mkdir(parents=True)
    legacy_meta.write_text("legacy\n")
    manifest = {
        "schema_version": 1,
        "task": "E2 leakage remediation",
        "freeze_id": freeze_id,
        "bundle_id": bundle_id,
        "dataset_id": "scannetpp_v2",
        "split_id": "nvs_sem_val",
        "scene_id": scene_id,
        "mode": "factory",
        "legacy_source": f"outputs/{scene_id}_factory",
        "legacy_source_immutable": True,
        "paper_ready": False,
        "git": {"commit": "a" * 40, "dirty": False},
        "config_artifact": {
            "sha256": _sha256(config_path),
            "canonical_sha256": exporter._canonical_hash(config),
        },
        "optimization_input_frames": ["train.JPG"],
        "targets": [{
            "legacy_index": 4,
            "gt_object_id": 16,
            "label": "mug",
            "old_test_frame": "test.JPG",
            "new_train_frame": "train.JPG",
            "optimization_input_frames": ["train.JPG"],
            "alignment_tier": "A",
            "legacy_artifacts": [_artifact(legacy_meta, tmp_path)],
            "generated_artifacts": generated,
        }],
        "validation": {
            "fixed_target_population": True,
            "mapped_by_gt_object_id": True,
            "preserved_legacy_indices": True,
            "generation_frames_official_train_only": True,
            "generation_eval_overlap": [],
            "all_regenerated_alignments_accepted": True,
            "legacy_sources_unchanged": True,
            "atomic_fresh_publish": True,
        },
    }
    manifest_path = bundle / "replacement_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    overlay, provenance = _load_replacement_overlay(
        config_path=config_path,
        replacement_root=root,
        freeze_id=freeze_id,
        scene_id=scene_id,
        mode="factory",
        code_commit="a" * 40,
    )
    assert overlay[4]["frame"] == "train.JPG"
    assert overlay[4]["gaussian_path"] == object_dir / "trellis_gs.ply"
    assert provenance["bundle_id"] == bundle_id
    assert provenance["manifest_sha256"] == _sha256(manifest_path)

    (object_dir / "trellis_gs.ply").write_bytes(b"drift")
    with pytest.raises(ValueError, match="size/path mismatch|hash mismatch"):
        _load_replacement_overlay(
            config_path=config_path,
            replacement_root=root,
            freeze_id=freeze_id,
            scene_id=scene_id,
            mode="factory",
            code_commit="a" * 40,
        )
