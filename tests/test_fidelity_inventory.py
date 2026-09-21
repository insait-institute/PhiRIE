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
import yaml
from PIL import Image

from robo.eval.fidelity_replacements import (
    EXPECTED_BUNDLES,
    EXPECTED_STAGE_PARAMETERS,
)

from robo.eval.fidelity_inventory import (
    AVAILABLE_ROOM_METHOD_IDS,
    KNOWN_ROOM_ANCHORS,
    OBJECT_METHODS,
    ROOM_METHODS,
    _canonical_hash,
    _config_population,
    _validate_config,
    build_inventory,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = REPO_ROOT / "tests/.fidelity-inventory-work"


@contextmanager
def _repo_workspace():
    WORK_ROOT.mkdir(exist_ok=True)
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _scenes(count: int) -> list[str]:
    return [f"{index:010x}" for index in range(count)]


def _test_tree_file(name: str) -> dict:
    return {
        "relative_path": (
            "sam3/__init__.py" if name == "sam3_source" else f"{name}.bin"
        ),
        "size_bytes": len(name),
        "sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
    }


def _test_tree_sha(record: dict) -> str:
    payload = (
        record["relative_path"].encode("utf-8") + b"\0"
        + str(record["size_bytes"]).encode("ascii") + b"\0"
        + record["sha256"].encode("ascii") + b"\n"
    )
    return hashlib.sha256(payload).hexdigest()


def _test_model_inputs() -> dict:
    tree_files = {
        name: _test_tree_file(name)
        for name in (
            "trellis_source", "trellis_snapshot", "dinov2_source", "sam3_source",
        )
    }
    return {
        "trellis_source": {
            "path": ".cache/test/trellis-source",
            "upstream_repository": "fixture/TRELLIS",
            "upstream_commit": "3" * 40,
            "flexicubes_commit": "4" * 40,
            "file_count": 1,
            "tree_sha256": _test_tree_sha(tree_files["trellis_source"]),
        },
        "trellis_snapshot": {
            "path": ".cache/test/trellis-snapshot",
            "upstream_revision": "1" * 40,
            "file_count": 1,
            "tree_sha256": _test_tree_sha(tree_files["trellis_snapshot"]),
        },
        "dinov2_source": {
            "path": ".cache/test/dinov2-source",
            "upstream_repository": "fixture/dinov2",
            "upstream_commit": None,
            "upstream_commit_reason": "fixture source is tree-addressed",
            "file_count": 1,
            "tree_sha256": _test_tree_sha(tree_files["dinov2_source"]),
        },
        "dinov2_checkpoint": {
            "path": ".cache/test/dinov2.pth",
            "size_bytes": 11,
            "sha256": "5" * 64,
        },
        "sam3_source": {
            "path": ".cache/test/sam3-source",
            "upstream_repository": "fixture/sam3",
            "upstream_commit": "7" * 40,
            "package_version": "0.1.0",
            "file_count": 1,
            "tree_sha256": _test_tree_sha(tree_files["sam3_source"]),
        },
        "sam3_checkpoint": {
            "path": ".cache/test/sam3.pt",
            "upstream_revision": "2" * 40,
            "size_bytes": 13,
            "sha256": "6" * 64,
        },
    }


def _test_model_manifest(declared: dict) -> dict:
    observed = {}
    for name in (
        "trellis_source", "trellis_snapshot", "dinov2_source", "sam3_source",
    ):
        record = declared[name]
        observed[name] = {
            "path": record["path"],
            "file_count": record["file_count"],
            "tree_sha256": record["tree_sha256"],
            "files": [_test_tree_file(name)],
            "source_identity": f"tree-sha256:{record['tree_sha256']}",
        }
    observed["trellis_source"].update({
        field: declared["trellis_source"][field]
        for field in ("upstream_repository", "upstream_commit", "flexicubes_commit")
    })
    observed["trellis_snapshot"]["upstream_revision"] = declared[
        "trellis_snapshot"
    ]["upstream_revision"]
    observed["dinov2_source"].update({
        field: declared["dinov2_source"][field]
        for field in ("upstream_repository", "upstream_commit", "upstream_commit_reason")
    })
    observed["dinov2_source"]["upstream_commit_provenance"] = (
        "unavailable_from_gitless_torch_hub_snapshot"
    )
    observed["sam3_source"].update({
        field: declared["sam3_source"][field]
        for field in ("upstream_repository", "upstream_commit", "package_version")
    })
    observed["dinov2_checkpoint"] = dict(declared["dinov2_checkpoint"])
    observed["sam3_checkpoint"] = dict(declared["sam3_checkpoint"])
    sam3_module = _test_tree_file("sam3_source")
    return {
        "declared": copy.deepcopy(declared),
        "observed": observed,
        "runtime": {
            "sam3": {
                "package_version": "0.1.0",
                "module_file": ".cache/test/sam3-source/sam3/__init__.py",
                "module_size_bytes": sam3_module["size_bytes"],
                "module_sha256": sam3_module["sha256"],
            }
        },
    }


def _config_payload(scenes: list[str]) -> dict:
    config = json.loads(
        (REPO_ROOT / "configs/experiments/icra2027/fidelity_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    config["population"]["planned_scenes"] = len(scenes)
    config["population"]["scene_ids"] = scenes
    config["population"]["smoke_scene_ids"] = scenes[:2]
    config["coverage"]["minimum_room_scenes_by_method"] = {
        method: len(scenes) for method in AVAILABLE_ROOM_METHOD_IDS
    } | {"harmonizer_option_c": 0}
    config["coverage"]["minimum_room_views_by_method"] = {
        method: len(scenes) * 8 for method in AVAILABLE_ROOM_METHOD_IDS
    } | {"harmonizer_option_c": 0}
    config["leakage_remediation"]["model_inputs"] = _test_model_inputs()
    return config


def _construction_payload(scenes: list[str]) -> dict:
    return {
        "schema_version": 1,
        "population": {
            "dataset": "scannetpp_v2",
            "split": "nvs_sem_val",
            "planned_scenes": len(scenes),
            "preserve_failed_scenes": True,
            "scene_ids": scenes,
        },
    }


def _metric_payload(bg, twin, *, frames: list[str]) -> dict:
    return {
        "n_objects_composited": 0,
        "split": "official-test",
        "eval_frames": frames,
        "frames": [
            {"frame": frame, "bg": dict(bg), "twin": dict(twin)}
            for frame in frames
        ],
        "mean": {"bg": dict(bg), "twin": dict(twin)},
    }


def _object_row(value: float = 0.7) -> dict:
    return {
        "index": 0,
        "label": "box",
        "criterion": "sym_score vs gt_points.ply",
        "trellis": {"f1_20": value},
        "rvg": {"f1_20": value + 0.1, "rejected": None},
        "winner": "rvg",
    }


def _object_review(n_scenes: int, *, value: float = 0.7) -> dict:
    return {
        "provenance": {"n_scenes": n_scenes, "n_objects_total": n_scenes},
        "dataset_wide": {
            "pooled_mean_f1_20_trellis_only": value,
            "pooled_mean_f1_20_rvg_all_scored": value + 0.1,
            "pooled_mean_f1_20_hybrid": value + 0.1,
            "gate_vs_f1_oracle": {"n_objects_both_scored": n_scenes},
            "rvg_collapses_f1_lt_0.1": {"n_collapses": 0},
        },
        "oracle_hybrid_upper_bound": {
            "pooled_mean_f1_20_oracle_nonrejected_candidates": value + 0.1,
        },
    }


def _make_contract(
    path: Path,
    *,
    freeze_id: str,
    config: Path,
    construction: Path,
    mode: str,
) -> None:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    payload = {
        "schema_version": 1,
        "freeze_id": freeze_id,
        "mode": mode,
        "created_utc": "ignored by contract digest",
        "environment": {"ignored": True},
        "code": {"commit": commit, "dirty": False},
        "configs": [
            {
                "field": "fidelity_manifest",
                "source_content_sha256": _sha(config),
            },
            {
                "field": "construction_config",
                "source_content_sha256": _sha(construction),
            },
        ],
    }
    payload["contract_sha256"] = _canonical_hash({
        key: value
        for key, value in payload.items()
        if key not in {"created_utc", "environment", "contract_sha256"}
    })
    _write_json(path, payload)


def _make_scene(
    root: Path,
    scene: str,
    freeze_id: str,
    *,
    object_value: float = 0.7,
) -> None:
    outputs = root / "outputs"
    room = root / "room_runs" / scene
    frames = [f"frame_{index:02d}.JPG" for index in range(8)]
    pngs = [f"frame_{index:02d}.png" for index in range(8)]

    construction_sources = {}
    for suffix, method in (
        ("factory", "factorized_gt_discovery"),
        ("auto", "factorized_auto_discovery"),
    ):
        objects = outputs / f"{scene}_{suffix}" / "objects/objects.json"
        _write_json(objects, [])
        construction_sources[method] = {
            "source": str(objects.parent.parent.relative_to(REPO_ROOT)),
            "accepted_object_ids": [],
            "n_objects_composited": 0,
            "artifacts": [{
                "path": str(objects.relative_to(REPO_ROOT)),
                "size_bytes": objects.stat().st_size,
                "sha256": _sha(objects),
            }],
        }
    hybrid = outputs / f"{scene}_factory/objects/hybrid_all.json"
    row = _object_row(object_value)
    _write_json(hybrid, {"n_objects": 1, "rows": [row]})

    bg = {
        "psnr": KNOWN_ROOM_ANCHORS["input_scene_gaussian"]["psnr"],
        "ssim": KNOWN_ROOM_ANCHORS["input_scene_gaussian"]["ssim"],
        "lpips": 0.16,
    }
    gt = {
        "psnr": KNOWN_ROOM_ANCHORS["factorized_gt_discovery"]["psnr"],
        "ssim": KNOWN_ROOM_ANCHORS["factorized_gt_discovery"]["ssim"],
        "lpips": 0.17,
    }
    automatic = {"psnr": 27.52, "ssim": 0.902, "lpips": 0.18}
    _write_json(
        outputs / f"{scene}_factory/render_metrics_v2.json",
        _metric_payload(bg, gt, frames=frames),
    )
    _write_json(
        outputs / f"{scene}_auto/render_metrics_v2.json",
        _metric_payload(bg, automatic, frames=frames),
    )

    output_artifacts = []
    for directory_index, directory in enumerate((
        "gt", "input_scene_gaussian", "factorized_gt_discovery",
        "factorized_auto_discovery",
    )):
        for frame_index, png in enumerate(pngs):
            path = room / directory / png
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB", (2, 2),
                color=(directory_index * 30, frame_index * 20, 10),
            ).save(path)
            output_artifacts.append({
                "relative_path": f"{directory}/{png}",
                "sha256": _sha(path),
            })
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    _write_json(room / "manifest.json", {
        "schema_version": 1,
        "freeze_id": freeze_id,
        "dataset_id": "scannetpp_v2",
        "split_id": "nvs_sem_val:official-test",
        "scene_id": scene,
        "evaluation_unit": "held_out_view",
        "n_eval_frames": 8,
        "eval_frames": frames,
        "official_train_frames": ["train_00.JPG", "train_01.JPG"],
        "optimization_input_frames": ["train_00.JPG", "train_01.JPG"],
        "methods": list(AVAILABLE_ROOM_METHOD_IDS),
        "image_width": 2,
        "image_height": 2,
        "color_space": "sRGB; 8-bit RGB PNG",
        "source_scene_gaussian": {
            "path": "/dataset/splat.ply",
            "size_bytes": 123,
            "sha256": "a" * 64,
        },
        "split_artifact": {"path": "/dataset/split.json", "sha256": "b" * 64},
        "camera_artifacts": {
            "intrinsics": {"path": "/dataset/transforms.json", "sha256": "c" * 64},
            "poses": {"path": "/dataset/images.txt", "sha256": "d" * 64},
        },
        "source_images": [
            {
                "frame": frame,
                "path": f"/dataset/{frame}",
                "sha256": f"{index + 1:064x}",
                "exported_png": png,
            }
            for index, (frame, png) in enumerate(zip(frames, pngs))
        ],
        "construction_sources": construction_sources,
        "output_artifacts": sorted(
            output_artifacts, key=lambda item: item["relative_path"]
        ),
        "git": {"commit": commit, "dirty": False, "branch": "fixture"},
        "created_utc": "2026-09-03T00:00:00+00:00",
        "paper_ready": False,
        "paper_ready_reason": "aggregation pending",
    })


def _fixture(
    root: Path,
    count: int = 2,
    *,
    mode: str = "paper",
    scene_ids: list[str] | None = None,
) -> dict:
    scenes = list(scene_ids) if scene_ids is not None else _scenes(count)
    freeze_id = "fixture-freeze"
    config = root / "fidelity_manifest.json"
    construction = root / "construction.yaml"
    review = root / "review.json"
    _write_json(config, _config_payload(scenes))
    construction.write_text(
        yaml.safe_dump(_construction_payload(scenes), sort_keys=False),
        encoding="utf-8",
    )
    _write_json(review, _object_review(len(scenes)))
    for scene in scenes:
        _make_scene(root, scene, freeze_id)
    contract = root / "contract/freeze_manifest.json"
    _make_contract(
        contract,
        freeze_id=freeze_id,
        config=config,
        construction=construction,
        mode=mode,
    )
    return {
        "scenes": scenes,
        "freeze_id": freeze_id,
        "config": config,
        "construction": construction,
        "review": review,
        "contract": contract,
        "outputs": root / "outputs",
        "room_runs": root / "room_runs",
    }


def _bare_artifact(path: Path, *, relative_to: Path | None = None) -> dict:
    return {
        "path": str(path.relative_to(relative_to)) if relative_to else str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _attach_valid_replacement(fixture: dict, bundle_id: str) -> Path:
    spec = EXPECTED_BUNDLES[bundle_id]
    scene_id = spec["scene_id"]
    mode = spec["mode"]
    method = (
        "factorized_gt_discovery" if mode == "factory"
        else "factorized_auto_discovery"
    )
    legacy = fixture["outputs"] / f"{scene_id}_{mode}"
    legacy_objects = []
    legacy_by_object = {}
    for index, object_id, label, old_frame in spec["targets"]:
        meta = {
            "index": index,
            "gt_object_id": object_id,
            "label": label,
            "frame": old_frame,
        }
        legacy_objects.append(meta)
        odir = legacy / "objects" / f"obj_{index:02d}"
        _write_json(odir / "meta.json", meta)
        _write_json(odir / "aligned.json", {
            "index": index, "label": label, "tier": "A", "rejected": None,
        })
        (odir / "rgba.png").write_bytes(f"legacy-{index}".encode())
        legacy_by_object[object_id] = [
            _bare_artifact(odir / name, relative_to=REPO_ROOT)
            for name in ("meta.json", "aligned.json", "rgba.png")
        ]
    legacy_objects_path = legacy / "objects/objects.json"
    _write_json(legacy_objects_path, legacy_objects)
    if mode == "auto":
        (legacy / "auto_instances.npz").write_bytes(b"authoritative-auto-instances")

    bundle_root = (
        fixture["outputs"] / "icra2027" / fixture["freeze_id"]
        / "fidelity/replacements" / bundle_id
    )
    train = ["train_00.JPG", "train_01.JPG"]
    controls = bundle_root / "provenance"
    _write_json(controls / "official_train_frames.json", {"frames": train})
    _write_json(controls / "factory_read_frames.json", {"frames": ["train_00.JPG"]})
    _write_json(controls / "preserved_index_by_gt_object_id.json", {
        str(object_id): index
        for index, object_id, _label, _old_frame in spec["targets"]
    })
    replacement_objects = []
    aligned_all = []
    targets = []
    for index, object_id, label, old_frame in spec["targets"]:
        odir = bundle_root / "objects" / f"obj_{index:02d}"
        meta = {
            "index": index,
            "gt_object_id": object_id,
            "label": label,
            "frame": "train_00.JPG",
        }
        replacement_objects.append(meta)
        alignment = {
            "index": index, "label": label, "tier": "A", "rejected": None,
        }
        aligned_all.append(alignment)
        _write_json(odir / "meta.json", meta)
        _write_json(odir / "aligned.json", alignment)
        for name in (
            "gt_points.ply", "mesh_sim.obj", "mesh_sim.ply", "rgba.png",
            "trellis_gs.ply", "trellis_mesh.ply",
        ):
            (odir / name).write_bytes(f"{name}-{index}".encode())
        generated = [
            _bare_artifact(path, relative_to=bundle_root)
            for path in sorted(odir.iterdir(), key=lambda item: item.name)
            if path.is_file()
        ]
        targets.append({
            "legacy_index": index,
            "gt_object_id": object_id,
            "label": label,
            "old_test_frame": old_frame,
            "new_train_frame": "train_00.JPG",
            "optimization_input_frames": ["train_00.JPG"],
            "alignment_tier": "A",
            "legacy_artifacts": legacy_by_object[object_id],
            "generated_artifacts": generated,
        })
    _write_json(bundle_root / "objects/objects.json", replacement_objects)
    _write_json(bundle_root / "objects/aligned_all.json", aligned_all)
    control_artifacts = [
        _bare_artifact(bundle_root / relative, relative_to=bundle_root)
        for relative in (
            "provenance/official_train_frames.json",
            "provenance/preserved_index_by_gt_object_id.json",
            "provenance/factory_read_frames.json",
            "objects/objects.json",
            "objects/aligned_all.json",
        )
    ]
    auto_record = None
    if mode == "auto":
        source_auto = legacy / "auto_instances.npz"
        copied_auto = bundle_root / "auto_instances.npz"
        copied_auto.write_bytes(source_auto.read_bytes())
        auto_record = {
            **_bare_artifact(copied_auto, relative_to=bundle_root),
            "source_path": str(source_auto.relative_to(REPO_ROOT)),
            "source_sha256": _sha(source_auto),
        }

    config_payload = json.loads(fixture["config"].read_text())
    contract_payload = json.loads(fixture["contract"].read_text())
    dataset_root = fixture["config"].parent / "replacement-dataset" / scene_id
    split_path = dataset_root / "dslr/train_test_lists.json"
    intrinsics_path = dataset_root / "dslr/nerfstudio/transforms_undistorted.json"
    poses_path = dataset_root / "dslr/colmap/images.txt"
    mesh_path = dataset_root / "scans/mesh_aligned_0.05.ply"
    segments_path = dataset_root / "scans/segments.json"
    anno_path = dataset_root / "scans/segments_anno.json"
    image_path = dataset_root / "dslr/resized_undistorted_images/train_00.JPG"
    _write_json(split_path, {"train": train, "test": [f"view_{i:02d}.JPG" for i in range(8)]})
    _write_json(intrinsics_path, {"camera_model": "PINHOLE"})
    poses_path.parent.mkdir(parents=True)
    poses_path.write_bytes(b"replacement fixture poses\n")
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_bytes(b"replacement fixture mesh\n")
    _write_json(segments_path, {"segIndices": []})
    _write_json(anno_path, {"segGroups": []})
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"replacement fixture train image\n")
    dataset_inputs = {
        "split": _bare_artifact(split_path),
        "intrinsics": _bare_artifact(intrinsics_path),
        "poses": _bare_artifact(poses_path),
        "mesh": _bare_artifact(mesh_path),
        "segments": _bare_artifact(segments_path),
        "segments_anno": _bare_artifact(anno_path),
        "read_frames_control": _bare_artifact(
            controls / "factory_read_frames.json", relative_to=bundle_root
        ),
        "read_images": [{
            **_bare_artifact(image_path),
            "frame": "train_00.JPG",
        }],
    }
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    manifest = {
        "schema_version": 1,
        "task": "E2 leakage remediation",
        "freeze_id": fixture["freeze_id"],
        "bundle_id": bundle_id,
        "dataset_id": "scannetpp_v2",
        "split_id": "nvs_sem_val",
        "scene_id": scene_id,
        "mode": mode,
        "created_utc": "2026-09-03T00:00:00Z",
        "git": {"commit": commit, "dirty": False, "status": []},
        "config_artifact": {
            **_bare_artifact(fixture["config"], relative_to=REPO_ROOT),
            "canonical_sha256": _canonical_hash(config_payload),
        },
        "contract_artifact": {
            "path": str(fixture["contract"]),
            "sha256": _sha(fixture["contract"]),
            "contract_sha256": contract_payload["contract_sha256"],
            "freeze_id": contract_payload["freeze_id"],
            "mode": contract_payload["mode"],
        },
        "model_inputs": _test_model_manifest(
            config_payload["leakage_remediation"]["model_inputs"]
        ),
        "stage_parameters": copy.deepcopy(EXPECTED_STAGE_PARAMETERS),
        "official_split_artifact": {
            **_bare_artifact(split_path),
        },
        "dataset_inputs": dataset_inputs,
        "official_train_frame_count": len(train),
        "official_test_frame_count": 8,
        "official_train_frames_sha256": _canonical_hash(train),
        "official_test_frames_sha256": "e" * 64,
        "legacy_source": spec["legacy_source"],
        "legacy_source_immutable": True,
        "auto_instances": auto_record,
        "optimization_input_frames": ["train_00.JPG"],
        "targets": targets,
        "stages": [
            {"name": name, "status": "passed"}
            for name in (
                "factory_prepare", "factory_refine_masks", "trellis", "factory_align",
            )
        ],
        "bundle_artifacts": control_artifacts,
        "validation": {
            "fixed_target_population": True,
            "target_count": len(targets),
            "mapped_by_gt_object_id": True,
            "preserved_legacy_indices": True,
            "generation_frames_official_train_only": True,
            "generation_eval_overlap": [],
            "all_regenerated_alignments_accepted": True,
            "legacy_sources_unchanged": True,
            "atomic_fresh_publish": True,
        },
        "paper_ready": False,
        "paper_ready_reason": "fresh held-out metrics pending",
    }
    replacement_manifest = bundle_root / "replacement_manifest.json"
    _write_json(replacement_manifest, manifest)

    room_manifest_path = fixture["room_runs"] / scene_id / "manifest.json"
    room_manifest = json.loads(room_manifest_path.read_text())
    room_manifest["split_artifact"] = {
        "path": str(split_path), "sha256": _sha(split_path),
    }
    room_manifest["camera_artifacts"] = {
        "intrinsics": {
            "path": str(intrinsics_path), "sha256": _sha(intrinsics_path),
        },
        "poses": {"path": str(poses_path), "sha256": _sha(poses_path)},
    }
    room_source = room_manifest["construction_sources"][method]
    room_source["accepted_object_ids"] = [target[0] for target in spec["targets"]]
    room_source["n_objects_composited"] = len(spec["targets"])
    room_source["optimization_input_frames"] = ["train_00.JPG"]
    room_source["artifacts"] = [{
        "path": str(legacy_objects_path.relative_to(REPO_ROOT)),
        "size_bytes": legacy_objects_path.stat().st_size,
        "sha256": _sha(legacy_objects_path),
    }]
    room_source["replacement_bundle"] = {
        "manifest_path": str(replacement_manifest.relative_to(REPO_ROOT)),
        "manifest_sha256": _sha(replacement_manifest),
        "bundle_id": bundle_id,
        "targets": targets,
    }
    _write_json(room_manifest_path, room_manifest)
    return replacement_manifest


def _build(fixture: dict, out: Path, *, smoke: bool = False, smoke_limit: int = 2):
    return build_inventory(
        fixture["outputs"],
        fixture["room_runs"],
        out,
        fixture["freeze_id"],
        config_path=fixture["config"],
        construction_config_path=fixture["construction"],
        reviewed_object_aggregate_path=fixture["review"],
        contract_manifest_path=fixture["contract"],
        expected_scenes=len(fixture["scenes"]),
        smoke=smoke,
        smoke_limit=smoke_limit,
    )


def test_checked_in_e2_config_pins_same_exact_fifty_scene_roster_as_e1():
    fidelity_path = REPO_ROOT / "configs/experiments/icra2027/fidelity_manifest.json"
    construction_path = REPO_ROOT / "configs/experiments/icra2027/construction_regimes.yaml"
    fidelity = json.loads(fidelity_path.read_text(encoding="utf-8"))
    construction = yaml.safe_load(construction_path.read_text(encoding="utf-8"))

    scenes = _validate_config(fidelity, 50)

    assert len(scenes) == 50
    assert scenes == _config_population(construction, label="construction config")
    assert {"38d58a7a31", "c4c04e6d6c"} <= set(scenes)


def test_full_inventory_publishes_two_scene_records_and_sixteen_views_per_method():
    with _repo_workspace() as root:
        fixture = _fixture(root)
        out = root / "inventory"

        result = _build(fixture, out)

        report = json.loads((out / "validation_report.json").read_text())
        manifest = json.loads((out / "manifests/fidelity_manifest.json").read_text())
        assert result["valid"] is True
        assert report["valid"] is True
        assert report["population"]["full_population_preserved"] is True
        assert report["coverage"]["observed_room_views_by_method"] == {
            **{method: 16 for method in AVAILABLE_ROOM_METHOD_IDS},
            "harmonizer_option_c": 0,
        }
        for label, method in ROOM_METHODS.items():
            records = manifest["room_methods"][label]["records"]
            if method == "harmonizer_option_c":
                assert records == []
                continue
            assert len(records) == 2
            assert all(record["n_views"] == 8 for record in records)
            assert all(record["coverage"] == {"render_only": [], "gt_only": []}
                       for record in records)
            assert all({"freeze_id", "scene_id", "source_build", "render_dir", "gt_dir"}
                       <= set(record) for record in records)
            assert all(len(record["views"]) == 8 for record in records)
            assert all(
                record["optimization_input_frames"]
                == ["train_00.JPG", "train_01.JPG"]
                for record in records
            )
        assert report["room_anchor_reproduction"]["all_reproduced"] is True
        assert report["object_anchor_reproduction"]["source_snapshot_coherent"] is True
        assert report["paper_ready"] is False
        assert len(report["manifest_sha256"]) == 64


def test_smoke_preserves_full_planned_roster_but_selects_one_scene():
    with _repo_workspace() as root:
        fixture = _fixture(root, mode="smoke")
        out = root / "inventory"

        result = _build(fixture, out, smoke=True, smoke_limit=1)

        report = json.loads((out / "validation_report.json").read_text())
        manifest = json.loads((out / "manifests/fidelity_manifest.json").read_text())
        assert result["valid"] is True
        assert report["mode"] == "smoke"
        assert report["population"]["planned_scene_count"] == 2
        assert report["population"]["selected_scene_count"] == 1
        assert report["population"]["full_population_preserved"] is False
        assert report["coverage"]["observed_room_views_by_method"] == {
            **{method: 8 for method in AVAILABLE_ROOM_METHOD_IDS},
            "harmonizer_option_c": 0,
        }
        assert manifest["coverage"]["minimum_room_scenes_by_method"] == {
            **{method: 1 for method in AVAILABLE_ROOM_METHOD_IDS},
            "harmonizer_option_c": 0,
        }
        assert manifest["coverage"]["minimum_room_views_by_method"] == {
            **{method: 8 for method in AVAILABLE_ROOM_METHOD_IDS},
            "harmonizer_option_c": 0,
        }
        assert report["room_anchor_reproduction"]["all_reproduced"] is False
        assert report["object_anchor_reproduction"]["source_snapshot_coherent"] is False


def test_one_corrupt_png_excludes_the_entire_scene_from_all_paired_methods():
    with _repo_workspace() as root:
        fixture = _fixture(root)
        corrupt = (
            fixture["room_runs"] / fixture["scenes"][0]
            / "factorized_auto_discovery/frame_00.png"
        )
        corrupt.write_bytes(b"not a png")

        _build(fixture, root / "inventory")

        report = json.loads((root / "inventory/validation_report.json").read_text())
        manifest = json.loads(
            (root / "inventory/manifests/fidelity_manifest.json").read_text()
        )
        assert report["valid"] is False
        assert report["room_exports"]["invalid_scene_count"] == 1
        assert report["coverage"]["observed_room_views_by_method"] == {
            **{method: 8 for method in AVAILABLE_ROOM_METHOD_IDS},
            "harmonizer_option_c": 0,
        }
        for label, method in ROOM_METHODS.items():
            if method != "harmonizer_option_c":
                assert len(manifest["room_methods"][label]["records"]) == 1


@pytest.mark.parametrize(
    ("replacement", "expected_issue"),
    [
        (None, "optimization_input_frames_missing"),
        (["frame_00.JPG"], "optimization_and_evaluation_frames_overlap"),
    ],
)
def test_missing_or_overlapping_optimization_frames_fail_the_whole_scene(
    replacement, expected_issue,
):
    with _repo_workspace() as root:
        fixture = _fixture(root)
        scene = fixture["scenes"][0]
        path = fixture["room_runs"] / scene / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if replacement is None:
            payload.pop("optimization_input_frames")
        else:
            payload["optimization_input_frames"] = replacement
        _write_json(path, payload)

        _build(fixture, root / "inventory")

        report = json.loads((root / "inventory/validation_report.json").read_text())
        manifest = json.loads(
            (root / "inventory/manifests/fidelity_manifest.json").read_text()
        )
        scene_record = next(
            item for item in report["room_exports"]["records"]
            if item["scene_id"] == scene
        )
        assert report["valid"] is False
        assert expected_issue in scene_record["issues"]
        for label, method in ROOM_METHODS.items():
            if method != "harmonizer_option_c":
                records = manifest["room_methods"][label]["records"]
                assert {record["scene_id"] for record in records} == {
                    fixture["scenes"][1]
                }


def test_legacy_metric_mean_mismatch_is_audit_only_not_a_fresh_export_failure():
    with _repo_workspace() as root:
        fixture = _fixture(root)
        scene = fixture["scenes"][0]
        path = fixture["outputs"] / f"{scene}_factory/render_metrics_v2.json"
        payload = json.loads(path.read_text())
        payload["mean"]["bg"]["psnr"] += 1.0
        _write_json(path, payload)

        _build(fixture, root / "inventory")

        report = json.loads((root / "inventory/validation_report.json").read_text())
        assert report["valid"] is True
        anchor = report["room_anchor_reproduction"]
        assert anchor["valid_scene_count"] == 1
        assert anchor["canonical_measurement_input"] is False
        assert "legacy_bg_psnr_mean_mismatch" in anchor["validity_reasons"]


def test_legacy_frame_roster_mismatch_is_explicitly_nonreproducible_but_audit_only():
    with _repo_workspace() as root:
        fixture = _fixture(root)
        scene = fixture["scenes"][0]
        for suffix in ("factory", "auto"):
            path = fixture["outputs"] / f"{scene}_{suffix}/render_metrics_v2.json"
            payload = json.loads(path.read_text())
            renamed = [f"legacy_{index:02d}.JPG" for index in range(8)]
            payload["eval_frames"] = renamed
            for row, frame in zip(payload["frames"], renamed):
                row["frame"] = frame
            _write_json(path, payload)

        _build(fixture, root / "inventory")

        report = json.loads((root / "inventory/validation_report.json").read_text())
        anchor = report["room_anchor_reproduction"]
        assert report["valid"] is True
        assert anchor["legacy_metric_values_reproduced"] is True
        assert anchor["fresh_protocol_frames_match"] is False
        assert anchor["all_reproduced"] is False
        assert anchor["status"] == (
            "legacy_metric_values_match_but_fresh_view_protocol_not_reproducible"
        )
        assert "legacy_and_fresh_export_frame_mismatch" in anchor["audit_only_reasons"]
        scene_audit = next(
            row for row in anchor["per_scene"] if row["scene_id"] == scene
        )
        assert scene_audit["valid"] is True
        assert scene_audit["audit_only_reasons"] == [
            "legacy_and_fresh_export_frame_mismatch"
        ]


def test_object_review_snapshot_mismatch_keeps_every_canonical_value_null():
    with _repo_workspace() as root:
        fixture = _fixture(root)
        review = json.loads(fixture["review"].read_text())
        review["provenance"]["n_objects_total"] = 99
        _write_json(fixture["review"], review)

        _build(fixture, root / "inventory")

        report = json.loads((root / "inventory/validation_report.json").read_text())
        object_report = report["object_anchor_reproduction"]
        assert object_report["source_snapshot_coherent"] is False
        assert "reviewed_object_snapshot_not_reproducible_from_enumerated_sources" in (
            object_report["validity_reasons"]
        )
        for value in object_report["canonical_legacy_diagnostics"].values():
            assert value["f1_20"] is None
            assert value["n_objects"] is None


def test_e2_and_e1_roster_mismatch_is_rejected_before_publication():
    with _repo_workspace() as root:
        fixture = _fixture(root)
        construction = yaml.safe_load(fixture["construction"].read_text())
        construction["population"]["scene_ids"][0] = "ffffffffff"
        fixture["construction"].write_text(
            yaml.safe_dump(construction, sort_keys=False), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="roster differs"):
            _build(fixture, root / "inventory")

        assert not (root / "inventory").exists()


def test_contract_hash_mismatch_is_archived_and_marks_inventory_invalid():
    with _repo_workspace() as root:
        fixture = _fixture(root)
        contract = json.loads(fixture["contract"].read_text())
        contract["contract_sha256"] = "f" * 64
        _write_json(fixture["contract"], contract)

        _build(fixture, root / "inventory")

        report = json.loads((root / "inventory/validation_report.json").read_text())
        assert report["valid"] is False
        assert report["upstream_e0_contract"]["valid"] is False
        assert "e0_contract_declared_hash_mismatch" in report["global_validity_reasons"]


def _affected_replacement_fixture(root: Path) -> tuple[dict, str, str]:
    scene = "3f15a9266d"
    fixture = _fixture(
        root,
        scene_ids=["0000000000", scene],
    )
    bundle_id = f"{scene}-factory"
    _attach_valid_replacement(fixture, bundle_id)
    return fixture, scene, bundle_id


def _room_issues(out: Path, scene: str) -> list[str]:
    report = json.loads((out / "validation_report.json").read_text())
    record = next(
        item for item in report["room_exports"]["records"]
        if item["scene_id"] == scene
    )
    return record["issues"]


def test_affected_source_requires_and_accepts_exact_train_only_replacement_closure():
    with _repo_workspace() as root:
        fixture, scene, bundle_id = _affected_replacement_fixture(root)
        out = root / "inventory"

        _build(fixture, out)

        report = json.loads((out / "validation_report.json").read_text())
        manifest = json.loads((out / "manifests/fidelity_manifest.json").read_text())
        assert report["valid"] is True, _room_issues(out, scene)
        assert _room_issues(out, scene) == []
        records = manifest["room_methods"][
            "Factorized composite, GT discovery"
        ]["records"]
        record = next(item for item in records if item["scene_id"] == scene)
        assert record["replacement_bundle"]["bundle_id"] == bundle_id
        assert len(record["replacement_bundle"]["manifest_sha256"]) == 64

        replacement_path = (
            REPO_ROOT / record["replacement_bundle"]["manifest_path"]
        )
        replacement = json.loads(replacement_path.read_text())
        assert replacement["stage_parameters"] == EXPECTED_STAGE_PARAMETERS


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    [
        ("missing", "missing_factory_replacement_bundle"),
        ("target", "factory_replacement_embedded_targets_mismatch"),
        ("not_train", "new_frame_not_official_train"),
        ("old_remains", "factory_leaked_old_frames_remain_in_source"),
        ("tamper", "replacement_artifact_size_mismatch"),
        ("model_input", "replacement_trellis_source_tree_hash_mismatch"),
        ("dataset_input", "replacement_read_image_0_hash_mismatch"),
        ("stage_parameters", "factory_replacement_stage_parameters_mismatch"),
        ("unexpected", "unexpected_auto_replacement_bundle"),
    ],
)
def test_replacement_provenance_and_leakage_fail_closed(mutation, expected_issue):
    with _repo_workspace() as root:
        fixture, scene, bundle_id = _affected_replacement_fixture(root)
        room_path = fixture["room_runs"] / scene / "manifest.json"
        room = json.loads(room_path.read_text())
        factory = room["construction_sources"]["factorized_gt_discovery"]
        declaration = factory["replacement_bundle"]
        replacement_path = REPO_ROOT / declaration["manifest_path"]

        if mutation == "missing":
            factory.pop("replacement_bundle")
        elif mutation == "target":
            declaration["targets"][0]["gt_object_id"] = 9999
        elif mutation == "not_train":
            replacement = json.loads(replacement_path.read_text())
            replacement["targets"][0]["new_train_frame"] = "NOT_TRAIN.JPG"
            replacement["targets"][0]["optimization_input_frames"] = ["NOT_TRAIN.JPG"]
            replacement["optimization_input_frames"] = ["NOT_TRAIN.JPG"]
            _write_json(replacement_path, replacement)
            declaration["targets"] = replacement["targets"]
            declaration["manifest_sha256"] = _sha(replacement_path)
            factory["optimization_input_frames"] = ["NOT_TRAIN.JPG"]
        elif mutation == "old_remains":
            old = EXPECTED_BUNDLES[bundle_id]["targets"][0][3]
            factory["optimization_input_frames"].append(old)
            room["optimization_input_frames"].append(old)
        elif mutation == "tamper":
            artifact = replacement_path.parent / "objects/obj_193/trellis_gs.ply"
            artifact.write_bytes(artifact.read_bytes() + b"tampered")
        elif mutation == "model_input":
            replacement = json.loads(replacement_path.read_text())
            replacement["model_inputs"]["observed"]["trellis_source"]["files"][0][
                "sha256"
            ] = "f" * 64
            _write_json(replacement_path, replacement)
            declaration["manifest_sha256"] = _sha(replacement_path)
        elif mutation == "dataset_input":
            replacement = json.loads(replacement_path.read_text())
            Path(replacement["dataset_inputs"]["read_images"][0]["path"]).write_bytes(
                b"tampered read image"
            )
        elif mutation == "stage_parameters":
            replacement = json.loads(replacement_path.read_text())
            replacement["stage_parameters"]["factory_align"][
                "source_up_hypotheses_order"
            ] = ["+z", "+x", "-z", "-x", "+y", "-y"]
            _write_json(replacement_path, replacement)
            declaration["manifest_sha256"] = _sha(replacement_path)
        elif mutation == "unexpected":
            room["construction_sources"]["factorized_auto_discovery"][
                "replacement_bundle"
            ] = declaration
        _write_json(room_path, room)

        out = root / "inventory"
        _build(fixture, out)

        issues = _room_issues(out, scene)
        assert any(expected_issue in issue for issue in issues)
        report = json.loads((out / "validation_report.json").read_text())
        assert report["valid"] is False


def test_injected_failure_removes_staging_and_never_publishes_partial_inventory():
    with _repo_workspace() as root:
        fixture = _fixture(root)
        out = root / "inventory"

        with pytest.raises(RuntimeError, match="injected fidelity inventory failure"):
            build_inventory(
                fixture["outputs"],
                fixture["room_runs"],
                out,
                fixture["freeze_id"],
                config_path=fixture["config"],
                construction_config_path=fixture["construction"],
                reviewed_object_aggregate_path=fixture["review"],
                contract_manifest_path=fixture["contract"],
                expected_scenes=2,
                _inject_failure_after="manifest",
            )

        assert not out.exists()
        assert list(root.glob(".inventory.staging-*")) == []


def test_existing_destination_is_never_overwritten():
    with _repo_workspace() as root:
        fixture = _fixture(root)
        out = root / "inventory"
        out.mkdir()
        marker = out / "keep.txt"
        marker.write_text("owned by earlier run", encoding="utf-8")

        with pytest.raises(ValueError, match="refusing to overwrite"):
            _build(fixture, out)

        assert marker.read_text(encoding="utf-8") == "owned by earlier run"
