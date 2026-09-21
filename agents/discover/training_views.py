"""Official training-view boundary for discovery and splat-derived surfaces.

The split validator is shared with the existing fidelity replacement producer.
This only controls camera inputs; it does not certify a mesh or checkpoint.
"""
import hashlib
import json
from pathlib import Path

from robo.eval.fidelity_replacements import _load_split


def select_training_views(views, split_path, stride, max_train_frames=None):
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("frame stride must be a positive integer")
    if max_train_frames is not None and (
        isinstance(max_train_frames, bool) or not isinstance(max_train_frames, int)
        or max_train_frames < 1
    ):
        raise ValueError("training frame limit must be a positive integer")
    path = Path(split_path).resolve()
    train, test = _load_split(path)
    names = [name for name, _ in views]
    if len(set(names)) != len(names):
        raise ValueError("duplicate registered camera names")
    missing = set(train) - set(names)
    if missing:
        raise ValueError(f"official training cameras missing poses: {sorted(missing)[:5]}")
    # Filter BEFORE subsampling: held-out frame positions cannot affect stride.
    bounded_train = sorted(train)[:max_train_frames]
    train_set = set(bounded_train)
    selected = sorted((name, pose) for name, pose in views if name in train_set)[::stride]
    if not selected:
        raise ValueError("no official training views selected")
    manifest = {
        "schema_version": 1,
        "scope": "planned_camera_inputs_only",
        "split": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "stride": stride,
        "max_train_frames": max_train_frames,
        "training_frames": bounded_train,
        "train_count": len(train),
        "test_count": len(test),
        "selected_frames": [name for name, _ in selected],
        "heldout_overlap": [],
        "paper_ready": False,
    }
    return selected, manifest


def write_training_manifest(path, manifest):
    # Exclusive creation also protects failed/partial attempts from reuse.
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")


def validate_fusion_frames(files, manifest, read_frame_name):
    expected = manifest["selected_frames"]
    split_path = Path(manifest["split"]["path"])
    if hashlib.sha256(split_path.read_bytes()).hexdigest() != manifest["split"]["sha256"]:
        raise ValueError("official split changed since depth rendering")
    train, _ = _load_split(split_path)
    stride = manifest["stride"]
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("invalid frozen frame stride")
    limit = manifest.get("max_train_frames")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("invalid frozen training frame limit")
    bounded = sorted(train)[:limit]
    if "training_frames" in manifest and manifest["training_frames"] != bounded:
        raise ValueError("training-frame boundary differs from official selection")
    if expected != bounded[::stride]:
        raise ValueError("depth roster is not the declared official training selection")
    if len(files) != len(expected):
        raise ValueError("incomplete or stale training-only depth render set")
    for index, (path, expected_frame) in enumerate(zip(files, expected)):
        if path.name != f"view_{index:04d}.npz" or read_frame_name(path) != expected_frame:
            raise ValueError("depth render frame differs from frozen training-view roster")


def main(argv=None):
    """Audit a declared scene roster without loading images/models/GT geometry."""
    import argparse
    import subprocess
    import yaml
    from agents.core.common import load_colmap_w2c

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    scene_ids = [str(value) for value in config["population"]["scene_ids"]]
    if len(set(scene_ids)) != len(scene_ids) or len(scene_ids) != config["population"]["planned_scenes"]:
        raise ValueError("declared scene roster is duplicate or incomplete")
    rows = []
    for scene_id in scene_ids:
        scene = Path(args.dataset_root) / "data" / scene_id
        poses = scene / "dslr/colmap/images.txt"
        split_path = scene / "dslr/train_test_lists.json"
        row = {"scene_id": scene_id}
        try:
            views = sorted(load_colmap_w2c(poses).items())
            _, test = _load_split(split_path)
            protocols = {}
            for role, stride in (("auto_discovery", 12), ("splat_depth", 3)):
                _, selection = select_training_views(views, split_path, stride)
                protocols[role] = {
                    "selection": selection,
                    "legacy_unfiltered_heldout_count": len(set(name for name, _ in views[::stride]) & set(test)),
                }
            row.update(status="PASS", poses_sha256=hashlib.sha256(poses.read_bytes()).hexdigest(), protocols=protocols)
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            row.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    result = {"schema_version": 1, "scope": "real_roster_input_audit_only", "paper_ready": False,
              "code": {"commit": commit, "dirty": dirty},
              "config": {"path": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()},
              "planned_scenes": len(rows), "passed_scenes": sum(row["status"] == "PASS" for row in rows), "scenes": rows}
    write_training_manifest(args.out, result)
    print(json.dumps({key: result[key] for key in ("scope", "planned_scenes", "passed_scenes", "paper_ready")}))
    return 0 if result["passed_scenes"] == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
