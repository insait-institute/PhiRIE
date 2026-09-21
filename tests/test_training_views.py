import json
from pathlib import Path

import pytest

from agents.discover.training_views import (
    select_training_views, validate_fusion_frames, write_training_manifest,
)
from robo.eval.fidelity_replacements import ReplacementError


def split(tmp_path, train=None, test=None):
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": train or ["b.jpg", "d.jpg", "f.jpg"],
                                "test": test or ["a.jpg", "c.jpg", "e.jpg"]}))
    return path


def test_train_filter_precedes_stride_and_never_uses_heldout_pose(tmp_path):
    views = [(f"{c}.jpg", c) for c in "abcdef"]
    selected, manifest = select_training_views(views, split(tmp_path), 2)
    assert selected == [("b.jpg", "b"), ("f.jpg", "f")]
    # Inserting additional test cameras must not change the training selection.
    assert select_training_views(views + [("aa.jpg", "heldout")], split(tmp_path), 2)[0] == selected
    assert manifest["paper_ready"] is False


@pytest.mark.parametrize("stride", [0, -1, True, 1.5])
def test_invalid_stride_fails(tmp_path, stride):
    with pytest.raises(ValueError, match="stride"):
        select_training_views([], split(tmp_path), stride)


def test_bad_split_or_missing_train_poses_fails_closed(tmp_path):
    with pytest.raises(ReplacementError, match="overlap"):
        select_training_views([], split(tmp_path, train=["a.jpg"], test=["a.jpg"]), 1)
    with pytest.raises(ValueError, match="missing poses"):
        select_training_views([("b.jpg", None)], split(tmp_path), 1)
    with pytest.raises(ValueError, match="duplicate"):
        select_training_views([("b.jpg", None), ("b.jpg", None)], split(tmp_path), 1)


def test_partial_mislabeled_and_changed_split_cannot_be_fused(tmp_path):
    path = split(tmp_path)
    _, manifest = select_training_views([(f"{c}.jpg", None) for c in "abcdef"], path, 2)
    files = [Path("view_0000.npz"), Path("view_0001.npz")]
    frames = {files[0]: "b.jpg", files[1]: "f.jpg"}
    validate_fusion_frames(files, manifest, frames.__getitem__)
    with pytest.raises(ValueError, match="incomplete"):
        validate_fusion_frames(files[:1], manifest, frames.__getitem__)
    frames[files[1]] = "e.jpg"
    with pytest.raises(ValueError, match="differs"):
        validate_fusion_frames(files, manifest, frames.__getitem__)
    manifest["selected_frames"][-1] = "e.jpg"
    with pytest.raises(ValueError, match="official training"):
        validate_fusion_frames(files, manifest, frames.__getitem__)
    path.write_text('{}')
    with pytest.raises(ValueError, match="changed"):
        validate_fusion_frames(files, manifest, frames.__getitem__)


def test_manifest_cannot_overwrite_previous_attempt(tmp_path):
    output = tmp_path / "manifest.json"
    write_training_manifest(output, {"status": "planned"})
    with pytest.raises(FileExistsError):
        write_training_manifest(output, {"status": "rerun"})


def test_audit_cli_accepts_numeric_yaml_scene_id(tmp_path):
    from agents.discover.training_views import main
    scene = tmp_path / "data/3864514494/dslr"
    (scene / "colmap").mkdir(parents=True)
    (scene / "train_test_lists.json").write_text(json.dumps({"train": ["b.jpg"], "test": ["a.jpg"]}))
    (scene / "colmap/images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 a.jpg\n0 0 -1\n2 1 0 0 0 0 0 0 1 b.jpg\n0 0 -1\n")
    config = tmp_path / "config.yaml"
    config.write_text("population:\n  planned_scenes: 1\n  scene_ids: [3864514494]\n")
    output = tmp_path / "audit.json"
    assert main(["--config", str(config), "--dataset-root", str(tmp_path), "--out", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["passed_scenes"] == 1
    assert result["scenes"][0]["scene_id"] == "3864514494"
    assert result["scenes"][0]["protocols"]["auto_discovery"]["legacy_unfiltered_heldout_count"] == 1


def test_pilot_limit_is_before_stride_and_fusion_replays_it(tmp_path):
    path = split(tmp_path, train=[f'{i:02d}.jpg' for i in range(60)], test=['test.jpg'])
    views = [(f'{i:02d}.jpg', i) for i in range(60)]
    selected, manifest = select_training_views(views, path, 12, max_train_frames=48)
    assert [v[0] for v in selected] == ['00.jpg', '12.jpg', '24.jpg', '36.jpg']
    assert len(manifest['training_frames']) == 48
    files = [Path(f'view_{i:04d}.npz') for i in range(4)]
    labels = dict(zip(files, manifest['selected_frames']))
    validate_fusion_frames(files, manifest, labels.__getitem__)
    manifest['max_train_frames'] = 60
    with pytest.raises(ValueError, match='boundary'):
        validate_fusion_frames(files, manifest, labels.__getitem__)


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_invalid_train_limit_rejected(tmp_path, limit):
    with pytest.raises(ValueError, match='limit'):
        select_training_views([], split(tmp_path), 1, max_train_frames=limit)
