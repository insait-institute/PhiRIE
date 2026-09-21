import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

from robo.eval import audit_loso, audit_metrics
from robo.eval import build_task_support_dataset as builder


def _read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_smoke_dataset_is_unlabeled_then_joined_one_to_one(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    out = tmp_path / "outputs" / "e6-smoke" / "features"
    summary = builder.generate_smoke(out)

    assert summary["query_rows"] == 6
    assert summary["paper_ready"] is False
    unlabeled = _read_csv(out / "features_unlabeled.csv")
    labels = _read_csv(out / "labels.csv")
    joined = _read_csv(out / "task_local_features_and_labels.csv")
    assert len(unlabeled) == len(labels) == len(joined) == 6
    assert "invalid_label" not in unlabeled[0]
    assert "invalid_reasons" not in unlabeled[0]

    def keys(rows):
        return {tuple(row[field] for field in builder.JOIN_KEY) for row in rows}
    assert keys(unlabeled) == keys(labels) == keys(joined)
    assert {row["scene_id"] for row in joined} == set(builder.SMOKE_SCENES)
    by_scene = defaultdict(set)
    for row in joined:
        by_scene[row["scene_id"]].add(int(row["invalid_label"]))
        assert (out / row["graph_path"]).is_file()
        assert (out / row["build_manifest_path"]).is_file()
    assert all(values == {0, 1} for values in by_scene.values())
    assert {
        row["condition_id"] for row in joined
    } == {"clean", "mild", "severe"}

    manifest = json.loads((out / "smoke_manifest.json").read_text())
    assert manifest["stage_order"] == [
        "features_unlabeled", "labels", "label_join"]
    assert manifest["feature_generation_read_labels"] is False
    assert manifest["synthetic_only"] is True
    assert manifest["paper_ready"] is False


def test_smoke_dataset_is_deterministic_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    builder.generate_smoke(first)
    builder.generate_smoke(second)
    for relative in (
        "features_unlabeled.csv",
        "labels.csv",
        "task_local_features_and_labels.csv",
        "smoke_manifest.json",
    ):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.generate_smoke(first)


def test_smoke_output_must_be_repository_local(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    monkeypatch.setattr(builder, "ROOT", repository)
    with pytest.raises(ValueError, match="inside repository"):
        builder.generate_smoke(tmp_path / "outside")


def test_smoke_runs_exact_six_loso_groups_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    dataset = tmp_path / "dataset"
    builder.generate_smoke(dataset)
    predictions_dir = tmp_path / "predictions"
    result = audit_loso.generate(
        dataset / "task_local_features_and_labels.csv", predictions_dir)
    assert tuple(result["signals"]) == audit_loso.SIGNAL_GROUP_NAMES
    assert result["n_queries"] == 6
    assert result["n_predictions"] == 36

    predictions = _read_csv(predictions_dir / "heldout_predictions.csv")
    assert all(row["scene_id"] == row["heldout_scene"] for row in predictions)
    assert all(row["heldout_scene"] not in row["training_scenes"].split(";")
               for row in predictions)
    table_dir = tmp_path / "table"
    metrics = audit_metrics.generate(
        predictions_dir / "heldout_predictions.csv", table_dir)
    assert len(metrics["rows"]) == 6
    assert all(math.isfinite(row["auroc"]) for row in metrics["rows"])
    assert all(math.isfinite(row["brier"]) for row in metrics["rows"])
    risk_rows = _read_csv(table_dir / "risk_coverage.csv")
    assert len(risk_rows) == 36
    assert {row["signal"] for row in risk_rows} == set(
        audit_loso.SIGNAL_GROUP_NAMES)
