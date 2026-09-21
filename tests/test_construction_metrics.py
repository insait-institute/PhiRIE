import json
import shutil
import uuid
from pathlib import Path

import pytest

from robo.eval import construction_metrics
from robo.eval.construction_metrics import PAPER_SCENE_IDS, REQUIRED_REGIMES, aggregate


REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = REPO_ROOT / "tests" / ".construction-metrics-work"
_ACTIVE_WORK_ROOT: Path | None = None


@pytest.fixture(autouse=True)
def _manifest_workspace():
    global _ACTIVE_WORK_ROOT
    WORK_ROOT.mkdir(exist_ok=True)
    root = WORK_ROOT / uuid.uuid4().hex
    root.mkdir()
    _ACTIVE_WORK_ROOT = root
    try:
        yield
    finally:
        _ACTIVE_WORK_ROOT = None
        shutil.rmtree(root)
        try:
            WORK_ROOT.rmdir()
        except OSError:
            pass


def _normalized_manifest_record(row):
    normalized = dict(row)
    for key in ("f1_20", "stable_instances", "tested_instances", "runtime_minutes"):
        if normalized[key] in {"", "--", "null", "none", None}:
            normalized[key] = None
    reasons = normalized["validity_reasons"]
    if isinstance(reasons, str):
        normalized["validity_reasons"] = [
            value.strip() for value in reasons.split(";") if value.strip()
        ]
    normalized["scene_status"] = str(normalized["scene_status"]).lower()
    return normalized


def _sync_manifest(row):
    path = REPO_ROOT / row["build_manifest_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _normalized_manifest_record(row)
    payload = {
        "schema_version": 1,
        "freeze_id": record["freeze_id"],
        "regime": record["regime"],
        "scene_id": record["scene_id"],
        "build_commit": record["build_commit"],
        "source_artifact_hash": record["source_artifact_hash"],
        "record_valid": record["record_valid"],
        "validity_reasons": record["validity_reasons"],
        "record": record,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _record(regime, scene_id="scene-0", **updates):
    assert _ACTIVE_WORK_ROOT is not None
    regime_id = REQUIRED_REGIMES.index(regime)
    manifest = _ACTIVE_WORK_ROOT / "builds" / f"regime-{regime_id}" / f"{scene_id}.json"
    row = {
        "freeze_id": "freeze-test",
        "regime": regime,
        "scene_id": scene_id,
        "scene_status": "success",
        "input_instances": 4,
        "accepted_instances": 2,
        "f1_20": 0.5,
        "f1_weight": 1,
        "stable_instances": 1,
        "tested_instances": 2,
        "runtime_minutes": 10,
        "build_commit": "0123456789abcdef",
        "build_manifest_path": str(manifest.relative_to(REPO_ROOT)),
        "failure_reason": "",
        "source_artifact_hash": "a" * 64,
        "record_valid": True,
        "validity_reasons": "",
        "geometry_reference_status": "independent_gt_evaluation",
    }
    row.update(updates)
    _sync_manifest(row)
    return row


def _complete_rows():
    return [
        _record(regime, scene_id)
        for regime in REQUIRED_REGIMES
        for scene_id in PAPER_SCENE_IDS
    ]


def _smoke_rows():
    return [
        _record(regime, scene_id)
        for regime in REQUIRED_REGIMES[:2]
        for scene_id in ("scene-a", "scene-b")
    ]


def _aggregate_row(rows, regime=REQUIRED_REGIMES[0]):
    return next(row for row in aggregate(rows) if row["regime"] == regime)


def test_requires_exact_five_regimes_and_uses_canonical_order():
    rows = list(reversed(_complete_rows()))
    result = aggregate(rows)
    assert [row["regime"] for row in result] == list(REQUIRED_REGIMES)
    assert result == aggregate(list(reversed(rows)))

    with pytest.raises(ValueError, match="exactly the five fixed regimes"):
        aggregate([row for row in rows if row["regime"] != REQUIRED_REGIMES[-1]])


def test_default_rejects_non_pinned_population_even_with_all_five_regimes():
    rows = [_record(regime, "not-the-paper-scene") for regime in REQUIRED_REGIMES]
    with pytest.raises(ValueError, match="exact pinned 50-scene roster"):
        aggregate(rows)


@pytest.mark.parametrize(
    "column", ["freeze_id", "scene_status", "build_commit", "build_manifest_path"]
)
def test_requires_status_and_provenance_columns(column):
    rows = _complete_rows()
    del rows[0][column]
    with pytest.raises(ValueError, match="missing required columns"):
        aggregate(rows)


@pytest.mark.parametrize("column", ["freeze_id", "build_commit", "build_manifest_path"])
def test_rejects_empty_provenance(column):
    rows = _complete_rows()
    rows[0][column] = ""
    with pytest.raises(ValueError, match=f"empty {column}"):
        aggregate(rows)


@pytest.mark.parametrize(
    "commit", ["not-a-commit", "ABCDEF1", "123456", "a" * 41]
)
def test_valid_record_requires_lowercase_git_commit(commit):
    rows = _complete_rows()
    rows[0]["build_commit"] = commit
    with pytest.raises(ValueError, match="lowercase 7-40.*Git SHA"):
        aggregate(rows)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("source_artifact_hash", "not-a-sha", "lowercase SHA-256"),
        ("source_artifact_hash", "A" * 64, "lowercase SHA-256"),
        ("geometry_reference_status", "", "empty geometry_reference_status"),
    ],
)
def test_requires_source_and_geometry_provenance(column, value, message):
    rows = _complete_rows()
    rows[0][column] = value
    with pytest.raises(ValueError, match=message):
        aggregate(rows)


def test_f1_weight_is_an_integer_instance_count():
    rows = _complete_rows()
    rows[0]["f1_weight"] = 0.5
    with pytest.raises(ValueError, match="non-negative integer f1_weight"):
        aggregate(rows)


def test_valid_accepted_assets_require_independent_geometry_evidence():
    rows = _complete_rows()
    rows[0]["geometry_reference_status"] = "unavailable"
    _sync_manifest(rows[0])
    with pytest.raises(ValueError, match="independent_gt_evaluation"):
        aggregate(rows)


def test_rejects_duplicate_regime_scene_pair():
    rows = _complete_rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate construction row"):
        aggregate(rows)


def test_requires_identical_planned_scene_ids_in_every_regime():
    rows = _complete_rows()
    rows.append(_record(REQUIRED_REGIMES[0], "scene-extra"))
    with pytest.raises(ValueError, match="identical planned scene IDs"):
        aggregate(rows)


def test_smoke_accepts_canonical_subset_in_canonical_order():
    rows = list(reversed(_smoke_rows()))
    result = aggregate(rows, smoke=True)
    assert [row["regime"] for row in result] == list(REQUIRED_REGIMES[:2])
    assert all(row["planned_scenes"] == 2 for row in result)


def test_default_still_rejects_smoke_regime_subset():
    with pytest.raises(ValueError, match="exactly the five fixed regimes"):
        aggregate(_smoke_rows())


def test_smoke_requires_identical_scene_ids_within_subset():
    rows = _smoke_rows()
    rows[-1] = _record(REQUIRED_REGIMES[1], "scene-c")
    with pytest.raises(ValueError, match="identical planned scene IDs"):
        aggregate(rows, smoke=True)


def test_smoke_does_not_implicitly_allow_invalid_legacy_rows():
    rows = _smoke_rows()
    rows[0].update(
        record_valid=False,
        validity_reasons="legacy provenance",
        build_commit="legacy-unrecorded",
    )
    _sync_manifest(rows[0])
    with pytest.raises(ValueError, match="preliminary/invalid.*refused"):
        aggregate(rows, smoke=True)


def test_failed_scene_is_preserved_in_planned_and_status_counts():
    rows = _complete_rows()
    target = next(row for row in rows if row["regime"] == REQUIRED_REGIMES[0])
    target.update(
        scene_status="failed",
        input_instances=3,
        accepted_instances=0,
        f1_20="",
        f1_weight=0,
        stable_instances="",
        tested_instances="",
        runtime_minutes="",
        failure_reason="reconstruction failed",
        geometry_reference_status="not_applicable_failure",
    )
    _sync_manifest(target)
    row = _aggregate_row(rows)
    assert row["scenes"] == 50
    assert row["planned_scenes"] == 50
    assert row["successful_scenes"] == 49
    assert row["completed_scenes"] == 49
    assert row["contributing_scenes"] == 50
    assert row["failed_scenes"] == 1
    assert row["instances"] == 199


def test_yield_uses_summed_counts_and_f1_uses_explicit_weights():
    rows = _smoke_rows()
    rows[0].update(
        input_instances=10,
        accepted_instances=6,
        f1_20=0.5,
        f1_weight=2,
    )
    rows[1].update(
        input_instances=5,
        accepted_instances=4,
        f1_20=0.8,
        f1_weight=1,
    )
    _sync_manifest(rows[0])
    _sync_manifest(rows[1])
    row = next(
        row for row in aggregate(rows, smoke=True)
        if row["regime"] == REQUIRED_REGIMES[0]
    )
    assert row["yield"] == 10 / 15
    assert row["f1_20"] == pytest.approx(0.6)
    assert row["f1_weight"] == 3


@pytest.mark.parametrize("weight", ["", 0])
def test_f1_value_requires_a_positive_explicit_weight(weight):
    rows = _complete_rows()
    rows[0]["f1_weight"] = weight
    with pytest.raises(ValueError, match="f1_weight"):
        aggregate(rows)


def test_missing_f1_requires_explicit_zero_weight():
    rows = _complete_rows()
    rows[0].update(f1_20="", f1_weight=1)
    with pytest.raises(ValueError, match="f1_weight=0"):
        aggregate(rows)


def test_missing_stability_is_null_and_not_a_pass():
    rows = _complete_rows()
    for record in rows:
        if record["regime"] == REQUIRED_REGIMES[0]:
            record.update(stable_instances="", tested_instances="")
            _sync_manifest(record)
    row = _aggregate_row(rows)
    assert row["stable_instances"] is None
    assert row["tested_instances"] is None
    assert row["stability"] is None


def test_empty_scene_is_completed_but_not_contributing_and_renders_planned_count():
    rows = _complete_rows()
    for regime in REQUIRED_REGIMES:
        record = next(row for row in rows if row["regime"] == regime)
        record.update(
            scene_status="empty",
            input_instances=0,
            accepted_instances=0,
            f1_20="",
            f1_weight=0,
            stable_instances="",
            tested_instances="",
            runtime_minutes=3,
            failure_reason="report-backed scene contains zero instances",
            geometry_reference_status="not_applicable_empty",
        )
        _sync_manifest(record)

    aggregated = aggregate(rows)
    row = aggregated[0]
    assert row["planned_scenes"] == 50
    assert row["successful_scenes"] == 49
    assert row["completed_scenes"] == 50
    assert row["contributing_scenes"] == 49
    latex = construction_metrics.render_latex(aggregated)
    assert "inst./planned scenes" in latex
    assert f"{REQUIRED_REGIMES[0]} & 196/50 &" in latex


def test_zero_attempted_stability_is_null():
    rows = _complete_rows()
    for record in rows:
        if record["regime"] == REQUIRED_REGIMES[0]:
            record.update(stable_instances=0, tested_instances=0)
            _sync_manifest(record)
    row = _aggregate_row(rows)
    assert row["stable_instances"] == 0
    assert row["tested_instances"] == 0
    assert row["stability"] is None


def test_default_refuses_invalid_preliminary_records():
    rows = _complete_rows()
    rows[0].update(
        record_valid=False,
        validity_reasons="legacy row lacks independent source evidence",
    )
    _sync_manifest(rows[0])
    with pytest.raises(ValueError, match="preliminary/invalid.*refused"):
        aggregate(rows)


def test_preliminary_override_keeps_invalidity_visible():
    rows = _complete_rows()
    rows[0].update(
        record_valid=False,
        validity_reasons=["legacy row", "missing independent match"],
    )
    _sync_manifest(rows[0])
    result = aggregate(rows, allow_preliminary=True)
    assert result[0]["valid_for_paper"] is False
    assert result[0]["invalid_record_count"] == 1
    assert result[0]["validity_reasons"] == (
        "legacy row; missing independent match"
    )


def test_invalid_preliminary_record_accepts_explicit_legacy_commit_sentinel():
    rows = _complete_rows()
    rows[0].update(
        record_valid=False,
        validity_reasons="legacy provenance unavailable",
        build_commit="legacy-unrecorded",
    )
    _sync_manifest(rows[0])
    result = aggregate(rows, allow_preliminary=True)
    assert result[0]["valid_for_paper"] is False


@pytest.mark.parametrize(
    "path", ["../../outside/manifest.json", "/outside/manifest.json"]
)
def test_rejects_manifest_paths_outside_repository(path):
    rows = _complete_rows()
    rows[0]["build_manifest_path"] = path
    with pytest.raises(ValueError, match="outside the repository"):
        aggregate(rows)


def test_rejects_missing_or_identity_mismatched_build_manifest():
    rows = _complete_rows()
    manifest = REPO_ROOT / rows[0]["build_manifest_path"]
    manifest.unlink()
    with pytest.raises(ValueError, match="build_manifest_path is not a file"):
        aggregate(rows)

    _sync_manifest(rows[0])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["scene_id"] = "wrong-scene"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest scene_id disagrees"):
        aggregate(rows)


def test_generate_publishes_one_atomic_output_directory(tmp_path):
    input_path = tmp_path / "scene_records.json"
    input_path.write_text(json.dumps(_complete_rows()), encoding="utf-8")
    out_dir = tmp_path / "table"
    payload = construction_metrics.generate(input_path, out_dir)

    assert [row["regime"] for row in payload["rows"]] == list(REQUIRED_REGIMES)
    assert {path.name for path in out_dir.iterdir()} == {
        "construction_table.json",
        "construction_table.csv",
        "construction_table.tex",
    }
    assert not list(tmp_path.glob(".table.staging-*"))
    assert payload["paper_ready"] is True
    assert payload["preliminary_override"] is False
    assert payload["provenance"]["build_manifest_count"] == 250
    assert len(payload["provenance"]["scene_records_sha256"]) == 64
    assert len(payload["provenance"]["build_manifest_set_sha256"]) == 64
    assert json.loads((out_dir / "construction_table.json").read_text()) == payload

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        construction_metrics.generate(input_path, out_dir)


def test_generate_failure_leaves_no_partial_bundle(tmp_path, monkeypatch):
    input_path = tmp_path / "scene_records.json"
    input_path.write_text(json.dumps(_complete_rows()), encoding="utf-8")
    out_dir = tmp_path / "table"

    def fail_fsync(_path):
        raise OSError("injected directory fsync failure")

    monkeypatch.setattr(construction_metrics, "_fsync_directory", fail_fsync)
    with pytest.raises(OSError, match="injected directory fsync failure"):
        construction_metrics.generate(input_path, out_dir)

    assert not out_dir.exists()
    assert not list(tmp_path.glob(".table.staging-*"))


def test_generate_preliminary_override_forces_nonpaper_artifacts(tmp_path):
    rows = _complete_rows()
    rows[0].update(record_valid=False, validity_reasons="legacy evidence")
    _sync_manifest(rows[0])
    input_path = tmp_path / "scene_records.json"
    input_path.write_text(json.dumps(rows), encoding="utf-8")

    payload = construction_metrics.generate(
        input_path, tmp_path / "table", allow_preliminary=True
    )

    assert payload["paper_ready"] is False
    assert payload["preliminary_override"] is True
    latex = (tmp_path / "table" / "construction_table.tex").read_text()
    assert "PRELIMINARY AUDIT OUTPUT" in latex


def test_generate_smoke_forces_visible_nonpaper_artifacts(tmp_path):
    input_path = tmp_path / "smoke_records.json"
    input_path.write_text(json.dumps(_smoke_rows()), encoding="utf-8")

    payload = construction_metrics.generate(
        input_path, tmp_path / "smoke-table", smoke=True
    )

    assert payload["paper_ready"] is False
    assert payload["smoke"] is True
    assert payload["preliminary_override"] is False
    assert len(payload["rows"]) == 2
    persisted = json.loads(
        (tmp_path / "smoke-table" / "construction_table.json").read_text()
    )
    assert persisted == payload
    latex = (tmp_path / "smoke-table" / "construction_table.tex").read_text()
    assert "SMOKE/PRELIMINARY OUTPUT" in latex
    assert "NOT VALID FOR PAPER" in latex


def test_smoke_cli_persists_nonpaper_marker(tmp_path, capsys):
    input_path = tmp_path / "smoke_records.json"
    input_path.write_text(json.dumps(_smoke_rows()), encoding="utf-8")
    out_dir = tmp_path / "cli-table"

    assert construction_metrics.main(
        ["--input", str(input_path), "--out", str(out_dir), "--smoke"]
    ) == 0

    printed = json.loads(capsys.readouterr().out)
    persisted = json.loads((out_dir / "construction_table.json").read_text())
    assert printed == persisted
    assert persisted["smoke"] is True
    assert persisted["paper_ready"] is False
