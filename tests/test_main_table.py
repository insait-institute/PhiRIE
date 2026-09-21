import json

import pytest

from robo.eval.harness_spec import load_harness_spec
from robo.eval.main_table import aggregate, generate, render_latex


def _record(treatment, reset, success, outcome="task_failure",
            family="object_to_receptacle"):
    return {
        "treatment_id": treatment, "reset_state_id": reset,
        "outcome": "success" if success else outcome, "success": success,
        "task_family": family,
        "stages": {"grasp": success, "lift": success, "place": success}}


def test_failures_stay_in_planned_denominator():
    records = [
        _record("a", "r0", True),
        _record("a", "r1", False, outcome="build_failure"),
        _record("b", "r0", True),
        _record("b", "r1", True)]
    rows, _ = aggregate(records, ["a", "b"])
    lookup = {row["treatment_id"]: row for row in rows}
    assert lookup["a"]["coverage"] == 0.5
    assert lookup["a"]["success"] == 0.5
    assert lookup["b"]["success"] == 1.0


def test_latex_is_generated_from_comparison_spec():
    spec = load_harness_spec({
        "treatments": [{"id": "a", "scene": "box_proxy"}, {"id": "b"}],
        "comparisons": [{"id": "scene", "axis": "scene",
                         "treatments": ["a", "b"]}]})
    rows, _ = aggregate([_record("a", "r", False), _record("b", "r", True)],
                        ["a", "b"])
    text = render_latex(rows, spec)
    assert "Paired harness evaluation" in text
    assert "100.0\\%" in text


def _config():
    return {"contract": {"reset_ids": ["r0", "r1"],
                "policy": {"id": "scripted_sinusoid", "kind": "scripted_smoke"},
                "robot": {"id": "fixture"}, "cameras": {"id": "fixture"},
                "action_convention": "absolute_joint_position",
                "controller": {"id": "fixture"}, "control_rate_hz": 15,
                "horizon_s": 16, "task_instruction": "fixed instruction",
                "rubric": {"id": "fixture"}},
            "treatments": [{"id": "a", "scene": "box_proxy"}, {"id": "b"}],
            "comparisons": [{"id": "scene", "axis": "scene",
                             "treatments": ["a", "b"]}]}


def _ledger(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    return path


def test_absent_family_is_unevaluated_not_zero_success(tmp_path):
    records = [_record(t, r, False, family="object_to_region")
               for t in ("a", "b") for r in ("r0", "r1")]
    report = generate(_config(), _ledger(tmp_path / "ledger.jsonl", records),
                      tmp_path / "table")
    for row in report["rows"]:
        assert row["object_to_receptacle"] is None
        assert row["object_to_receptacle_planned"] == 0
        assert row["object_to_region_planned"] == 2
        assert row["object_to_region"] == 0.0
    assert report["denominator_source"] == "contract.reset_ids"
    assert [r["kind"] for r in report["degenerate_success_blocks"]] == ["all_zero"] * 2
    assert " & -- & 0.0\\%" in (tmp_path / "table/main_table.tex").read_text()
    import csv
    with (tmp_path / "table/main_table.csv").open() as stream:
        assert all(row["object_to_receptacle"] == "" for row in csv.DictReader(stream))


@pytest.mark.parametrize("damage", ["drop_one", "drop_both", "duplicate", "extra"])
def test_population_damage_cannot_inflate_table_coverage(tmp_path, damage):
    records = [_record(t, r, True) for t in ("a", "b") for r in ("r0", "r1")]
    if damage == "drop_one":
        records.pop()
    elif damage == "drop_both":
        records = [r for r in records if r["reset_state_id"] == "r0"]
    elif damage == "duplicate":
        records.append(dict(records[0]))
    else:
        records.append(_record("a", "unplanned", True))
    with pytest.raises(ValueError, match="invalid manipulation table population"):
        generate(_config(), _ledger(tmp_path / "ledger.jsonl", records), tmp_path / "table")
    assert not (tmp_path / "table").exists()


def test_table_output_is_immutable(tmp_path):
    records = [_record(t, r, True) for t in ("a", "b") for r in ("r0", "r1")]
    ledger = _ledger(tmp_path / "ledger.jsonl", records)
    output = tmp_path / "table"
    generate(_config(), ledger, output)
    original = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match="refusing to overwrite"):
        generate(_config(), ledger, output)
    assert original == {p.name: p.read_bytes() for p in output.iterdir()}


def test_explicit_resume_reuses_matching_table_without_any_write(tmp_path):
    records = [_record(t, r, True) for t in ("a", "b") for r in ("r0", "r1")]
    ledger = _ledger(tmp_path / "ledger.jsonl", records)
    output = tmp_path / "table"
    report = generate(_config(), ledger, output)
    original = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
    assert generate(_config(), ledger, output, reuse_existing=True) == report
    assert original == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}


@pytest.mark.parametrize("damage", ["config", "ledger", "csv", "missing_provenance", "implementation"])
def test_resume_rejects_drift_and_leaves_existing_table_untouched(tmp_path,damage):
    import json
    records = [_record(t, r, True) for t in ("a", "b") for r in ("r0", "r1")]
    ledger = _ledger(tmp_path / "ledger.jsonl", records)
    output = tmp_path / "table";config = _config()
    generate(config, ledger, output)
    if damage == "config": config["resume_annotation"] = "different"
    elif damage == "ledger": ledger.write_text(ledger.read_text()+"\n")
    elif damage == "csv": (output/"main_table.csv").write_text("changed")
    elif damage == "missing_provenance": (output/"table_provenance.json").unlink()
    else:
        path=output/"table_provenance.json";value=json.loads(path.read_text())
        value["source"]["implementation_sha256"]["metric_utils.py"]="0"*64
        path.write_text(json.dumps(value))
    original = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError,match="existing manipulation table"):
        generate(config, ledger, output, reuse_existing=True)
    assert original == {p.name: p.read_bytes() for p in output.iterdir()}


def test_empty_arm_has_no_measured_rates():
    rows, _ = aggregate([], ["a"])
    assert rows[0]["planned"] == 0
    assert all(rows[0][k] is None for k in ("coverage", "success", "grasp", "lift", "place"))


def test_unknown_family_cannot_disappear_from_family_denominators():
    with pytest.raises(ValueError, match="unknown manipulation task family"):
        aggregate([_record("a", "r0", True, family="typo")], ["a"])


def test_legacy_smoke_population_is_explicitly_not_predeclared(tmp_path):
    config = _config()
    config.pop("contract")
    records = [_record(t, "r0", True) for t in ("a", "b")]
    report = generate(config, _ledger(tmp_path / "ledger.jsonl", records), tmp_path / "table")
    assert report["denominator_source"] == "observed_union_smoke_only"
    assert report["population_validation"]["planned_per_treatment"] == 1


@pytest.mark.parametrize("missing", ["task_family", "reset_ids"])
def test_paper_export_cannot_use_legacy_inference(tmp_path, monkeypatch, missing):
    # The full policy contract is tested by harness_spec; exercise export's
    # additional requirements with an already parsed spec at this boundary.
    from robo.eval import main_table
    spec = load_harness_spec(_config())
    spec.raw["paper_mode"] = True
    records = [_record(t, r, True) for t in ("a", "b") for r in ("r0", "r1")]
    if missing == "task_family":
        records[0].pop("task_family")
    else:
        spec.raw["contract"].pop("reset_ids")
    monkeypatch.setattr(main_table, "load_harness_spec", lambda _: spec)
    with pytest.raises(ValueError, match="paper table requires"):
        generate({}, _ledger(tmp_path / "ledger.jsonl", records), tmp_path / "table")
    assert not (tmp_path / "table").exists()
