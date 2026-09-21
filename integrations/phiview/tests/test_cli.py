"""User-facing CLI contracts, including actual subprocess failure propagation."""

import json
import sys
from dataclasses import replace

import pytest
import yaml

from physicalview import cli
from physicalview.config import load_config
from physicalview.gpu import GpuInfo
from physicalview.jobs import JobSpec


def test_viewer_plan_is_portable_and_preserves_backend_args(capsys, monkeypatch):
    monkeypatch.setenv("HF_HOME", "/chosen/model-cache")
    assert (
        cli.main(["plan", "viewer", "--scene", "room_factory", "--", "--port", "8123"])
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert len(plan) == 1
    command = plan[0]["argv"]
    assert command[command.index("-m") + 1] == "physicalview.phiview"
    assert "--demo" in command and command[-2:] == ["--port", "8123"]
    assert plan[0]["env"]["HF_HOME"] == "/chosen/model-cache"
    assert plan[0]["where"] == "local"


def test_init_preserves_an_existing_user_config(tmp_path):
    path = tmp_path / "local.yaml"
    args = ["init", "--simany-root", str(tmp_path / "backend"), "--config", str(path)]
    assert cli.main(args) == 0
    saved = path.read_bytes()
    assert yaml.safe_load(saved)["simany_root"] == str(tmp_path / "backend")
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2 and path.read_bytes() == saved


@pytest.mark.backend
def test_full_plan_includes_construction_and_respects_environment(capsys, monkeypatch):
    monkeypatch.setenv("HF_HOME", "/chosen/model-cache")
    assert (
        cli.main(["plan", "full", "--scene", "room", "--discovery", "gt_segments"]) == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert len(plan) >= 7
    assert all(s["tags"]["pipeline"] == "full" for s in plan)
    assert all(s["env"]["HF_HOME"] == "/chosen/model-cache" for s in plan)
    assert any("export" in s["name"] for s in plan)


@pytest.mark.parametrize("failure", ["exit_code", "missing_artifact", None])
def test_run_waits_and_stops_after_a_failed_stage(tmp_path, monkeypatch, failure):
    cfg = replace(load_config(), repo_root=tmp_path, studio_out=tmp_path / "receipts")
    monkeypatch.setattr(cli, "load_config", lambda _: cfg)
    import physicalview.gpu

    monkeypatch.setattr(physicalview.gpu, "detect_gpu", lambda: GpuInfo(present=False))
    marker = tmp_path / "second-stage.txt"
    first = JobSpec(
        name="first",
        argv=[
            sys.executable,
            "-c",
            "raise SystemExit(3)" if failure == "exit_code" else "pass",
        ],
        needs_gpu=False,
        cwd=tmp_path,
        where="local",
        artifacts=[tmp_path / "absent"] if failure == "missing_artifact" else [],
    )
    second = JobSpec(
        name="second",
        argv=[
            sys.executable,
            "-c",
            f'from pathlib import Path; Path({str(marker)!r}).write_text("done")',
        ],
        needs_gpu=False,
        cwd=tmp_path,
        where="local",
    )
    monkeypatch.setattr(cli, "build_plan", lambda *_: [first, second])
    assert cli.main(["run", "full"]) == (1 if failure else 0)
    assert marker.exists() is (failure is None)
    results = list((cfg.studio_out / "jobs").glob("*/result.json"))
    assert len(results) == (1 if failure else 2)
    assert all(json.loads(p.read_text())["finished_utc"] for p in results)
