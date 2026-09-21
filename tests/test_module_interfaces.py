"""CPU tests for operational contracts, isolation, failures and adapter compatibility."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from phiroom.cli import main
from phiroom.core.execution import Command, execute, pipeline, plan
from phiroom.core.registry import catalog
from phiroom.core.runtime import interpreter, load_config

ROOT = Path(__file__).absolute().parents[1]


def command(tmp_path, script, **kw):
    return Command(
        "fixture",
        "cpu",
        (sys.executable, "-c", script),
        str(tmp_path),
        {},
        "control",
        False,
        **kw,
    )


def test_catalog_does_not_import_gpu_backends():
    script = """
import sys
from phiroom.core.registry import catalog
assert len(catalog()) == 16
assert not any(x in sys.modules for x in ('torch', 'mujoco', 'gsplat', 'agents.core.common', 'physicalview.phiview'))
"""
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)


def test_every_declared_action_resolves_to_shipped_code():
    # PhiView is private and optional in parent-only CI checkouts.
    for name, block in catalog().items():
        for action, spec in block["actions"].items():
            if (
                spec.get("kind") == "phiview"
                and not (ROOT / "integrations/phiview/physicalview/cli.py").exists()
            ):
                continue
            result = plan(name, action, [], ROOT, {})
            assert result.cwd == str(ROOT)
            assert result.argv


def test_plan_preserves_prompt_and_has_no_side_effects(tmp_path, capsys):
    prompt = 'remove the red cup; $(touch NEVER_EXECUTE) "keep lighting"'
    before = set(tmp_path.iterdir())
    assert (
        main(
            [
                "plan",
                "inpainting",
                "views",
                "--root",
                str(ROOT),
                "--receipts",
                str(tmp_path),
                "--",
                "--prompt",
                prompt,
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["steps"][0]["argv"][-1] == prompt
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("module,action", [("absent", "go"), ("physics", "absent")])
def test_unknown_actions_fail_before_execution(module, action):
    with pytest.raises(ValueError, match="unknown"):
        plan(module, action, [], ROOT, {})


def test_interpreter_keeps_venv_symlink_and_config_precedence(tmp_path, monkeypatch):
    venv = tmp_path / "venv/bin/python"
    venv.parent.mkdir(parents=True)
    venv.symlink_to(sys.executable)
    monkeypatch.setenv("SIMANY_PY", "/wrong/python")
    assert interpreter(
        "main", tmp_path, {"runtimes": {"main": "venv/bin/python"}}
    ) == str(venv)


def test_execution_preserves_literal_arguments(tmp_path):
    text = 'red cup; $(touch forbidden) "double"'
    cmd = Command(
        "fixture",
        "argv",
        (
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; Path('result').write_text(sys.argv[1])",
            text,
        ),
        str(tmp_path),
        {},
        "control",
        False,
        produces=(str(tmp_path / "result"),),
    )
    code, receipt = execute([cmd], tmp_path / "receipts")
    assert code == 0
    assert (tmp_path / "result").read_text() == text
    assert not (tmp_path / "forbidden").exists()
    assert json.loads(receipt.read_text())["scientific_validation"] == "NOT_RUN"


def test_failed_step_stops_pipeline_and_preserves_logs(tmp_path):
    bad = command(
        tmp_path, "print('failure evidence', flush=True); raise SystemExit(7)"
    )
    later = command(tmp_path, "from pathlib import Path; Path('later').touch()")
    code, receipt = execute([bad, later], tmp_path / "receipts")
    data = json.loads(receipt.read_text())
    assert code == 7 and data["status"] == "failed" and data["steps_not_run"] == 1
    assert "failure evidence" in Path(data["steps"][0]["log"]).read_text()
    assert not (tmp_path / "later").exists()


@pytest.mark.parametrize(
    "failure", ["missing_input", "missing_output", "stale_output", "missing_runtime"]
)
def test_fail_closed_artifact_and_runtime_checks(tmp_path, failure):
    marker = tmp_path / "output"
    options = {}
    if failure == "missing_input":
        options["requires"] = (str(marker),)
    elif failure in {"missing_output", "stale_output"}:
        options["produces"] = (str(marker),)
        if failure == "stale_output":
            marker.write_text("preserve me")
    cmd = command(tmp_path, "print('executed')", **options)
    if failure == "missing_runtime":
        from dataclasses import replace

        cmd = replace(cmd, argv=(str(tmp_path / "absent-python"),))
    code, receipt = execute([cmd], tmp_path / "receipts")
    assert code != 0 and json.loads(receipt.read_text())["status"] == "failed"
    if failure == "stale_output":
        assert marker.read_text() == "preserve me"


def test_pipeline_accepts_only_explicit_steps(tmp_path):
    path = tmp_path / "pipeline.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "steps": [
                    {"module": "experiments", "action": "finalize", "args": ["--help"]}
                ],
            }
        )
    )
    commands = pipeline(path, ROOT, {})
    assert len(commands) == 1
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "steps": [
                    {"module": "experiments", "action": "finalize", "args": "--help"}
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="list"):
        pipeline(path, ROOT, {})


def test_runtime_config_rejects_unknown_aliases_and_secrets(tmp_path):
    path = tmp_path / "runtime.json"
    for cfg in ({"runtimes": {"wrong": "python"}}, {"env": {"SIMANY_TOKEN": "secret"}}):
        path.write_text(json.dumps(cfg))
        with pytest.raises(ValueError):
            load_config(str(path))


def test_split_adapters_keep_generation_order_and_isolation(monkeypatch, tmp_path):
    from simfactory.blocks import generation
    from simfactory.registry import Context, get

    calls = []
    monkeypatch.setattr(
        generation, "sh", lambda ctx, module, *a, **kw: calls.append((module, kw))
    )
    ctx = Context(
        ROOT, "scene", "scannetpp", {}, tmp_path, tmp_path, tmp_path, tmp_path
    )
    get("generation", "hybrid").run(ctx, {"candidates": ["trellis", "sam3d"]})
    assert [c[0] for c in calls] == [
        "models.s4_trellis",
        "models.s4_sam3d",
        "agents.assets.factory_align",
        "agents.assets.factory_hybrid",
        "agents.assets.s6_physics",
    ]
    assert calls[0][1]["env_kind"] == "venv"
    assert calls[1][1]["env_kind"] == "sam3d"


def test_simfactory_list_has_no_circular_import():
    result = subprocess.run(
        [sys.executable, "-m", "simfactory.runner", "list"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "reconstruction:" in result.stdout and "evaluation:" in result.stdout


def test_nested_factory_receives_explicit_runtime_bindings(tmp_path):
    cfg = {"runtimes": {"sam3": "sam3/bin/python", "main": "main/bin/python"}}
    spec = plan("experiments", "factory", ["list"], ROOT, cfg)
    assert spec.env["SIMANY_SAM3_PY"] == str(ROOT / "sam3/bin/python")
    assert spec.env["SIMANY_PY"] == str(ROOT / "main/bin/python")
