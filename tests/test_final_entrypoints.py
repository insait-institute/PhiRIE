"""Compatibility/CLI tests. These never submit jobs or run model inference."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(os.environ.get("SIMANY_TEST_REPO_ROOT", Path(__file__).resolve().parents[1]))


def test_compatibility_entry_preserves_arguments_and_interpreter(tmp_path):
    old_entry = tmp_path / "run/finalize/run.sh"
    old_entry.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "run/finalize/run.sh", old_entry)
    current = tmp_path / "run/campaign/finalize.sh"
    current.parent.mkdir(parents=True)
    current.write_text('#!/usr/bin/env bash\nset -euo pipefail\nprintf "%s\\n" "$SIMANY_CAMPAIGN_PY" "$@"\n')
    env = dict(os.environ, SIMANY_FINAL_PY="/existing/venv with spaces/bin/python")
    env.pop("SIMANY_CAMPAIGN_PY", None)
    result = subprocess.run(["bash", str(old_entry), "verify", "--root", "/tmp/path with spaces"],
                            env=env, text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == [env["SIMANY_FINAL_PY"], "verify", "--root", "/tmp/path with spaces"]
    env["SIMANY_CAMPAIGN_PY"] = "/current/interpreter"
    result = subprocess.run(["bash", str(old_entry), "--help"], env=env,
                            text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == ["/current/interpreter", "--help"]


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_preflight_help_does_not_resolve_runtime(flag, tmp_path):
    env = dict(os.environ, SIMANY_CAMPAIGN_PY="/nonexistent/interpreter")
    result = subprocess.run(["bash", str(ROOT / "run/campaign/final_preflight.sh"), flag],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0
    assert "does not run experiments" in result.stdout


def test_preflight_rejects_submit_flag():
    result = subprocess.run(["bash", str(ROOT / "run/campaign/final_preflight.sh"), "--submit"],
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert "No experiment has been launched" in result.stderr

