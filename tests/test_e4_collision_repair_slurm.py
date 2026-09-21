from __future__ import annotations

import csv
import os
import re
import shlex
import stat
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "run/icra2027/submit_e4_collision_repair_fleet_noarray.sh"
WORKER = ROOT / "run/slurm/icra2027_e4_collision_repair_cpu.sbatch"
PILOT = ROOT / "run/icra2027/e4_collision_repair_pilot.py"


@pytest.mark.parametrize("script", (SUBMITTER, WORKER, PILOT))
def test_repair_fleet_scripts_are_executable(script: Path) -> None:
    assert script.stat().st_mode & stat.S_IXUSR


@pytest.mark.parametrize("script", (SUBMITTER, WORKER))
def test_repair_shell_is_valid_bash(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], cwd=ROOT, check=True)


@pytest.mark.parametrize("script", (SUBMITTER, WORKER))
def test_repair_fleet_has_no_array_or_gpu_request(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    assert not re.search(r"(?:^|\s)--array(?:=|\s)", source)
    assert "--gpus" not in source
    assert "--gres" not in source
    for variable in (
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_ARRAY_TASK_COUNT",
        "SBATCH_ARRAY_INX",
        "SBATCH_ARRAY",
    ):
        assert variable in source


def test_submitter_declares_exact_29_plus_21_job_dag() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")
    assert "PILOT_SCENES=(3db0a1c8f3 d755b3d9d8)" in source
    assert "CANDIDATE_SCENES=(3db0a1c8f3 27dd4da69e d755b3d9d8 acd95847c5 40aec5fffa)" in source
    assert "[[ $SEQUENCE -eq 50 ]]" in source
    assert "submitted 50 ordinary CPU jobs at once" in source
    assert "afterok:$REPAIR_AGGREGATE_JOB,afterany:$qualifier_ids" in source
    assert "paired_materializations=\"afterok:" in source
    assert "REPAIR_MATERIALIZE_JOBS[$scene/$mode/A0]" in source
    assert "REPAIR_MATERIALIZE_JOBS[$scene/$mode/A4]" in source
    assert 'dependency="afterok:$REPAIR_AGGREGATE_JOB:' in source


def test_worker_binds_exact_profiles_and_cpu_only_provenance() -> None:
    source = WORKER.read_text(encoding="utf-8")
    for phase, profile in {
        "materialize": ("8", "65536", "64G", "01:00:00"),
        "export": ("16", "98304", "96G", "04:00:00"),
        "validate": ("4", "32768", "32G", "01:00:00"),
        "compare": ("2", "16384", "16G", "00:30:00"),
        "aggregate": ("2", "16384", "16G", "00:30:00"),
    }.items():
        block = source.split(f"  {phase})", 1)[1].split("    ;;", 1)[0]
        cpus, mem_mb, mem_label, time_limit = profile
        assert f"expected_cpus={cpus}" in block
        assert f"expected_mem_mb={mem_mb}" in block
        assert f"expected_mem_label={mem_label}" in block
        assert f"expected_time={time_limit}" in block
    assert "Requeue=0" in source
    assert "repair job unexpectedly owns GRES resources" in source
    assert "status --porcelain --untracked-files=normal" in source
    assert "E4_CODE_COMMIT" in source
    assert '"${E4_REPAIR_ID}-${mode}"' in source


def test_fake_sbatch_records_exact_50_jobs_and_direct_gate_dependencies(tmp_path: Path) -> None:
    code_root = tmp_path / "code"
    evidence_root = tmp_path / "evidence"
    fake_bin = tmp_path / "bin"
    expected_e3 = (
        evidence_root
        / "outputs/icra2027"
        / "icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic"
    )
    menagerie = code_root / "menagerie"
    for directory in (code_root, fake_bin, expected_e3, menagerie):
        directory.mkdir(parents=True, exist_ok=True)

    fake_python = fake_bin / "python"
    fake_repair_worker = fake_bin / "repair-worker"
    fake_candidate_worker = fake_bin / "candidate-worker"
    fake_pilot = fake_bin / "pilot.py"
    fake_git = fake_bin / "git"
    fake_sacctmgr = fake_bin / "sacctmgr"
    fake_sbatch = fake_bin / "sbatch"
    counter = tmp_path / "counter"
    invocations = tmp_path / "sbatch-invocations"
    counter.write_text("0\n", encoding="utf-8")
    invocations.write_text("", encoding="utf-8")
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_pilot.write_text("# fake pilot\n", encoding="utf-8")
    fake_repair_worker.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_candidate_worker.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_git.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case " $* " in
  *" rev-parse HEAD "*) printf '%s\n' "$TEST_CODE_COMMIT" ;;
  *" rev-parse --show-toplevel "*) printf '%s\n' "$TEST_CODE_ROOT" ;;
  *" status --porcelain "*|*" status --short "*) ;;
  *) printf 'unexpected git: %s\n' "$*" >&2; exit 90 ;;
esac
""",
        encoding="utf-8",
    )
    fake_sacctmgr.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf 'runyi_yang|%s|batch|normal|\n' "$TEST_SUBMIT_USER"
""",
        encoding="utf-8",
    )
    fake_sbatch.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
count=$(<"$TEST_COUNTER")
count=$((count + 1))
printf '%s\n' "$count" > "$TEST_COUNTER"
printf '%q ' "$@" >> "$TEST_INVOCATIONS"
printf '\n' >> "$TEST_INVOCATIONS"
printf '%s\n' "$((10000 + count))"
""",
        encoding="utf-8",
    )
    for executable in (
        fake_python,
        fake_repair_worker,
        fake_candidate_worker,
        fake_git,
        fake_sacctmgr,
        fake_sbatch,
    ):
        executable.chmod(0o700)

    rendered = SUBMITTER.read_text(encoding="utf-8")
    replacements = {
        "CODE_ROOT=/group/worldcept/PhiRIE/code/SimAny-wt/e4-paired-pilot": f"CODE_ROOT={shlex.quote(str(code_root))}",
        "EVIDENCE_ROOT=/group/worldcept/PhiRIE/code/SimAny": f"EVIDENCE_ROOT={shlex.quote(str(evidence_root))}",
        "PYTHON=/group/worldcept/PhiRIE/code/SimAny/.venv/bin/python": f"PYTHON={shlex.quote(str(fake_python))}",
        'REPAIR_LAUNCHER="$CODE_ROOT/run/slurm/icra2027_e4_collision_repair_cpu.sbatch"': f"REPAIR_LAUNCHER={shlex.quote(str(fake_repair_worker))}",
        'CANDIDATE_LAUNCHER="$CODE_ROOT/run/slurm/icra2027_e4_candidate_cpu.sbatch"': f"CANDIDATE_LAUNCHER={shlex.quote(str(fake_candidate_worker))}",
        'PILOT_SCRIPT="$CODE_ROOT/run/icra2027/e4_collision_repair_pilot.py"': f"PILOT_SCRIPT={shlex.quote(str(fake_pilot))}",
        'MENAGERIE_ROOT="$CODE_ROOT/third_party/mujoco_menagerie"': f"MENAGERIE_ROOT={shlex.quote(str(menagerie))}",
    }
    for old, new in replacements.items():
        assert rendered.count(old) == 1
        rendered = rendered.replace(old, new)
    rendered_submitter = tmp_path / "submit.sh"
    rendered_submitter.write_text(rendered, encoding="utf-8")
    rendered_submitter.chmod(0o700)

    submit_user = subprocess.check_output(["id", "-un"], text=True).strip()
    environment = {
        **os.environ,
        "E4_REPAIR_ID": "repair-fleet-test",
        "E4_SCREEN_ID": "candidate-screen-test",
        "E4_TEST_TOOL_OVERRIDES": "1",
        "SBATCH_BIN": str(fake_sbatch),
        "SACCTMGR_BIN": str(fake_sacctmgr),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TEST_CODE_COMMIT": "a" * 40,
        "TEST_CODE_ROOT": str(code_root),
        "TEST_COUNTER": str(counter),
        "TEST_INVOCATIONS": str(invocations),
        "TEST_SUBMIT_USER": submit_user,
    }
    result = subprocess.run(
        ["bash", str(rendered_submitter)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "submitted 50 ordinary CPU jobs at once" in result.stdout
    assert counter.read_text(encoding="utf-8").strip() == "50"

    submission = (
        evidence_root
        / "outputs/icra2027/submissions/repair-fleet-test-collision-repair-fleet"
    )
    with (submission / "jobs.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 50
    assert [row["pipeline"] for row in rows[:29]] == ["repair"] * 29
    assert [row["pipeline"] for row in rows[29:]] == ["candidate"] * 21
    assert rows[28]["stage"] == "aggregate"
    repair_gate_job = rows[28]["job_id"]
    assert repair_gate_job == "10029"
    assert all(
        f"afterok:{repair_gate_job}" in row["dependency"] for row in rows[29:]
    )
    assert rows[-1]["dependency"].startswith(
        f"afterok:{repair_gate_job},afterany:"
    )
    assert sum(row["stage"] == "materialize" for row in rows[:29]) == 8
    assert sum(row["stage"] == "export" for row in rows[:29]) == 8
    assert sum(row["stage"] == "validate" for row in rows[:29]) == 8
    assert sum(row["stage"] == "compare" for row in rows[:29]) == 4
    assert all(row["gres"] == "none" for row in rows)
    assert len(invocations.read_text(encoding="utf-8").splitlines()) == 50

    for private in (
        submission,
        submission / "logs",
        submission / "sbatch_receipts",
        submission / "runtime",
        evidence_root
        / "outputs/icra2027/submissions/candidate-screen-test-candidate-screen/runtime",
    ):
        assert private.is_dir() and not private.is_symlink()
        assert stat.S_IMODE(private.stat().st_mode) == 0o700
        acl = subprocess.check_output(["getfacl", "-cp", str(private)], text=True)
        assert "default:" not in acl

    repeated = subprocess.run(
        ["bash", str(rendered_submitter)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert repeated.returncode == 2
    assert "refusing to reuse fleet path" in repeated.stderr
    assert counter.read_text(encoding="utf-8").strip() == "50"

