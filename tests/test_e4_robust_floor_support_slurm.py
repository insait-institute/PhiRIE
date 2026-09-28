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
SUBMITTER = ROOT / "run/icra2027/submit_e4_robust_floor_support_noarray.sh"
WORKER = ROOT / "run/slurm/icra2027_e4_robust_floor_support_cpu.sbatch"
RUNNER = ROOT / "run/icra2027/e4_robust_floor_support_diagnostic.py"
CONTROL_SWEEP_ID = (
    "icra2027-contract-v1-e4-46bab70bc69c-collision-diagnostic-"
    "20260904T181737Z"
)


@pytest.mark.parametrize("script", (SUBMITTER, WORKER, RUNNER))
def test_robust_diagnostic_scripts_are_executable(script: Path) -> None:
    assert script.stat().st_mode & stat.S_IXUSR


@pytest.mark.parametrize("script", (SUBMITTER, WORKER))
def test_robust_diagnostic_shell_is_valid_bash(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], cwd=ROOT, check=True)


@pytest.mark.parametrize("script", (SUBMITTER, WORKER))
def test_robust_diagnostic_fleet_is_nonarray_cpu_only(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    assert not re.search(r"(?:^|\s)--array(?:=|\s)", source)
    assert "--gpus" not in source
    assert "--gres" not in source
    assert "--no-requeue" in source or "Requeue=0" in source
    for variable in (
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_ARRAY_TASK_COUNT",
        "SBATCH_ARRAY_INX",
        "SBATCH_ARRAY",
    ):
        assert variable in source


def test_submitter_declares_exact_17_job_dag() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")
    assert "[[ $SEQUENCE -eq 17 ]]" in source
    assert "submitted 17 ordinary CPU robust diagnostic jobs" in source
    assert "common_build_jobs=4" in source
    assert "policy_eval_jobs=8" in source
    assert "pair_compare_jobs=4" in source
    assert 'dependency="afterok:${BUILD_JOBS[$variant]}"' in source
    assert (
        'dependency="afterany:${EVAL_A0_JOBS[$variant]}:'
        '${EVAL_A4_JOBS[$variant]}"'
    ) in source
    assert "aggregate_dependency=$(IFS=:; printf 'afterany:%s'" in source
    assert source.count("CONTROL_SWEEP_ID=") == 1


def test_worker_binds_exact_profiles_and_clean_cpu_provenance() -> None:
    source = WORKER.read_text(encoding="utf-8")
    for phase, profile in {
        "build-common": ("16", "98304", "96G", "04:00:00"),
        "eval-policy": ("4", "65536", "64G", "01:00:00"),
        "compare-pair": ("2", "16384", "16G", "00:30:00"),
        "aggregate": ("2", "16384", "16G", "00:30:00"),
    }.items():
        block = source.split(f"  {phase})", 1)[1].split("    ;;", 1)[0]
        cpus, mem_mb, mem_label, time_limit = profile
        assert f"expected_cpus={cpus}" in block
        assert f"expected_mem_mb={mem_mb}" in block
        assert f"expected_mem_label={mem_label}" in block
        assert f"expected_time={time_limit}" in block
    assert "Requeue=0" in source
    assert "diagnostic job unexpectedly owns GRES resources" in source
    assert "status --porcelain --untracked-files=normal" in source
    assert "E4_CODE_COMMIT" in source
    assert "E4_ROBUST_SWEEP_ID" in source


def test_fake_sbatch_records_exact_17_job_dependency_graph(tmp_path: Path) -> None:
    code_root = tmp_path / "code"
    evidence_root = tmp_path / "evidence"
    fake_bin = tmp_path / "bin"
    expected_e3 = (
        evidence_root
        / "outputs/icra2027"
        / "icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic"
    )
    control_aggregate = (
        evidence_root / "outputs/icra2027" / CONTROL_SWEEP_ID / "aggregate"
    )
    for directory in (code_root, fake_bin, expected_e3, control_aggregate):
        directory.mkdir(parents=True, exist_ok=True)

    fake_python = fake_bin / "python"
    fake_worker = fake_bin / "worker"
    fake_runner = fake_bin / "runner.py"
    fake_git = fake_bin / "git"
    fake_sacctmgr = fake_bin / "sacctmgr"
    fake_sbatch = fake_bin / "sbatch"
    counter = tmp_path / "counter"
    invocations = tmp_path / "sbatch-invocations"
    counter.write_text("0\n", encoding="utf-8")
    invocations.write_text("", encoding="utf-8")
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_worker.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_runner.write_text("# fake runner\n", encoding="utf-8")
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
printf 'phirie|%s|batch|normal|\n' "$TEST_SUBMIT_USER"
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
        fake_worker,
        fake_git,
        fake_sacctmgr,
        fake_sbatch,
    ):
        executable.chmod(0o700)

    rendered = SUBMITTER.read_text(encoding="utf-8")
    replacements = {
        "CODE_ROOT=${SIMANY_ROOT:-$PWD}/worktrees/e4-paired-pilot": (
            f"CODE_ROOT={shlex.quote(str(code_root))}"
        ),
        "EVIDENCE_ROOT=${SIMANY_ROOT:-$PWD}": (
            f"EVIDENCE_ROOT={shlex.quote(str(evidence_root))}"
        ),
        "PYTHON=${SIMANY_ROOT:-$PWD}/.venv/bin/python": (
            f"PYTHON={shlex.quote(str(fake_python))}"
        ),
        'LAUNCHER="$CODE_ROOT/run/slurm/'
        'icra2027_e4_robust_floor_support_cpu.sbatch"': (
            f"LAUNCHER={shlex.quote(str(fake_worker))}"
        ),
        'SWEEP_SCRIPT="$CODE_ROOT/run/icra2027/'
        'e4_robust_floor_support_diagnostic.py"': (
            f"SWEEP_SCRIPT={shlex.quote(str(fake_runner))}"
        ),
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
        "E4_ROBUST_SWEEP_ID": "robust-test",
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
    assert "submitted 17 ordinary CPU robust diagnostic jobs" in result.stdout
    assert counter.read_text(encoding="utf-8").strip() == "17"

    submission = (
        evidence_root
        / "outputs/icra2027/submissions/robust-test-robust-floor-support-diagnostic"
    )
    with (submission / "jobs.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 17
    assert [row["stage"] for row in rows[:4]] == ["build-common"] * 4
    assert [row["stage"] for row in rows[4:12]] == ["eval-policy"] * 8
    assert [row["stage"] for row in rows[12:16]] == ["compare-pair"] * 4
    assert rows[16]["stage"] == "aggregate"
    assert rows[16]["job_id"] == "10017"
    assert all(row["gres"] == "none" for row in rows)
    assert all(row["dependency"] == "-" for row in rows[:4])

    build_by_variant = {row["variant_id"]: row for row in rows[:4]}
    eval_by_cell = {
        (row["variant_id"], row["policy_id"]): row for row in rows[4:12]
    }
    compare_by_variant = {row["variant_id"]: row for row in rows[12:16]}
    assert tuple(build_by_variant) == ("3db-f", "3db-fs", "d755-f", "d755-fs")
    assert len(eval_by_cell) == 8
    for variant, build in build_by_variant.items():
        for policy in ("A0", "A4"):
            evaluation = eval_by_cell[(variant, policy)]
            assert evaluation["dependency"] == f"afterok:{build['job_id']}"
            assert evaluation["kill_on_invalid_dep"] == "yes"
        comparison = compare_by_variant[variant]
        assert comparison["dependency"] == (
            f"afterany:{eval_by_cell[(variant, 'A0')]['job_id']}:"
            f"{eval_by_cell[(variant, 'A4')]['job_id']}"
        )
    aggregate_dependencies = rows[16]["dependency"].split(":")
    assert aggregate_dependencies[0] == "afterany"
    assert aggregate_dependencies[1:] == [
        compare_by_variant[variant]["job_id"] for variant in build_by_variant
    ]
    assert len(invocations.read_text(encoding="utf-8").splitlines()) == 17

    for private in (
        submission,
        submission / "logs",
        submission / "sbatch_receipts",
        submission / "runtime",
        evidence_root / "outputs/icra2027/robust-test",
        evidence_root / "outputs/icra2027/robust-test/variants/3db-f",
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
    assert "refusing to reuse robust diagnostic path" in repeated.stderr
    assert counter.read_text(encoding="utf-8").strip() == "17"
