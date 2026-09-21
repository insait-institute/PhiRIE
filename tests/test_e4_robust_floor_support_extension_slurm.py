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
SUBMITTER = (
    ROOT / "run/icra2027/submit_e4_robust_floor_support_extension_noarray.sh"
)
WORKER = (
    ROOT / "run/slurm/icra2027_e4_robust_floor_support_extension_cpu.sbatch"
)
RUNNER = ROOT / "run/icra2027/e4_robust_floor_support_extension.py"
EXTERNAL_SWEEP_ID = (
    "icra2027-contract-v1-e4-8c7b796b3f25-robust-floor-support-"
    "20260904T192849Z"
)
EXTERNAL_AGGREGATE_SHA = (
    "7d12d1cf7c212c97da0f97596970c0da983f450a89da7abe17035a98b4a7d5a6"
)
EXTERNAL_COMPARISON_SHA = (
    "99b979ded06854426f2cab4942086401ef6a27f7b0c499ca4f80a1f341c0bc19"
)
FAKE_MATRIX = (
    ("27dd4da69e", "27dd-f", "27dd-fs"),
    ("acd95847c5", "acd-f", "acd-fs"),
    ("1ada7a0617", "1ada-f", "1ada-fs"),
    ("25f3b7a318", "25f3-f", "25f3-fs"),
)


@pytest.mark.parametrize("script", (SUBMITTER, WORKER, RUNNER))
def test_extension_scripts_are_executable(script: Path) -> None:
    assert script.stat().st_mode & stat.S_IXUSR


@pytest.mark.parametrize("script", (SUBMITTER, WORKER))
def test_extension_shell_is_valid_bash(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], cwd=ROOT, check=True)


@pytest.mark.parametrize("script", (SUBMITTER, WORKER))
def test_extension_is_ordinary_cpu_only(script: Path) -> None:
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


def test_submitter_declares_exact_33_job_dag_and_external_pins() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")
    assert "[[ $SEQUENCE -eq 33 ]]" in source
    assert "submitted 33 ordinary CPU robust extension jobs" in source
    assert "common_build_jobs=8" in source
    assert "policy_eval_jobs=16" in source
    assert "pair_compare_jobs=8" in source
    assert 'dependency="afterok:${BUILD_JOBS[$variant]}"' in source
    assert (
        'dependency="afterany:${EVAL_A0_JOBS[$variant]}:'
        '${EVAL_A4_JOBS[$variant]}"'
    ) in source
    assert "aggregate_dependency=$(IFS=:; printf 'afterany:%s'" in source
    assert f"EXTERNAL_SWEEP_ID={EXTERNAL_SWEEP_ID}" in source
    assert f"EXTERNAL_AGGREGATE_MANIFEST_SHA256={EXTERNAL_AGGREGATE_SHA}" in source
    assert f"EXTERNAL_COMPARISON_MANIFEST_SHA256={EXTERNAL_COMPARISON_SHA}" in source


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
    assert "job unexpectedly owns GRES resources" in source
    assert "status --porcelain --untracked-files=normal" in source
    assert "variant_id not in module.VARIANT_IDS" in source


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o700)


def test_fake_sbatch_records_exact_33_job_dependency_graph(tmp_path: Path) -> None:
    code_root = tmp_path / "code"
    evidence_root = tmp_path / "evidence"
    fake_bin = tmp_path / "bin"
    expected_e3 = (
        evidence_root
        / "outputs/icra2027"
        / "icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic"
    )
    external_root = (
        evidence_root / "outputs/icra2027" / EXTERNAL_SWEEP_ID
    )
    for directory in (
        code_root,
        fake_bin,
        expected_e3,
        external_root / "aggregate",
        external_root / "variants/d755-fs/comparison",
        external_root / "variants/d755-fs/common_static",
    ):
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
    matrix_arguments = " ".join(shlex.quote("\t".join(row)) for row in FAKE_MATRIX)
    _write_executable(
        fake_python,
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"printf '%s\\n' {matrix_arguments}\n",
    )
    _write_executable(fake_worker, "#!/usr/bin/env bash\nexit 0\n")
    fake_runner.write_text("# fake runner\n", encoding="utf-8")
    _write_executable(
        fake_git,
        """#!/usr/bin/env bash
set -euo pipefail
case " $* " in
  *" rev-parse HEAD "*) printf '%s\n' "$TEST_CODE_COMMIT" ;;
  *" rev-parse --show-toplevel "*) printf '%s\n' "$TEST_CODE_ROOT" ;;
  *" status --porcelain "*|*" status --short "*) ;;
  *) printf 'unexpected git: %s\n' "$*" >&2; exit 90 ;;
esac
""",
    )
    _write_executable(
        fake_sacctmgr,
        """#!/usr/bin/env bash
set -euo pipefail
printf 'runyi_yang|%s|batch|normal|\n' "$TEST_SUBMIT_USER"
""",
    )
    _write_executable(
        fake_sbatch,
        """#!/usr/bin/env bash
set -euo pipefail
count=$(<"$TEST_COUNTER")
count=$((count + 1))
printf '%s\n' "$count" > "$TEST_COUNTER"
printf '%q ' "$@" >> "$TEST_INVOCATIONS"
printf '\n' >> "$TEST_INVOCATIONS"
printf '%s\n' "$((12000 + count))"
""",
    )

    rendered = SUBMITTER.read_text(encoding="utf-8")
    replacements = {
        "CODE_ROOT=/group/worldcept/PhiRIE/code/SimAny-wt/e4-paired-pilot": (
            f"CODE_ROOT={shlex.quote(str(code_root))}"
        ),
        "EVIDENCE_ROOT=/group/worldcept/PhiRIE/code/SimAny": (
            f"EVIDENCE_ROOT={shlex.quote(str(evidence_root))}"
        ),
        "PYTHON=/group/worldcept/PhiRIE/code/SimAny/.venv/bin/python": (
            f"PYTHON={shlex.quote(str(fake_python))}"
        ),
        'LAUNCHER="$CODE_ROOT/run/slurm/'
        'icra2027_e4_robust_floor_support_extension_cpu.sbatch"': (
            f"LAUNCHER={shlex.quote(str(fake_worker))}"
        ),
        'SWEEP_SCRIPT="$CODE_ROOT/run/icra2027/'
        'e4_robust_floor_support_extension.py"': (
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
        "E4_ROBUST_EXTENSION_ID": "robust-extension-test",
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
    assert "submitted 33 ordinary CPU robust extension jobs" in result.stdout
    assert counter.read_text(encoding="utf-8").strip() == "33"

    submission = (
        evidence_root
        / "outputs/icra2027/submissions/robust-extension-test-robust-floor-support-extension"
    )
    with (submission / "jobs.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 33
    assert [row["stage"] for row in rows[:8]] == ["build-common"] * 8
    assert [row["stage"] for row in rows[8:24]] == ["eval-policy"] * 16
    assert [row["stage"] for row in rows[24:32]] == ["compare-pair"] * 8
    assert rows[32]["stage"] == "aggregate"
    assert rows[32]["job_id"] == "12033"
    assert all(row["gres"] == "none" for row in rows)
    assert all(row["dependency"] == "-" for row in rows[:8])
    assert all(row["external_sweep_id"] == EXTERNAL_SWEEP_ID for row in rows)
    assert all(
        row["external_aggregate_manifest_sha256"] == EXTERNAL_AGGREGATE_SHA
        for row in rows
    )
    assert all(
        row["external_comparison_manifest_sha256"] == EXTERNAL_COMPARISON_SHA
        for row in rows
    )

    variants = [variant for _, floor, support in FAKE_MATRIX for variant in (floor, support)]
    variant_scene = {
        variant: scene
        for scene, floor, support in FAKE_MATRIX
        for variant in (floor, support)
    }
    build_by_variant = {row["variant_id"]: row for row in rows[:8]}
    eval_by_cell = {
        (row["variant_id"], row["policy_id"]): row for row in rows[8:24]
    }
    compare_by_variant = {row["variant_id"]: row for row in rows[24:32]}
    assert list(build_by_variant) == variants
    for variant, build in build_by_variant.items():
        assert build["scene_id"] == variant_scene[variant]
        for policy in ("A0", "A4"):
            evaluation = eval_by_cell[(variant, policy)]
            assert evaluation["dependency"] == f"afterok:{build['job_id']}"
            assert evaluation["kill_on_invalid_dep"] == "yes"
        comparison = compare_by_variant[variant]
        assert comparison["dependency"] == (
            f"afterany:{eval_by_cell[(variant, 'A0')]['job_id']}:"
            f"{eval_by_cell[(variant, 'A4')]['job_id']}"
        )
    aggregate_dependencies = rows[32]["dependency"].split(":")
    assert aggregate_dependencies[0] == "afterany"
    assert aggregate_dependencies[1:] == [
        compare_by_variant[variant]["job_id"] for variant in variants
    ]

    invocation_lines = invocations.read_text(encoding="utf-8").splitlines()
    assert len(invocation_lines) == 33
    assert all("--no-requeue" in line for line in invocation_lines)
    assert all("--array" not in line and "--gres" not in line for line in invocation_lines)
    receipts = list((submission / "sbatch_receipts").iterdir())
    assert len(receipts) == 66
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in receipts)

    for private in (
        submission,
        submission / "logs",
        submission / "sbatch_receipts",
        submission / "runtime",
        evidence_root / "outputs/icra2027/robust-extension-test",
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
    assert "refusing to reuse robust extension path" in repeated.stderr
    assert counter.read_text(encoding="utf-8").strip() == "33"
