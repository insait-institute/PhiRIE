from __future__ import annotations

import os
import re
import shlex
import stat
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "run/icra2027/submit_e4_candidate_screen_noarray.sh"
LAUNCHER = ROOT / "run/slurm/icra2027_e4_candidate_cpu.sbatch"
SCENES = (
    "3db0a1c8f3",
    "27dd4da69e",
    "d755b3d9d8",
    "acd95847c5",
    "40aec5fffa",
)


@pytest.fixture(scope="module")
def submitter_text() -> str:
    return SUBMITTER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def launcher_text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


@pytest.mark.parametrize("script", (SUBMITTER, LAUNCHER))
def test_candidate_shell_is_executable_and_valid_bash(script: Path) -> None:
    assert script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(script)], cwd=ROOT, check=True)


@pytest.mark.parametrize("script", (SUBMITTER, LAUNCHER))
def test_candidate_fleet_has_no_slurm_array_or_gpu_request(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    assert not re.search(r"(?:^|\s)--array(?:=|\s)", source)
    assert "--gpus" not in source
    assert "--gres" not in source
    for variable in (
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_ARRAY_TASK_COUNT",
        "SLURM_ARRAY_TASK_MIN",
        "SLURM_ARRAY_TASK_MAX",
        "SLURM_ARRAY_TASK_STEP",
        "SBATCH_ARRAY_INX",
        "SBATCH_ARRAY",
    ):
        assert variable in source


def test_submitter_enqueues_exact_21_job_dag(submitter_text: str) -> None:
    roster = re.search(r"^SCENES=\(([^)]*)\)$", submitter_text, re.MULTILINE)
    assert roster is not None
    assert tuple(roster.group(1).split()) == SCENES
    assert "for scene_id in \"${SCENES[@]}\"; do" in submitter_text
    assert submitter_text.count('for index in "${!SCENES[@]}"; do') == 2
    assert submitter_text.count("submit_job materialize") == 2
    assert submitter_text.count("submit_job prepare") == 1
    assert submitter_text.count("submit_job qualify") == 1
    assert submitter_text.count("submit_job aggregate") == 1
    assert "afterok:${MATERIALIZE_A0_JOBS[$index]}:${MATERIALIZE_A4_JOBS[$index]}" in submitter_text
    assert 'dependency="afterok:${PREPARE_JOBS[$index]}"' in submitter_text
    assert 'submit_job aggregate - - "afterany:$qualifier_ids"' in submitter_text
    assert "[[ $SEQUENCE -eq 21 ]]" in submitter_text
    assert "submitted 21 ordinary CPU jobs at once" in submitter_text


def test_submitter_uses_exact_cpu_profiles(submitter_text: str) -> None:
    assert "ACCOUNT=${SLURM_ACCOUNT:-phirie}" in submitter_text
    assert "CPU_NODELIST='sof1-h200-[0-7]'" in submitter_text
    assert "--partition=batch --qos=normal --nodelist=\"$CPU_NODELIST\"" in submitter_text
    assert "--parsable --no-requeue" in submitter_text
    assert re.search(
        r"submit_job materialize .* 8 64G 01:00:00", submitter_text
    )
    assert re.search(r"submit_job prepare .* 16 96G 04:00:00", submitter_text)
    assert re.search(r"submit_job qualify .* 4 32G 01:00:00", submitter_text)
    assert re.search(r"submit_job aggregate .* 4 32G 00:30:00", submitter_text)
    assert 'local export_spec="E4_SCREEN_ID=' in submitter_text
    assert 'local export_spec="ALL,' not in submitter_text
    assert "E4_E3_ROOT=$EXPECTED_E3_ROOT" in submitter_text


def test_submitter_persists_receipts_and_atomic_partial_ledger(
    submitter_text: str,
) -> None:
    assert "sbatch_receipts" in submitter_text
    assert '> "$stdout_receipt" 2> "$stderr_receipt"' in submitter_text
    assert 'chmod 0444 "$stdout_receipt" "$stderr_receipt"' in submitter_text
    assert 'sync -f "$stdout_receipt"' in submitter_text
    assert 'sync -f "$stderr_receipt"' in submitter_text
    assert 'cp -- "$LEDGER" "$LEDGER_TEMP"' in submitter_text
    assert 'mv -T -- "$LEDGER_TEMP" "$LEDGER"' in submitter_text
    assert "jobs.tsv.incomplete" in submitter_text
    assert "candidate submission stopped after $SEQUENCE recorded ordinary jobs" in submitter_text
    assert "refusing to reuse candidate-screen output" in submitter_text
    assert "refusing to reuse candidate submission directory" in submitter_text
    assert 'RUNTIME_DIR="$SUBMISSION_DIR/runtime"' in submitter_text
    assert 'mkdir "$LOG_DIR" "$RECEIPT_DIR" "$RUNTIME_DIR"' in submitter_text
    assert '"$SETFACL" -k -- "$durable_directory"' in submitter_text
    assert 'chmod 00700 "$durable_directory"' in submitter_text
    assert '"$durable_acl" != *"default:"*' in submitter_text
    assert 'EXPERIMENT_PARENT_DIRS=(' in submitter_text
    for relative in (
        "construction_variants/A0",
        "construction_variants/A4",
        "candidate_suites",
        "task_freezes",
        "scene_prepares",
        "scene_qualifiers",
    ):
        assert f'"$EXPERIMENT_ROOT/{relative}"' in submitter_text
    assert 'mkdir "$experiment_directory"' in submitter_text
    assert '"$SETFACL" -k -- "$experiment_directory"' in submitter_text
    assert 'chmod 00700 "$experiment_directory"' in submitter_text
    assert '"$experiment_acl" != *"default:"*' in submitter_text
    assert 'sync -f "$RUNTIME_DIR"' in submitter_text


def test_fake_sbatch_can_start_every_worker_after_durable_runtime_parent(
    tmp_path: Path, submitter_text: str,
) -> None:
    """Exercise the race boundary: fake sbatch starts each worker immediately."""
    code_root = tmp_path / "code"
    evidence_root = tmp_path / "evidence"
    fake_bin = tmp_path / "bin"
    menagerie_root = code_root / "menagerie"
    expected_e3 = (
        evidence_root
        / "outputs/icra2027"
        / "icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic"
    )
    for directory in (code_root, fake_bin, menagerie_root, expected_e3):
        directory.mkdir(parents=True, exist_ok=True)
    acl_inheritance_root = evidence_root / "outputs/icra2027"
    subprocess.run(
        ["setfacl", "-m", "d:g::rwx", str(acl_inheritance_root)], check=True
    )
    assert "default:group::rwx" in subprocess.check_output(
        ["getfacl", "-cp", str(acl_inheritance_root)], text=True
    )

    fake_python = fake_bin / "python"
    fake_worker = fake_bin / "worker"
    fake_git = fake_bin / "git"
    fake_sacctmgr = fake_bin / "sacctmgr"
    fake_sbatch = fake_bin / "sbatch"
    counter = tmp_path / "sbatch-count"
    counter.write_text("0\n", encoding="utf-8")

    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_worker.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
parent="$TEST_EVIDENCE_ROOT/outputs/icra2027/submissions/${E4_SCREEN_ID}-candidate-screen/runtime"
[[ -d "$parent" && ! -L "$parent" ]]
[[ "$(stat -c '%a' "$parent")" == 700 ]]
[[ "$(getfacl -cp "$parent")" != *"default:"* ]]
experiment="$TEST_EVIDENCE_ROOT/outputs/icra2027/$E4_SCREEN_ID"
for shared_parent in \
  "$experiment/construction_variants/A0" \
  "$experiment/construction_variants/A4" \
  "$experiment/candidate_suites" \
  "$experiment/task_freezes" \
  "$experiment/scene_prepares" \
  "$experiment/scene_qualifiers"
do
  [[ -d "$shared_parent" && ! -L "$shared_parent" ]]
  [[ "$(stat -c '%a' "$shared_parent")" == 700 ]]
  [[ "$(getfacl -cp "$shared_parent")" != *"default:"* ]]
done
runtime_root="$parent/$SLURM_JOB_ID"
mkdir "$runtime_root"
setfacl -k -- "$runtime_root"
chmod 00700 "$runtime_root"
[[ -d "$runtime_root" && ! -L "$runtime_root" ]]
[[ "$(stat -c '%a' "$runtime_root")" == 700 ]]
[[ "$(getfacl -cp "$runtime_root")" != *"default:"* ]]
mkdir "$runtime_root/tmp" "$runtime_root/xdg-cache"
for runtime_directory in "$runtime_root/tmp" "$runtime_root/xdg-cache"; do
  setfacl -k -- "$runtime_directory"
  chmod 00700 "$runtime_directory"
  [[ -d "$runtime_directory" && ! -L "$runtime_directory" ]]
  [[ "$(stat -c '%a' "$runtime_directory")" == 700 ]]
  [[ "$(getfacl -cp "$runtime_directory")" != *"default:"* ]]
done
""",
        encoding="utf-8",
    )
    fake_git.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case " $* " in
  *" rev-parse HEAD "*) printf '%s\n' "$TEST_CODE_COMMIT" ;;
  *" rev-parse --show-toplevel "*) printf '%s\n' "$TEST_CODE_ROOT" ;;
  *" status --porcelain "*|*" status --short "*) ;;
  *) printf 'unexpected fake git argv: %s\n' "$*" >&2; exit 90 ;;
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
count=$(<"$TEST_SBATCH_COUNTER")
count=$((count + 1))
printf '%s\n' "$count" > "$TEST_SBATCH_COUNTER"
job_id=$((9000 + count))
parent="$TEST_EVIDENCE_ROOT/outputs/icra2027/submissions/${E4_SCREEN_ID}-candidate-screen/runtime"
[[ -d "$parent" && ! -L "$parent" ]]
[[ "$(stat -c '%a' "$parent")" == 700 ]]
[[ "$(getfacl -cp "$parent")" != *"default:"* ]]
SLURM_JOB_ID="$job_id" "$TEST_WORKER"
printf '%s\n' "$job_id"
""",
        encoding="utf-8",
    )
    for executable in (
        fake_python, fake_worker, fake_git, fake_sacctmgr, fake_sbatch
    ):
        executable.chmod(0o700)

    rendered = submitter_text
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
        'LAUNCHER="$CODE_ROOT/run/slurm/icra2027_e4_candidate_cpu.sbatch"': (
            f"LAUNCHER={shlex.quote(str(fake_worker))}"
        ),
        'MENAGERIE_ROOT="$CODE_ROOT/third_party/mujoco_menagerie"': (
            f"MENAGERIE_ROOT={shlex.quote(str(menagerie_root))}"
        ),
    }
    for original, replacement in replacements.items():
        assert rendered.count(original) == 1
        rendered = rendered.replace(original, replacement)
    rendered_submitter = tmp_path / "submit.sh"
    rendered_submitter.write_text(rendered, encoding="utf-8")
    rendered_submitter.chmod(0o700)

    submit_user = subprocess.check_output(["id", "-un"], text=True).strip()
    screen_id = "runtime-parent-regression"
    environment = {
        **os.environ,
        "E4_SCREEN_ID": screen_id,
        "E4_TEST_TOOL_OVERRIDES": "1",
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SBATCH_BIN": str(fake_sbatch),
        "SACCTMGR_BIN": str(fake_sacctmgr),
        "TEST_CODE_COMMIT": "a" * 40,
        "TEST_CODE_ROOT": str(code_root),
        "TEST_EVIDENCE_ROOT": str(evidence_root),
        "TEST_SBATCH_COUNTER": str(counter),
        "TEST_SUBMIT_USER": submit_user,
        "TEST_WORKER": str(fake_worker),
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
    assert "submitted 21 ordinary CPU jobs at once" in result.stdout
    assert counter.read_text(encoding="utf-8").strip() == "21"

    submission = (
        evidence_root / "outputs/icra2027/submissions"
        / f"{screen_id}-candidate-screen"
    )
    runtime = submission / "runtime"
    assert runtime.is_dir() and not runtime.is_symlink()
    assert stat.S_IMODE(runtime.stat().st_mode) == 0o700
    private_directories = [
        submission,
        submission / "logs",
        submission / "sbatch_receipts",
        runtime,
    ]
    experiment = evidence_root / "outputs/icra2027" / screen_id
    experiment_parents = [
        experiment,
        experiment / "construction_variants",
        experiment / "construction_variants/A0",
        experiment / "construction_variants/A4",
        experiment / "candidate_suites",
        experiment / "task_freezes",
        experiment / "scene_prepares",
        experiment / "scene_qualifiers",
    ]
    assert all(
        path.is_dir()
        and not path.is_symlink()
        and stat.S_IMODE(path.stat().st_mode) == 0o700
        for path in experiment_parents
    )
    private_directories.extend(experiment_parents)
    assert all(
        "default:" not in subprocess.check_output(
            ["getfacl", "-cp", str(path)], text=True
        )
        for path in private_directories
    )
    assert [path.name for path in sorted(runtime.iterdir())] == [
        str(job_id) for job_id in range(9001, 9022)
    ]
    assert all(
        path.is_dir()
        and not path.is_symlink()
        and stat.S_IMODE(path.stat().st_mode) == 0o700
        for path in runtime.iterdir()
    )
    for job_runtime in runtime.iterdir():
        for child_name in ("tmp", "xdg-cache"):
            child = job_runtime / child_name
            assert child.is_dir() and not child.is_symlink()
            assert stat.S_IMODE(child.stat().st_mode) == 0o700
            assert "default:" not in subprocess.check_output(
                ["getfacl", "-cp", str(child)], text=True
            )
    assert len((submission / "jobs.tsv").read_text(encoding="utf-8").splitlines()) == 22

    # Precreating the shared output parents must not weaken the immutable
    # screen-ID boundary: a second invocation refuses before another sbatch.
    repeated = subprocess.run(
        ["bash", str(rendered_submitter)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert repeated.returncode == 2
    assert "refusing to reuse candidate-screen output" in repeated.stderr
    assert counter.read_text(encoding="utf-8").strip() == "21"


def test_launcher_binds_exact_roots_commit_import_and_cpu_allocation(
    launcher_text: str,
) -> None:
    assert "CODE_ROOT=${SIMANY_ROOT:-$PWD}/worktrees/e4-paired-pilot" in launcher_text
    assert "EVIDENCE_ROOT=${SIMANY_ROOT:-$PWD}" in launcher_text
    assert '[[ "$CODE_ROOT" != "$EVIDENCE_ROOT" ]]' in launcher_text
    assert '"$observed_commit" == "$E4_CODE_COMMIT"' in launcher_text
    assert "status --porcelain --untracked-files=normal" in launcher_text
    assert "Path(candidate.__file__).resolve(strict=True)" in launcher_text
    assert "candidate.CODE_ROOT != code_root" in launcher_text
    assert "candidate.EXPECTED_EVIDENCE_ROOT != evidence_root" in launcher_text
    assert "candidate.EXPECTED_MENAGERIE_ROOT != menagerie_root" in launcher_text
    assert "candidate.EXPECTED_MENAGERIE_COMMIT != menagerie_commit" in launcher_text
    assert 'export PYTHONPATH="$CODE_ROOT"' in launcher_text
    assert "export PYTHONNOUSERSITE=1" in launcher_text
    assert '"${SLURM_JOB_ACCOUNT:-}" == "$ACCOUNT"' in launcher_text
    assert '"${SLURM_JOB_PARTITION:-}" == batch' in launcher_text
    assert '"${SLURM_JOB_QOS:-}" == normal' in launcher_text
    assert "Requeue=0" in launcher_text
    assert "MinMemoryNode=" in launcher_text
    assert "Command=$CODE_ROOT/run/slurm/icra2027_e4_candidate_cpu.sbatch" in launcher_text
    assert "candidate job unexpectedly owns GRES resources" in launcher_text
    runtime_block = launcher_text.split('runtime_root="$runtime_parent/$SLURM_JOB_ID"', 1)[1]
    assert runtime_block.index('mkdir "$runtime_root"') < runtime_block.index(
        '"$SETFACL" -k -- "$runtime_root"'
    ) < runtime_block.index('chmod 00700 "$runtime_root"') < runtime_block.index(
        'mkdir "$runtime_root/tmp" "$runtime_root/xdg-cache"'
    )
    assert '"$GETFACL" -cp -- "$runtime_root"' in runtime_block
    assert '"$SETFACL" -k -- "$runtime_directory"' in runtime_block


def test_launcher_phase_cli_is_closed_and_matches_module(launcher_text: str) -> None:
    assert "SCENES=(3db0a1c8f3 27dd4da69e d755b3d9d8 acd95847c5 40aec5fffa)" in launcher_text
    assert launcher_text.count("-m robo.eval.e4_candidate_screen") == 4

    materialize = launcher_text.rsplit("  materialize)", 1)[1].split("    ;;", 1)[0]
    assert "e4_candidate_screen materialize" in materialize
    assert '--screen-id "$E4_SCREEN_ID"' in materialize
    assert '--scene-id "$scene_id"' in materialize
    assert '--policy-id "$policy_id"' in materialize
    assert '--e3-root "$E4_E3_ROOT"' in materialize
    assert "--out" not in materialize

    aggregate = launcher_text.rsplit("  aggregate)", 1)[1].split("    ;;", 1)[0]
    assert "e4_candidate_screen aggregate" in aggregate
    assert '--screen-id "$E4_SCREEN_ID"' in aggregate
    assert "--scene-id" not in aggregate
    assert "--policy-id" not in aggregate
    assert '--expected-code-commit "$E4_CODE_COMMIT"' in aggregate


def test_launcher_phase_resource_contracts_are_exact(launcher_text: str) -> None:
    expected = {
        "materialize": ("8", "65536", "64G", "01:00:00"),
        "prepare-scene": ("16", "98304", "96G", "04:00:00"),
        "qualify-scene": ("4", "32768", "32G", "01:00:00"),
        "aggregate": ("4", "32768", "32G", "00:30:00"),
    }
    phase_gate = launcher_text.split("# Bind every phase", 1)[1].split(
        ': "${SLURM_JOB_ID', 1
    )[0]
    for index, (phase, values) in enumerate(expected.items()):
        start = phase_gate.index(f"  {phase})")
        next_starts = [
            phase_gate.find(f"  {other})", start + 1)
            for other in expected
            if phase_gate.find(f"  {other})", start + 1) >= 0
        ]
        end = min(next_starts) if next_starts else len(phase_gate)
        block = phase_gate[start:end]
        cpus, mem_mb, mem_label, time_limit = values
        assert f"expected_cpus={cpus}" in block
        assert f"expected_mem_mb={mem_mb}" in block
        assert f"expected_mem_label={mem_label}" in block
        assert f"expected_time={time_limit}" in block
