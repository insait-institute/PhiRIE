from __future__ import annotations

import re
import subprocess
import os
from pathlib import Path

import pytest
import yaml

from robo.eval import agentic_ablation


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run/slurm/icra2027_e3_agentic.sbatch"
SUBMITTER = ROOT / "run/icra2027/submit_e3_noarray.sh"
MONITOR = ROOT / "run/icra2027/monitor_e3_noarray.sh"
ROSTER = ROOT / "configs/experiments/icra2027/construction_regimes.yaml"
JOBS = ROOT / "configs/experiments/icra2027/agentic_jobs.yaml"
POLICIES = ROOT / "configs/experiments/icra2027/agentic_policies.yaml"
MVPY = Path(os.environ.get("SIMANY_GSPLAT_PY", "/nonexistent/mini-viewer/bin/python"))


@pytest.fixture(scope="module")
def launcher_text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def submitter_text() -> str:
    return SUBMITTER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def monitor_text() -> str:
    return MONITOR.read_text(encoding="utf-8")


@pytest.mark.parametrize("script", [LAUNCHER, SUBMITTER, MONITOR])
def test_e3_shell_scripts_have_valid_bash_syntax(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], cwd=ROOT, check=True)


@pytest.mark.parametrize("script", [LAUNCHER, SUBMITTER])
def test_e3_launch_path_forbids_slurm_arrays(script: Path) -> None:
    text = script.read_text(encoding="utf-8")
    assert not re.search(r"(?:^|\s)--array(?:=|\s)", text)
    for variable in (
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_ARRAY_TASK_COUNT",
        "SLURM_ARRAY_TASK_MIN",
        "SLURM_ARRAY_TASK_MAX",
        "SLURM_ARRAY_TASK_STEP",
    ):
        assert variable in text
    assert '[[ -v "$ARRAY_VARIABLE" ]]' in text
    if script == SUBMITTER:
        assert "SBATCH_ARRAY_INX" in text


def test_launcher_exposes_gt_isolated_phases(launcher_text: str) -> None:
    for flag in (
        "--inventory",
        "--smoke",
        "--observe",
        "--control",
        "--evaluate",
        "--aggregate",
    ):
        assert flag in launcher_text
    for scene_phase in ("observe", "control", "evaluate"):
        assert re.search(
            rf"{scene_phase}\) COMMAND\+=\(\s*--{scene_phase} "
            rf'--scene-id "\$E3_SCENE_ID"\s*\)',
            launcher_text,
        )
    assert "inventory|control|evaluate" in launcher_text
    assert "smoke|observe" in launcher_text
    assert (
        'if [[ "$E3_PHASE" == observe || "$E3_PHASE" == control '
        '|| "$E3_PHASE" == evaluate ]]'
    ) in launcher_text
    assert "not in the exact frozen 50-scene roster" in launcher_text


def test_launcher_keeps_gpu_and_cpu_environments_separate(
    launcher_text: str,
) -> None:
    assert "MVPY=${SIMANY_GSPLAT_PY:-${SIMANY_ROOT:-$PWD}/.envs/mini-viewer/bin/python}" in launcher_text
    assert "EXEC_PYTHON=\"$MVPY\"" in launcher_text
    assert "EXEC_PYTHON=\"$PYTHON\"" in launcher_text
    assert "gcp-eu1-a100-80g-qrfh" in launcher_text
    assert 'export E3_EXPECTED_GPU_NAME="NVIDIA A100-SXM4-80GB"' in launcher_text
    assert "GPU_ARCH=8.0" in launcher_text
    assert '[[ "$SLURMD_NODENAME" == hala ]]' in launcher_text
    assert 'export E3_EXPECTED_GPU_NAME="NVIDIA RTX A6000"' in launcher_text
    assert "GPU_ARCH=8.6" in launcher_text
    assert 'export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"' in launcher_text
    assert 'export CUMM_CUDA_ARCH_LIST="$GPU_ARCH"' in launcher_text
    assert 'expected_name = os.environ["E3_EXPECTED_GPU_NAME"]' in launcher_text
    assert (
        'expected_capability = os.environ["E3_EXPECTED_GPU_CAPABILITY"]'
        in launcher_text
    )
    assert "sof1-h200-[0-7]" in launcher_text
    assert "E3 CPU phase=$E3_PHASE rejects leaked E3_GPU_PROFILE" in launcher_text
    assert "E3 CPU profile must not request a GPU" in launcher_text
    assert "E3 CPU profile must not expose CUDA devices" in launcher_text
    assert "smoke|observe|control) export SIMANY_NO_GT=1" in launcher_text


@pytest.mark.parametrize(
    ("phase", "cpus"),
    [("inventory", "8"), ("control", "8"), ("evaluate", "8"), ("aggregate", "16")],
)
def test_launcher_rejects_gpu_profile_leak_into_every_cpu_phase(
    phase: str,
    cpus: str,
) -> None:
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "SLURM_JOB_ID": "1",
            "SLURMD_NODENAME": "sof1-h200-0",
            "SLURM_JOB_PARTITION": "batch",
            "SLURM_JOB_QOS": "normal",
            "SLURM_CPUS_PER_TASK": cpus,
            "SLURM_MEM_PER_NODE": "65536",
            "E3_PHASE": phase,
            "E3_GPU_PROFILE": "hala-a6000",
            "E3_FREEZE_ID": "profile-leak-test",
            "E3_CODE_COMMIT": "a" * 40,
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert f"E3 CPU phase={phase} rejects leaked E3_GPU_PROFILE" in result.stderr


@pytest.mark.parametrize(
    ("phase", "scene_id", "value", "expected_error"),
    [
        (
            "inventory",
            None,
            "1",
            "E3_REQUIRE_NONEMPTY_PROPOSALS is valid only for phase=control",
        ),
        (
            "control",
            "5748ce6f01",
            "0",
            "E3_REQUIRE_NONEMPTY_PROPOSALS must equal 1 when set",
        ),
        (
            "control",
            "09a6c4d13e",
            "1",
            "E3_REQUIRE_NONEMPTY_PROPOSALS is valid only for pilot scene=5748ce6f01",
        ),
    ],
)
def test_launcher_restricts_nonempty_proposal_postcondition_to_pilot_control(
    phase: str,
    scene_id: str | None,
    value: str,
    expected_error: str,
) -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "SLURM_JOB_ID": "1",
        "SLURMD_NODENAME": "sof1-h200-0",
        "SLURM_JOB_PARTITION": "batch",
        "SLURM_JOB_QOS": "normal",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_MEM_PER_NODE": "65536",
        "E3_PHASE": phase,
        "E3_REQUIRE_NONEMPTY_PROPOSALS": value,
        "E3_FREEZE_ID": "pilot-postcondition-test",
        "E3_CODE_COMMIT": "a" * 40,
    }
    if scene_id is not None:
        env["E3_SCENE_ID"] = scene_id
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert expected_error in result.stderr


def test_launcher_fail_closes_on_empty_sealed_pilot_shard(
    launcher_text: str,
) -> None:
    command = launcher_text.index('"${COMMAND[@]}"')
    postcondition = launcher_text.index(
        "if [[ -v E3_REQUIRE_NONEMPTY_PROPOSALS ]]; then", command
    )
    completion = launcher_text.index('echo "E3_PHASE_COMPLETE=$E3_PHASE"')
    assert command < postcondition < completion
    assert "from robo.eval.agentic_ablation import _load_control_scene" in launcher_text
    assert "_, shard, seal = _load_control_scene(out_dir, scene_id)" in launcher_text
    assert 'members = seal.get("members")' in launcher_text
    assert '"controller_shard.json"' in launcher_text
    assert '"job_ledger.jsonl"' in launcher_text
    assert "E3 pilot control seal has invalid member coverage" in launcher_text
    assert 'job_count = shard.get("job_count")' in launcher_text
    assert 'proposals = shard.get("proposals")' in launcher_text
    assert "type(job_count) is not int or job_count < 1" in launcher_text
    assert "not isinstance(proposals, list) or not proposals" in launcher_text
    assert "E3 pilot controller shard contains no real proposals" in launcher_text


def test_launcher_runtime_and_cache_paths_stay_in_repository(
    launcher_text: str,
) -> None:
    assert "ROOT=${SIMANY_ROOT:-$PWD}" in launcher_text
    assert 'RUNTIME="$ROOT/outputs/icra2027/' in launcher_text
    assert 'export TMPDIR="$RUNTIME/tmp"' in launcher_text
    assert 'export XDG_CACHE_HOME="$RUNTIME/xdg"' in launcher_text
    assert 'export TORCHINDUCTOR_CACHE_DIR="$RUNTIME/torchinductor"' in launcher_text
    assert 'export NUMBA_CACHE_DIR="$RUNTIME/numba"' in launcher_text
    assert 'export TORCH_EXTENSIONS_DIR="$RUNTIME/' in launcher_text
    assert 'export CUDA_CACHE_PATH="$RUNTIME/' in launcher_text
    assert 'export TRITON_CACHE_DIR="$RUNTIME/' in launcher_text
    assert not re.search(r'(?<![A-Za-z0-9_$}])/tmp(?:/|["\'\s])', launcher_text)


def test_submitter_builds_expected_154_job_dag(submitter_text: str) -> None:
    assert 'submit_e3 inventory inventory - 8 01:00:00 "$E0_JOB"' in submitter_text
    assert 'submit_e3 smoke smoke - 8 00:30:00 "$INVENTORY_JOB"' in submitter_text
    assert "PILOT_SCENE=5748ce6f01" in submitter_text
    assert re.search(
        r'submit_e3 "observe-pilot-\$PILOT_SCENE" observe "\$PILOT_SCENE" 8 '
        r'02:00:00 "\$SMOKE_JOB"',
        submitter_text,
    )
    assert re.search(
        r'submit_e3 "control-pilot-\$PILOT_SCENE" control "\$PILOT_SCENE" 8 '
        r'04:00:00\s+\\\s+"\$PILOT_OBSERVE_JOB" 1',
        submitter_text,
    )
    assert re.search(
        r'submit_e3 "evaluate-pilot-\$PILOT_SCENE" evaluate "\$PILOT_SCENE" 8 '
        r'02:00:00 "\$PILOT_CONTROL_JOB"',
        submitter_text,
    )
    assert re.search(
        r'submit_e3 "observe-\$SCENE_ID" observe "\$SCENE_ID" 8 '
        r'02:00:00 "\$PILOT_EVALUATE_JOB"',
        submitter_text,
    )
    assert re.search(
        r'submit_e3 "control-\$SCENE_ID" control "\$SCENE_ID" 8 '
        r'04:00:00 "\$OBSERVE_JOB"',
        submitter_text,
    )
    assert re.search(
        r'submit_e3 "evaluate-\$SCENE_ID" evaluate "\$SCENE_ID" 8 '
        r'02:00:00 "\$CONTROL_JOB"',
        submitter_text,
    )
    assert 'submit_e3 aggregate aggregate - 16 01:00:00 "$EVALUATE_DEPENDENCY"' in submitter_text
    assert 'OBSERVE_JOBS=("$PILOT_OBSERVE_JOB")' in submitter_text
    assert 'CONTROL_JOBS=("$PILOT_CONTROL_JOB")' in submitter_text
    assert 'EVALUATE_JOBS=("$PILOT_EVALUATE_JOB")' in submitter_text
    assert 'if [[ "$SCENE_ID" == "$PILOT_SCENE" ]]; then\n    continue' in submitter_text
    assert "[[ $SEQUENCE -eq 154 ]]" in submitter_text
    assert "154 ordinary jobs (51 %s GPU, 103 CPU)" in submitter_text
    assert '--dependency="afterok:$dependency"' in submitter_text
    assert "--kill-on-invalid-dep=yes" in submitter_text
    assert 'local require_nonempty_proposals="${7:-0}"' in submitter_text
    assert (
        submitter_text.count(
            'export_spec+=",E3_REQUIRE_NONEMPTY_PROPOSALS=1"'
        )
        == 1
    )
    assert "E3_GPU_PROFILE E3_REQUIRE_NONEMPTY_PROPOSALS" in submitter_text


def test_submitter_supports_exact_dual_gpu_profiles(submitter_text: str) -> None:
    assert "CPU_NODELIST=${E3_CPU_NODELIST:-sof1-h200-[0-7]}" in submitter_text
    assert "GPU_PROFILE=${E3_GPU_PROFILE:-gcp-a100}" in submitter_text
    assert "gcp-a100)" in submitter_text
    assert "GPU_NODE=gcp-eu1-a100-80g-qrfh" in submitter_text
    assert "GPU_GRES=a100-80g:1" in submitter_text
    assert 'GPU_RESOURCE_ARGS=(--nodelist="$GPU_NODE" --gpus="$GPU_GRES")' in submitter_text
    assert "hala-a6000)" in submitter_text
    assert "GPU_NODE=hala" in submitter_text
    assert "GPU_GRES=gpu:a6000:1" in submitter_text
    assert 'GPU_RESOURCE_ARGS=(--nodelist="$GPU_NODE" --gres="$GPU_GRES")' in submitter_text
    assert 'if [[ "$phase" == smoke || "$phase" == observe ]]' in submitter_text
    assert 'resource_args=("${GPU_RESOURCE_ARGS[@]}")' in submitter_text
    assert 'resource_args=(--nodelist="$CPU_NODELIST")' in submitter_text
    assert 'export_spec+=",E3_GPU_PROFILE=$GPU_PROFILE"' in submitter_text
    assert "E3_E0_JOB_ID E3_E0_LOG E3_GPU_PROFILE" in submitter_text
    capture = submitter_text.index("GPU_PROFILE=${E3_GPU_PROFILE:-gcp-a100}")
    clear_inherited = submitter_text.index(
        "E3_E0_JOB_ID E3_E0_LOG E3_GPU_PROFILE"
    )
    first_submission = submitter_text.index("submit_e0\nE0_JOB=")
    assert capture < clear_inherited < first_submission


def test_submitter_persists_each_ordinary_job_atomically(
    submitter_text: str,
) -> None:
    assert "--parsable --no-requeue" in submitter_text
    assert '[[ "$raw" =~ ^([0-9]+)' in submitter_text
    assert 'job_id="${BASH_REMATCH[1]}"' in submitter_text
    assert 'cp -- "$LEDGER" "$LEDGER_TEMP"' in submitter_text
    assert 'mv -T -- "$LEDGER_TEMP" "$LEDGER"' in submitter_text
    assert "refusing to reuse E3 experiment root" in submitter_text
    assert "refusing to reuse E3 submission directory" in submitter_text
    assert submitter_text.index("validate_job_id \"$raw\"") < submitter_text.index(
        'record_job e0 e0 - "$SUBMITTED_JOB_ID"'
    )
    assert "E3_E0_JOB_ID=$E0_JOB" in submitter_text
    assert "E3_E0_LOG=$E0_LOG" in submitter_text


def test_monitor_is_read_only_and_understands_partial_ledgers(
    monitor_text: str,
) -> None:
    assert '"$SQUEUE_BIN"' in monitor_text
    assert '"$SACCT_BIN"' in monitor_text
    assert "--watch" in monitor_text
    assert "between 1 and 154 jobs" in monitor_text
    assert "non-contiguous sequence" in monitor_text
    assert "squeue query failed" in monitor_text
    assert "sacct query failed" in monitor_text
    assert "scancel" not in monitor_text
    assert not re.search(r"(?:^|\s)--array(?:=|\s)", monitor_text)


def test_frozen_scene_roster_has_exactly_50_safe_ids() -> None:
    config = yaml.safe_load(ROSTER.read_text(encoding="utf-8"))
    population = config["population"]
    scene_ids = [str(value) for value in population["scene_ids"]]
    assert population["planned_scenes"] == 50
    assert len(scene_ids) == len(set(scene_ids)) == 50
    assert all(re.fullmatch(r"[0-9a-f]{10}", scene_id) for scene_id in scene_ids)


def test_rgb_ed_pilot_is_a_frozen_one_object_scene() -> None:
    jobs = yaml.safe_load(
        (ROOT / "configs/experiments/icra2027/agentic_jobs.yaml").read_text(
            encoding="utf-8"
        )
    )
    snapshots = {
        str(row["scene_id"]): row for row in jobs["population"]["scene_snapshots"]
    }
    assert snapshots["5748ce6f01"]["accepted_jobs"] == 1


@pytest.mark.parametrize(
    ("phase", "runner_name", "needs_scene"),
    [
        ("--smoke", "run_smoke", False),
        ("--observe", "run_observe", True),
        ("--control", "run_control", True),
        ("--evaluate", "run_evaluate", True),
    ],
)
def test_preaggregate_phases_do_not_open_inventory_only_jobs_yaml(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    phase: str,
    runner_name: str,
    needs_scene: bool,
) -> None:
    monkeypatch.setenv("SIMANY_NO_GT", "0")
    main_globals = agentic_ablation.main.__globals__
    # This focused routing test stubs the downstream runner and its E0 gate.
    # Real source/freeze/config rejection is covered in test_agentic_evidence_root.
    monkeypatch.setitem(main_globals, "_validate_cli_execution",
                        lambda *a, **kw: {"code": {"commit": "a" * 40}})
    original_yaml_loader = main_globals["_load_yaml_repo"]

    def guarded_yaml_loader(path: str | Path, identity: str):
        assert Path(path).name != JOBS.name, "pre-aggregate phase opened GT-laden jobs YAML"
        return original_yaml_loader(path, identity)

    monkeypatch.setitem(main_globals, "_load_yaml_repo", guarded_yaml_loader)
    monkeypatch.setitem(
        main_globals,
        "_load_json_repo",
        lambda *_args, **_kwargs: {"code": {"commit": "a" * 40}},
    )
    monkeypatch.setitem(
        main_globals,
        "_load_controller_json",
        lambda *_args, **_kwargs: {"code": {"commit": "a" * 40}},
    )
    monkeypatch.setitem(
        main_globals,
        runner_name,
        lambda *_args, **_kwargs: {"phase_isolation_test": phase},
    )
    freeze_id = f"phase-isolation-{phase.removeprefix('--')}"
    argv = [
        "--jobs",
        str(JOBS),
        "--policies",
        str(POLICIES),
        "--contract-manifest",
        str(ROOT / "outputs/icra2027/nonexistent-test-contract.json"),
        "--freeze-id",
        freeze_id,
        "--out",
        str(ROOT / f"outputs/icra2027/{freeze_id}/agentic"),
        phase,
    ]
    if needs_scene:
        argv.extend(["--scene-id", "5748ce6f01"])
    assert agentic_ablation.main(argv) == 0
    assert '"phase"' in capsys.readouterr().out


@pytest.mark.skipif(not MVPY.is_file(), reason="cluster gsplat environment is unavailable")
def test_mvpy_import_does_not_pull_control_runtime() -> None:
    command = """
import json
import sys
import robo.eval.agentic_ablation
forbidden = sorted(
    name for name in sys.modules
    if name == 'pybullet' or name.startswith('open3d')
    or name == 'agents.orchestrator.runtime'
)
print(json.dumps(forbidden))
"""
    result = subprocess.run(
        [str(MVPY), "-c", command],
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "[]"
