"""CPU-only tests for physicalview/jobs.py (JobManager) using fake commands.

Nothing here needs viser, torch, a GPU or Slurm: the Slurm path is tested by
building the srun argv without running it (JobManager._build_argv)."""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from physicalview.config import load_config
from physicalview.gpu import GpuInfo
from physicalview.jobs import JobManager, JobSpec, JobState, new_job_id, tail_file

PY = sys.executable
A6000 = GpuInfo(present=True, name="NVIDIA RTX A6000", compute_cap="8.6", memory_mib=49140)
BLACKWELL = GpuInfo(present=True, name="NVIDIA RTX PRO 6000", compute_cap="12.0", memory_mib=97887)
NOGPU = GpuInfo(present=False)


@pytest.fixture
def cfg(tmp_path):
    base = load_config()
    return replace(base, repo_root=tmp_path, interpreters={k: Path(PY) for k in base.interpreters},
                   studio_out=tmp_path / "studio")


@pytest.fixture
def mgr(cfg, tmp_path):
    events = []
    m = JobManager(cfg, A6000, on_update=lambda j: events.append((j.id, j.state.value)),
                   jobs_root=tmp_path / "jobs")
    m.events = events
    yield m
    m.shutdown()


def _py(code: str, **kw) -> JobSpec:
    return JobSpec(name=kw.pop("name", "fake"), argv=[PY, "-c", code], **kw)


def _wait(job, timeout=30.0):
    job.wait(timeout)
    assert job.state.terminal, f"job did not finish: {job.state}"
    return job


# ------------------------------------------------------------------ basics
def test_job_id_format():
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{4}", new_job_id())


def test_success_and_persistence(mgr, tmp_path):
    job = mgr.submit(_py("print('hello studio'); print('line two')", name="ok job",
                         tags={"stage": "test"}))
    _wait(job)
    assert job.state == JobState.SUCCEEDED
    assert job.exit_code == 0
    assert job.where == "local"
    assert job.node
    assert "hello studio" in job.tail()
    jdir = tmp_path / "jobs" / job.id
    assert (jdir / "spec.json").exists() and (jdir / "log.txt").exists() and (jdir / "result.json").exists()
    spec = json.loads((jdir / "spec.json").read_text())
    assert spec["name"] == "ok job" and spec["argv"][0] == PY and spec["tags"]["stage"] == "test"
    res = json.loads((jdir / "result.json").read_text())
    assert res["state"] == "SUCCEEDED" and res["exit_code"] == 0
    assert res["started_utc"] and res["finished_utc"] and res["duration_s"] is not None
    states = [s for jid, s in mgr.events if jid == job.id]
    assert states[0] == "QUEUED" and "RUNNING" in states and states[-1] == "SUCCEEDED"
    assert mgr.get(job.id) is job and job in mgr.list()


def test_failure_exit_code(mgr):
    job = mgr.submit(_py("import sys; print('boom'); sys.exit(3)"))
    _wait(job)
    assert job.state == JobState.FAILED
    assert job.exit_code == 3
    assert "exit code 3" in job.error
    assert "boom" in job.tail()


def test_spec_env_reaches_child(mgr):
    job = mgr.submit(_py("import os; print('VAL=' + os.environ['SIMANY_TEST_VAR'])",
                         env={"SIMANY_TEST_VAR": "forty two"}))
    _wait(job)
    assert job.state == JobState.SUCCEEDED
    assert "VAL=forty two" in job.tail()


def test_artifact_missing_fails_even_with_exit_zero(mgr, tmp_path):
    missing = tmp_path / "never_written.ply"
    job = mgr.submit(_py("print('done')", artifacts=[missing]))
    _wait(job)
    assert job.state == JobState.FAILED
    assert job.exit_code == 0
    assert "missing artifacts" in job.error and str(missing) in job.error
    assert job.artifacts_present[str(missing)] is False


def test_artifact_present_succeeds(mgr, tmp_path):
    art = tmp_path / "out.txt"
    job = mgr.submit(_py(f"open({str(art)!r}, 'w').write('x')", artifacts=[art]))
    _wait(job)
    assert job.state == JobState.SUCCEEDED
    assert job.artifacts_present[str(art)] is True


def test_on_done_callback(mgr):
    seen = []
    job = mgr.submit(_py("pass", on_done=lambda j: seen.append(j.state.value)))
    _wait(job)
    time.sleep(0.1)
    assert seen == ["SUCCEEDED"]


# ------------------------------------------------------------------ cancel
def test_cancel_running_job(mgr):
    job = mgr.submit(_py("import time\nprint('sleeping', flush=True)\ntime.sleep(60)"))
    t0 = time.time()
    while job.state == JobState.QUEUED and time.time() - t0 < 10:
        time.sleep(0.05)
    assert job.state == JobState.RUNNING
    assert mgr.cancel(job.id) is True
    _wait(job, timeout=20)
    assert job.state == JobState.CANCELLED
    assert job.error == "cancelled"
    assert mgr.cancel(job.id) is False  # already terminal


def test_cancel_queued_job_behind_gpu_slot(cfg, tmp_path):
    m = JobManager(cfg, A6000, max_local_gpu=1, jobs_root=tmp_path / "jobs")
    try:
        first = m.submit(_py("import time; time.sleep(60)", needs_gpu=True))
        second = m.submit(_py("print('never')", needs_gpu=True))
        t0 = time.time()
        while first.state == JobState.QUEUED and time.time() - t0 < 10:
            time.sleep(0.05)
        assert second.state == JobState.QUEUED
        assert m.cancel(second.id) is True
        _wait(second, timeout=5)
        assert second.state == JobState.CANCELLED
        assert m.cancel(first.id)
        _wait(first, timeout=20)
        assert first.state == JobState.CANCELLED
        assert "never" not in second.tail()
    finally:
        m.shutdown()


# ------------------------------------------------------------------ chains
def test_chain_runs_sequentially_and_stops_on_failure(mgr, tmp_path):
    marker = tmp_path / "order.txt"
    specs = [
        _py(f"import time; time.sleep(0.5); open({str(marker)!r}, 'a').write('A\\n')", name="A"),
        _py(f"import sys; open({str(marker)!r}, 'a').write('B\\n'); sys.exit(2)", name="B"),
        _py(f"open({str(marker)!r}, 'a').write('C\\n')", name="C"),
    ]
    jobs = mgr.submit_chain(specs)
    assert len(jobs) == 3
    chain = jobs[0].spec.tags["chain"]
    assert all(j.spec.tags["chain"] == chain for j in jobs)
    for j in jobs:
        _wait(j)
    assert [j.state for j in jobs] == [JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED]
    assert "chain" in jobs[2].error and "B" in jobs[2].error
    assert marker.read_text().splitlines() == ["A", "B"]
    # sequential: B started after A finished
    assert jobs[1].started_utc >= jobs[0].finished_utc


def test_chain_by_tag_on_plain_submit(mgr, tmp_path):
    marker = tmp_path / "order2.txt"
    a = mgr.submit(_py(f"import time; time.sleep(0.4); open({str(marker)!r}, 'a').write('a')",
                       tags={"chain": "mychain"}))
    b = mgr.submit(_py(f"open({str(marker)!r}, 'a').write('b')", tags={"chain": "mychain"}))
    _wait(a)
    _wait(b)
    assert a.state == b.state == JobState.SUCCEEDED
    assert marker.read_text() == "ab"


# ------------------------------------------------------------------ log tail
def test_tail_returns_last_lines(mgr):
    job = mgr.submit(_py("for i in range(300): print(f'line {i:03d}')"))
    _wait(job)
    tail = job.tail(5).splitlines()
    assert tail[-1].startswith("[studio] exit code 0")
    assert tail[-2] == "line 299" and tail[-6 + 1] == "line 296"
    assert len(job.tail(1000).splitlines()) >= 300


def test_tail_file_small_block(tmp_path):
    p = tmp_path / "log.txt"
    p.write_text("\n".join(f"row{i}" for i in range(50)) + "\n")
    assert tail_file(p, 3, block=7).splitlines() == ["row47", "row48", "row49"]
    assert tail_file(tmp_path / "missing.txt") == ""


# ---------------------------------------------------------------- dispatch
def test_dispatch_local_for_cpu_and_compatible_gpu(cfg, tmp_path):
    m = JobManager(cfg, A6000, jobs_root=tmp_path / "jobs")
    argv, where = m._build_argv(JobSpec(name="cpu", argv=[PY, "-c", "pass"], needs_gpu=False))
    assert where == "local" and argv[0] == PY
    argv, where = m._build_argv(JobSpec(name="gpu", argv=[PY, "-c", "pass"], env_key="main", needs_gpu=True))
    assert where == "local"


def test_dispatch_slurm_when_env_incompatible(cfg, tmp_path):
    m = JobManager(cfg, BLACKWELL, jobs_root=tmp_path / "jobs")
    spec = JobSpec(name="generate:trellis obj 3", argv=[PY, "-m", "models.s4_trellis", "--objects", "3"],
                   env={"SIMANY_SCENE": "c50d2d1d42", "SIMANY_OUT": "/tmp/o u t"},
                   env_key="main", needs_gpu=True)
    argv, where = m._build_argv(spec)
    assert where == "slurm"
    assert argv[0] == "srun"
    tgt = cfg.gpu_targets["a6000"]  # main env supports 8.6; default remote is a6000
    assert argv[1:3] == ["-p", tgt.partition]
    assert f"--gres={tgt.gres}" in argv
    assert f"--mem={tgt.mem}" in argv and f"--time={tgt.time}" in argv
    assert "--cpus-per-task=8" in argv
    assert any(a.startswith("--job-name=studio-generate") for a in argv)
    assert argv[-3:-1] == ["bash", "-lc"]
    cmd = argv[-1]
    for k, v in cfg.env_exports.items():
        assert f"export {k}=" in cmd and v in cmd
    assert "export SIMANY_SCENE=c50d2d1d42" in cmd
    assert "export SIMANY_OUT='/tmp/o u t'" in cmd          # shlex-quoted
    assert f"cd {cfg.repo_root}" in cmd
    assert cmd.rstrip().endswith(f"exec {PY} -m models.s4_trellis --objects 3")
    # the sam3 env supports 12.0 -> stays local on Blackwell
    _, where2 = m._build_argv(JobSpec(name="sam3", argv=[PY], env_key="sam3", needs_gpu=True))
    assert where2 == "local"


def test_dispatch_forced_slurm_uses_spec_gpu_type_and_extra(cfg, tmp_path):
    m = JobManager(cfg, A6000, jobs_root=tmp_path / "jobs")
    spec = JobSpec(name="policy", argv=["bash", "run/pi05_serve.sh"], env_key="openpi",
                   needs_gpu=True, where="slurm", gpu_type="h200", tags={"slurm_mem": "100G"})
    argv, where = m._build_argv(spec)
    h200 = cfg.gpu_targets["h200"]
    assert where == "slurm" and argv[1:3] == ["-p", h200.partition]
    assert f"--gres={h200.gres}" in argv
    for extra in h200.extra:
        assert extra in argv
    assert "--mem=100G" in argv                              # tag override
    assert f"--time={h200.time}" in argv


def test_dispatch_no_gpu_node_goes_remote_and_default_gpu_pref(cfg, tmp_path):
    m = JobManager(cfg, NOGPU, jobs_root=tmp_path / "jobs")
    spec = JobSpec(name="g", argv=[PY], env_key="gsplat", needs_gpu=True)
    _, where = m._build_argv(spec)
    assert where == "slurm"
    assert m.resolve_gpu_type(spec) == cfg.default_remote_gpu
    m.default_gpu_type = "h200"
    assert m.resolve_gpu_type(spec) == "h200"
    m.default_gpu_type = "rtx6000"            # gsplat env lacks 12.0 -> ignored
    assert m.resolve_gpu_type(spec) == cfg.default_remote_gpu
    m.force_remote = True
    _, where = m._build_argv(JobSpec(name="cpu", argv=[PY], needs_gpu=False))
    assert where == "local"                   # force_remote only affects GPU-less? no: CPU jobs stay local
    _, where = m._build_argv(JobSpec(name="gpu", argv=[PY], env_key="main", needs_gpu=True))
    assert where == "slurm"


def test_force_remote_with_compatible_gpu(cfg, tmp_path):
    m = JobManager(cfg, A6000, jobs_root=tmp_path / "jobs")
    spec = JobSpec(name="gpu", argv=[PY], env_key="main", needs_gpu=True)
    assert m.decide_where(spec) == "local"
    m.force_remote = True
    assert m.decide_where(spec) == "slurm"
    assert m.decide_where(JobSpec(name="cpu", argv=[PY], needs_gpu=False)) == "local"


def test_bad_where_and_unknown_gpu_type(cfg, tmp_path):
    m = JobManager(cfg, A6000, jobs_root=tmp_path / "jobs")
    with pytest.raises(ValueError):
        m._build_argv(JobSpec(name="x", argv=[PY], where="cloud"))
    with pytest.raises(ValueError):
        m._build_argv(JobSpec(name="x", argv=[PY], where="slurm", gpu_type="tpu"))
