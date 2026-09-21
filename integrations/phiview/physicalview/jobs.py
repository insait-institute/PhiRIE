"""Job execution for pipeline stages: local subprocess or Slurm, with live logs.

CONTRACT (implemented by the jobs/pipeline agent; app.py and panels code against it):

    mgr = JobManager(config, gpu, on_update=callable(Job))
    job = mgr.submit(JobSpec(...))           -> Job (state QUEUED/RUNNING immediately)
    mgr.submit_chain([JobSpec, ...])         -> list[Job] (sequential, stop on failure)
    mgr.cancel(job.id)                       -> bool
    mgr.list() -> list[Job]; mgr.get(job_id) -> Job
    job.tail(n=200) -> str                   (last n log lines, thread-safe)
    job.wait(timeout=None) -> Job            (blocks until terminal)
    mgr.shutdown()                           (cancels running jobs, joins threads)

JobSpec fields:
    name: str                      human label ("generate:trellis obj_03")
    argv: list[str]                full command; argv[0] is the interpreter path
    env: dict[str, str]            extra environment (SIMANY_SCENE, SIMANY_OUT, ...)
    cwd: Path                      defaults to config.repo_root
    env_key: str                   which interpreter env this is ("main", "sam3", ...)
    needs_gpu: bool
    where: "auto" | "local" | "slurm"
    gpu_type: str | None           Slurm GPU type when where=="slurm"/"auto"-remote
    on_done: callable(Job) | None  invoked once when terminal
    artifacts: list[Path]          paths whose appearance marks success (checked at end)
    tags: dict[str, str]           free-form (scene, object_ids, stage, model);
                                   recognised keys: "chain" (sequential group name),
                                   "slurm_mem" / "slurm_time" (override the GPU
                                   target's --mem / --time for this job only)

Dispatch rule (where == "auto"): run locally if not needs_gpu, or if
gpu.env_compatible(env_key); otherwise submit via `srun` to gpu_type_for(env_key)
(or the spec's gpu_type) using config.gpu_targets[...] (partition, gres, extra,
mem, time) with config.env_exports exported first. Slurm jobs run as a foreground
`srun ... bash -lc '<cmd>'` subprocess so the same log/cancel path works; the Slurm
job id is parsed from srun's stderr ("srun: job N queued and waiting") when present.

Persistence: every job writes outputs/studio/jobs/<job_id>/{spec.json,log.txt,result.json}
(job_id = UTC timestamp + short random). result.json: state, exit_code, started/finished,
duration_s, node, slurm_job_id, artifacts_present.

States: QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED. `Job.state` transitions are
published through on_update from the worker thread; consumers must be thread-safe
(viser GUI handles are).

Concurrency: a bounded worker pool (default 2 local GPU jobs, 4 remote, unlimited CPU
jobs) — see JobManager(max_local_gpu=..., max_remote=...).

Chains: every JobSpec whose tags carry the same "chain" value runs sequentially in
submission order; when one fails (or is cancelled) the remaining queued members are
CANCELLED with an explanatory error. ``submit_chain(specs)`` assigns a fresh chain name
when the specs do not carry one.

This module must stay importable in any interpreter: no viser, no torch.
"""
from __future__ import annotations

import enum
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from physicalview.config import StudioConfig
from physicalview.gpu import GpuInfo, env_compatible, gpu_type_for

_SRUN_JOB_RE = re.compile(r"srun: job (\d+)")
_STUDIO_NODE_RE = re.compile(r"\[studio\] node=(\S+)(?: slurm_job=(\d*))?")
_TERM_GRACE_S = 10.0
_LOG_TICK_S = 1.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def new_job_id() -> str:
    """UTC 'YYYYmmddTHHMMSSZ-<4 hex>'."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(2)


def _slug(name: str, n: int = 48) -> str:
    s = re.sub(r"[^A-Za-z0-9_.:-]+", "-", name).strip("-")
    return (s or "job")[:n]


def tail_file(path: Path | None, n: int = 200, block: int = 65536) -> str:
    """Last ``n`` lines of a (possibly growing) text file without reading it whole."""
    if path is None:
        return ""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            data = b""
            pos = size
            while pos > 0 and data.count(b"\n") <= n:
                step = min(block, pos)
                pos -= step
                fh.seek(pos)
                data = fh.read(step) + data
    except OSError:
        return ""
    lines = data.decode("utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


class JobState(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def terminal(self) -> bool:
        return self in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED)


@dataclass
class JobSpec:
    name: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path | None = None
    env_key: str = "main"
    needs_gpu: bool = False
    where: str = "auto"                 # auto | local | slurm
    gpu_type: str | None = None
    on_done: Callable[["Job"], None] | None = None
    artifacts: list[Path] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "argv": [str(a) for a in self.argv],
                "env": dict(self.env), "cwd": str(self.cwd) if self.cwd else None,
                "env_key": self.env_key, "needs_gpu": self.needs_gpu,
                "where": self.where, "gpu_type": self.gpu_type,
                "artifacts": [str(a) for a in self.artifacts], "tags": dict(self.tags)}


@dataclass
class Job:
    id: str
    spec: JobSpec
    state: JobState = JobState.QUEUED
    exit_code: int | None = None
    started_utc: str | None = None
    finished_utc: str | None = None
    node: str | None = None
    slurm_job_id: str | None = None
    where: str = "local"
    log_path: Path | None = None
    error: str | None = None
    submitted_utc: str = field(default_factory=_utc_now)
    argv_effective: list[str] = field(default_factory=list, repr=False)
    artifacts_present: dict[str, bool] = field(default_factory=dict, repr=False)
    _done: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def dir(self) -> Path | None:
        return self.log_path.parent if self.log_path is not None else None

    @property
    def duration_s(self) -> float | None:
        t0 = _parse_utc(self.started_utc)
        if t0 is None:
            return None
        t1 = _parse_utc(self.finished_utc) if self.finished_utc else time.time()
        return max(0.0, (t1 or time.time()) - t0)

    def tail(self, n: int = 200) -> str:
        return tail_file(self.log_path, n)

    def wait(self, timeout: float | None = None) -> "Job":
        self._done.wait(timeout)
        return self

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.spec.name, "state": self.state.value,
                "exit_code": self.exit_code, "submitted_utc": self.submitted_utc,
                "started_utc": self.started_utc, "finished_utc": self.finished_utc,
                "duration_s": self.duration_s, "node": self.node,
                "slurm_job_id": self.slurm_job_id, "where": self.where,
                "log_path": str(self.log_path) if self.log_path else None,
                "error": self.error, "artifacts_present": dict(self.artifacts_present),
                "argv_effective": list(self.argv_effective), "tags": dict(self.spec.tags)}


class JobManager:
    def __init__(self, config: StudioConfig, gpu: GpuInfo,
                 on_update: Callable[[Job], None] | None = None,
                 max_local_gpu: int = 2, max_remote: int = 4,
                 jobs_root: Path | None = None):
        self.config = config
        self.gpu = gpu
        self.on_update = on_update
        self.jobs_root = Path(jobs_root) if jobs_root is not None else config.studio_out / "jobs"
        self.default_gpu_type: str | None = None   # UI-selected remote GPU (jobs panel)
        self.force_remote: bool = False            # UI toggle: auto -> slurm
        self.srun_exe: str = "srun"
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._procs: dict[str, subprocess.Popen] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._cancel_req: set[str] = set()
        self._chains: dict[str, deque[Job]] = {}
        self._chain_threads: dict[str, threading.Thread] = {}
        self._local_gpu_sem = threading.BoundedSemaphore(max(1, max_local_gpu))
        self._remote_sem = threading.BoundedSemaphore(max(1, max_remote))
        self._shutdown = False

    # ------------------------------------------------------------------ public API
    def submit(self, spec: JobSpec) -> Job:
        if self._shutdown:
            raise RuntimeError("JobManager is shut down")
        argv, where = self._build_argv(spec)  # validates the target up front
        job_id = new_job_id()
        with self._lock:
            while job_id in self._jobs:
                job_id = new_job_id()
            jdir = self.jobs_root / job_id
            jdir.mkdir(parents=True, exist_ok=True)
            job = Job(id=job_id, spec=spec, where=where, log_path=jdir / "log.txt",
                      argv_effective=list(argv))
            self._jobs[job_id] = job
            self._order.append(job_id)
        self._write_json(jdir / "spec.json", {**spec.to_dict(), "id": job_id, "where": where,
                                              "argv_effective": list(argv),
                                              "submitted_utc": job.submitted_utc})
        job.log_path.touch()
        self._notify(job)
        chain = spec.tags.get("chain")
        if chain:
            self._enqueue_chain(chain, job)
        else:
            t = threading.Thread(target=self._run_job, args=(job,), name=f"job-{job_id}", daemon=True)
            with self._lock:
                self._threads[job_id] = t
            t.start()
        return job

    def submit_chain(self, specs: list[JobSpec]) -> list[Job]:
        specs = list(specs)
        if not specs:
            return []
        name = next((s.tags.get("chain") for s in specs if s.tags.get("chain")), None) \
            or f"chain-{new_job_id()}"
        for s in specs:
            s.tags.setdefault("chain", name)
            s.tags["chain"] = s.tags["chain"] or name
        return [self.submit(s) for s in specs]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state.terminal:
                return False
            self._cancel_req.add(job_id)
            proc = self._procs.get(job_id)
            slurm_id = job.slurm_job_id
            queued = job.state == JobState.QUEUED
        if proc is None:
            if queued:
                # worker (or chain runner) sees the request before starting
                self._finish(job, JobState.CANCELLED, error="cancelled before start")
            return True
        self._terminate(proc, slurm_id)
        return True

    def get(self, job_id: str) -> Job:
        with self._lock:
            return self._jobs[job_id]

    def list(self) -> list[Job]:
        with self._lock:
            return [self._jobs[i] for i in self._order]

    def active(self) -> list[Job]:
        return [j for j in self.list() if not j.state.terminal]

    def shutdown(self) -> None:
        self._shutdown = True
        for job in self.list():
            if not job.state.terminal:
                try:
                    self.cancel(job.id)
                except Exception:  # noqa: BLE001
                    pass
        with self._lock:
            threads = list(self._threads.values()) + list(self._chain_threads.values())
        for t in threads:
            t.join(timeout=_TERM_GRACE_S + 5)

    # ------------------------------------------------------------- dispatch build
    def decide_where(self, spec: JobSpec) -> str:
        if spec.where == "local":
            return "local"
        if spec.where == "slurm":
            return "slurm"
        if spec.where != "auto":
            raise ValueError(f"JobSpec.where must be auto|local|slurm, got {spec.where!r}")
        if not spec.needs_gpu:
            return "local"
        if self.force_remote:
            return "slurm"
        return "local" if env_compatible(self.config, spec.env_key, self.gpu) else "slurm"

    def resolve_gpu_type(self, spec: JobSpec) -> str:
        cfg = self.config
        if spec.gpu_type:
            key = spec.gpu_type
        else:
            key = None
            archs = cfg.env_arch_support.get(spec.env_key)
            pref = self.default_gpu_type
            if pref and pref in cfg.gpu_targets and (
                    archs is None or cfg.gpu_targets[pref].compute_cap in archs):
                key = pref
            key = key or gpu_type_for(cfg, spec.env_key) or cfg.default_remote_gpu
        if key not in cfg.gpu_targets:
            raise ValueError(f"unknown Slurm GPU type {key!r}; known: {sorted(cfg.gpu_targets)}")
        return key

    def _build_argv(self, spec: JobSpec) -> tuple[list[str], str]:
        """(effective argv, where). Never executes anything."""
        where = self.decide_where(spec)
        argv = [str(a) for a in spec.argv]
        if not argv:
            raise ValueError("JobSpec.argv is empty")
        if where == "local":
            return argv, "local"
        cfg = self.config
        target = cfg.gpu_targets[self.resolve_gpu_type(spec)]
        cwd = Path(spec.cwd) if spec.cwd else cfg.repo_root
        exports = {**cfg.env_exports, **spec.env}
        parts = ['echo "[studio] node=$(hostname) slurm_job=${SLURM_JOB_ID:-}"']
        parts += [f"export {k}={shlex.quote(str(v))}" for k, v in exports.items()]
        parts.append(f"cd {shlex.quote(str(cwd))}")
        parts.append("exec " + " ".join(shlex.quote(a) for a in argv))
        cmd = "; ".join(parts)
        mem = spec.tags.get("slurm_mem") or target.mem
        tlim = spec.tags.get("slurm_time") or target.time
        srun = [self.srun_exe, "-p", target.partition, f"--gres={target.gres}", *target.extra]
        account = (cfg.raw.get("slurm") or {}).get("account")
        if account:
            srun += ["-A", str(account)]
        srun += [f"--mem={mem}", f"--time={tlim}", "--cpus-per-task=8",
                 f"--job-name=studio-{_slug(spec.name)}", "bash", "-lc", cmd]
        return srun, "slurm"

    # ------------------------------------------------------------------- chains
    def _enqueue_chain(self, name: str, job: Job) -> None:
        with self._lock:
            self._chains.setdefault(name, deque()).append(job)
            runner = self._chain_threads.get(name)
            need_start = runner is None or not runner.is_alive()
            if need_start:
                runner = threading.Thread(target=self._run_chain, args=(name,),
                                          name=f"chain-{_slug(name)}", daemon=True)
                self._chain_threads[name] = runner
        if need_start:
            runner.start()

    def _run_chain(self, name: str) -> None:
        stopped_by: Job | None = None
        while True:
            with self._lock:
                q = self._chains.get(name)
                if not q:
                    self._chains.pop(name, None)
                    return
                job = q.popleft()
            if stopped_by is not None:
                self._finish(job, JobState.CANCELLED,
                             error=f"chain {name!r} stopped: {stopped_by.spec.name} "
                                   f"({stopped_by.id}) {stopped_by.state.value}")
                continue
            self._run_job(job)
            if job.state != JobState.SUCCEEDED:
                stopped_by = job

    # ------------------------------------------------------------------- worker
    def _run_job(self, job: Job) -> None:
        spec = job.spec
        sem = None
        if job.where == "slurm":
            sem = self._remote_sem
        elif spec.needs_gpu:
            sem = self._local_gpu_sem
        if sem is not None:
            while not sem.acquire(timeout=0.5):
                if job.state.terminal:
                    return
                if job.id in self._cancel_req or self._shutdown:
                    self._finish(job, JobState.CANCELLED, error="cancelled while queued")
                    return
        try:
            if job.state.terminal:
                return
            if job.id in self._cancel_req or self._shutdown:
                self._finish(job, JobState.CANCELLED, error="cancelled while queued")
                return
            self._execute(job)
        except Exception as exc:  # noqa: BLE001 - never kill the worker silently
            self._append_log(job, f"\n[studio] internal error: {traceback.format_exc()}")
            self._finish(job, JobState.FAILED, error=f"{type(exc).__name__}: {exc}")
        finally:
            if sem is not None:
                sem.release()

    def _execute(self, job: Job) -> None:
        spec = job.spec
        argv = job.argv_effective or self._build_argv(spec)[0]
        cwd = Path(spec.cwd) if spec.cwd else self.config.repo_root
        env = dict(os.environ)
        if job.where == "local":
            env.update({k: str(v) for k, v in spec.env.items()})
        env.setdefault("PYTHONUNBUFFERED", "1")
        with open(job.log_path, "a", buffering=1, encoding="utf-8", errors="replace") as log:
            log.write(f"[studio] job {job.id} {spec.name}\n[studio] where={job.where} "
                      f"cwd={cwd}\n[studio] argv: {' '.join(shlex.quote(a) for a in argv)}\n")
            if spec.env:
                log.write("[studio] env: " + " ".join(f"{k}={v}" for k, v in spec.env.items()) + "\n")
            if job.state.terminal:
                return
            try:
                proc = subprocess.Popen(argv, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            except OSError as exc:
                log.write(f"[studio] failed to start: {exc}\n")
                self._finish(job, JobState.FAILED, error=f"failed to start: {exc}")
                return
            with self._lock:
                self._procs[job.id] = proc
                job.state = JobState.RUNNING
                job.started_utc = _utc_now()
                job.node = socket.gethostname() if job.where == "local" else None
            self._notify(job)
            if job.id in self._cancel_req:
                self._terminate(proc, None)

            reader = threading.Thread(target=self._pump, args=(job, proc, log),
                                      name=f"log-{job.id}", daemon=True)
            reader.start()
            last_size, last_tick = 0, time.time()
            while proc.poll() is None:
                time.sleep(0.2)
                now = time.time()
                if now - last_tick >= _LOG_TICK_S:
                    last_tick = now
                    try:
                        size = job.log_path.stat().st_size
                    except OSError:
                        size = last_size
                    if size != last_size:
                        last_size = size
                        self._notify(job)
            reader.join(timeout=5.0)
            code = proc.returncode
            log.write(f"[studio] exit code {code}\n")
        with self._lock:
            self._procs.pop(job.id, None)
            cancelled = job.id in self._cancel_req
        job.exit_code = code
        if cancelled:
            self._finish(job, JobState.CANCELLED, error="cancelled")
            return
        if code != 0:
            self._finish(job, JobState.FAILED, error=f"exit code {code}")
            return
        present = {str(p): Path(p).exists() for p in spec.artifacts}
        job.artifacts_present = present
        missing = [p for p, ok in present.items() if not ok]
        if missing:
            self._append_log(job, "[studio] missing artifacts: " + ", ".join(missing) + "\n")
            self._finish(job, JobState.FAILED,
                         error="exit 0 but missing artifacts: " + ", ".join(missing))
            return
        self._finish(job, JobState.SUCCEEDED)

    def _pump(self, job: Job, proc: subprocess.Popen, log) -> None:
        """Copy child output to the log line by line; parse srun / node markers."""
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace")
            log.write(line)
            if job.where == "slurm":
                m = _SRUN_JOB_RE.search(line)
                if m and job.slurm_job_id is None:
                    with self._lock:
                        job.slurm_job_id = m.group(1)
                m2 = _STUDIO_NODE_RE.search(line)
                if m2:
                    with self._lock:
                        job.node = m2.group(1)
                        if m2.group(2):
                            job.slurm_job_id = m2.group(2)
        try:
            proc.stdout.close()
        except OSError:
            pass

    def _terminate(self, proc: subprocess.Popen, slurm_job_id: str | None) -> None:
        def _kill():
            if slurm_job_id and shutil.which("scancel"):
                try:
                    subprocess.run(["scancel", slurm_job_id], timeout=30,
                                   capture_output=True)
                except (subprocess.SubprocessError, OSError):
                    pass
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                return
            t0 = time.time()
            while proc.poll() is None and time.time() - t0 < _TERM_GRACE_S:
                time.sleep(0.1)
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        threading.Thread(target=_kill, name="job-kill", daemon=True).start()

    # ------------------------------------------------------------------ helpers
    def _finish(self, job: Job, state: JobState, error: str | None = None) -> None:
        with self._lock:
            if job.state.terminal:
                return
            job.state = state
            job.finished_utc = _utc_now()
            if job.started_utc is None:
                job.started_utc = job.finished_utc
            if error:
                job.error = error
            self._cancel_req.discard(job.id)
            self._procs.pop(job.id, None)
        if job.dir is not None:
            self._write_json(job.dir / "result.json", job.to_dict())
        job._done.set()
        self._notify(job)
        if job.spec.on_done is not None:
            try:
                job.spec.on_done(job)
            except Exception:  # noqa: BLE001
                self._append_log(job, f"[studio] on_done failed:\n{traceback.format_exc()}")

    def _notify(self, job: Job) -> None:
        if self.on_update is None:
            return
        try:
            self.on_update(job)
        except Exception:  # noqa: BLE001 - UI failures must not kill the worker
            traceback.print_exc()

    def _append_log(self, job: Job, text: str) -> None:
        if job.log_path is None:
            return
        try:
            with open(job.log_path, "a", encoding="utf-8") as fh:
                fh.write(text if text.endswith("\n") else text + "\n")
        except OSError:
            pass

    @staticmethod
    def _write_json(path: Path, obj: dict) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(obj, indent=1, default=str))
            os.replace(tmp, path)
        except OSError:
            pass
