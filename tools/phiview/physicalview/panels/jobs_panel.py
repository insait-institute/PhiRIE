"""Jobs tab — see docs/ARCHITECTURE.md for the required controls.

CONTRACT: build(ctx) creates the tab's GUI and wires events:
  * job table (markdown, newest first: id, name, state, where/node, duration)
  * job picker dropdown + live log tail (refreshed every second while the job runs
    by a daemon thread), Cancel button
  * GPU compatibility matrix (gpu.compatibility_matrix) and a "run remotely on"
    dropdown whose value is stored on ctx (``ctx.remote_gpu``) and mirrored into
    ``ctx.jobs.default_gpu_type``; a "force remote" checkbox mirrors into
    ``ctx.jobs.force_remote``.
viser is imported lazily inside build() so this module stays importable in any env.
"""
from __future__ import annotations

import threading
import time
import traceback

from physicalview import gpu as gpu_mod

_NONE = "(none)"
_MAX_ROWS = 40


def _fmt_duration(sec) -> str:
    if sec is None:
        return "-"
    sec = int(sec)
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def _state_icon(state: str) -> str:
    return {"QUEUED": "⏳", "RUNNING": "▶", "SUCCEEDED": "✅", "FAILED": "❌",
            "CANCELLED": "⛔"}.get(state, "")


def jobs_table(jobs, max_rows: int = _MAX_ROWS) -> str:
    """Markdown table, newest first."""
    rows = ["| id | name | state | where / node | duration |", "|---|---|---|---|---|"]
    for j in list(jobs)[::-1][:max_rows]:
        where = j.where + (f" / {j.node}" if j.node else "")
        if j.slurm_job_id:
            where += f" (slurm {j.slurm_job_id})"
        name = j.spec.name.replace("|", "/")
        err = f" — {j.error}" if j.error and j.state.value in ("FAILED", "CANCELLED") else ""
        rows.append(f"| `{j.id}` | {name} | {_state_icon(j.state.value)} {j.state.value}{err} "
                    f"| {where} | {_fmt_duration(j.duration_s)} |")
    if len(rows) == 2:
        rows.append("| — | no jobs yet | | | |")
    return "\n".join(rows)


def matrix_markdown(config, gpu) -> str:
    m = gpu_mod.compatibility_matrix(config, gpu)
    local = (f"{gpu.name} cc {gpu.compute_cap}, {gpu.memory_mib // 1024} GB"
             if gpu.present else "no GPU on this node")
    rows = [f"**Local GPU:** {local}", "", "| env | interpreter | local | remote GPU |",
            "|---|---|---|---|"]
    for key, info in m.items():
        interp = config.interpreters.get(key)
        exists = "" if interp is None or interp.exists() else " (missing)"
        rows.append(f"| `{key}` | `{interp}`{exists} | {'yes' if info['local'] else 'no'} "
                    f"| {info['remote_gpu'] or '-'} |")
    return "\n".join(rows)


def _label(job) -> str:
    return f"{job.id} · {job.spec.name[:40]}"


def build(ctx) -> None:
    import viser  # noqa: F401 - lazy so jobs/pipeline tests never need viser

    gui = ctx.server.gui
    config, gpu = ctx.config, ctx.gpu
    mgr = ctx.jobs

    # --- GPU matrix + remote target ---------------------------------------------
    with gui.add_folder("GPU compatibility", expand_by_default=False):
        gui.add_markdown(matrix_markdown(config, gpu))
    gpu_keys = list(config.gpu_targets) or [config.default_remote_gpu]
    initial = config.default_remote_gpu if config.default_remote_gpu in gpu_keys else gpu_keys[0]
    remote = gui.add_dropdown("Run remotely on", options=gpu_keys, initial_value=initial,
                              hint="Slurm GPU type used when a stage cannot run on this node")
    ctx.remote_gpu = remote.value
    mgr.default_gpu_type = remote.value
    force_remote = gui.add_checkbox("Force remote (Slurm) for GPU stages", False)

    @remote.on_update
    def _(_evt) -> None:
        ctx.remote_gpu = remote.value
        mgr.default_gpu_type = remote.value
        ctx.log(f"remote GPU target: {remote.value}")

    @force_remote.on_update
    def _(_evt) -> None:
        mgr.force_remote = bool(force_remote.value)

    # --- table + picker -------------------------------------------------------------
    table = gui.add_markdown(jobs_table(mgr.list()))
    picker = gui.add_dropdown("Job", options=[_NONE], initial_value=_NONE)
    with gui.add_folder("Log", expand_by_default=True):
        tail_n = gui.add_number("Tail lines", 60, min=10, max=2000, step=10)
        follow = gui.add_checkbox("Follow newest job", True)
        log_md = gui.add_markdown("```\n(no job selected)\n```")
    cancel_btn = gui.add_button("Cancel selected job", color="red")
    refresh_btn = gui.add_button("Refresh")

    lock = threading.Lock()
    labels: dict[str, str] = {}   # label -> job id

    def _refresh_table() -> None:
        jobs = mgr.list()
        try:
            table.content = jobs_table(jobs)
        except Exception:  # noqa: BLE001
            pass
        opts = [_NONE] + [_label(j) for j in jobs[::-1]]
        with lock:
            labels.clear()
            labels.update({_label(j): j.id for j in jobs})
        try:
            cur = picker.value
            picker.options = opts
            if follow.value and jobs:
                picker.value = _label(jobs[-1])
            elif cur in opts:
                picker.value = cur
        except Exception:  # noqa: BLE001
            pass

    def _selected_job():
        with lock:
            jid = labels.get(picker.value)
        if jid is None:
            return None
        try:
            return mgr.get(jid)
        except KeyError:
            return None

    def _refresh_log() -> None:
        job = _selected_job()
        if job is None:
            text = "(no job selected)"
        else:
            head = (f"{job.spec.name} — {job.state.value}" + (f" [{job.error}]" if job.error else "")
                    + (f" on {job.node}" if job.node else "") + f"\n{job.log_path}\n")
            text = head + (job.tail(int(tail_n.value)) or "(log empty)")
        try:
            log_md.content = "```\n" + text.replace("```", "'''")[-20000:] + "\n```"
        except Exception:  # noqa: BLE001
            pass

    @picker.on_update
    def _(_evt) -> None:
        _refresh_log()

    @refresh_btn.on_click
    def _(_evt) -> None:
        _refresh_table()
        _refresh_log()

    @cancel_btn.on_click
    def _(_evt) -> None:
        job = _selected_job()
        if job is None:
            ctx.set_status("no job selected")
            return
        ok = mgr.cancel(job.id)
        ctx.set_status(f"cancel {job.id}: {'requested' if ok else 'not running'}")

    def _on_job_updated(_job) -> None:
        _refresh_table()
        _refresh_log()

    ctx.events.subscribe("job.updated", _on_job_updated)

    def _ticker() -> None:
        while True:
            time.sleep(1.0)
            try:
                job = _selected_job()
                if job is not None and not job.state.terminal:
                    _refresh_log()
                    table.content = jobs_table(mgr.list())
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    threading.Thread(target=_ticker, name="jobs-panel-tick", daemon=True).start()
    _refresh_table()
    _refresh_log()
