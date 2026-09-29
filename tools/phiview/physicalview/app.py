"""SimAny Studio application: viser server, shared Context, event bus, tab layout.

Run (studio env, GPU node):
    source run/env.sh
    $STUDIO_PY -m physicalview.app --port 8080 [--config configs/default.yaml]
Then tunnel:  ssh -L 8080:<node>:8080 <login>   and open http://localhost:8080

Panels register themselves via ``build(ctx)``; all cross-panel communication goes through
``ctx`` and ``ctx.events`` (topics documented in docs/ARCHITECTURE.md).
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from physicalview.config import StudioConfig, load_config
from physicalview.gpu import GpuInfo, detect_gpu

log = logging.getLogger("studio")


class EventBus:
    """Minimal thread-safe publish/subscribe. Handlers run on the publisher's thread and
    exceptions are logged, never propagated (a broken panel must not kill the loop)."""

    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[[Any], None]]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> Callable[[], None]:
        with self._lock:
            self._subs[topic].append(handler)

        def unsubscribe() -> None:
            with self._lock:
                if handler in self._subs[topic]:
                    self._subs[topic].remove(handler)
        return unsubscribe

    def publish(self, topic: str, payload: Any = None) -> None:
        with self._lock:
            handlers = list(self._subs.get(topic, ()))
        for h in handlers:
            try:
                h(payload)
            except Exception:  # noqa: BLE001 - isolate panels from each other
                log.error("event handler for %s failed:\n%s", topic, traceback.format_exc())


@dataclass
class Selection:
    kind: str = "none"                 # none | object | box
    object_ids: list[int] = field(default_factory=list)
    box_center: tuple[float, float, float] | None = None
    box_size: tuple[float, float, float] | None = None
    box_quat_wxyz: tuple[float, float, float, float] | None = None
    camera_frame: str | None = None

    @property
    def object_names(self) -> list[str]:
        return [f"obj_{i:02d}" for i in self.object_ids]


class Context:
    """Shared state handed to every panel."""

    def __init__(self, server, config: StudioConfig, gpu: GpuInfo, jobs) -> None:
        self.server = server
        self.config = config
        self.gpu = gpu
        self.jobs = jobs
        self.events = EventBus()
        self.scene = None            # scene_state.SceneState | None
        self.renderer = None         # render.Renderer | None (GPU)
        self.robot = None            # robot.RobotSession | None
        self.selection = Selection()
        self.display_mode = getattr(config, "display_mode", "server")   # server | client (scene panel keeps it current)
        self.initial_scene: str | None = None   # --scene: result set the Scene tab preselects and loads at startup
        self.stream = None           # streaming.ServerRenderStream (owned by the scene panel)
        self.panel_status: dict[str, str] = {}   # tab -> built | placeholder | failed: <err>
        self._status_handle = None
        self._log_lines: list[str] = []
        self._log_lock = threading.Lock()

    # --- status / logging -------------------------------------------------------
    def attach_status(self, handle) -> None:
        self._status_handle = handle

    def set_status(self, msg: str) -> None:
        self.log(msg)
        if self._status_handle is not None:
            try:
                self._status_handle.content = f"**{time.strftime('%H:%M:%S')}** {msg}"
            except Exception:  # noqa: BLE001
                pass

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        log.info(msg)
        with self._log_lock:
            self._log_lines.append(line)
            del self._log_lines[:-500]

    def recent_log(self, n: int = 50) -> str:
        with self._log_lock:
            return "\n".join(self._log_lines[-n:])

    # --- selection ----------------------------------------------------------------
    def set_selection(self, sel: Selection) -> None:
        self.selection = sel
        self.events.publish("selection.changed", sel)

    def set_scene(self, scene) -> None:
        self.scene = scene
        self.events.publish("scene.loaded", scene)


class StudioApp:
    TABS = (
        ("Scene", "physicalview.panels.scene_panel"),
        ("Generate", "physicalview.panels.generate_panel"),
        ("Inpaint", "physicalview.panels.inpaint_panel"),
        ("Robot", "physicalview.panels.robot_panel"),
        ("Jobs", "physicalview.panels.jobs_panel"),
    )

    def __init__(self, config: StudioConfig, port: int, host: str = "0.0.0.0",
                 share: bool = False, scene: str | None = None) -> None:
        import viser
        from physicalview.jobs import JobManager

        self.config = config
        self.gpu = detect_gpu()
        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.set_up_direction("+z")
        jobs = JobManager(config, self.gpu, on_update=self._on_job_update)
        self.ctx = Context(self.server, config, self.gpu, jobs)
        self.ctx.initial_scene = scene
        self._build_layout()
        if share:
            try:
                self.ctx.log(f"share URL: {self.server.request_share_url()}")
            except Exception as exc:  # noqa: BLE001
                self.ctx.log(f"share url failed: {exc}")

    # --- layout -----------------------------------------------------------------------
    def _build_layout(self) -> None:
        gui = self.server.gui
        gpu_txt = (f"{self.gpu.name} (cc {self.gpu.compute_cap}, {self.gpu.memory_mib // 1024} GB)"
                   if self.gpu.present else "no GPU (client-side splats only)")
        gui.add_markdown(f"### SimAny Studio\n`{gpu_txt}`")
        status = gui.add_markdown("ready")
        self.ctx.attach_status(status)
        tabs = gui.add_tab_group()
        for title, module_path in self.TABS:
            with tabs.add_tab(title):
                try:
                    module = __import__(module_path, fromlist=["build"])
                    module.build(self.ctx)
                    self.ctx.panel_status[title] = "built"
                except NotImplementedError:
                    gui.add_markdown(f"_{title} panel not implemented yet_")
                    self.ctx.panel_status[title] = "placeholder"
                except Exception as exc:  # noqa: BLE001
                    gui.add_markdown(f"**{title} panel failed to build** — see server log")
                    self.ctx.panel_status[title] = f"failed: {type(exc).__name__}: {exc}"
                    log.error("panel %s failed:\n%s", title, traceback.format_exc())

    def _on_job_update(self, job) -> None:
        self.ctx.events.publish("job.updated", job)

    # --- lifecycle ----------------------------------------------------------------------
    def serve_forever(self) -> None:
        self.ctx.set_status("Studio up. Load a scene in the Scene tab.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        try:
            if self.ctx.stream is not None:
                try:
                    self.ctx.stream.stop()
                except Exception:  # noqa: BLE001
                    log.debug("stream stop failed", exc_info=True)
            if self.ctx.robot is not None:
                self.ctx.robot.close()
        finally:
            self.ctx.jobs.shutdown()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--share", action="store_true", help="request a viser share URL")
    ap.add_argument("--scene", default=None,
                    help="result set to preselect and load at startup (e.g. behavior_task-0020)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config = load_config(args.config)
    app = StudioApp(config, port=args.port or config.viewer_port, host=args.host, share=args.share,
                    scene=args.scene)
    print(f"[studio] listening on http://{args.host}:{args.port or config.viewer_port}  "
          f"(tunnel: ssh -L {args.port or config.viewer_port}:$(hostname):{args.port or config.viewer_port} <login>)",
          flush=True)
    app.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
