"""Scene tab — see docs/ARCHITECTURE.md for the required controls.

CONTRACT: build(ctx) creates the tab's GUI and wires events.

Display modes ("Display" dropdown at the top; default from config ``viewer.display_mode``):
  * **Server render (JPEG stream, recommended)** — ``physicalview.streaming.
    ServerRenderStream`` renders the browser camera's view on the GPU (gsplat) and streams
    JPEG frames as the viewer background: the browser holds NO Gaussian data. The scene
    graph only carries lightweight helpers: a small frame per accepted object
    (``/helpers/obj_XX``, re-posed on gizmo drags and robot.tick), the highlight AABB, the
    transform gizmo, the selected camera's frustum (``/helpers/camera``) and the optional
    mesh / collision layers. "JPEG quality" (50-95) and "stream resolution"
    (720p/1080p/native) publish ``display.invalidate``.
  * **Client splats (WebGL, high memory)** — the previous behaviour: splat arrays capped
    by ``viewer.max_splats_background`` / ``max_splats_object`` are uploaded as
    ``/background`` and ``/objects/obj_XX`` gaussian nodes.
  Switching at runtime tears the other mode down (splat nodes removed / background image
  cleared) and publishes ``display.mode_changed`` ("server" | "client"); the mode is
  mirrored on ``ctx.display_mode`` and the stream on ``ctx.stream``. Without a GPU the
  panel falls back to client mode (the stream needs the CUDA renderer).

Controls: result-set dropdown (+ refresh) and Load; layer checkboxes (raw background,
clean background, scene mesh, object splats, object meshes, collision parts); object table
+ object dropdown that highlights the object's discovery AABB; "Select object" ->
ctx.set_selection; per-object transform gizmo whose drags re-pose the object's nodes and
record ``ctx.scene.edited_poses[obj_id]`` (4x4, scale folded in) for other panels; camera
dropdown + "Snap viewer camera"; "Photoreal snapshot" (exactly what the stream shows,
robot included, saved as PNG).

Scene graph: /background (gaussian splats, client mode), /objects/obj_XX (canonical
gaussians scaled by s, node pose = R,t of aligned T; client mode), /helpers/obj_XX (server
mode frames), /helpers/camera, /object_meshes/obj_XX, /collision/obj_XX/part_i,
/scene_mesh, /highlight, /gizmo/obj_XX.

Events: subscribes to scene.objects_changed (reload + refresh), robot.tick (re-pose the
object nodes with splats.frame_transform from body_poses / reset_poses), selection.changed
(highlight) and inpaint.version_selected (swap the clean background). Loading runs in a
daemon thread; the viser thread is never blocked. Every scene.add_* on a callback path
replaces the previous handle, so reloads / objects_changed / ticks never accumulate nodes.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from physicalview import scene_state as S
from physicalview import splats as SP
from physicalview.app import Selection
from physicalview.streaming import RESOLUTION_CHOICES, ServerRenderStream, camera_state, stream_size

log = logging.getLogger("studio.scene_panel")

_MAX_MESH_FACES = 400_000
_HIGHLIGHT_RGB = (255, 200, 0)
_FRUSTUM_RGB = (80, 160, 255)
DISPLAY_SERVER = "Server render (JPEG stream, recommended)"
DISPLAY_CLIENT = "Client splats (WebGL, high memory)"
_MODE_LABEL = {"server": DISPLAY_SERVER, "client": DISPLAY_CLIENT}
_LABEL_MODE = {v: k for k, v in _MODE_LABEL.items()}
_RESOLUTIONS = tuple(RESOLUTION_CHOICES)
_KIND_ALL = "all"
# Dataset-filter order: robot datasets first (the demo entry points), scans after.
_KIND_ORDER = ("behavior", "droid", "video", "factory", "auto", "full", "unbuilt", "other")

_rot_to_wxyz = SP.matrix_to_quat_wxyz   # old private name, kept for callers


def _aabb_segments(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """12 edges of an axis-aligned box as [12, 2, 3]."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=np.float32)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return np.stack([np.stack([c[a], c[b]]) for a, b in edges])


def _object_table(objects: dict[str, S.ObjectRecord]) -> str:
    any_eval = any(o.f1_20 is not None for o in objects.values())
    head = "| id | label | source | tier | F1@20 (eval) | stable |" if any_eval else "| id | label | source | tier | stable |"
    sep = "|---|---|---|---|---|---|" if any_eval else "|---|---|---|---|---|"
    rows = [head, sep]
    for name, o in sorted(objects.items()):
        tier = o.tier or "-"
        status = f"{tier} ✓" if o.accepted else (f"{tier} ✗ {o.rejected_reason}" if o.rejected_reason else f"{tier} –")
        stable = "-" if o.stable is None else ("yes" if o.stable else "no")
        cells = [name, o.label, o.chosen_source or "-", status]
        if any_eval:
            cells.append("-" if o.f1_20 is None else f"{o.f1_20:.2f}")
        cells.append(stable)
        rows.append("| " + " | ".join(str(c).replace("|", "/") for c in cells) + " |")
    if len(rows) == 2:
        return "_no objects_"
    return "\n".join(rows)


def _decimated(mesh, max_faces: int = _MAX_MESH_FACES):
    if len(mesh.faces) <= max_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=max_faces)
    except Exception:  # noqa: BLE001 - fast_simplification missing
        pass
    try:
        import open3d as o3d
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
                                      o3d.utility.Vector3iVector(np.asarray(mesh.faces)))
        if mesh.visual.kind == "vertex":
            m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0)
        m = m.simplify_quadric_decimation(max_faces)
        import trimesh
        colors = (np.asarray(m.vertex_colors) * 255).astype(np.uint8) if m.has_vertex_colors() else None
        return trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles), vertex_colors=colors, process=False)
    except Exception:  # noqa: BLE001
        log.warning("mesh decimation unavailable; sending %d faces", len(mesh.faces))
        return mesh


def _resolution_label(max_width) -> str:
    for label, w in RESOLUTION_CHOICES.items():
        if w == max_width:
            return label
    return "native" if not max_width else "720p"


class _ScenePanel:
    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.server = ctx.server
        self.cfg = ctx.config
        self.cache = SP.SplatCache(max_total=max(6_000_000, 2 * ctx.config.max_splats_background))
        self.result_sets: dict[str, S.ResultSet] = {}
        self.state: S.SceneState | None = None
        self.nodes: dict[str, dict] = {}       # obj_id -> {"splat": h, "frame": h, "mesh": h|None, "collision": [h]}
        self.bg_handle = None
        self.mesh_handle = None
        self.highlight = None
        self.gizmo = None
        self.gizmo_obj: str | None = None
        self.frustum = None
        self.snapshot_img = None
        self.busy = threading.Lock()
        self._syncing = False                 # programmatic GUI writes (viser fires callbacks)
        self._clean_bg_path: Path | None = None
        wanted = getattr(ctx, "display_mode", None) or getattr(self.cfg, "display_mode", "server")
        if wanted == "server" and not ctx.gpu.present:
            ctx.log("display: no GPU on this node -> client splats (server render needs the CUDA renderer)")
            wanted = "client"
        self.mode = wanted
        ctx.display_mode = self.mode
        self.stream = ServerRenderStream(ctx, lambda: ctx.renderer, lambda: ctx.robot, self.cfg)
        self.stream.on_frame = self._on_stream_stats
        ctx.stream = self.stream
        self._build_gui()
        ctx.events.subscribe("scene.objects_changed", self._on_objects_changed)
        ctx.events.subscribe("robot.tick", self._on_robot_tick)
        ctx.events.subscribe("selection.changed", self._on_selection_changed)
        ctx.events.subscribe("inpaint.version_selected", self._on_inpaint_version)
        ctx.log(f"display mode: {self.mode} ({'GPU render -> JPEG stream, no splat data in the browser' if self.mode == 'server' else 'WebGL splats in the browser'})")

    # ------------------------------------------------------------------ GUI ----
    def _build_gui(self) -> None:
        gui = self.server.gui
        cfg = self.cfg
        self.result_sets = {r.name: r for r in self._discover()}
        # --scene preselects a result set; its kind also preselects the dataset filter.
        wanted = getattr(self.ctx, "initial_scene", None)
        self.kind_filter = self.result_sets[wanted].kind if wanted in self.result_sets else _KIND_ALL
        names = self._visible_names()
        initial = wanted if wanted in names else names[0]
        with gui.add_folder("Display"):
            self.dd_display = gui.add_dropdown(
                "mode", (DISPLAY_SERVER, DISPLAY_CLIENT), initial_value=_MODE_LABEL[self.mode],
                hint="server: the GPU renders and the browser shows a JPEG stream (no splat data in the browser)")
            self.sl_quality = gui.add_slider("JPEG quality", 50, 95, 1, int(np.clip(cfg.stream.jpeg_quality, 50, 95)))
            self.dd_res = gui.add_dropdown("stream resolution", _RESOLUTIONS,
                                           initial_value=_resolution_label(cfg.stream.max_width))
            self.md_display = gui.add_markdown(self._display_note())
        with gui.add_folder("Result set"):
            self.dd_kind = gui.add_dropdown("dataset", self._kind_options(), initial_value=self.kind_filter,
                                            hint="filter the result-set list by source dataset")
            self.dd_set = gui.add_dropdown("result set", names, initial_value=initial)
            self.md_set = gui.add_markdown(self._describe(initial))
            self.btn_refresh = gui.add_button("Refresh list")
            self.btn_load = gui.add_button("Load scene")
        with gui.add_folder("Layers"):
            self.cb_bg = gui.add_checkbox("background splat", True,
                                          hint="client mode only; the server render always shows the background")
            self.cb_clean = gui.add_checkbox("clean (inpainted) background", False)
            self.cb_mesh = gui.add_checkbox("scene mesh", False)
            self.cb_obj_splats = gui.add_checkbox("object splats", True)
            self.cb_obj_meshes = gui.add_checkbox("object meshes", False)
            self.cb_collision = gui.add_checkbox("collision parts", False)
        with gui.add_folder("Objects"):
            self.md_table = gui.add_markdown("_load a scene_")
            self.dd_obj = gui.add_dropdown("object", ("<none>",), initial_value="<none>")
            self.btn_select = gui.add_button("Select object")
            self.cb_gizmo = gui.add_checkbox("transform gizmo", False)
            self.btn_reset_poses = gui.add_button("Reset edited poses")
        with gui.add_folder("Cameras"):
            self.dd_cam = gui.add_dropdown("camera frame", ("<none>",), initial_value="<none>")
            self.btn_snap_cam = gui.add_button("Snap viewer camera")
            self.btn_photo = gui.add_button("Photoreal snapshot")
            self.md_photo = gui.add_markdown("")

        self.dd_display.on_update(lambda _: self._on_display_dropdown())
        self.sl_quality.on_update(lambda _: self._set_quality())
        self.dd_res.on_update(lambda _: self._set_resolution())
        self.dd_kind.on_update(lambda _: self._on_kind_filter())
        self.dd_set.on_update(lambda _: setattr(self.md_set, "content", self._describe(self.dd_set.value)))
        self.btn_refresh.on_click(lambda _: self._refresh_list())
        self.btn_load.on_click(lambda _: self._start_load(self.dd_set.value))
        self.cb_bg.on_update(lambda _: self._set_visible(self.bg_handle, self.cb_bg.value))
        self.cb_clean.on_update(lambda _: None if self._syncing else self._run_bg(self._swap_background))
        self.cb_mesh.on_update(lambda _: self._run_bg(self._toggle_scene_mesh))
        self.cb_obj_splats.on_update(lambda _: self._on_obj_splats())
        self.cb_obj_meshes.on_update(lambda _: self._run_bg(self._toggle_object_meshes))
        self.cb_collision.on_update(lambda _: self._run_bg(self._toggle_collision))
        self.dd_obj.on_update(lambda _: self._highlight(self.dd_obj.value))
        self.btn_select.on_click(lambda _: self._select_current())
        self.cb_gizmo.on_update(lambda _: self._toggle_gizmo())
        self.btn_reset_poses.on_click(lambda _: self._reset_edited_poses())
        self.dd_cam.on_update(lambda _: self._show_frustum(self.dd_cam.value))
        self.btn_snap_cam.on_click(lambda _: self._snap_camera())
        self.btn_photo.on_click(lambda _: self._run_bg(self._photoreal_snapshot))

    def _discover(self) -> list[S.ResultSet]:
        try:
            return S.discover_result_sets(self.cfg)
        except Exception as exc:  # noqa: BLE001
            self.ctx.log(f"result-set discovery failed: {exc}")
            return []

    def _describe(self, name: str) -> str:
        rs = self.result_sets.get(name)
        if rs is None:
            return "_no result sets found_"
        flags = [f for f, ok in (("clean bg", rs.has_clean_bg), ("sim export", rs.has_sim_export),
                                 ("tasks", rs.has_tasks)) if ok]
        return (f"`{rs.kind}` scene `{rs.scene_id}` — {rs.n_accepted}/{rs.n_objects} objects accepted"
                + (f" · {', '.join(flags)}" if flags else "")
                + ("" if rs.splat_ply else " · **no splat**"))

    def _kind_options(self) -> tuple[str, ...]:
        """"all" plus every kind actually present, in _KIND_ORDER then alphabetically."""
        kinds = {r.kind for r in self.result_sets.values()}
        ordered = [k for k in _KIND_ORDER if k in kinds]
        ordered += sorted(kinds - set(_KIND_ORDER))
        return (_KIND_ALL,) + tuple(ordered)

    def _visible_names(self) -> tuple[str, ...]:
        """Result-set names passing the dataset filter, sorted; ("<none>",) when empty."""
        names = tuple(sorted(n for n, r in self.result_sets.items()
                             if self.kind_filter == _KIND_ALL or r.kind == self.kind_filter))
        return names or ("<none>",)

    def _on_kind_filter(self) -> None:
        self.kind_filter = self.dd_kind.value
        names = self._visible_names()
        cur = self.dd_set.value
        self.dd_set.options = names
        self.dd_set.value = cur if cur in names else names[0]
        self.md_set.content = self._describe(self.dd_set.value)
        self.ctx.set_status(f"{len(names)} result sets in `{self.kind_filter}`")

    def _refresh_list(self) -> None:
        self.result_sets = {r.name: r for r in self._discover()}
        kinds = self._kind_options()
        self.dd_kind.options = kinds
        if self.kind_filter not in kinds:
            self.kind_filter = _KIND_ALL
        self.dd_kind.value = self.kind_filter
        names = self._visible_names()
        cur = self.dd_set.value
        self.dd_set.options = names
        self.dd_set.value = cur if cur in names else names[0]
        self.md_set.content = self._describe(self.dd_set.value)
        self.ctx.set_status(f"{len(self.result_sets)} result sets "
                            f"({len(names)} in `{self.kind_filter}`)")

    def _run_bg(self, fn, *args) -> None:
        def run():
            try:
                fn(*args)
            except Exception:  # noqa: BLE001
                self.ctx.log(f"scene panel: {fn.__name__} failed:\n{traceback.format_exc()}")
        threading.Thread(target=run, daemon=True).start()

    def _set_value(self, handle, value) -> None:
        """Write a GUI value without re-entering our own callbacks."""
        self._syncing = True
        try:
            handle.value = value
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._syncing = False

    @staticmethod
    def _set_visible(handle, visible: bool) -> None:
        if handle is not None:
            try:
                handle.visible = bool(visible)
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _remove(handle) -> None:
        if handle is None:
            return
        for h in (handle if isinstance(handle, (list, tuple)) else [handle]):
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------ display mode ----
    def _display_note(self, stats: dict | None = None) -> str:
        cfg = self.cfg
        if self.mode == "server":
            s = self.stream.settings
            res = self.dd_res.value if hasattr(self, "dd_res") else _resolution_label(s.max_width)
            txt = (f"**server render** — GPU gsplat → JPEG q{s.jpeg_quality} ≤ {s.max_fps:g} fps ({res}); "
                   f"the browser holds 0 gaussians")
            if stats:
                parts = [f"client {cid}: {v['wh'][0]}×{v['wh'][1]} · {v['fps']} fps · {v['ms']} ms/frame"
                         for cid, v in stats.items() if v.get("wh")]
                if parts:
                    txt += "  \n" + " · ".join(parts)
            return txt
        return (f"**client splats** — WebGL in the browser; caps {cfg.max_splats_background:,} background / "
                f"{cfg.max_splats_object:,} per-object gaussians (browser memory)")

    def _on_stream_stats(self, stats: dict) -> None:
        try:
            self.md_display.content = self._display_note(stats)
        except Exception:  # noqa: BLE001
            pass

    def _on_display_dropdown(self) -> None:
        if self._syncing:
            return
        self._run_bg(self._set_display_mode, _LABEL_MODE.get(self.dd_display.value, "client"))

    def _set_display_mode(self, mode: str) -> None:
        if mode == self.mode:
            return
        if mode == "server" and not self.ctx.gpu.present:
            self.ctx.set_status("server render needs a GPU node (CUDA renderer) — staying on client splats")
            self._set_value(self.dd_display, DISPLAY_CLIENT)
            return
        cfg = self.cfg
        with self.busy:                      # never while a load is running
            self.mode = mode
            self.ctx.display_mode = mode
            st = self.state
            if mode == "server":
                self._remove_nodes(("splat",))
                self._remove(self.bg_handle)
                self.bg_handle = None
                n = self._add_helper_frames() if st is not None else 0
                if st is not None:
                    self.stream.start()
                self.ctx.log(f"display mode: server render — splat nodes removed, 0 gaussians in the browser "
                             f"({n} helper frames)")
            else:
                self.stream.stop()
                self._remove_nodes(("frame",))
                n = 0
                if st is not None:
                    self._add_background()
                    for obj_id, rec in sorted(st.objects.items()):
                        if rec.accepted and rec.T_world is not None and self._add_object_splat(obj_id, rec):
                            n += 1
                self.ctx.log(f"display mode: client splats — /background + {n} object splat nodes uploaded "
                             f"(caps {cfg.max_splats_background:,}/{cfg.max_splats_object:,})")
        self.ctx.events.publish("display.mode_changed", mode)
        self.ctx.events.publish("display.invalidate")
        self.md_display.content = self._display_note()
        self.ctx.set_status(f"display: {mode} mode")

    def _set_quality(self) -> None:
        self.stream.jpeg_quality = int(self.sl_quality.value)
        self.ctx.events.publish("display.invalidate")
        self.md_display.content = self._display_note()

    def _set_resolution(self) -> None:
        self.stream.max_width = RESOLUTION_CHOICES.get(self.dd_res.value, self.cfg.stream.max_width)
        self.ctx.events.publish("display.invalidate")
        self.md_display.content = self._display_note()

    # ------------------------------------------------------------- loading ----
    def _start_load(self, name: str) -> None:
        if name not in self.result_sets:
            self.ctx.set_status("nothing to load")
            return
        if not self.busy.acquire(blocking=False):
            self.ctx.set_status("a load is already running")
            return
        self.btn_load.disabled = True

        def run():
            try:
                self._load(self.result_sets[name])
            except Exception:  # noqa: BLE001
                self.ctx.set_status(f"load of {name} failed — see log")
                self.ctx.log(traceback.format_exc())
            finally:
                self.btn_load.disabled = False
                self.busy.release()
        threading.Thread(target=run, daemon=True, name="scene-load").start()

    def _clear_scene(self) -> None:
        self._remove(self.bg_handle)
        self._remove(self.mesh_handle)
        self._remove(self.highlight)
        self._remove(self.gizmo)
        self._remove(self.frustum)
        for n in self.nodes.values():
            for kind in ("splat", "frame", "mesh", "collision"):
                self._remove(n.get(kind))
        self.bg_handle = self.mesh_handle = self.highlight = self.gizmo = self.frustum = None
        self.gizmo_obj = None
        self.nodes = {}
        self.stream.clear_background()

    def _load(self, rs: S.ResultSet) -> None:
        ctx = self.ctx
        cfg = self.cfg
        ctx.set_status(f"loading {rs.name}: cameras + splats …")
        state = S.load_scene(cfg, rs, device="cpu", load_mesh=False)
        if ctx.renderer is not None:
            ctx.renderer = None
        self._clear_scene()
        self.state = state
        clean_path = Path(rs.out_dir) / "inpaint" / "clean_background.ply"
        self._clean_bg_path = clean_path if state.clean_bg_gs is not None else None
        if state.clean_bg_gs is None and self.cb_clean.value:
            self._set_value(self.cb_clean, False)
        self.cb_clean.disabled = state.clean_bg_gs is None
        n = 0
        if self.mode == "client":
            ctx.set_status(f"{rs.name}: uploading background splat …")
            self._add_background()
            for obj_id, rec in sorted(state.objects.items()):
                if not rec.accepted or rec.T_world is None:
                    continue
                ctx.set_status(f"{rs.name}: object splats {obj_id} …")
                if self._add_object_splat(obj_id, rec):
                    n += 1
            ctx.log(f"display mode: client splats — /background + {n} object splat nodes uploaded "
                    f"(caps {cfg.max_splats_background:,}/{cfg.max_splats_object:,})")
        else:
            n = self._add_helper_frames()
            ctx.log(f"display mode: server render — 0 gaussian splat nodes created ({n} helper frames); "
                    "frames are streamed as JPEG background images")
        self._refresh_objects_gui()
        self.dd_cam.options = tuple(sorted(state.cameras)) or ("<none>",)
        self.dd_cam.value = self.dd_cam.options[0]
        self._show_frustum(self.dd_cam.value)
        if ctx.gpu.present and (state.splat_gs is not None or state.clean_bg_gs is not None):
            ctx.set_status(f"{rs.name}: starting CUDA renderer …")
            try:
                from physicalview.render import Renderer, RenderUnavailable
                try:
                    ctx.renderer = Renderer(state, use_clean_bg=self.cb_clean.value or state.splat_gs is None)
                except RenderUnavailable as exc:
                    ctx.log(f"photoreal renderer unavailable: {exc}")
            except Exception as exc:  # noqa: BLE001
                ctx.log(f"photoreal renderer failed: {exc}")
        ctx.set_scene(state)
        if self.mode == "server":
            if ctx.renderer is None:
                ctx.log("server render idle: no CUDA renderer for this scene — switch Display to client splats")
            self.stream.start()
        if self.cb_mesh.value:
            self._run_bg(self._toggle_scene_mesh)
        if self.cb_obj_meshes.value:
            self._run_bg(self._toggle_object_meshes)
        if self.cb_collision.value:
            self._run_bg(self._toggle_collision)
        ctx.set_status(f"loaded {rs.name}: {n} {'object splats' if self.mode == 'client' else 'object frames'}, "
                       f"{len(state.cameras)} cameras"
                       + (", photoreal on" if ctx.renderer is not None else "")
                       + (" · server render stream" if self.mode == "server" else ""))

    # ---------------------------------------------------------- scene nodes ----
    def _bg_arrays(self, clean: bool) -> SP.SplatArrays | None:
        st = self.state
        if st is None:
            return None
        gs = st.clean_bg_gs if clean else st.splat_gs
        if gs is None:
            return None
        key = f"{st.result_set.name}:bg:{'clean' if clean else 'raw'}"
        return self.cache.get(key, lambda: SP.to_viser_arrays(gs, self.cfg.max_splats_background))

    def _add_background(self) -> None:
        """Client mode: (re)upload the capped background splat; replaces the old node."""
        use_clean = self.cb_clean.value and self.state is not None and self.state.clean_bg_gs is not None
        arr = self._bg_arrays(use_clean) or self._bg_arrays(False)
        self._remove(self.bg_handle)
        self.bg_handle = None
        if arr is None:
            self.ctx.log("no background splat for this scene")
            return
        self.bg_handle = self.server.scene.add_gaussian_splats(
            "/background", centers=arr.centers, covariances=arr.covariances, rgbs=arr.rgbs,
            opacities=arr.opacities, visible=self.cb_bg.value)

    def _swap_background(self) -> None:
        if self.state is None:
            return
        if self.mode == "client":
            self._add_background()
        if self.ctx.renderer is not None:
            try:
                self.ctx.renderer.set_background("clean" if self.cb_clean.value else "raw")
            except Exception as exc:  # noqa: BLE001
                self.ctx.log(f"renderer background switch failed: {exc}")
        self.ctx.events.publish("display.invalidate")
        self.ctx.set_status("clean background" if self.cb_clean.value else "raw background")

    def _object_pose(self, obj_id: str) -> tuple[float, np.ndarray, np.ndarray] | None:
        st = self.state
        if st is None or obj_id not in st.objects:
            return None
        rec = st.objects[obj_id]
        T = st.edited_poses.get(obj_id, rec.T_world)
        if T is None:
            return None
        return SP.decompose_similarity(T)

    def _add_object_splat(self, obj_id: str, rec: S.ObjectRecord) -> bool:
        """Client mode: one capped splat node per object; an existing node is replaced."""
        pose = self._object_pose(obj_id)
        if pose is None:
            return False
        s, R, t = pose
        try:
            gs = S.object_canonical_gs(self.state, obj_id)
        except (FileNotFoundError, KeyError) as exc:
            self.ctx.log(f"{obj_id}: {exc}")
            return False
        src = S.canonical_gs_path(rec)[0]
        key = f"{self.state.result_set.name}:{obj_id}:{src}:{s:.6f}"
        arr = self.cache.get(key, lambda: SP.to_viser_arrays(gs, self.cfg.max_splats_object, extra_scale=s))
        n = self.nodes.setdefault(obj_id, {})
        self._remove(n.get("splat"))
        n["splat"] = self.server.scene.add_gaussian_splats(
            f"/objects/{obj_id}", centers=arr.centers, covariances=arr.covariances, rgbs=arr.rgbs,
            opacities=arr.opacities, position=tuple(t), wxyz=tuple(SP.matrix_to_quat_wxyz(R)),
            visible=self.cb_obj_splats.value)
        return True

    def _add_object_frame(self, obj_id: str, rec: S.ObjectRecord) -> bool:
        """Server mode: a small axes frame marks the object pose (no gaussians)."""
        pose = self._object_pose(obj_id)
        if pose is None:
            return False
        s, R, t = pose
        dims = rec.world_dims or [0.2, 0.2, 0.2]
        L = float(np.clip(0.5 * max(dims), 0.05, 0.3))
        n = self.nodes.setdefault(obj_id, {})
        self._remove(n.get("frame"))
        n["frame"] = None
        try:
            n["frame"] = self.server.scene.add_frame(
                f"/helpers/{obj_id}", axes_length=L, axes_radius=L / 25.0, origin_radius=L / 10.0,
                position=tuple(t), wxyz=tuple(SP.matrix_to_quat_wxyz(R)), visible=self.cb_obj_splats.value)
        except Exception as exc:  # noqa: BLE001
            self.ctx.log(f"{obj_id} frame: {exc}")
            return False
        return True

    def _add_object_visual(self, obj_id: str, rec: S.ObjectRecord) -> bool:
        if self.mode == "client":
            return self._add_object_splat(obj_id, rec)
        return self._add_object_frame(obj_id, rec)

    def _add_helper_frames(self) -> int:
        st = self.state
        if st is None:
            return 0
        n = 0
        for obj_id, rec in sorted(st.objects.items()):
            if rec.accepted and rec.T_world is not None and self._add_object_frame(obj_id, rec):
                n += 1
        return n

    def _remove_nodes(self, kinds: tuple[str, ...]) -> None:
        for n in self.nodes.values():
            for kind in kinds:
                self._remove(n.pop(kind, None))

    def _set_layer(self, layer: str, visible: bool) -> None:
        for n in self.nodes.values():
            h = n.get(layer)
            if isinstance(h, list):
                for hh in h:
                    self._set_visible(hh, visible)
            else:
                self._set_visible(h, visible)

    def _on_obj_splats(self) -> None:
        vis = bool(self.cb_obj_splats.value)
        self._set_layer("splat", vis)
        self._set_layer("frame", vis)
        self.stream.show_objects = vis
        self.ctx.events.publish("display.invalidate")

    def _pose_nodes(self, obj_id: str, T: np.ndarray) -> None:
        s, R, t = SP.decompose_similarity(T)
        q = tuple(SP.matrix_to_quat_wxyz(R))
        n = self.nodes.get(obj_id, {})
        for h in [n.get("splat"), n.get("frame"), n.get("mesh")] + list(n.get("collision") or []):
            if h is None:
                continue
            try:
                h.position = tuple(t)
                h.wxyz = q
            except Exception:  # noqa: BLE001
                pass

    def _toggle_scene_mesh(self) -> None:
        st = self.state
        if st is None:
            return
        if not self.cb_mesh.value:
            self._set_visible(self.mesh_handle, False)
            return
        if self.mesh_handle is not None:
            self._set_visible(self.mesh_handle, True)
            return
        if st.mesh is None:
            mp = st.result_set.mesh_ply
            if mp is None or not Path(mp).exists():
                self.ctx.set_status("no scene mesh for this result set")
                return
            self.ctx.set_status(f"loading scene mesh {Path(mp).name} …")
            import trimesh
            st.mesh = trimesh.load(str(mp), process=False, force="mesh")
        mesh = _decimated(st.mesh)
        self.mesh_handle = self.server.scene.add_mesh_trimesh("/scene_mesh", mesh, visible=True)
        self.ctx.set_status(f"scene mesh: {len(mesh.faces)} faces")

    def _toggle_object_meshes(self) -> None:
        st = self.state
        if st is None:
            return
        if not self.cb_obj_meshes.value:
            self._set_layer("mesh", False)
            return
        import trimesh
        for obj_id, rec in sorted(st.objects.items()):
            n = self.nodes.setdefault(obj_id, {})
            if n.get("mesh") is not None:
                self._set_visible(n["mesh"], True)
                continue
            pose = self._object_pose(obj_id)
            mp = rec.dir / "mesh_sim.ply"
            if pose is None or not rec.accepted or not mp.exists():
                continue
            s, R, t = pose
            try:
                tm = trimesh.load(str(mp), process=False, force="mesh")
                n["mesh"] = self.server.scene.add_mesh_trimesh(
                    f"/object_meshes/{obj_id}", tm, scale=s, position=tuple(t), wxyz=tuple(SP.matrix_to_quat_wxyz(R)))
            except Exception as exc:  # noqa: BLE001
                self.ctx.log(f"{obj_id} mesh: {exc}")

    def _toggle_collision(self) -> None:
        st = self.state
        if st is None:
            return
        if not self.cb_collision.value:
            self._set_layer("collision", False)
            return
        import trimesh
        rng = np.random.default_rng(0)
        for obj_id, rec in sorted(st.objects.items()):
            n = self.nodes.setdefault(obj_id, {})
            if n.get("collision"):
                for h in n["collision"]:
                    self._set_visible(h, True)
                continue
            pose = self._object_pose(obj_id)
            if pose is None or not rec.accepted or rec.collision_parts == 0:
                continue
            s, R, t = pose
            q = tuple(SP.matrix_to_quat_wxyz(R))
            handles = []
            for part in sorted((rec.dir / "collision").glob("part_*.obj")):
                try:
                    tm = trimesh.load(str(part), process=False, force="mesh")
                    color = (rng.integers(80, 255, 3)).astype(np.uint8)
                    tm.visual.face_colors = np.tile(np.append(color, 200), (len(tm.faces), 1))
                    handles.append(self.server.scene.add_mesh_trimesh(
                        f"/collision/{obj_id}/{part.stem}", tm, scale=s, position=tuple(t), wxyz=q))
                except Exception as exc:  # noqa: BLE001
                    self.ctx.log(f"{obj_id}/{part.name}: {exc}")
            n["collision"] = handles

    # -------------------------------------------------------------- objects ----
    def _refresh_objects_gui(self) -> None:
        st = self.state
        if st is None:
            return
        self.md_table.content = _object_table(st.objects)
        names = tuple(sorted(st.objects)) or ("<none>",)
        cur = self.dd_obj.value
        self.dd_obj.options = names
        self.dd_obj.value = cur if cur in names else names[0]

    def _highlight(self, obj_id: str | None) -> None:
        self._remove(self.highlight)
        self.highlight = None
        st = self.state
        if st is None or not obj_id or obj_id not in st.objects:
            return
        rec = st.objects[obj_id]
        aabb = rec.meta.get("aabb")
        if aabb is None and rec.T_world is not None and rec.world_dims:
            c = rec.T_world[:3, 3]
            half = np.asarray(rec.world_dims, dtype=np.float64) / 2.0
            aabb = [c - half, c + half]
        if aabb is None:
            return
        lo, hi = np.asarray(aabb[0], dtype=np.float64), np.asarray(aabb[1], dtype=np.float64)
        pad = 0.01
        seg = _aabb_segments(lo - pad, hi + pad)
        colors = np.tile(np.array(_HIGHLIGHT_RGB, dtype=np.uint8), (seg.shape[0], 2, 1))
        try:
            self.highlight = self.server.scene.add_line_segments("/highlight", seg, colors, line_width=3.0)
        except Exception as exc:  # noqa: BLE001
            self.ctx.log(f"highlight failed: {exc}")

    def _select_current(self) -> None:
        obj_id = self.dd_obj.value
        st = self.state
        if st is None or obj_id not in st.objects:
            return
        self.ctx.set_selection(Selection(kind="object", object_ids=[st.objects[obj_id].index]))
        self.ctx.set_status(f"selected {obj_id} ({st.objects[obj_id].label})")

    def _on_selection_changed(self, sel) -> None:
        if getattr(sel, "kind", None) == "object" and sel.object_ids:
            name = f"obj_{sel.object_ids[0]:02d}"
            if self.state is not None and name in self.state.objects:
                if self.dd_obj.value != name:
                    self.dd_obj.value = name
                self._highlight(name)
                if self.cb_gizmo.value:
                    self._toggle_gizmo()
        elif getattr(sel, "kind", None) == "none":
            self._highlight(None)

    def _toggle_gizmo(self) -> None:
        self._remove(self.gizmo)
        self.gizmo = None
        self.gizmo_obj = None
        st = self.state
        obj_id = self.dd_obj.value
        if not self.cb_gizmo.value or st is None or obj_id not in st.objects:
            return
        pose = self._object_pose(obj_id)
        if pose is None:
            self.ctx.set_status(f"{obj_id} has no registered pose to edit")
            return
        s, R, t = pose
        dims = st.objects[obj_id].world_dims or [0.2, 0.2, 0.2]
        try:
            self.gizmo = self.server.scene.add_transform_controls(
                f"/gizmo/{obj_id}", scale=float(max(0.15, 1.6 * max(dims))), line_width=1.5,
                position=tuple(t), wxyz=tuple(SP.matrix_to_quat_wxyz(R)))
        except Exception as exc:  # noqa: BLE001
            self.ctx.log(f"gizmo failed: {exc}")
            return
        self.gizmo_obj = obj_id

        @self.gizmo.on_update
        def _(event) -> None:
            if self.state is None or self.gizmo_obj != obj_id:
                return
            tc = event.target if hasattr(event, "target") else self.gizmo
            T = np.eye(4)
            T[:3, :3] = s * SP.quat_wxyz_to_matrix(np.asarray(tc.wxyz, dtype=np.float64)[None])[0]
            T[:3, 3] = np.asarray(tc.position, dtype=np.float64)
            self.state.edited_poses[obj_id] = T
            self._pose_nodes(obj_id, T)
            self.ctx.events.publish("display.invalidate")      # live in the stream (coalesced)
            if getattr(event, "phase", "end") == "end":
                self.ctx.events.publish("scene.pose_edited", {"object": obj_id, "T": T})

    def _reset_edited_poses(self) -> None:
        st = self.state
        if st is None:
            return
        edited = list(st.edited_poses)
        st.edited_poses.clear()
        for obj_id in edited:
            rec = st.objects.get(obj_id)
            if rec is not None and rec.T_world is not None:
                self._pose_nodes(obj_id, rec.T_world)
        if self.cb_gizmo.value:
            self._toggle_gizmo()
        self.ctx.events.publish("display.invalidate")
        self.ctx.set_status(f"reset {len(edited)} edited pose(s)")

    def _on_objects_changed(self, obj_ids) -> None:
        st = self.state
        if st is None:
            return

        def run():
            changed = S.reload_objects(st)
            ids = set(changed) | set(obj_ids or [])
            for obj_id in sorted(ids):
                n = self.nodes.pop(obj_id, {})
                for kind in ("splat", "frame", "mesh", "collision"):
                    self._remove(n.get(kind))
                for key in list(self.cache._items):  # invalidate per-object arrays
                    if key.startswith(f"{st.result_set.name}:{obj_id}:"):
                        self.cache.pop(key)
                rec = st.objects.get(obj_id)
                if rec is not None and rec.accepted:
                    self._add_object_visual(obj_id, rec)
            if self.ctx.renderer is not None:
                try:
                    self.ctx.renderer.invalidate(ids)
                except Exception:  # noqa: BLE001
                    pass
            self._refresh_objects_gui()
            if self.cb_obj_meshes.value:
                self._toggle_object_meshes()
            if self.cb_collision.value:
                self._toggle_collision()
            self.ctx.events.publish("display.invalidate")
            self.ctx.set_status(f"objects refreshed: {', '.join(sorted(ids)) or 'none changed'}")
        self._run_bg(run)

    def _on_inpaint_version(self, path) -> None:
        """Inpaint tab picked a clean-background version (None = original splat)."""
        st = self.state
        if st is None:
            return

        def run():
            if path is None:
                if self.cb_clean.value:
                    self._set_value(self.cb_clean, False)
                    self._swap_background()
                return
            p = Path(path)
            if not p.exists():
                self.ctx.set_status(f"inpaint version missing: {p}")
                return
            if st.clean_bg_gs is None or self._clean_bg_path is None or Path(self._clean_bg_path) != p:
                self.ctx.set_status(f"loading clean background {p.name} …")
                from agents.core import common as C
                st.clean_bg_gs = C.load_gaussians(p, device="cpu")
                self._clean_bg_path = p
                self.cache.pop(f"{st.result_set.name}:bg:clean")
                if self.ctx.renderer is not None:
                    try:
                        self.ctx.renderer.set_background("clean", force=True)
                    except Exception as exc:  # noqa: BLE001
                        self.ctx.log(f"renderer clean background reload failed: {exc}")
            self.cb_clean.disabled = False
            if not self.cb_clean.value:
                self._set_value(self.cb_clean, True)
            self._swap_background()
        self._run_bg(run)

    # ----------------------------------------------------------------- robot ----
    def _on_robot_tick(self, payload) -> None:
        st = self.state
        if st is None or payload is None:
            return
        body_poses = getattr(payload, "body_poses", None)
        reset_poses = getattr(payload, "reset_poses", None)
        if isinstance(payload, dict):
            body_poses = payload.get("body_poses", body_poses)
            reset_poses = payload.get("reset_poses", reset_poses)
        if not body_poses:
            return
        try:
            with self.server.atomic():
                for obj_id, (pos, quat) in body_poses.items():
                    rec = st.objects.get(obj_id)
                    if rec is None or rec.T_world is None or obj_id not in self.nodes:
                        continue
                    base_T = st.edited_poses.get(obj_id, rec.T_world)
                    if reset_poses and obj_id in reset_poses:
                        p0, q0 = reset_poses[obj_id]
                        T = SP.frame_transform(base_T, pos, quat, p0, q0)
                    else:
                        T = SP.body_pose_transform(base_T, pos, quat)
                    self._pose_nodes(obj_id, T)
        except Exception as exc:  # noqa: BLE001
            log.debug("robot.tick re-pose failed: %s", exc)

    # ---------------------------------------------------------------- cameras ----
    def _clients(self):
        try:
            return list(self.server.get_clients().values())
        except Exception:  # noqa: BLE001
            return []

    def _show_frustum(self, name: str | None) -> None:
        """Frustum helper for the selected scene camera (replaces the previous one)."""
        self._remove(self.frustum)
        self.frustum = None
        st = self.state
        if st is None or not name or name not in st.cameras:
            return
        c2w = np.linalg.inv(st.cameras[name])
        fov = (float(2.0 * np.arctan(st.H / (2.0 * st.K[1, 1]))) if (st.K is not None and st.H)
               else float(np.radians(60.0)))
        aspect = (st.W / st.H) if (st.W and st.H) else 16.0 / 9.0
        try:
            self.frustum = self.server.scene.add_camera_frustum(
                "/helpers/camera", fov=fov, aspect=aspect, scale=0.15, color=_FRUSTUM_RGB,
                position=tuple(c2w[:3, 3]), wxyz=tuple(SP.matrix_to_quat_wxyz(c2w[:3, :3])))
        except Exception as exc:  # noqa: BLE001
            self.ctx.log(f"camera frustum failed: {exc}")

    def _snap_camera(self) -> None:
        st = self.state
        name = self.dd_cam.value
        if st is None or name not in st.cameras:
            self.ctx.set_status("no camera frame selected")
            return
        c2w = np.linalg.inv(st.cameras[name])
        clients = self._clients()
        if not clients:
            self.ctx.set_status("no browser client connected")
            return
        for client in clients:
            try:
                cam = client.camera
                cam.position = tuple(c2w[:3, 3])
                cam.wxyz = tuple(SP.matrix_to_quat_wxyz(c2w[:3, :3]))
                if st.K is not None and st.H:
                    cam.fov = float(2.0 * np.arctan(st.H / (2.0 * st.K[1, 1])))
            except Exception as exc:  # noqa: BLE001
                self.ctx.log(f"camera snap failed: {exc}")
        self._show_frustum(name)
        self.ctx.set_selection(Selection(kind=self.ctx.selection.kind, object_ids=list(self.ctx.selection.object_ids),
                                         camera_frame=name))
        self.ctx.set_status(f"viewer camera -> {name}")

    def viewer_camera(self) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
        """(w2c, K, (W,H)) of the first connected browser camera at config.render_wh."""
        clients = self._clients()
        if not clients:
            return None
        cam = clients[0].camera
        c2w = np.eye(4)
        c2w[:3, :3] = SP.quat_wxyz_to_matrix(np.asarray(cam.wxyz, dtype=np.float64)[None])[0]
        c2w[:3, 3] = np.asarray(cam.position, dtype=np.float64)
        W, H = self.cfg.render_wh
        fov = float(cam.fov) or np.radians(60.0)
        fy = H / (2.0 * np.tan(fov / 2.0))
        K = np.array([[fy, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]])
        return np.linalg.inv(c2w), K, (W, H)

    def _photoreal_snapshot(self) -> None:
        r = self.ctx.renderer
        if r is None:
            self.md_photo.content = "_photoreal renderer unavailable (needs a GPU node with gsplat)_"
            return
        clients = self._clients()
        if not clients:
            self.md_photo.content = "_no browser client connected_"
            return
        self.ctx.set_status("rendering photoreal snapshot …")
        cam = camera_state(clients[0])
        if cam is not None:   # same path as the stream: objects posed live, robot composited
            wh = stream_size(cam["image_width"], cam["aspect"], 1920)
            img = self.stream.render_for_camera(cam["position"], cam["wxyz"], cam["fov"], cam["aspect"], wh)
        else:
            w2c, K, wh = self.viewer_camera()
            img = r.render(w2c, K, wh)
        if self.snapshot_img is None:
            self.snapshot_img = self.server.gui.add_image(img, label="photoreal snapshot", format="jpeg")
        else:
            self.snapshot_img.image = img
        out = Path(self.cfg.studio_out) / "snapshots"
        try:
            path = r.save_png(img, out / f"{self.state.result_set.name}_{time.strftime('%Y%m%d_%H%M%S')}.png")
            self.md_photo.content = f"saved `{path}`"
        except Exception as exc:  # noqa: BLE001
            self.md_photo.content = f"snapshot rendered (save failed: {exc})"
        self.ctx.set_status("photoreal snapshot done")


def build(ctx) -> None:
    import viser  # noqa: F401 - only needed at build time (the GUI lives in ctx.server)
    panel = _ScenePanel(ctx)
    ctx.scene_panel = panel  # handy for tests / other panels (viewer_camera helper)
    # --scene: load it now so the first browser to connect sees the scene already up.
    wanted = getattr(ctx, "initial_scene", None)
    if wanted:
        if wanted in panel.result_sets:
            ctx.log(f"--scene {wanted}: loading at startup")
            panel._start_load(wanted)
        else:
            ctx.log(f"--scene {wanted}: no such result set; pick one in the Scene tab")
