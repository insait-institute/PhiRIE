"""Inpaint tab — see docs/ARCHITECTURE.md for the required controls.

CONTRACT: build(ctx) creates the tab's GUI and wires events:
  * selection mode (object from ctx.selection | 3D box: a viser box mesh under a
    transform-controls gizmo with an editable size vector3; moving it publishes a
    box Selection through ctx.set_selection)
  * text prompt, negative prompt, backend dropdown (config.inpaint_backends; the
    Qwen backend is withheld when the edit stage would run locally on a GPU with
    < 40 GB), refine iterations
  * "Run inpainting" submits pipeline.inpaint(...) as a chain via ctx.jobs
  * before/after toggle + versions dropdown (inpaint/versions/index.json); both
    publish "inpaint.version_selected" with the chosen ply path (None = original
    splat) so the scene panel can swap the background splat.
viser is imported lazily inside build().
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

from physicalview import pipeline
from physicalview.jobs import JobState

MODE_OBJECT = "object (from selection)"
MODE_BOX = "3D box"
_BEFORE = "(before: original splat)"
_CURRENT = "current clean_background.ply"
QWEN_MIN_MIB = 40 * 1024


def read_versions(out_dir: Path) -> list[dict]:
    """Records of inpaint/versions/index.json (newest first) whose file exists."""
    inp = Path(out_dir) / "inpaint"
    idx = inp / "versions" / "index.json"
    if not idx.exists():
        return []
    try:
        recs = json.loads(idx.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    out = []
    for r in recs:
        f = inp / "versions" / str(r.get("file", ""))
        if f.exists():
            out.append({**r, "path": f})
    return out[::-1]


def version_label(rec: dict) -> str:
    sel = rec.get("selection")
    if isinstance(sel, dict):
        objs = sel.get("objects")
        regs = sel.get("regions")
        sel_s = ("all" if objs == "all" else
                 ",".join(f"obj_{int(i):02d}" for i in (objs or [])) if objs else "")
        if regs:
            sel_s = (sel_s + " " if sel_s else "") + ",".join(regs)
    else:
        sel_s = str(sel) if sel else ""
    return f"{rec.get('stamp', '?')} · {sel_s or '?'} · {rec.get('backend') or '?'}"


def qwen_allowed(ctx, backend_id: str, spec_env_key: str, needs_gpu: bool) -> bool:
    """False when the qwen edit stage would run locally on a < 40 GB GPU."""
    if backend_id != "qwen_image_edit":
        return True
    from physicalview.jobs import JobSpec
    probe = JobSpec(name="probe", argv=["x"], env_key=spec_env_key, needs_gpu=needs_gpu)
    try:
        where = ctx.jobs.decide_where(probe)
    except Exception:  # noqa: BLE001
        where = "local"
    if where != "local":
        return True
    return bool(ctx.gpu.present and ctx.gpu.memory_mib >= QWEN_MIN_MIB)


def build(ctx) -> None:
    import viser  # noqa: F401
    from physicalview.app import Selection
    from physicalview.panels.generate_panel import stage_context

    gui = ctx.server.gui
    scene_api = ctx.server.scene
    config = ctx.config
    backends = {c.label: c for c in config.inpaint_backends}
    state = {"tc": None, "box": None, "syncing": False}

    info = gui.add_markdown("_no scene loaded_")
    mode = gui.add_dropdown("Selection mode", options=[MODE_OBJECT, MODE_BOX], initial_value=MODE_OBJECT)
    sel_md = gui.add_markdown("_selection: none_")
    with gui.add_folder("3D box", expand_by_default=False):
        box_size = gui.add_vector3("Box size (m)", initial_value=(0.30, 0.30, 0.30), step=0.01, min=(0.01,) * 3)
        box_label = gui.add_text("Region label (SAM3 prompt; 'region' = none)", "region")
        center_btn = gui.add_button("Move box to selected object")
    with gui.add_folder("Edit"):
        prompt = gui.add_text("Prompt", "remove the {label} completely from the scene; show the empty flat "
                              "surface behind it, seamlessly continuing the table top and background, "
                              "photorealistic, same lighting")
        neg_prompt = gui.add_text("Negative prompt", " ")
        backend_dd = gui.add_dropdown("Backend", options=list(backends) or ["-"])
        backend_note = gui.add_markdown("")
        iters = gui.add_number("Refine iterations", 1500, min=50, max=10000, step=50)
        run_btn = gui.add_button("Run inpainting", color="green")
    with gui.add_folder("Versions"):
        after = gui.add_checkbox("Show inpainted (after)", False)
        versions_dd = gui.add_dropdown("Version", options=[_BEFORE], initial_value=_BEFORE)
        refresh_btn = gui.add_button("Refresh versions")
        ver_md = gui.add_markdown("")

    vmap: dict[str, Path | None] = {_BEFORE: None}

    # ------------------------------------------------------------ box gizmo
    def _ensure_box(visible: bool) -> None:
        if state["tc"] is None:
            try:
                tc = scene_api.add_transform_controls("/studio/inpaint_box", scale=0.4,
                                                      position=(0.0, 0.0, 0.8))
                box = scene_api.add_box("/studio/inpaint_box/mesh", dimensions=tuple(box_size.value),
                                        color=(255, 140, 0), opacity=0.35)
                state["tc"], state["box"] = tc, box

                @tc.on_update
                def _(_evt) -> None:
                    _publish_box()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                return
        for h in (state["tc"], state["box"]):
            try:
                h.visible = visible
            except Exception:  # noqa: BLE001
                pass

    def _publish_box() -> None:
        tc = state["tc"]
        if tc is None or mode.value != MODE_BOX or state["syncing"]:
            return
        sel = Selection(kind="box", box_center=tuple(float(x) for x in tc.position),
                        box_size=tuple(float(x) for x in box_size.value),
                        box_quat_wxyz=tuple(float(x) for x in tc.wxyz))
        sel.label = box_label.value.strip() or "region"  # consumed by pipeline.selection_box_json
        state["syncing"] = True
        try:
            ctx.set_selection(sel)
        finally:
            state["syncing"] = False
        _update_sel_md()

    @box_size.on_update
    def _(_evt) -> None:
        if state["box"] is not None:
            try:
                state["box"].dimensions = tuple(box_size.value)
            except Exception:  # noqa: BLE001
                pass
        _publish_box()

    @mode.on_update
    def _(_evt) -> None:
        _ensure_box(mode.value == MODE_BOX)
        if mode.value == MODE_BOX:
            _publish_box()
        _update_sel_md()
        _update_backend_note()

    @center_btn.on_click
    def _(_evt) -> None:
        scene = ctx.scene
        ids = ctx.selection.object_ids if ctx.selection.kind == "object" else []
        if scene is None or not ids or state["tc"] is None:
            ctx.set_status("select an object first (Scene/Generate tab) and switch to 3D box mode")
            return
        rec = scene.objects.get(f"obj_{ids[0]:02d}")
        if rec is None or rec.T_world is None:
            ctx.set_status("selected object has no world pose")
            return
        T = rec.T_world
        try:
            state["tc"].position = tuple(float(x) for x in T[:3, 3])
            if rec.world_dims:
                box_size.value = tuple(float(d) * 1.3 + 0.04 for d in rec.world_dims)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        _publish_box()

    # --------------------------------------------------------------- status
    def _current_selection():
        if mode.value == MODE_BOX:
            if state["tc"] is None:
                _ensure_box(True)
            tc = state["tc"]
            if tc is None:
                raise RuntimeError("box gizmo unavailable")
            sel = Selection(kind="box", box_center=tuple(float(x) for x in tc.position),
                            box_size=tuple(float(x) for x in box_size.value),
                            box_quat_wxyz=tuple(float(x) for x in tc.wxyz))
            sel.label = box_label.value.strip() or "region"
            return sel
        sel = ctx.selection
        if sel.kind != "object" or not sel.object_ids:
            raise RuntimeError("select one or more objects first")
        return sel

    def _update_sel_md() -> None:
        try:
            sel = _current_selection()
        except Exception as exc:  # noqa: BLE001
            sel_md.content = f"_selection: none ({exc})_"
            return
        if sel.kind == "box":
            c = ", ".join(f"{x:.2f}" for x in sel.box_center)
            s = ", ".join(f"{x:.2f}" for x in sel.box_size)
            sel_md.content = f"selection: **box** center ({c}) size ({s}) label `{sel.label}`"
        else:
            sel_md.content = "selection: " + ", ".join(f"obj_{i:02d}" for i in sel.object_ids)

    def _update_backend_note() -> None:
        b = backends.get(backend_dd.value)
        if b is None:
            return
        ok = qwen_allowed(ctx, b.id, b.env or "sam3", b.id != "lama")
        gb = ctx.gpu.memory_mib // 1024 if ctx.gpu.present else 0
        backend_note.content = ("" if ok else
                                f"**Qwen-Image-Edit needs ≥ 40 GB; local GPU has {gb} GB** — "
                                "pick LaMa or enable 'Force remote' in the Jobs tab")

    backend_dd.on_update(lambda _e: _update_backend_note())

    def _on_selection(sel) -> None:
        if state["syncing"]:
            return
        if sel is not None and sel.kind == "box" and mode.value != MODE_BOX:
            mode.value = MODE_BOX
            _ensure_box(True)
        _update_sel_md()

    ctx.events.subscribe("selection.changed", _on_selection)

    # ------------------------------------------------------------- versions
    def _refresh_versions() -> None:
        scene = ctx.scene
        vmap.clear()
        vmap[_BEFORE] = None
        if scene is not None:
            inp = Path(scene.result_set.out_dir) / "inpaint"
            if (inp / pipeline.DEFAULT_CLEAN_BG).exists():
                vmap[_CURRENT] = inp / pipeline.DEFAULT_CLEAN_BG
            for rec in read_versions(scene.result_set.out_dir):
                vmap[version_label(rec)] = rec["path"]
            info.content = (f"**{scene.result_set.name}** · {len(vmap) - 1} inpaint version(s) · "
                            f"out `{scene.result_set.out_dir}`")
        else:
            info.content = "_no scene loaded_"
        try:
            cur = versions_dd.value
            versions_dd.options = list(vmap)
            versions_dd.value = cur if cur in vmap else (list(vmap)[1] if len(vmap) > 1 else _BEFORE)
        except Exception:  # noqa: BLE001
            pass
        _publish_version()

    def _publish_version() -> None:
        path = vmap.get(versions_dd.value) if after.value else None
        ver_md.content = f"showing: `{path}`" if path else "showing: original splat"
        ctx.events.publish("inpaint.version_selected", str(path) if path else None)

    after.on_update(lambda _e: _publish_version())
    versions_dd.on_update(lambda _e: (_publish_version() if after.value else None))
    refresh_btn.on_click(lambda _e: _refresh_versions())
    ctx.events.subscribe("scene.loaded", lambda _s: _refresh_versions())

    # ------------------------------------------------------------------ run
    @run_btn.on_click
    def _(_evt) -> None:
        try:
            sctx = stage_context(ctx)
            sel = _current_selection()
            b = backends[backend_dd.value]
            if not qwen_allowed(ctx, b.id, b.env or "sam3", b.id != "lama"):
                raise RuntimeError("Qwen backend not allowed on this GPU locally; use LaMa or force remote")
            specs = pipeline.inpaint(sctx, sel, prompt.value, b.id, int(iters.value),
                                     negative_prompt=neg_prompt.value)
            ids = list(sel.object_ids) if sel.kind == "object" else []

            def _on_done(job) -> None:
                ctx.set_status(f"{job.spec.name}: {job.state.value}" + (f" ({job.error})" if job.error else ""))
                if job.state == JobState.SUCCEEDED and job.spec.tags.get("step") == "fill":
                    _refresh_versions()
                    try:
                        after.value = True
                    except Exception:  # noqa: BLE001
                        pass
                    _publish_version()
                    ctx.events.publish("scene.objects_changed", [f"obj_{i:02d}" for i in ids] or ["background"])

            for s in specs:
                s.on_done = _on_done
            jobs = ctx.jobs.submit_chain(specs)
            ctx.set_status(f"inpainting submitted ({len(jobs)} stages, backend {b.id})")
        except Exception as exc:  # noqa: BLE001
            ctx.set_status(f"inpaint: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    _update_backend_note()
    _update_sel_md()
    _refresh_versions()
