"""Generate tab — see docs/ARCHITECTURE.md for the required controls.

CONTRACT: build(ctx) creates the tab's GUI and wires events:
  * discovery model dropdown + "Run discovery" (whole scene)
  * generation model dropdown, registration mode dropdown, collision mode dropdown
  * object multi-select (checkbox per object from ctx.scene.objects, kept in sync
    with ctx.selection both ways)
  * buttons: Generate selected, Register selected, Physics selected, Report,
    Export MJCF, Generate tasks, Run full pipeline
  * proposals section for the first selected object: one markdown card per
    proposal (source, chamfer / sym_chamfer / scale / size_ratio from the evidence,
    eval-only F1 labelled 'eval-only', gen_seconds, chosen marker) + "Accept
    proposal <source>" which re-materializes that source via
    agents.assets.factory_hybrid (SIMANY_HYBRID_FORCE=<source>) followed by physics.
Every action submits JobSpecs through ctx.jobs (pipeline.py builders); when a job
finishes, "scene.objects_changed" is published with the affected object ids.
viser is imported lazily inside build().
"""
from __future__ import annotations

import threading
import traceback
from pathlib import Path

from physicalview import pipeline
from physicalview.jobs import JobSpec, JobState

_ALL = "all"


# ----------------------------------------------------------------- pure helpers
def stage_context(ctx) -> pipeline.StageContext:
    """StageContext from the loaded scene's ResultSet (raises if no scene)."""
    scene = ctx.scene
    if scene is None:
        raise RuntimeError("load a scene in the Scene tab first")
    rs = scene.result_set
    scene_dir = Path(rs.scene_dir) if rs.scene_dir else None
    images = scene_dir / "dslr" / "resized_undistorted_images" if scene_dir else None
    return pipeline.StageContext(ctx.config, rs.scene_id, Path(rs.out_dir),
                                 auto=(rs.kind == "auto"), scene_dir=scene_dir, images_dir=images)


def object_id_from_name(name: str) -> int:
    return int(str(name).rsplit("_", 1)[-1])


def _fmt(v, nd=3, scale=1.0, unit=""):
    if v is None:
        return "-"
    try:
        return f"{float(v) * scale:.{nd}f}{unit}"
    except (TypeError, ValueError):
        return str(v)


def proposal_cards(record) -> str:
    """Markdown cards for an ObjectRecord's proposals (numbers read from artifacts)."""
    if record is None:
        return "_select an object to see its proposals_"
    props = getattr(record, "proposals", {}) or {}
    if not props:
        return f"**{record.name}** ({record.label}): _no proposals on disk_"
    chosen = getattr(record, "chosen_source", None)
    out = [f"**{record.name}** — {record.label} — chosen source: `{chosen or '?'}`", ""]
    for src, p in props.items():
        ev = getattr(p, "evidence", {}) or {}
        mark = " ✅ chosen" if src == chosen else ""
        eval_only = getattr(p, "eval_only_fields", ("f1_20", "f1_40"))
        lines = [f"**{src}**{mark}",
                 f"- sym_chamfer: {_fmt(ev.get('sym_chamfer_m'), 1, 1000, ' mm')} · "
                 f"chamfer_med: {_fmt(ev.get('chamfer_med_m'), 1, 1000, ' mm')} · "
                 f"scale: {_fmt(ev.get('scale'))} · size_ratio: {_fmt(ev.get('size_ratio'), 2)}",
                 f"- gen_seconds: {_fmt(ev.get('gen_seconds'), 0)} · tier: {ev.get('tier', '-')}"
                 + (f" · rejected: {ev['rejected']}" if ev.get("rejected") else "")]
        f1s = [f"{k}={_fmt(ev.get(k))}" for k in eval_only if ev.get(k) is not None]
        if f1s:
            lines.append("- eval-only (GT F1, never used for selection): " + ", ".join(f1s))
        if getattr(p, "mesh_ply", None):
            lines.append(f"- mesh: `{Path(p.mesh_ply).name}`" +
                         (f" · gs: `{Path(p.gs_ply).name}`" if getattr(p, "gs_ply", None) else ""))
        out.append("\n".join(lines))
        out.append("")
    return "\n".join(out)


def accept_proposal_specs(sctx: pipeline.StageContext, obj_idx: int, source: str,
                          available: list[str]) -> list[JobSpec]:
    """factory_hybrid with SIMANY_HYBRID_FORCE=<source> (candidates = sources with
    assets on disk) followed by physics, as one chain."""
    source = source.lower()
    cands = ["trellis"] + [s for s in ("rvg", "sam3d") if s in available or s == source]
    hy = pipeline.generate(sctx, "hybrid", [obj_idx])
    hy.name = f"accept:{source} obj_{obj_idx:02d}"
    hy.env["SIMANY_HYBRID_FORCE"] = source
    hy.env["SIMANY_HYBRID_CANDIDATES"] = ",".join(dict.fromkeys(cands))
    hy.tags.update({"stage": "accept_proposal", "source": source})
    ph = pipeline.physics(sctx, [obj_idx])
    chain = f"accept:{sctx.scene_id}:obj_{obj_idx:02d}:{pipeline._stamp()}"
    for s in (hy, ph):
        s.tags["chain"] = chain
    return [hy, ph]


# ------------------------------------------------------------------------ build
def build(ctx) -> None:
    import viser  # noqa: F401
    from physicalview.app import Selection

    gui = ctx.server.gui
    config = ctx.config
    state = {"boxes": {}, "folder": None, "syncing": False, "props_src": None}

    def _labels(choices):
        return {c.label: c.id for c in choices}

    disc_map = _labels(config.discovery)
    gen_map = _labels(config.generation)
    reg_map = _labels(config.registration)

    info = gui.add_markdown("_no scene loaded_")
    with gui.add_folder("Discovery"):
        disc_dd = gui.add_dropdown("Discovery model", options=list(disc_map) or ["-"])
        disc_btn = gui.add_button("Run discovery (whole scene)")
    with gui.add_folder("Objects"):
        sel_md = gui.add_markdown("_selection: none_")
        all_btn = gui.add_button("Select all")
        none_btn = gui.add_button("Clear selection")
    with gui.add_folder("Generate / register / physics"):
        gen_dd = gui.add_dropdown("Generation model", options=list(gen_map) or ["-"])
        reg_dd = gui.add_dropdown("Registration mode", options=list(reg_map) or ["-"])
        gen_btn = gui.add_button("Generate selected")
        reg_btn = gui.add_button("Register selected")
        phys_btn = gui.add_button("Physics selected")
    with gui.add_folder("Scene outputs"):
        coll_dd = gui.add_dropdown("Collision mode", options=list(config.collision_modes))
        max_tasks = gui.add_number("Max tasks", 10, min=1, max=50, step=1)
        report_btn = gui.add_button("Report (drop test + yield)")
        mjcf_btn = gui.add_button("Export MJCF (+ settle test)")
        tasks_btn = gui.add_button("Generate tasks")
        full_btn = gui.add_button("Run full pipeline", color="green")
    with gui.add_folder("Proposals (selected object)"):
        props_md = gui.add_markdown(proposal_cards(None))
        src_dd = gui.add_dropdown("Proposal source", options=["trellis", "rvg", "sam3d"])
        accept_btn = gui.add_button("Accept proposal")

    # ------------------------------------------------------------- selection
    def _selected_ids() -> list[int]:
        return sorted(i for i, cb in state["boxes"].items() if cb.value)

    def _update_sel_md() -> None:
        ids = _selected_ids()
        sel_md.content = ("_selection: none_" if not ids else
                          "selection: " + ", ".join(f"obj_{i:02d}" for i in ids))

    def _push_selection() -> None:
        if state["syncing"]:
            return
        ids = _selected_ids()
        state["syncing"] = True
        try:
            ctx.set_selection(Selection(kind="object" if ids else "none", object_ids=ids))
        finally:
            state["syncing"] = False
        _update_sel_md()
        _refresh_props()

    def _rebuild_objects() -> None:
        if state["folder"] is not None:
            try:
                state["folder"].remove()
            except Exception:  # noqa: BLE001
                pass
            state["folder"] = None
        state["boxes"] = {}
        scene = ctx.scene
        if scene is None:
            info.content = "_no scene loaded_"
            return
        rs = scene.result_set
        info.content = (f"**{rs.name}** · scene `{rs.scene_id}` · kind `{rs.kind}` · "
                        f"{len(scene.objects)} objects · out `{rs.out_dir}`")
        folder = gui.add_folder("Object list", expand_by_default=False)
        state["folder"] = folder
        with folder:
            for name, rec in sorted(scene.objects.items()):
                idx = getattr(rec, "index", object_id_from_name(name))
                tag = "" if getattr(rec, "accepted", True) else " (rejected)"
                src = getattr(rec, "chosen_source", None)
                cb = gui.add_checkbox(f"{name} {rec.label}{tag}" + (f" [{src}]" if src else ""), False)
                state["boxes"][idx] = cb

                @cb.on_update
                def _(_evt) -> None:
                    _push_selection()
        _apply_selection(ctx.selection)

    def _apply_selection(sel) -> None:
        if state["syncing"]:
            return
        state["syncing"] = True
        try:
            ids = set(sel.object_ids) if sel is not None and sel.kind == "object" else set()
            for idx, cb in state["boxes"].items():
                want = idx in ids
                if cb.value != want:
                    cb.value = want
        finally:
            state["syncing"] = False
        _update_sel_md()
        _refresh_props()

    def _refresh_props() -> None:
        scene = ctx.scene
        ids = _selected_ids()
        if scene is None or not ids:
            props_md.content = proposal_cards(None)
            return
        rec = scene.objects.get(f"obj_{ids[0]:02d}")
        props_md.content = proposal_cards(rec)
        avail = list((getattr(rec, "proposals", {}) or {}).keys()) if rec else []
        opts = [s for s in ("trellis", "rvg", "sam3d") if s in avail] or ["trellis", "rvg", "sam3d"]
        try:
            src_dd.options = opts
            if src_dd.value not in opts:
                src_dd.value = opts[0]
        except Exception:  # noqa: BLE001
            pass

    @all_btn.on_click
    def _(_evt) -> None:
        for cb in state["boxes"].values():
            cb.value = True
        _push_selection()

    @none_btn.on_click
    def _(_evt) -> None:
        for cb in state["boxes"].values():
            cb.value = False
        _push_selection()

    ctx.events.subscribe("scene.loaded", lambda _s: _rebuild_objects())
    ctx.events.subscribe("selection.changed", _apply_selection)
    ctx.events.subscribe("scene.objects_changed", lambda _ids: _refresh_props())

    # ---------------------------------------------------------------- submit
    def _done_publisher(ids: list[int] | None):
        def _on_done(job) -> None:
            if job.state == JobState.SUCCEEDED:
                affected = [f"obj_{i:02d}" for i in ids] if ids else (
                    list(ctx.scene.objects) if ctx.scene is not None else [])
                ctx.events.publish("scene.objects_changed", affected)
            ctx.set_status(f"{job.spec.name}: {job.state.value}" + (f" ({job.error})" if job.error else ""))
        return _on_done

    def _submit(specs, ids=None, chain=False) -> None:
        if not isinstance(specs, list):
            specs = [specs]
        for s in specs:
            s.on_done = _done_publisher(ids)
        try:
            jobs = ctx.jobs.submit_chain(specs) if (chain or len(specs) > 1) else [ctx.jobs.submit(specs[0])]
        except Exception as exc:  # noqa: BLE001
            ctx.set_status(f"submit failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return
        ctx.set_status(f"submitted {len(jobs)} job(s): " + ", ".join(j.spec.name for j in jobs))

    def _guarded(fn):
        def _run(_evt=None) -> None:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                ctx.set_status(f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
        return _run

    def _ids_or_none() -> list[int] | None:
        ids = _selected_ids()
        return ids or None

    def _ids_required() -> list[int]:
        ids = _selected_ids()
        if not ids:
            raise RuntimeError("select at least one object")
        return ids

    disc_btn.on_click(_guarded(lambda: _submit(
        pipeline.discover(stage_context(ctx), disc_map[disc_dd.value]), None, chain=True)))
    gen_btn.on_click(_guarded(lambda: _submit(
        pipeline.generate(stage_context(ctx), gen_map[gen_dd.value], _ids_required()), _ids_required())))
    reg_btn.on_click(_guarded(lambda: _submit(
        pipeline.register(stage_context(ctx), reg_map[reg_dd.value], _ids_required()), _ids_required())))
    phys_btn.on_click(_guarded(lambda: _submit(
        pipeline.physics(stage_context(ctx), _ids_or_none()), _ids_or_none())))
    report_btn.on_click(_guarded(lambda: _submit(pipeline.report(stage_context(ctx)))))
    mjcf_btn.on_click(_guarded(lambda: _submit(
        pipeline.export_mjcf(stage_context(ctx), coll_dd.value, test=True))))
    tasks_btn.on_click(_guarded(lambda: _submit(
        pipeline.tasks(stage_context(ctx), int(max_tasks.value)))))
    full_btn.on_click(_guarded(lambda: _submit(
        pipeline.full_pipeline(stage_context(ctx), disc_map[disc_dd.value], gen_map[gen_dd.value],
                               reg_map[reg_dd.value], coll_dd.value), None, chain=True)))

    def _accept() -> None:
        ids = _ids_required()
        scene = ctx.scene
        rec = scene.objects.get(f"obj_{ids[0]:02d}") if scene else None
        avail = list((getattr(rec, "proposals", {}) or {}).keys()) if rec else []
        specs = accept_proposal_specs(stage_context(ctx), ids[0], src_dd.value, avail)
        _submit(specs, [ids[0]], chain=True)

    accept_btn.on_click(_guarded(_accept))

    # initial state (scene may already be loaded when the tab is built)
    threading.Thread(target=_rebuild_objects, name="generate-panel-init", daemon=True).start()
