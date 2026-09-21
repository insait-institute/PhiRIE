"""Command builders for every pipeline stage, parameterised by model/mode choices.

Each builder returns a JobSpec (physicalview.jobs) and never executes anything, so
it is unit-testable on CPU. All stage modules are invoked exactly as run/run_factory.sh,
run/run_auto.sh and run/run_inpaint.sh invoke them (same module paths, same interpreter
routing, `python -m <module>` from the repo root with PYTHONPATH=repo_root), with the
scene selected through SIMANY_SCENE / SIMANY_OUT (and SIMANY_AUTO=1 for automatic
discovery result sets) because agents.core.common binds those at import time.

CONTRACT (implemented by the jobs/pipeline agent):

    ctx = StageContext(config, scene_id, out_dir, auto=bool, scene_dir=Path, images_dir=Path)

    discover(ctx, mode)                     mode in {"gt_segments","sam3_auto"} -> list[JobSpec]
                                            (auto: auto_segment [sam3] -> factory_prepare [main]
                                             -> factory_refine_masks [sam3]; gt: factory_prepare
                                             -> factory_refine_masks)
    generate(ctx, model_id, object_ids)     model_id from config.generation; object_ids
                                            list[int] or None for all. trellis has no
                                            --objects flag today: when object_ids is given,
                                            pass SIMANY_OBJECTS="0,3,7" in env AND add the
                                            flag once models/s4_trellis.py supports it (the
                                            agent adds `--objects` to s4_trellis.py, default
                                            all, backwards compatible). reconviagen/sam3d/hybrid
                                            already accept --objects.
    register(ctx, mode_id, object_ids)      mode_id from config.registration -> JobSpec
                                            (factory_align; the agent adds `--objects` and
                                            `--source-up {default,signed,alternative}` flags
                                            mapping to s5_align.align_object /
                                            align_object_with_signed_source_up /
                                            align_object_with_alternative_source_up,
                                            backwards compatible, GT-free when no GT files)
    physics(ctx, object_ids)                agents.assets.s6_physics [main]
    report(ctx)                             agents.eval.factory_report [main]
    export_mjcf(ctx, collision_mode, test)  robo.sim.export_mjcf [--test] --collision-mode X
    tasks(ctx, max_tasks)                   robo.tasks.pi05_tasks --out-dir <out_dir>
    inpaint(ctx, selection, prompt, backend_id, refine_iters)
                                            -> list[JobSpec]: inpaint_prepare [main]
                                            (with --objects for object selections, or
                                            --region-box json for a 3D box selection),
                                            inpaint_masks [sam3], inpaint_qwen [sam3|main]
                                            with --prompt "<text>" --backend qwen|lama
                                            --objects/--region, inpaint_fill [gsplat]
                                            --objects/--region --out-name clean_background.ply
                                            (writes also inpaint/versions/<utc>_clean_background.ply)
    policy_server(config, policy_id, port, gpu_type) -> JobSpec running run/pi05_serve.sh
                                            with SIMANY_PI05_CKPT/SIMANY_PI05_CONFIG per
                                            configs/policies/<id>.yaml, where="slurm"
    full_pipeline(ctx, discovery, generation, registration, collision_mode) -> list[JobSpec]
                                            ordered chain (each JobSpec.tags["chain"]=name;
                                            JobManager runs a chain sequentially, stopping on
                                            first failure).

Selection model shared with the UI (dataclass Selection in app.py):
    kind: "none" | "object" | "box"
    object_ids: list[int]
    box_center: (x,y,z) | None ; box_size: (sx,sy,sz) | None ; box_quat_wxyz | None
    camera_frame: str | None

The agent must extend these stage scripts minimally and with tests:
    models/s4_trellis.py            --objects
    agents/assets/factory_align.py  --objects, --source-up
    agents/edit/inpaint_prepare.py  --objects, --region-box (JSON string or file)
    agents/edit/inpaint_qwen.py     --prompt, --negative-prompt, --backend, --objects, --region
    agents/edit/inpaint_fill.py     --objects, --region, --out-name, version copy
Defaults must reproduce today's behaviour byte-for-byte when no new flag is given.

Build-time validation: every builder checks that the stage module file exists under
config.repo_root and that the interpreter for its env key exists (FileNotFoundError naming
the key/path); unknown model/mode ids raise KeyError (from StudioConfig.choice).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from physicalview.config import StudioConfig
from physicalview.jobs import JobSpec

log = logging.getLogger("studio.pipeline")

REGION_NAME = "region_00"          # inpaint_prepare's fixed name for a 3D-box removal set
DEFAULT_CLEAN_BG = "clean_background.ply"
OPENPI_ASSETS_MARK = "openpi-assets-simeval/"


@dataclass
class StageContext:
    config: StudioConfig
    scene_id: str
    out_dir: Path
    auto: bool = False
    scene_dir: Path | None = None
    images_dir: Path | None = None

    def base_env(self) -> dict[str, str]:
        env = {"SIMANY_SCENE": self.scene_id, "SIMANY_OUT": str(self.out_dir),
               "PYTHONPATH": str(self.config.repo_root),
               "SIMANY_SCANNETPP_ROOT": str(self.config.scannetpp_root),
               "SIMANY_SPLATS_ROOT": str(self.config.splats_root)}
        if self.auto:
            env["SIMANY_AUTO"] = "1"
        if self.scene_dir is not None:
            env["SIMANY_SCENE_DIR"] = str(self.scene_dir)
        return env

    def resolved_images_dir(self) -> Path:
        if self.images_dir is not None:
            return Path(self.images_dir)
        if self.scene_dir is not None:
            return Path(self.scene_dir) / "dslr" / "resized_undistorted_images"
        return self.config.scannetpp_root / "data" / self.scene_id / "dslr" / "resized_undistorted_images"

    def resolved_scene_dir(self) -> Path:
        if self.scene_dir is not None:
            return Path(self.scene_dir)
        return self.config.scannetpp_root / "data" / self.scene_id


# --------------------------------------------------------------------------- helpers
def _csv(ids: list[int] | None) -> str | None:
    if ids is None:
        return None
    return ",".join(str(int(i)) for i in ids)


def _module_path(config: StudioConfig, module: str) -> Path:
    return config.repo_root.joinpath(*module.split(".")).with_suffix(".py")


def _check_module(config: StudioConfig, module: str) -> None:
    p = _module_path(config, module)
    if not p.is_file():
        raise FileNotFoundError(f"stage module {module!r} not found at {p}")


STRICT_INTERPRETERS = False   # True -> a missing interpreter raises at build time


def interpreter_missing(config: StudioConfig, env_key: str) -> bool:
    """True when the env's interpreter path is absent on THIS node (e.g. an unmounted
    shared filesystem); the job may still run remotely, so builders only tag it."""
    return not Path(config.interpreter(env_key)).exists()


def _interpreter(config: StudioConfig, env_key: str) -> Path:
    path = Path(config.interpreter(env_key))       # KeyError for unknown key
    if not path.exists():
        msg = f"interpreter {env_key!r} not found at {path}"
        if STRICT_INTERPRETERS:
            raise FileNotFoundError(msg)
        log.warning("%s (job will be tagged interpreter_missing and may only run remotely)", msg)
    return path


def _stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _spec(ctx: StageContext, name: str, env_key: str, module: str, args: list[str],
          needs_gpu: bool, artifacts: list[Path] | None = None,
          tags: dict[str, str] | None = None, extra_env: dict[str, str] | None = None,
          where: str = "auto") -> JobSpec:
    _check_module(ctx.config, module)
    interp = _interpreter(ctx.config, env_key)
    env = ctx.base_env()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    t = {"scene": ctx.scene_id, "out_dir": str(ctx.out_dir), "module": module,
         "env_key": env_key}
    if interpreter_missing(ctx.config, env_key):
        t["interpreter_missing"] = "1"
    if tags:
        t.update({k: str(v) for k, v in tags.items() if v is not None})
    return JobSpec(name=name, argv=[str(interp), "-m", module, *[str(a) for a in args]],
                   env=env, cwd=ctx.config.repo_root, env_key=env_key, needs_gpu=needs_gpu,
                   where=where, artifacts=[Path(a) for a in (artifacts or [])], tags=t)


def _known_object_ids(ctx: StageContext) -> list[int] | None:
    """Indices from <out_dir>/objects/objects.json when it exists (else None)."""
    p = ctx.out_dir / "objects" / "objects.json"
    if not p.is_file():
        return None
    try:
        return [int(m["index"]) for m in json.loads(p.read_text())]
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _obj_dir(ctx: StageContext, idx: int) -> Path:
    return ctx.out_dir / "objects" / f"obj_{int(idx):02d}"


# ------------------------------------------------------------------------- discover
def discover(ctx: StageContext, mode: str) -> list[JobSpec]:
    """gt_segments: factory_prepare -> factory_refine_masks; sam3_auto: auto_segment first."""
    choice = ctx.config.choice("discovery", mode)
    auto = choice.stage == "discover_auto" or mode == "sam3_auto"
    dctx = replace(ctx, auto=auto)
    chain = f"discover:{mode}:{ctx.scene_id}:{_stamp()}"
    tags = {"stage": "discover", "model": mode, "chain": chain}
    images = dctx.resolved_images_dir()
    specs: list[JobSpec] = []
    if auto:
        specs.append(_spec(dctx, f"discover:auto_segment {ctx.scene_id}", "sam3",
                           "agents.discover.auto_segment",
                           ["--scene-dir", str(dctx.resolved_scene_dir()), "--out-dir", str(ctx.out_dir)],
                           needs_gpu=True, artifacts=[ctx.out_dir / "auto_instances.npz"], tags=tags))
    specs.append(_spec(dctx, f"discover:factory_prepare {ctx.scene_id}", "main",
                       "agents.discover.factory_prepare", [], needs_gpu=False,
                       artifacts=[ctx.out_dir / "objects" / "objects.json"], tags=tags))
    specs.append(_spec(dctx, f"discover:refine_masks {ctx.scene_id}", "sam3",
                       "agents.discover.factory_refine_masks",
                       ["--images-dir", str(images), "--out-dir", str(ctx.out_dir)],
                       needs_gpu=True, tags=tags))
    return specs


# ------------------------------------------------------------------------- generate
_GEN_ARTIFACT = {
    "trellis": lambda d: d / "trellis_gs.ply",
    "reconviagen": lambda d: d / "rvg" / "rvg_gs.ply",
    "sam3d": lambda d: d / "sam3d" / "sam3d_gs.ply",
    "hybrid": lambda d: d / "hybrid.json",
}


def generate(ctx: StageContext, model_id: str, object_ids: list[int] | None) -> JobSpec:
    choice = ctx.config.choice("generation", model_id)
    if not choice.module or not choice.env:
        raise ValueError(f"generation choice {model_id!r} has no module/env in the config")
    ids = list(object_ids) if object_ids is not None else None
    args: list[str] = list(choice.args)
    env: dict[str, str] = {}
    if model_id == "reconviagen" and ids is None:
        ids = _known_object_ids(ctx)
        if ids is None:
            raise ValueError("reconviagen requires --objects: pass object_ids or run discovery "
                             f"first ({ctx.out_dir / 'objects' / 'objects.json'} missing)")
    csv = _csv(ids)
    if csv is not None:
        args += ["--objects", csv]
        if model_id == "trellis":
            env["SIMANY_OBJECTS"] = csv
    art_fn = _GEN_ARTIFACT.get(model_id)
    arts = [art_fn(_obj_dir(ctx, i)) for i in ids] if (ids is not None and art_fn) else []
    label = f"generate:{model_id} " + (f"obj {csv}" if csv else "all")
    return _spec(ctx, label, choice.env, choice.module, args, needs_gpu=True, artifacts=arts,
                 tags={"stage": "generate", "model": model_id, "object_ids": csv or "all"},
                 extra_env=env)


# ------------------------------------------------------------------------- register
def register(ctx: StageContext, mode_id: str, object_ids: list[int] | None) -> JobSpec:
    choice = ctx.config.choice("registration", mode_id)
    module = choice.module or "agents.assets.factory_align"
    env_key = choice.env or "main"
    args = list(choice.args)
    csv = _csv(list(object_ids) if object_ids is not None else None)
    if csv is not None:
        args += ["--objects", csv]
    arts = [_obj_dir(ctx, i) / "aligned.json" for i in object_ids] if object_ids else []
    label = f"register:{mode_id} " + (f"obj {csv}" if csv else "all")
    return _spec(ctx, label, env_key, module, args, needs_gpu=False, artifacts=arts,
                 tags={"stage": "register", "model": mode_id, "object_ids": csv or "all"})


# -------------------------------------------------------------------------- physics
def physics(ctx: StageContext, object_ids: list[int] | None) -> JobSpec:
    """s6_physics annotates every non-rejected object (no per-object flag in the script);
    SIMANY_OBJECTS is exported for forward compatibility and recorded in the tags."""
    csv = _csv(list(object_ids) if object_ids is not None else None)
    env = {"SIMANY_OBJECTS": csv} if csv else {}
    arts = [_obj_dir(ctx, i) / "physics.json" for i in object_ids] if object_ids else []
    return _spec(ctx, "physics " + (f"obj {csv}" if csv else "all"), "main",
                 "agents.assets.s6_physics", [], needs_gpu=True, artifacts=arts,
                 tags={"stage": "physics", "object_ids": csv or "all"}, extra_env=env)


# --------------------------------------------------------------------------- report
def report(ctx: StageContext) -> JobSpec:
    return _spec(ctx, f"report {ctx.scene_id}", "main", "agents.eval.factory_report", [],
                 needs_gpu=False, artifacts=[ctx.out_dir / "report.json"], tags={"stage": "report"})


# ---------------------------------------------------------------------- export_mjcf
def export_mjcf(ctx: StageContext, collision_mode: str = "room", test: bool = True) -> JobSpec:
    if collision_mode not in ctx.config.collision_modes:
        raise ValueError(f"collision mode {collision_mode!r} not in {ctx.config.collision_modes}")
    args = (["--test"] if test else []) + ["--collision-mode", collision_mode]
    return _spec(ctx, f"export_mjcf {collision_mode}", "main", "robo.sim.export_mjcf", args,
                 needs_gpu=False, artifacts=[ctx.out_dir / "sim_export" / "scene.xml"],
                 tags={"stage": "export_mjcf", "collision_mode": collision_mode, "test": str(test)})


# ---------------------------------------------------------------------------- tasks
def tasks(ctx: StageContext, max_tasks: int = 10) -> JobSpec:
    args = ["--out-dir", str(ctx.out_dir), "--max-tasks", str(int(max_tasks))]
    return _spec(ctx, f"tasks {ctx.scene_id}", "main", "robo.tasks.pi05_tasks", args,
                 needs_gpu=False, artifacts=[ctx.out_dir / "sim_export" / "pi05_tasks.json"],
                 tags={"stage": "tasks", "max_tasks": str(max_tasks)})


# -------------------------------------------------------------------------- inpaint
def selection_box_json(selection) -> str:
    """Serialize a box Selection for inpaint_prepare --region-box."""
    if selection.box_center is None or selection.box_size is None:
        raise ValueError("box selection needs box_center and box_size")
    quat = selection.box_quat_wxyz or (1.0, 0.0, 0.0, 0.0)
    box = {"center": [float(x) for x in selection.box_center],
           "size": [float(x) for x in selection.box_size],
           "quat_wxyz": [float(x) for x in quat]}
    label = getattr(selection, "label", None)
    if label:
        box["label"] = str(label)
    return json.dumps(box, separators=(",", ":"))


def inpaint(ctx: StageContext, selection, prompt: str, backend_id: str,
            refine_iters: int | None = None, negative_prompt: str | None = None) -> list[JobSpec]:
    backend = ctx.config.choice("inpaint_backends", backend_id)
    if backend_id == "lama" or backend.id == "lama":
        backend_flag, qwen_env, qwen_gpu = "lama", backend.env or "main", False
    else:
        backend_flag, qwen_env, qwen_gpu = "qwen", backend.env or "sam3", True

    kind = getattr(selection, "kind", "none")
    if kind == "object":
        ids = [int(i) for i in selection.object_ids]
        if not ids:
            raise ValueError("object selection has no object ids")
        csv = _csv(ids)
        prep_args = ["--objects", csv]
        sel_args = ["--objects", csv]
        sel_desc = f"obj {csv}"
        prep_arts = [ctx.out_dir / "inpaint" / "prepare_meta.json"]
    elif kind == "box":
        prep_args = ["--region-box", selection_box_json(selection)]
        sel_args = ["--region", REGION_NAME]
        sel_desc = REGION_NAME
        prep_arts = [ctx.out_dir / "inpaint" / REGION_NAME / "removal_idx.npy"]
    else:
        raise ValueError(f"inpaint needs an object or box selection, got kind={kind!r}")

    stamp = _stamp()
    chain = f"inpaint:{ctx.scene_id}:{stamp}"
    common = {"stage": "inpaint", "chain": chain, "selection": sel_desc, "backend": backend_id}
    images = ctx.resolved_images_dir()
    inp = ctx.out_dir / "inpaint"

    edit_args = ["--backend", backend_flag, "--prompt", prompt, *sel_args]
    if negative_prompt is not None:
        edit_args += ["--negative-prompt", negative_prompt]
    fill_args = [*sel_args, "--out-name", DEFAULT_CLEAN_BG]
    if refine_iters is not None:
        fill_args += ["--iters", str(int(refine_iters))]

    return [
        _spec(ctx, f"inpaint:prepare {sel_desc}", "main", "agents.edit.inpaint_prepare",
              prep_args, needs_gpu=False, artifacts=prep_arts, tags={**common, "step": "prepare"}),
        _spec(ctx, f"inpaint:masks {sel_desc}", "sam3", "agents.edit.inpaint_masks",
              ["--images-dir", str(images), "--out-dir", str(ctx.out_dir), *sel_args],
              needs_gpu=True, tags={**common, "step": "masks"}),
        _spec(ctx, f"inpaint:edit[{backend_flag}] {sel_desc}", qwen_env, "agents.edit.inpaint_qwen",
              edit_args, needs_gpu=qwen_gpu, tags={**common, "step": "edit"}),
        _spec(ctx, f"inpaint:fill {sel_desc}", "gsplat", "agents.edit.inpaint_fill", fill_args,
              needs_gpu=True, artifacts=[inp / DEFAULT_CLEAN_BG], tags={**common, "step": "fill"}),
    ]


# -------------------------------------------------------------------- policy server
def load_policy_entry(config: StudioConfig, policy_id: str) -> dict:
    pdir = config.repo_root / "configs" / "policies"
    for yml in sorted(pdir.glob("*.yaml")):
        data = yaml.safe_load(yml.read_text()) or {}
        for entry in data.get("policies") or []:
            if entry.get("id") == policy_id:
                return dict(entry)
    raise KeyError(f"policy {policy_id!r} not found in {pdir}/*.yaml")


def policy_server(config: StudioConfig, policy_id: str, port: int,
                  gpu_type: str | None = None) -> JobSpec:
    entry = load_policy_entry(config, policy_id)
    if entry.get("client_kind") != "pi05_server":
        raise ValueError(f"policy {policy_id!r} (client_kind={entry.get('client_kind')!r}) "
                         "does not use a policy server")
    ckpt = entry.get("checkpoint_path")
    if not ckpt:
        raise ValueError(f"policy {policy_id!r} has no checkpoint_path")
    ckpt = str(ckpt)
    if OPENPI_ASSETS_MARK in ckpt:
        ckpt_sub = ckpt.split(OPENPI_ASSETS_MARK, 1)[1].strip("/")
    else:
        ckpt_sub = ckpt.rstrip("/").rsplit("/", 1)[-1]
    training_config = str(entry.get("training_config") or "pi05_droid_jointpos")
    script = config.policy_server_script
    if not Path(script).is_file():
        raise FileNotFoundError(f"policy server script not found: {script}")
    _interpreter(config, "openpi")
    gpu = gpu_type or (config.policy_server_gpu_types[0] if config.policy_server_gpu_types
                       else config.default_remote_gpu)
    if gpu not in config.gpu_targets:
        raise ValueError(f"unknown GPU type {gpu!r} for policy server; known: {sorted(config.gpu_targets)}")
    env = {"SIMANY_PI05_CKPT": ckpt_sub, "SIMANY_PI05_CONFIG": training_config,
           "PYTHONPATH": str(config.repo_root)}
    return JobSpec(name=f"policy_server:{policy_id} :{port}",
                   argv=["bash", str(script), "--port", str(int(port))], env=env,
                   cwd=config.repo_root, env_key="openpi", needs_gpu=True, where="slurm",
                   gpu_type=gpu,
                   tags={"kind": "policy_server", "policy": policy_id, "port": str(int(port)),
                         "slurm_mem": "100G", "ckpt": ckpt_sub, "training_config": training_config})


# -------------------------------------------------------------------- full pipeline
def full_pipeline(ctx: StageContext, discovery: str, generation: str, registration: str,
                  collision_mode: str = "room") -> list[JobSpec]:
    dchoice = ctx.config.choice("discovery", discovery)
    auto = dchoice.stage == "discover_auto" or discovery == "sam3_auto"
    pctx = replace(ctx, auto=auto)
    chain = f"full:{ctx.scene_id}:{_stamp()}"
    specs = list(discover(pctx, discovery))
    specs += [generate(pctx, generation, None), register(pctx, registration, None),
              physics(pctx, None), report(pctx), export_mjcf(pctx, collision_mode, test=True),
              tasks(pctx)]
    for s in specs:
        s.tags["chain"] = chain
        s.tags["pipeline"] = "full"
    return specs
