"""CPU-only tests for physicalview/pipeline.py builders (no execution).

Interpreters are redirected to sys.executable so the existence check passes on any
node; the scene is c50d2d1d42 with a temporary out_dir."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from physicalview import pipeline as P
from physicalview.app import Selection
from physicalview.config import load_config
from physicalview.jobs import JobSpec

PY = str(Path(sys.executable))
SCENE = "c50d2d1d42"
pytestmark = pytest.mark.backend


@pytest.fixture
def cfg():
    base = load_config()
    return replace(base, interpreters={k: Path(PY) for k in base.interpreters})


@pytest.fixture
def ctx(cfg, tmp_path):
    out = tmp_path / f"{SCENE}_factory"
    out.mkdir()
    scene_dir = cfg.scannetpp_root / "data" / SCENE
    return P.StageContext(cfg, SCENE, out, auto=False, scene_dir=scene_dir,
                          images_dir=scene_dir / "dslr" / "resized_undistorted_images")


def _module(spec: JobSpec) -> str:
    assert spec.argv[1] == "-m"
    return spec.argv[2]


def _flags(spec: JobSpec) -> list[str]:
    return spec.argv[3:]


def _check_common(spec: JobSpec, ctx, env_key: str):
    assert isinstance(spec, JobSpec)
    assert spec.argv[0] == PY
    assert spec.env_key == env_key
    assert spec.env["SIMANY_SCENE"] == SCENE
    assert spec.env["SIMANY_OUT"] == str(ctx.out_dir)
    assert spec.env["PYTHONPATH"] == str(ctx.config.repo_root)
    assert spec.cwd == ctx.config.repo_root
    assert (ctx.config.repo_root.joinpath(*_module(spec).split(".")).with_suffix(".py")).is_file()
    assert spec.tags["scene"] == SCENE


# ------------------------------------------------------------------ discover
def test_discover_gt(ctx):
    specs = P.discover(ctx, "gt_segments")
    assert [_module(s) for s in specs] == ["agents.discover.factory_prepare",
                                          "agents.discover.factory_refine_masks"]
    assert [s.env_key for s in specs] == ["main", "sam3"]
    assert [s.needs_gpu for s in specs] == [False, True]
    assert "SIMANY_AUTO" not in specs[0].env
    assert _flags(specs[1]) == ["--images-dir", str(ctx.images_dir), "--out-dir", str(ctx.out_dir)]
    assert len({s.tags["chain"] for s in specs}) == 1
    assert Path(str(ctx.out_dir / "objects" / "objects.json")) in specs[0].artifacts


def test_discover_auto(ctx):
    specs = P.discover(ctx, "sam3_auto")
    assert [_module(s) for s in specs] == ["agents.discover.auto_segment",
                                          "agents.discover.factory_prepare",
                                          "agents.discover.factory_refine_masks"]
    assert all(s.env["SIMANY_AUTO"] == "1" for s in specs)
    assert _flags(specs[0]) == ["--scene-dir", str(ctx.scene_dir), "--out-dir", str(ctx.out_dir)]
    assert specs[0].env_key == "sam3" and specs[0].needs_gpu


def test_discover_unknown_mode(ctx):
    with pytest.raises(KeyError):
        P.discover(ctx, "clip_magic")


# ------------------------------------------------------------------ generate
def test_generate_trellis_objects_env_and_flag(ctx):
    spec = P.generate(ctx, "trellis", [0, 3, 7])
    _check_common(spec, ctx, "main")
    assert _module(spec) == "models.s4_trellis"
    assert _flags(spec) == ["--objects", "0,3,7"]
    assert spec.env["SIMANY_OBJECTS"] == "0,3,7"
    assert spec.needs_gpu
    assert spec.tags["model"] == "trellis" and spec.tags["object_ids"] == "0,3,7"
    assert spec.artifacts == [ctx.out_dir / "objects" / f"obj_{i:02d}" / "trellis_gs.ply" for i in (0, 3, 7)]


def test_generate_trellis_all(ctx):
    spec = P.generate(ctx, "trellis", None)
    assert _flags(spec) == []
    assert "SIMANY_OBJECTS" not in spec.env
    assert spec.artifacts == []


def test_generate_reconviagen_requires_objects(ctx):
    with pytest.raises(ValueError):
        P.generate(ctx, "reconviagen", None)
    (ctx.out_dir / "objects").mkdir()
    (ctx.out_dir / "objects" / "objects.json").write_text(json.dumps([{"index": 1}, {"index": 4}]))
    spec = P.generate(ctx, "reconviagen", None)
    assert _flags(spec) == ["--objects", "1,4"]
    assert spec.artifacts[0] == ctx.out_dir / "objects" / "obj_01" / "rvg" / "rvg_gs.ply"


def test_generate_sam3d_and_hybrid(ctx):
    s = P.generate(ctx, "sam3d", [2])
    _check_common(s, ctx, "sam3d")
    assert _module(s) == "models.s4_sam3d" and _flags(s) == ["--objects", "2"]
    h = P.generate(ctx, "hybrid", [2, 5])
    _check_common(h, ctx, "main")
    assert _module(h) == "agents.assets.factory_hybrid" and _flags(h) == ["--objects", "2,5"]
    assert h.artifacts == [ctx.out_dir / "objects" / "obj_02" / "hybrid.json",
                           ctx.out_dir / "objects" / "obj_05" / "hybrid.json"]


def test_generate_unknown_model(ctx):
    with pytest.raises(KeyError):
        P.generate(ctx, "nerf", None)


# ------------------------------------------------------------------ register
@pytest.mark.parametrize("mode, extra", [("yaw_sweep_icp", []),
                                         ("signed_source_up", ["--source-up", "signed"]),
                                         ("alt_source_up", ["--source-up", "alternative"])])
def test_register_modes(ctx, mode, extra):
    spec = P.register(ctx, mode, [3, 8])
    _check_common(spec, ctx, "main")
    assert _module(spec) == "agents.assets.factory_align"
    assert _flags(spec) == extra + ["--objects", "3,8"]
    assert not spec.needs_gpu
    assert spec.artifacts == [ctx.out_dir / "objects" / "obj_03" / "aligned.json",
                              ctx.out_dir / "objects" / "obj_08" / "aligned.json"]
    all_spec = P.register(ctx, mode, None)
    assert _flags(all_spec) == extra


# ------------------------------------------- physics / report / mjcf / tasks
def test_physics_report_mjcf_tasks(ctx):
    ph = P.physics(ctx, [1])
    _check_common(ph, ctx, "main")
    assert _module(ph) == "agents.assets.s6_physics" and _flags(ph) == []
    assert ph.env["SIMANY_OBJECTS"] == "1" and ph.needs_gpu
    rp = P.report(ctx)
    assert _module(rp) == "agents.eval.factory_report" and not rp.needs_gpu
    assert rp.artifacts == [ctx.out_dir / "report.json"]
    mx = P.export_mjcf(ctx, "shim", test=True)
    assert _module(mx) == "robo.sim.export_mjcf" and _flags(mx) == ["--test", "--collision-mode", "shim"]
    assert mx.artifacts == [ctx.out_dir / "sim_export" / "scene.xml"]
    assert _flags(P.export_mjcf(ctx, "room", test=False)) == ["--collision-mode", "room"]
    with pytest.raises(ValueError):
        P.export_mjcf(ctx, "convex_soup")
    tk = P.tasks(ctx, 7)
    assert _module(tk) == "robo.tasks.pi05_tasks"
    assert _flags(tk) == ["--out-dir", str(ctx.out_dir), "--max-tasks", "7"]
    assert tk.artifacts == [ctx.out_dir / "sim_export" / "pi05_tasks.json"]


# ------------------------------------------------------------------- inpaint
def test_inpaint_objects_qwen(ctx):
    sel = Selection(kind="object", object_ids=[4, 9])
    specs = P.inpaint(ctx, sel, "remove the {label}, keep the desk", "qwen_image_edit", 800,
                      negative_prompt="blurry")
    assert [_module(s) for s in specs] == ["agents.edit.inpaint_prepare", "agents.edit.inpaint_masks",
                                          "agents.edit.inpaint_qwen", "agents.edit.inpaint_fill"]
    assert [s.env_key for s in specs] == ["main", "sam3", "sam3", "gsplat"]
    assert [s.needs_gpu for s in specs] == [False, True, True, True]
    assert len({s.tags["chain"] for s in specs}) == 1
    assert _flags(specs[0]) == ["--objects", "4,9"]
    assert _flags(specs[1]) == ["--images-dir", str(ctx.images_dir), "--out-dir", str(ctx.out_dir),
                                "--objects", "4,9"]
    assert _flags(specs[2]) == ["--backend", "qwen", "--prompt", "remove the {label}, keep the desk",
                                "--objects", "4,9", "--negative-prompt", "blurry"]
    assert _flags(specs[3]) == ["--objects", "4,9", "--out-name", "clean_background.ply", "--iters", "800"]
    assert specs[3].artifacts == [ctx.out_dir / "inpaint" / "clean_background.ply"]
    for s in specs:
        _check_common(s, ctx, s.env_key)


def test_inpaint_box_lama(ctx):
    sel = Selection(kind="box", box_center=(1.0, 2.0, 0.8), box_size=(0.3, 0.2, 0.25),
                    box_quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    specs = P.inpaint(ctx, sel, "erase it", "lama")
    prep, masks, edit, fill = specs
    flags = _flags(prep)
    assert flags[0] == "--region-box"
    box = json.loads(flags[1])
    assert box == {"center": [1.0, 2.0, 0.8], "size": [0.3, 0.2, 0.25], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}
    assert prep.artifacts == [ctx.out_dir / "inpaint" / "region_00" / "removal_idx.npy"]
    assert _flags(masks)[-2:] == ["--region", "region_00"]
    assert edit.env_key == "main" and edit.needs_gpu is False
    assert _flags(edit) == ["--backend", "lama", "--prompt", "erase it", "--region", "region_00"]
    assert _flags(fill) == ["--region", "region_00", "--out-name", "clean_background.ply"]


def test_inpaint_rejects_empty_selection(ctx):
    with pytest.raises(ValueError):
        P.inpaint(ctx, Selection(), "x", "lama")
    with pytest.raises(ValueError):
        P.inpaint(ctx, Selection(kind="object", object_ids=[]), "x", "lama")
    with pytest.raises(KeyError):
        P.inpaint(ctx, Selection(kind="object", object_ids=[1]), "x", "dalle")


# ------------------------------------------------------------- policy server
def test_policy_server_pi05(cfg):
    spec = P.policy_server(cfg, "pi05_droid_jointpos", 8123)
    assert spec.argv == ["bash", str(cfg.policy_server_script), "--port", "8123"]
    assert spec.where == "slurm" and spec.needs_gpu and spec.env_key == "openpi"
    assert spec.gpu_type == cfg.policy_server_gpu_types[0]
    assert spec.env["SIMANY_PI05_CKPT"] == "pi05_droid_jointpos"
    assert spec.env["SIMANY_PI05_CONFIG"] == "pi05_droid_jointpos"
    assert spec.tags["kind"] == "policy_server" and spec.tags["port"] == "8123"
    assert spec.tags["slurm_mem"] == "100G"


def test_policy_server_sim_cotrained_and_gpu_override(cfg):
    spec = P.policy_server(cfg, "droid_pi05_jointpos_with_web_and_sim", 8000, gpu_type="h200")
    assert spec.env["SIMANY_PI05_CKPT"] == "droid_pi05_jointpos_with_web_and_sim/80000"
    assert spec.env["SIMANY_PI05_CONFIG"] == "pi05_droid_jointpos_sim"
    assert spec.gpu_type == "h200"


def test_policy_server_rejects_scripted_and_unknown(cfg):
    with pytest.raises(ValueError):
        P.policy_server(cfg, "scripted_sinusoid", 8000)
    with pytest.raises(KeyError):
        P.policy_server(cfg, "not_a_policy", 8000)
    with pytest.raises(ValueError):
        P.policy_server(cfg, "pi05_droid_jointpos", 8000, gpu_type="tpu")


# ------------------------------------------------------------- full pipeline
def test_full_pipeline_chain_order(ctx):
    specs = P.full_pipeline(ctx, "sam3_auto", "trellis", "signed_source_up", "shim")
    mods = [_module(s) for s in specs]
    assert mods == ["agents.discover.auto_segment", "agents.discover.factory_prepare",
                    "agents.discover.factory_refine_masks", "models.s4_trellis",
                    "agents.assets.factory_align", "agents.assets.s6_physics",
                    "agents.eval.factory_report", "robo.sim.export_mjcf", "robo.tasks.pi05_tasks"]
    chains = {s.tags["chain"] for s in specs}
    assert len(chains) == 1 and next(iter(chains)).startswith("full:c50d2d1d42:")
    assert all(s.env["SIMANY_AUTO"] == "1" for s in specs)   # auto derived from discovery
    assert _flags(specs[4]) == ["--source-up", "signed"]
    assert _flags(specs[7]) == ["--test", "--collision-mode", "shim"]
    gt = P.full_pipeline(ctx, "gt_segments", "hybrid", "yaw_sweep_icp")
    assert all("SIMANY_AUTO" not in s.env for s in gt)
    assert len(gt) == 8


# ------------------------------------------------------------ validation
def test_missing_interpreter_is_tagged_not_fatal(cfg, tmp_path, monkeypatch):
    # A shared-FS env (e.g. /group/streetsplat sam3) may be unmounted on this node while
    # the job can still run remotely: builders tag the spec instead of failing ...
    bad = replace(cfg, interpreters={**cfg.interpreters, "sam3": tmp_path / "nope" / "python"})
    ctx = P.StageContext(bad, SCENE, tmp_path / "out")
    specs = P.discover(ctx, "gt_segments")
    sam3_specs = [s for s in specs if s.env_key == "sam3"]
    assert sam3_specs and all(s.tags.get("interpreter_missing") == "1" for s in sam3_specs)
    assert "interpreter_missing" not in P.report(ctx).tags  # main still fine
    # ... unless strict mode is requested.
    monkeypatch.setattr(P, "STRICT_INTERPRETERS", True)
    with pytest.raises(FileNotFoundError) as ei:
        P.discover(ctx, "gt_segments")
    assert "sam3" in str(ei.value)


def test_unknown_interpreter_key_and_missing_module(cfg, tmp_path):
    ctx = P.StageContext(cfg, SCENE, tmp_path / "out")
    with pytest.raises(KeyError):
        P._interpreter(cfg, "cuda13")
    with pytest.raises(FileNotFoundError):
        P._check_module(cfg, "agents.nope.missing_stage")


def test_base_env_and_images_dir(cfg, tmp_path):
    ctx = P.StageContext(cfg, SCENE, tmp_path / "out", auto=True)
    env = ctx.base_env()
    assert env["SIMANY_AUTO"] == "1" and "SIMANY_SCENE_DIR" not in env
    assert env["SIMANY_SCANNETPP_ROOT"] == str(cfg.scannetpp_root)
    assert ctx.resolved_images_dir() == cfg.scannetpp_root / "data" / SCENE / "dslr" / "resized_undistorted_images"
