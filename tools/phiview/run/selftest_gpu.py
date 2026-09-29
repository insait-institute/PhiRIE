#!/usr/bin/env python
"""GPU self-test for the SimAny Studio env. Run on a GPU node:

    source run/env.sh
    $STUDIO_PY run/selftest_gpu.py --out outputs/selftest/$(hostname).json

Checks (critical unless noted): GPU/driver/compute capability, torch CUDA
matmul, gsplat import + cubin archs of the JIT-built .so + a real 640x360
rasterization of a ScanNet++ splat, mujoco EGL offscreen render, viser and
openpi_client imports, and that the repo modules the studio needs import
under this env. Writes one JSON report and exits non-zero on any critical
failure.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # PhysicalView checkout
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import physicalview  # noqa: E402,F401  (adds the SimAny root to sys.path)

SPLAT_PLY = Path(os.environ.get("STUDIO_SELFTEST_PLY",
                                "/data/ScanNetppv2_gsplat/splats/c50d2d1d42.ply"))
REPO_MODULES = [
    "agents.core.common",
    "robo.envs.pi05_env",
    "robo.policy.clients.pi05_client",
    "robo.rendering.pi05_render",
    "interface.mujoco_live_viewer",
]
CRITICAL = ("gpu", "torch", "gsplat", "mujoco", "viser", "openpi_client", "repo_modules")

MJ_XML = """
<mujoco>
  <visual><global offwidth="320" offheight="240"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body pos="0 0 0.3"><freejoint/><geom type="box" size="0.1 0.1 0.1" rgba="0.9 0.2 0.2 1"/></body>
    <camera name="cam" pos="1.2 -1.2 0.9" xyaxes="1 1 0 -0.4 0.4 1"/>
  </worldbody>
</mujoco>
"""


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def check_gpu() -> dict:
    r: dict = {"ok": False}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=60, check=True)
        name, cc, drv, mem = [s.strip() for s in out.stdout.splitlines()[0].split(",")]
        r.update(name=name, compute_capability=cc, driver=drv, memory_total=mem, ok=True)
    except Exception as e:  # noqa: BLE001
        r["error"] = _err(e)
    return r


def check_torch() -> dict:
    r: dict = {"ok": False}
    try:
        import torch
        r.update(version=torch.__version__, cuda=torch.version.cuda,
                 cudnn=torch.backends.cudnn.version(),
                 arch_list=torch.cuda.get_arch_list(),
                 cuda_available=torch.cuda.is_available())
        if not torch.cuda.is_available():
            r["error"] = "torch.cuda.is_available() is False"
            return r
        cap = torch.cuda.get_device_capability(0)
        r["device"] = torch.cuda.get_device_name(0)
        r["device_capability"] = f"{cap[0]}.{cap[1]}"
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        (a @ b).sum().item()  # warm-up: cuBLAS handle/kernel load dominates the first call
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()
        r["matmul_1024_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
        ref = a.cpu() @ b.cpu()
        r["matmul_max_abs_err"] = float((c.cpu() - ref).abs().max())
        r["ok"] = bool(r["matmul_max_abs_err"] < 1e-1)
        if not r["ok"]:
            r["error"] = "cuda matmul does not match cpu reference"
    except Exception as e:  # noqa: BLE001
        r["error"] = _err(e)
        r["traceback"] = traceback.format_exc(limit=3)
    return r


def _so_archs(so: str) -> list[str]:
    cuda_home = os.environ.get("CUDA_HOME", "/opt/modules/nvidia-cuda-12.8.1")
    cuobjdump = Path(cuda_home) / "bin" / "cuobjdump"
    if not cuobjdump.exists():
        return [f"<{cuobjdump} missing>"]
    out = subprocess.run([str(cuobjdump), "--list-elf", so], capture_output=True, text=True,
                         timeout=120)
    import re
    return sorted(set(re.findall(r"sm_\d+[a-z]*", out.stdout)),
                  key=lambda s: int(re.sub(r"\D", "", s)))


def check_gsplat() -> dict:
    r: dict = {"ok": False}
    try:
        import torch
        t0 = time.perf_counter()
        import gsplat
        from gsplat.cuda._backend import _C
        r["import_s"] = round(time.perf_counter() - t0, 2)
        r["version"] = gsplat.__version__
        if _C is None:
            r["error"] = "gsplat CUDA backend is None (no toolkit / build failed)"
            return r
        so = getattr(_C, "__file__", None)
        r["backend_so"] = so
        r["backend_kind"] = ("jit:" + os.environ.get("TORCH_EXTENSIONS_DIR", "~/.cache")
                             if so and "torch_extensions" in so else "installed csrc")
        if so:
            r["so_archs"] = _so_archs(so)
        # real rasterization through the repo's own helpers
        if not SPLAT_PLY.exists():
            subprocess.run(["ls", "/data/ScanNetpp"], capture_output=True, timeout=60)  # autofs
        if not SPLAT_PLY.exists():
            r["error"] = f"{SPLAT_PLY} not found (autofs?)"
            return r
        from agents.core import common as C
        t0 = time.perf_counter()
        gs = C.load_gaussians(SPLAT_PLY, device="cuda")
        torch.cuda.synchronize()
        r["load_s"] = round(time.perf_counter() - t0, 2)
        r["n_gaussians"] = int(gs["means"].shape[0])
        r["sh_degree"] = int(gs["sh_degree"])
        import numpy as np
        w, h = 640, 360
        K = np.array([[0.8 * w, 0, w / 2], [0, 0.8 * w, h / 2], [0, 0, 1]], dtype=np.float64)
        # identity rotation, camera placed at the splat centroid (inside the room),
        # so the +z view direction always hits geometry regardless of the scene.
        center = gs["means"].mean(0).cpu().numpy().astype(np.float64)
        w2c = np.eye(4)
        w2c[:3, 3] = -center
        C.render_view(gs, w2c, K, w, h)  # warm-up (first launch of each kernel)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rgb, _, alpha = C.render_view(gs, w2c, K, w, h)
        torch.cuda.synchronize()
        r["render_640x360_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
        r["rgb_mean"] = float(rgb.mean())
        r["rgb_std"] = float(rgb.std())
        r["alpha_mean"] = float(alpha.mean())
        r["nonblack_frac"] = float((rgb.max(axis=-1) > 0.02).mean())
        r["image_not_black"] = bool(r["nonblack_frac"] > 0.05 and r["rgb_std"] > 0.01)
        r["ok"] = bool(r["image_not_black"] and np.isfinite(rgb).all())
        if not r["ok"]:
            r["error"] = "rendered image is (nearly) all black or non-finite"
    except Exception as e:  # noqa: BLE001
        r["error"] = _err(e)
        r["traceback"] = traceback.format_exc(limit=5)
    return r


def check_mujoco() -> dict:
    r: dict = {"ok": False, "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
               "MUJOCO_EGL_DEVICE_ID": os.environ.get("MUJOCO_EGL_DEVICE_ID")}
    try:
        import mujoco
        import numpy as np
        r["version"] = mujoco.__version__
        model = mujoco.MjModel.from_xml_string(MJ_XML)
        data = mujoco.MjData(model)
        for _ in range(50):
            mujoco.mj_step(model, data)
        t0 = time.perf_counter()
        renderer = mujoco.Renderer(model, height=240, width=320)
        r["renderer_init_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
        renderer.update_scene(data, camera="cam")
        t0 = time.perf_counter()
        img = renderer.render()
        r["render_320x240_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
        renderer.close()
        r["img_shape"] = list(img.shape)
        r["img_mean"] = float(img.mean())
        r["img_not_black"] = bool(img.std() > 1.0 and img.mean() > 1.0)
        r["ok"] = r["img_not_black"] and tuple(img.shape) == (240, 320, 3)
        if not r["ok"]:
            r["error"] = "EGL offscreen render came back black/wrong shape"
    except Exception as e:  # noqa: BLE001
        r["error"] = _err(e)
        r["traceback"] = traceback.format_exc(limit=3)
    return r


def check_import(name: str) -> dict:
    r: dict = {"ok": False}
    try:
        t0 = time.perf_counter()
        m = importlib.import_module(name)
        r["import_s"] = round(time.perf_counter() - t0, 2)
        v = getattr(m, "__version__", None)
        if v:
            r["version"] = v
        r["ok"] = True
    except Exception as e:  # noqa: BLE001
        r["error"] = _err(e)
        r["traceback"] = traceback.format_exc(limit=3)
    return r


def check_repo_modules() -> dict:
    mods = {m: check_import(m) for m in REPO_MODULES}
    # the mujoco env additionally needs the menagerie checkout next to the repo
    try:
        from robo.rigs import pi05_rig
        mods["robo.rigs.pi05_rig"] = {"ok": True, "menagerie": str(pi05_rig.MENAGERIE),
                                      "menagerie_present": pi05_rig.PANDA_XML.exists()
                                      and pi05_rig.GRIPPER_XML.exists()}
        if not mods["robo.rigs.pi05_rig"]["menagerie_present"]:
            mods["robo.rigs.pi05_rig"]["warning"] = (
                "third_party/mujoco_menagerie missing in this checkout (gitignored; "
                "symlink it from the main checkout)")
    except Exception as e:  # noqa: BLE001
        mods["robo.rigs.pi05_rig"] = {"ok": False, "error": _err(e)}
    return {"ok": all(v["ok"] for v in mods.values()), "modules": mods,
            "failed": [k for k, v in mods.items() if not v["ok"]]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="path of the JSON report to write")
    ap.add_argument("--skip", default="", help="comma-separated check names to skip")
    args = ap.parse_args()
    skip = {s for s in args.skip.split(",") if s}

    report: dict = {
        "host": platform.node(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "repo_root": str(ROOT),
        "env": {k: os.environ.get(k) for k in
                ("CUDA_HOME", "TORCH_CUDA_ARCH_LIST", "TORCH_EXTENSIONS_DIR", "MUJOCO_GL",
                 "PYOPENGL_PLATFORM", "PYTHONPATH", "HF_HUB_OFFLINE")},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checks": {},
    }
    steps = [
        ("gpu", check_gpu),
        ("torch", check_torch),
        ("gsplat", check_gsplat),
        ("mujoco", check_mujoco),
        ("viser", lambda: check_import("viser")),
        ("openpi_client", lambda: check_import("openpi_client")),
        ("repo_modules", check_repo_modules),
    ]
    t_all = time.perf_counter()
    for name, fn in steps:
        if name in skip:
            report["checks"][name] = {"ok": None, "skipped": True}
            continue
        t0 = time.perf_counter()
        res = fn()
        res["wall_s"] = round(time.perf_counter() - t0, 2)
        report["checks"][name] = res
        print(f"[{'PASS' if res['ok'] else 'FAIL'}] {name:14s} {res['wall_s']:6.1f}s"
              + (f"  {res.get('error')}" if not res["ok"] else ""), flush=True)
    report["total_s"] = round(time.perf_counter() - t_all, 1)
    report["failed_critical"] = [n for n in CRITICAL
                                 if n in report["checks"] and report["checks"][n]["ok"] is False]
    report["pass"] = not report["failed_critical"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"{'PASS' if report['pass'] else 'FAIL'}  report -> {out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
