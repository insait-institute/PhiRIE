"""Portable command line for PhiView blocks and recorded pipeline execution."""

from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from physicalview import __version__
from physicalview.config import DEFAULT_CONFIG, load_config


def blocks():
    return {
        p.stem: json.loads(p.read_text())
        for p in sorted(
            (Path(__file__).with_name("resources") / "pipelines").glob("*.json")
        )
    }


def runtime_env(cfg):
    paths = [cfg.package_root, cfg.repo_root, cfg.repo_root / "third_party/TRELLIS"]
    exports = {
        key: os.environ.get(key, value) for key, value in cfg.env_exports.items()
    }
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    return {
        **exports,
        "SIMANY_ROOT": str(cfg.repo_root),
        "PYTHONPATH": os.pathsep.join(map(str, paths)),
        "SIMANY_SCANNETPP_ROOT": str(cfg.scannetpp_root),
        "SIMANY_SPLATS_ROOT": str(cfg.splats_root),
    }


def build_plan(a, cfg):
    from physicalview import pipeline as p
    from physicalview.jobs import JobSpec

    out = Path(a.out).resolve() if a.out else cfg.outputs_root / (a.scene + "_factory")
    scene_dir = (
        Path(a.scene_dir).resolve()
        if a.scene_dir
        else cfg.scannetpp_root / "data" / a.scene
    )
    ctx = p.StageContext(cfg, a.scene, out, auto=a.auto, scene_dir=scene_dir)
    ids = [int(x) for x in a.objects.split(",")] if a.objects else None
    extra = a.args[1:] if a.args[:1] == ["--"] else a.args
    action = a.action or (
        "construct" if a.block == "full" else blocks()[a.block]["default_action"]
    )

    def direct(module, env, argv, gpu=True, artifacts=()):
        cwd = cfg.package_root if module.startswith("physicalview.") else cfg.repo_root
        return [
            JobSpec(
                name=f"{a.block}:{action}",
                argv=[str(cfg.interpreter(env)), "-m", module, *map(str, argv), *extra],
                env=runtime_env(cfg),
                cwd=cwd,
                env_key=env,
                needs_gpu=gpu,
                artifacts=list(artifacts),
            )
        ]

    if a.block == "viewer":
        if action == "studio":
            specs = direct(
                "physicalview.app",
                "studio",
                ["--config", a.config, "--host", a.host]
                + (["--scene", a.scene] if a.scene != "scene" else []),
            )
        elif action == "demo":
            specs = direct(
                "physicalview.phiview",
                "studio",
                [
                    "--demo",
                    "--config",
                    a.config,
                    "--scene",
                    a.scene,
                    "--out",
                    out,
                    "--host",
                    a.host,
                ],
            )
        else:
            raise ValueError("viewer action must be demo or studio")
    elif a.block == "datasets":
        modules = {
            "libero": ("physicalview.paper_libero", "studio"),
            "behavior": ("physicalview.paper_behavior", "main"),
        }
        if action not in modules:
            raise ValueError(
                "datasets action must be libero or behavior; ScanNet++ and DROID use the documented scene input contract"
            )
        module, env = modules[action]
        specs = direct(module, env, [], gpu=action == "libero")
    elif a.block == "reconstruction":
        argv = ["--scene-dir", scene_dir, "--out", out]
        specs = direct("agents.recon.gsplat_train", "gsplat", argv, artifacts=[out])
    elif a.block == "discovery":
        specs = p.discover(ctx, a.model or "sam3_auto")
    elif a.block == "generation":
        specs = [p.generate(ctx, a.model or "trellis", ids)]
    elif a.block == "registration":
        specs = [p.register(ctx, a.model or "yaw_sweep_icp", ids)]
    elif a.block == "inpainting":
        specs = p.inpaint(
            ctx,
            SimpleNamespace(kind="object", object_ids=ids),
            a.prompt,
            a.model or "qwen_image_edit",
            a.iters,
        )
        for spec in specs:
            if (
                "agents.edit.inpaint_qwen" in spec.argv
                and (a.model or "qwen_image_edit") == "qwen_image_edit"
            ):
                spec.argv[spec.argv.index("agents.edit.inpaint_qwen")] = (
                    "physicalview.phiview_inpaint"
                )
                spec.env["SIMANY_REQUIRE_QWEN"] = "1"
    elif a.block == "simulation":
        if action == "annotate":
            specs = [p.physics(ctx, ids)]
        elif action == "export":
            specs = [p.export_mjcf(ctx, a.collision, True)]
        else:
            raise ValueError("simulation action must be annotate or export")
    elif a.block == "robotics":
        if action == "tasks":
            specs = [p.tasks(ctx, a.max_tasks)]
        elif action == "policy-server":
            specs = [p.policy_server(cfg, a.model or "pi05_droid_jointpos", a.port)]
        else:
            raise ValueError("robotics action must be tasks or policy-server")
    elif a.block == "paper":
        if action == "capture":
            specs = direct(
                "physicalview.paper_capture",
                "studio",
                ["--config", a.config, "--scene", a.scene, "--out", out],
            )
        elif action in ("pack", "zip", "campaign"):
            if not a.root:
                raise ValueError("--root is required for paper pack, zip, or campaign")
            module = {
                "pack": "paper_pack",
                "zip": "paper_download_pack",
                "campaign": "paper_campaign",
            }[action]
            argv = ["--root", Path(a.root).resolve()]
            if action == "zip":
                argv += ["--out", out]
            specs = direct(
                "physicalview." + module, "studio", argv, gpu=action == "campaign"
            )
        else:
            raise ValueError("paper action must be capture, pack, zip, or campaign")
    elif a.block == "full":
        specs = p.full_pipeline(
            ctx, a.discovery, a.model or "trellis", a.registration, a.collision
        )
    else:
        raise ValueError(f"Unknown block: {a.block}")
    if extra and a.block in (
        "discovery",
        "generation",
        "registration",
        "inpainting",
        "simulation",
        "robotics",
        "full",
    ):
        raise ValueError(
            "Pass structured options for composed stages; backend -- arguments are supported only by direct module blocks"
        )
    for spec in specs:
        spec.env = {**runtime_env(cfg), **spec.env}
        for key, value in cfg.env_exports.items():
            if spec.env.get(key) == value:
                spec.env[key] = os.environ.get(key, value)
        spec.env["PYTHONPATH"] = runtime_env(cfg)["PYTHONPATH"]
        spec.where = a.where
    return specs


def doctor(a):
    cfg = load_config(a.config)
    profile = a.profile
    checks = {
        "python": sys.version.split()[0],
        "version": __version__,
        "profile": profile,
    }
    required = {}
    if profile == "cpu":
        for name in ("numpy", "mujoco", "viser", "pyyaml"):
            try:
                required[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                required[name] = None
    else:
        env = {"studio": "studio", "inference": "sam3", "generation": "main"}[profile]
        required["interpreter"] = (
            str(cfg.interpreter(env)) if cfg.interpreter(env).is_file() else None
        )
        required["backend"] = (
            str(cfg.repo_root) if (cfg.repo_root / "agents").is_dir() else None
        )
        if profile == "generation":
            required["trellis_source"] = (
                str(cfg.repo_root / "third_party/TRELLIS")
                if (cfg.repo_root / "third_party/TRELLIS/trellis").is_dir()
                else None
            )
    checks.update(
        checks=required,
        ready=all(required.values()),
        boundary="Prerequisite paths/package metadata only; GPU kernels, weights and model quality require separate validation.",
    )
    print(json.dumps(checks, indent=2))
    return 0 if checks["ready"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser(
        "blocks", help="List pipeline blocks, tools and development branches"
    )
    b.add_argument("--json", action="store_true")
    init = sub.add_parser(
        "init", help="Write a local configuration without changing defaults"
    )
    init.add_argument("--simany-root", required=True)
    init.add_argument("--data-root")
    init.add_argument("--splats-root")
    init.add_argument("--outputs-root")
    init.add_argument("--config", default="configs/local.yaml")
    doc = sub.add_parser("doctor", help="Check explicit runtime prerequisites")
    doc.add_argument("--config", default=str(DEFAULT_CONFIG))
    doc.add_argument(
        "--profile", choices=["cpu", "studio", "inference", "generation"], default="cpu"
    )
    for verb in ("plan", "run"):
        p = sub.add_parser(
            verb,
            help="Show commands as JSON"
            if verb == "plan"
            else "Execute stages sequentially and retain job receipts",
        )
        p.add_argument("block", choices=list(blocks()) + ["full"])
        p.add_argument("--config", default=str(DEFAULT_CONFIG))
        p.add_argument("--scene", default="scene")
        p.add_argument("--scene-dir")
        p.add_argument("--out")
        p.add_argument("--root")
        p.add_argument("--action")
        p.add_argument("--model")
        p.add_argument("--objects")
        p.add_argument(
            "--prompt",
            default="Remove the selected object and restore the empty surface.",
        )
        p.add_argument("--iters", type=int, default=1000)
        p.add_argument("--auto", action="store_true")
        p.add_argument("--collision", default="room")
        p.add_argument("--discovery", default="sam3_auto")
        p.add_argument("--registration", default="yaw_sweep_icp")
        p.add_argument("--where", choices=["local", "slurm", "auto"], default="local")
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8000)
        p.add_argument("--max-tasks", type=int, default=10)
    raw = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in raw:
        idx = raw.index("--")
        extra = raw[idx + 1 :]
        raw = raw[:idx]
    a = ap.parse_args(raw)
    a.args = extra
    try:
        if a.command == "blocks":
            data = blocks()
            if a.json:
                print(json.dumps(data, indent=2))
            else:
                for name, item in data.items():
                    print(f"{name:16} {item['branch']:26} {item['description']}")
            return 0
        if a.command == "init":
            import yaml

            data = yaml.safe_load(DEFAULT_CONFIG.read_text())
            data["simany_root"] = str(Path(a.simany_root).expanduser().resolve())
            for arg, key in [
                ("data_root", "scannetpp_root"),
                ("splats_root", "splats_root"),
                ("outputs_root", "outputs_root"),
            ]:
                if getattr(a, arg):
                    data[key] = str(Path(getattr(a, arg)).expanduser().resolve())
            path = Path(a.config)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            print(path.resolve())
            return 0
        if a.command == "doctor":
            return doctor(a)
        a.config = str(Path(a.config).resolve())
        cfg = load_config(a.config)
        specs = build_plan(a, cfg)
        if a.command == "plan":
            print(json.dumps([s.to_dict() for s in specs], indent=2))
            return 0
        from physicalview.jobs import JobManager, JobState
        from physicalview.gpu import detect_gpu

        manager = JobManager(cfg, detect_gpu(), max_local_gpu=1, max_remote=1)
        try:
            for spec in specs:
                job = manager.submit(spec)
                print(f"{job.id}: {job.name}\nLog: {job.log_path}", flush=True)
                while not job.state.terminal:
                    job.wait(timeout=0.5)
                print(f"{job.state.value}: {job.name}", flush=True)
                if job.state != JobState.SUCCEEDED:
                    print(job.tail(30), file=sys.stderr)
                    return 1
                missing = [str(path) for path in spec.artifacts if not path.exists()]
                if missing:
                    print(
                        "Missing declared artifacts: " + ", ".join(missing),
                        file=sys.stderr,
                    )
                    return 1
        finally:
            manager.shutdown()
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
