"""StudioConfig: typed view over configs/default.yaml.

Contract: ``load_config(path=None, repo_root=None)`` returns a StudioConfig whose path
fields are absolute. Unknown keys are preserved in ``raw`` so panels can read extras
without schema churn. Nothing here touches the GPU or the filesystem beyond reading the
YAML and resolving paths.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]          # the PhysicalView checkout
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "default.yaml"
if not DEFAULT_CONFIG.is_file():
    DEFAULT_CONFIG = Path(__file__).with_name("resources") / "default.yaml"
DEFAULT_SIMANY_ROOT = PACKAGE_ROOT / "backends" / "simany"


def resolve_simany_root(raw: dict | None = None) -> Path:
    """SimAny checkout that provides agents/, robo/, models/ and the stage scripts.
    Precedence: $SIMANY_ROOT env > config `simany_root` > DEFAULT_SIMANY_ROOT."""
    env = os.environ.get("SIMANY_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    if raw and raw.get("simany_root"):
        value = _expand(raw["simany_root"], PACKAGE_ROOT)
        path = Path(value).expanduser()
        return (path if path.is_absolute() else PACKAGE_ROOT / path).resolve()
    return Path(DEFAULT_SIMANY_ROOT)


@dataclass(frozen=True)
class ModelChoice:
    id: str
    label: str
    module: str | None = None
    env: str | None = None
    args: tuple[str, ...] = ()
    stage: str | None = None
    min_vram_gb: float = 0.0


@dataclass(frozen=True)
class GpuTarget:
    key: str
    partition: str
    gres: str
    compute_cap: str
    extra: tuple[str, ...] = ()
    mem: str = "64G"
    time: str = "03:50:00"


@dataclass(frozen=True)
class StreamConfig:
    """viewer.stream: defaults of the server-render JPEG stream (physicalview.streaming)."""
    max_width: int = 1280          # frame width cap (720p); the Scene tab offers 720p/1080p/native
    jpeg_quality: int = 80
    max_fps: float = 15.0
    moving_scale: float = 0.5      # render scale while the camera moves; full-res frame when it settles


@dataclass
class StudioConfig:
    repo_root: Path                 # SimAny checkout (stage scripts, agents/robo/models)
    package_root: Path              # PhysicalView checkout
    outputs_root: Path
    studio_out: Path
    scannetpp_root: Path
    splats_root: Path
    interpreters: dict[str, Path]
    env_arch_support: dict[str, tuple[str, ...]]
    gpu_targets: dict[str, GpuTarget]
    default_remote_gpu: str
    env_exports: dict[str, str]
    discovery: list[ModelChoice]
    generation: list[ModelChoice]
    registration: list[ModelChoice]
    inpaint_backends: list[ModelChoice]
    collision_modes: list[str]
    policies: list[str]
    policy_server_script: Path
    policy_server_port: int
    policy_server_gpu_types: list[str]
    viewer_port: int
    max_splats_background: int
    max_splats_object: int
    render_wh: tuple[int, int]
    control_hz: int
    display_mode: str = "server"            # viewer.display_mode: server (GPU render -> JPEG stream) | client (WebGL splats)
    stream: StreamConfig = field(default_factory=StreamConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    def interpreter(self, key: str) -> Path:
        try:
            return self.interpreters[key]
        except KeyError as exc:
            raise KeyError(f"unknown interpreter key {key!r}; known: {sorted(self.interpreters)}") from exc

    def choice(self, group: str, choice_id: str) -> ModelChoice:
        for item in getattr(self, group):
            if item.id == choice_id:
                return item
        raise KeyError(f"unknown {group} choice {choice_id!r}")


def _expand(value: str | os.PathLike, root: Path) -> str:
    return os.path.expandvars(str(value).replace("${PHYSICALVIEW_ROOT}", str(PACKAGE_ROOT))
                              .replace("${SIMANY_ROOT}", str(root)))


def _abs(root: Path, value: str | os.PathLike) -> Path:
    p = Path(_expand(value, root)).expanduser()
    return p if p.is_absolute() else (root / p).resolve()


def _interpreter_path(root: Path, value: str | os.PathLike) -> Path:
    # Resolving bin/python itself follows a venv symlink into the base Python,
    # losing pyvenv.cfg and every package installed in that environment.
    p = Path(_expand(value, root)).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.parent.resolve() / p.name


def _choices(items: list[dict] | None) -> list[ModelChoice]:
    out = []
    for it in items or []:
        out.append(ModelChoice(
            id=str(it["id"]), label=str(it.get("label", it["id"])),
            module=it.get("module"), env=it.get("env"),
            args=tuple(str(a) for a in it.get("args", ())), stage=it.get("stage"),
            min_vram_gb=float(it.get("min_vram_gb", 0.0))))
    return out


def load_config(path: str | os.PathLike | None = None,
                repo_root: str | os.PathLike | None = None) -> StudioConfig:
    path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(Path(path).read_text()) or {}
    root = Path(repo_root).resolve() if repo_root else resolve_simany_root(raw)
    slurm = raw.get("slurm", {})
    targets = {}
    for key, t in (slurm.get("gpu_types") or {}).items():
        targets[key] = GpuTarget(
            key=key, partition=str(t["partition"]), gres=str(t["gres"]),
            compute_cap=str(t["compute_cap"]), extra=tuple(t.get("extra", ())),
            mem=str(t.get("mem", "64G")), time=str(t.get("time", "03:50:00")))
    models = raw.get("models", {})
    viewer = raw.get("viewer", {})
    ps = raw.get("policy_server", {})
    display_mode = str(viewer.get("display_mode", "server")).lower()
    if display_mode not in ("server", "client"):
        raise ValueError(f"viewer.display_mode must be 'server' or 'client', got {display_mode!r}")
    st = viewer.get("stream") or {}
    stream = StreamConfig(
        max_width=int(st.get("max_width", 1280)), jpeg_quality=int(st.get("jpeg_quality", 80)),
        max_fps=float(st.get("max_fps", 15)), moving_scale=float(st.get("moving_scale", 0.5)))
    return StudioConfig(
        repo_root=root,
        package_root=PACKAGE_ROOT,
        outputs_root=_abs(root, raw.get("outputs_root", "outputs")),
        studio_out=_abs(root, raw.get("studio_out", "outputs/studio")),
        scannetpp_root=_abs(root, raw.get("scannetpp_root", "/data/ScanNetpp")),
        splats_root=_abs(root, raw.get("splats_root", "/data/ScanNetppv2_gsplat/splats")),
        interpreters={k: _interpreter_path(root, v) for k, v in (raw.get("interpreters") or {}).items()},
        env_arch_support={k: tuple(str(a) for a in v)
                          for k, v in (raw.get("env_arch_support") or {}).items()},
        gpu_targets=targets,
        default_remote_gpu=str(slurm.get("default_remote_gpu", "a6000")),
        env_exports={k: _expand(v, root) for k, v in (slurm.get("env_exports") or {}).items()},
        discovery=_choices(models.get("discovery")),
        generation=_choices(models.get("generation")),
        registration=_choices(models.get("registration")),
        inpaint_backends=_choices(models.get("inpaint_backends")),
        collision_modes=[str(m) for m in models.get("collision_modes", ["room", "shim"])],
        policies=[str(p) for p in models.get("policies", [])],
        policy_server_script=_abs(root, ps.get("script", "run/pi05_serve.sh")),
        policy_server_port=int(ps.get("default_port", 8000)),
        policy_server_gpu_types=[str(g) for g in ps.get("gpu_types", ["a6000"])],
        viewer_port=int(viewer.get("port", 8080)),
        max_splats_background=int(viewer.get("max_splats_background", 2_500_000)),
        max_splats_object=int(viewer.get("max_splats_object", 200_000)),
        render_wh=tuple(int(x) for x in viewer.get("render_wh", (640, 360))),  # type: ignore[arg-type]
        control_hz=int(viewer.get("control_hz", 15)),
        display_mode=display_mode,
        stream=stream,
        raw=raw,
    )
