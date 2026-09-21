"""CPU tests for physicalview.config and physicalview.gpu (no viser, no GPU)."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from physicalview import gpu as G
from physicalview.config import DEFAULT_CONFIG, load_config


def test_default_config_loads_and_resolves_paths():
    cfg = load_config()
    assert cfg.repo_root.is_absolute()
    assert cfg.outputs_root.is_absolute()
    assert cfg.interpreter("main").name.startswith("python")
    assert {c.id for c in cfg.generation} >= {"trellis", "reconviagen", "sam3d", "hybrid"}
    assert {c.id for c in cfg.registration} >= {"yaw_sweep_icp", "signed_source_up", "alt_source_up"}
    assert {c.id for c in cfg.inpaint_backends} == {"qwen_image_edit", "lama"}
    assert cfg.collision_modes == ["room", "shim"]
    assert "pi05_droid_jointpos" in cfg.policies
    assert set(cfg.gpu_targets) == {"a6000", "h200", "a100", "rtx6000"}
    assert cfg.gpu_targets["h200"].extra == ()
    assert cfg.render_wh == (640, 360)


def test_unknown_choice_and_interpreter_raise():
    cfg = load_config()
    with pytest.raises(KeyError):
        cfg.choice("generation", "nope")
    with pytest.raises(KeyError):
        cfg.interpreter("nope")


def test_virtualenv_interpreter_does_not_resolve_to_base_python(tmp_path):
    from physicalview.config import _interpreter_path
    base = tmp_path / "base-python"
    base.write_text("base")
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(base)
    assert _interpreter_path(tmp_path, "venv/bin/python") == venv_bin / "python"


def test_relative_paths_resolve_against_repo_root(tmp_path: Path):
    cfg_text = textwrap.dedent("""
        outputs_root: out
        interpreters: {main: bin/python}
        slurm: {gpu_types: {a6000: {partition: debug, gres: gpu:a6000:1, compute_cap: "8.6"}}}
    """)
    p = tmp_path / "c.yaml"
    p.write_text(cfg_text)
    cfg = load_config(p, repo_root=tmp_path)
    assert cfg.outputs_root == (tmp_path / "out").resolve()
    assert cfg.interpreter("main") == (tmp_path / "bin/python").resolve()
    assert cfg.default_remote_gpu == "a6000"


def _gpu(cc: str) -> G.GpuInfo:
    return G.GpuInfo(present=True, name="x", compute_cap=cc, memory_mib=1, driver="d")


def test_env_compatibility_matrix_by_compute_capability():
    cfg = load_config()
    blackwell, a6000, none = _gpu("12.0"), _gpu("8.6"), G.GpuInfo(present=False)
    # cu124 envs cannot run on Blackwell; the studio env can run everywhere.
    assert not G.env_compatible(cfg, "main", blackwell)
    assert G.env_compatible(cfg, "studio", blackwell)
    assert G.env_compatible(cfg, "main", a6000)
    assert not G.env_compatible(cfg, "main", none)
    # env without an arch entry is treated as CPU-only / compatible
    assert G.env_compatible(cfg, "not_listed_env", none)


def test_remote_gpu_type_selection_prefers_default_then_compatible():
    cfg = load_config()
    assert G.gpu_type_for(cfg, "main") == cfg.default_remote_gpu
    # gsplat wheel supports 8.0/8.6/9.0 -> default a6000 is fine
    assert G.gpu_type_for(cfg, "gsplat") == "a6000"
    # an env supporting only 12.0 must route to rtx6000
    cfg.env_arch_support["only_blackwell"] = ("12.0",)
    assert G.gpu_type_for(cfg, "only_blackwell") == "rtx6000"
    cfg.env_arch_support["nowhere"] = ("1.0",)
    assert G.gpu_type_for(cfg, "nowhere") is None


def test_detect_gpu_never_raises(monkeypatch):
    monkeypatch.setattr(G.shutil, "which", lambda _: None)
    info = G.detect_gpu()
    assert info.present is False
    assert info.to_dict()["present"] is False


def test_default_config_file_is_the_committed_one():
    assert DEFAULT_CONFIG.name == "default.yaml" and DEFAULT_CONFIG.exists()
