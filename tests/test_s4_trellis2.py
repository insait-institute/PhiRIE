"""CPU contract tests; no model, remote service, or GPU is simulated as evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from models import s4_trellis2 as backend


def mesh():
    return SimpleNamespace(
        vertices=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
        faces=np.array([[0, 1, 2]]), attrs=np.ones((2, 6)) * .5,
        coords=np.array([[0, 0, 0], [1, 0, 0]]), origin=np.array([-.5] * 3),
        voxel_size=1 / 512, voxel_shape=(1, 6, 512, 512, 512),
        layout={"base_color": slice(0, 3), "metallic": slice(3, 4),
                "roughness": slice(4, 5), "alpha": slice(5, 6)})


@pytest.fixture
def setup(tmp_path, monkeypatch):
    out = tmp_path / "out"
    (out / "objects").mkdir(parents=True)
    (out / "objects/objects.json").write_text(json.dumps([{"index": 1}, {"index": 2}]))
    for index in (1, 2):
        odir = out / "objects" / f"obj_{index:02d}"
        odir.mkdir()
        data = np.full((8, 8, 4), 255, dtype=np.uint8)
        data[0, :, 3] = 0
        Image.fromarray(data).save(odir / "rgba.png")
    counters = {"loads": 0, "calls": [], "preprocess": 0}

    class Pipe:
        def preprocess_image(self, image):
            counters["preprocess"] += 1
            return image.convert("RGB")

        def run(self, image, **kwargs):
            counters["calls"].append(kwargs)
            return [mesh()]

    pipe = Pipe()
    torch = SimpleNamespace(cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 123, empty_cache=lambda: None))

    def loader(*args):
        counters["loads"] += 1
        return pipe, torch

    monkeypatch.setattr(backend, "validate_sources", lambda *args: ({}, {}, {"source_commit": "pinned"}))
    monkeypatch.setattr(backend, "load_pipeline", loader)
    kwargs = dict(out_root=out, records_dir=tmp_path / "records", source_dir="source", model_dir="model",
                  dinov3_model="dino", ss_decoder="decoder", source_commit="pinned")
    return kwargs, pipe, counters


def test_generate_native_mesh_pbr_once_loaded_no_gaussian(setup):
    kwargs, _, counts = setup
    records = backend.generate_objects(**kwargs)
    assert counts["loads"] == 1 and counts["preprocess"] == 2
    assert counts["calls"] == [dict(seed=42, pipeline_type="512", preprocess_image=False)] * 2
    for rec in records:
        assert rec["status"] == "generated" and rec["wall_s"] >= 0
        assert rec["capabilities"] == dict(mesh=True, pbr_voxels=True, gaussian=False, baked_glb=False)
        odir = kwargs["out_root"] / "objects" / f"obj_{rec['object_index']:02d}"
        for name in backend.OUTPUTS:
            assert rec["artifacts"][name]["sha256"] == backend._sha(odir / name)
        with np.load(odir / "trellis2_pbr.npz", allow_pickle=False) as saved:
            np.testing.assert_equal(saved["attrs"], mesh().attrs)
            np.testing.assert_equal(saved["origin"], mesh().origin)
            assert json.loads(str(saved["layout_json"]))["base_color"] == [0, 3]
        assert not list(odir.glob("*gs.ply"))


def test_failure_retained_next_object_runs(setup):
    kwargs, pipe, _ = setup
    good_run = pipe.run
    count = 0

    def run(*args, **kw):
        nonlocal count
        count += 1
        if count == 1:
            raise RuntimeError("empty sparse coordinates")
        return good_run(*args, **kw)

    pipe.run = run
    records = backend.generate_objects(**kwargs)
    assert [r["status"] for r in records] == ["generation_failed", "generated"]
    assert records[0]["error_type"] == "RuntimeError"
    assert "empty sparse coordinates" in records[0]["reason"]
    assert (kwargs["records_dir"] / "object_01.json").is_file()
    assert not (kwargs["out_root"] / "objects/obj_01/mesh_sim.ply").exists()


def test_no_overwrite_existing_record_and_claim(setup):
    kwargs, _, counts = setup
    backend.generate_objects(**kwargs)
    record = kwargs["records_dir"] / "object_01.json"
    original = record.read_bytes()
    with pytest.raises(FileExistsError):
        backend.generate_objects(**kwargs)
    assert record.read_bytes() == original and counts["loads"] == 1


def test_interrupted_claim_blocks_model_load(setup):
    kwargs, _, counts = setup
    kwargs["records_dir"].mkdir()
    (kwargs["records_dir"] / "object_02.claim").write_text("interrupted")
    with pytest.raises(FileExistsError):
        backend.generate_objects(**kwargs)
    assert counts["loads"] == 0


@pytest.mark.parametrize("mode", ["RGB", "opaque", "empty"])
def test_input_requires_authentic_premasked_rgba(setup, mode):
    kwargs, _, counts = setup
    path = kwargs["out_root"] / "objects/obj_01/rgba.png"
    image = Image.open(path).copy()
    if mode == "RGB":
        image = image.convert("RGB")
    else:
        image.putalpha(255 if mode == "opaque" else 0)
    image.save(path)
    records = backend.generate_objects(**kwargs)
    assert records[0]["status"] == "generation_failed"
    assert records[1]["status"] == "generated" and len(counts["calls"]) == 1


@pytest.mark.parametrize("field", ["vertices", "attrs", "origin"])
def test_nonfinite_mesh_or_pbr_rejected(field):
    candidate = mesh()
    getattr(candidate, field).flat[0] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        backend.validate_mesh_pbr(candidate)


def test_invalid_topology_and_layout_rejected():
    candidate = mesh()
    candidate.faces[0, 0] = 999
    with pytest.raises(ValueError, match="indices"):
        backend.validate_mesh_pbr(candidate)
    candidate = mesh()
    candidate.layout["base_color"] = slice(1, 3)
    with pytest.raises(ValueError, match="layout"):
        backend.validate_mesh_pbr(candidate)


def test_invalid_proposal_records_failure_without_publishing(setup):
    kwargs, pipe, _ = setup
    candidate = mesh()
    candidate.attrs[0, 0] = np.nan
    pipe.run = lambda *args, **kw: [candidate]
    records = backend.generate_objects(**kwargs)
    assert all(r["status"] == "generation_failed" for r in records)
    assert not (kwargs["out_root"] / "objects/obj_01/trellis2_mesh.ply").exists()


def test_model_init_failure_is_process_failure(setup, monkeypatch):
    kwargs, _, _ = setup

    def failure(*args):
        raise RuntimeError("checkpoint unavailable")

    monkeypatch.setattr(backend, "load_pipeline", failure)
    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        backend.generate_objects(**kwargs)
    assert not list(kwargs["records_dir"].glob("object_*.json"))


def test_duplicate_indices_fail_before_model_load(setup):
    kwargs, _, counts = setup
    (kwargs["out_root"] / "objects/objects.json").write_text('[{"index": 1}, {"index": 1}]')
    with pytest.raises(ValueError, match="unique"):
        backend.generate_objects(**kwargs)
    assert counts["loads"] == 0


def test_atomic_json_never_replaces(tmp_path):
    path = tmp_path / "record.json"
    backend._atomic_json(path, {"status": "first"})
    with pytest.raises(FileExistsError):
        backend._atomic_json(path, {"status": "second"})
    assert json.loads(path.read_text())["status"] == "first"


def test_source_dirty_or_wrong_commit_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(backend.subprocess, "check_output", lambda cmd, **kw: "dirty" if "status" in cmd else "wrong")
    with pytest.raises(ValueError, match="clean and match"):
        backend.validate_sources(tmp_path, tmp_path, tmp_path, tmp_path, "pinned")


def test_factory_uses_local_components_and_disables_rembg(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    recorded = {"models": [], "dino": [], "kwargs": None, "cuda": 0}
    package = ModuleType("trellis2")
    package.__file__ = str(tmp_path / "trellis2/__init__.py")
    package.models = SimpleNamespace(from_pretrained=lambda path: recorded["models"].append(path) or path)
    pipeline_module = ModuleType("trellis2.pipelines")

    class Pipe:
        def __init__(self, **kwargs):
            recorded["kwargs"] = kwargs

        def cuda(self):
            recorded["cuda"] += 1

    pipeline_module.Trellis2ImageTo3DPipeline = Pipe
    pipeline_module.samplers = SimpleNamespace(OfficialSampler=lambda **kw: kw)
    features = ModuleType("trellis2.modules.image_feature_extractor")
    features.DinoV3FeatureExtractor = lambda path: recorded["dino"].append(path) or path
    monkeypatch.setitem(sys.modules, "trellis2", package)
    monkeypatch.setitem(sys.modules, "trellis2.pipelines", pipeline_module)
    monkeypatch.setitem(sys.modules, "trellis2.modules.image_feature_extractor", features)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    args = {stage + "_sampler": {"name": "OfficialSampler", "args": {"sigma_min": 1e-5},
                                "params": {"steps": 12}} for stage in
            ("sparse_structure", "shape_slat", "tex_slat")}
    args.update(shape_slat_normalization={"mean": [0]}, tex_slat_normalization={"mean": [0]})
    prefixes = {"decoder": str(tmp_path / "ss_decoder"), "flow": str(tmp_path / "flow")}
    backend.load_pipeline(args, prefixes, {"source_dir": str(tmp_path), "dinov3_model": str(tmp_path / "dino")})
    assert recorded["models"] == list(prefixes.values())
    assert recorded["dino"] == [str(tmp_path / "dino")]
    assert recorded["kwargs"]["rembg_model"] is None
    assert recorded["kwargs"]["sparse_structure_sampler_params"] == {"steps": 12}
    assert recorded["cuda"] == 1
    assert backend.os.environ["HF_HUB_OFFLINE"] == "1"


def test_factory_rejects_another_imported_source(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "trellis2", SimpleNamespace(__file__="/other/trellis2/__init__.py"))
    with pytest.raises(RuntimeError, match="different"):
        backend.load_pipeline({}, {}, {"source_dir": str(tmp_path)})
