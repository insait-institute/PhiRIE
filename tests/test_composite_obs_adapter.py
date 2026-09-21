import hashlib
import json

import numpy as np
import pytest

from robo.rendering import composite_obs, harness_composite_obs, pi05_render


def test_adapter_resolves_the_maintained_composite_renderer():
    assert composite_obs._candidate_class() is pi05_render.CompositeObs


def test_camera_scoped_renderer_is_adapted_to_droid_observation_dict():
    class Renderer:
        def render(self, camera):
            return np.full((3, 4, 3), len(camera), dtype=np.uint8)

    class Env:
        def joint_position(self):
            return np.arange(7, dtype=float)

        def gripper_position(self):
            return np.array([0.25])

    wrapped = harness_composite_obs.HarnessCompositeObserver(
        Renderer(), Env(),
        {
            "observation/exterior_image_1_left": "ext_cam",
            "observation/wrist_image_left": "wrist_cam",
        },
        ("robot",),
    )
    observation = wrapped._obs()
    assert observation["observation/exterior_image_1_left"].shape == (3, 4, 3)
    assert observation["observation/joint_position"].tolist() == list(range(7))
    assert observation["observation/gripper_position"].tolist() == [0.25]


def test_materialized_background_uses_exact_scene_bound_identity(tmp_path):
    source = tmp_path / "scene.ply"
    source.write_bytes(b"scene-specific-gaussian")
    identity = {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "size_bytes": source.stat().st_size,
    }
    factory = tmp_path / "factory"
    factory.mkdir()
    (factory / "materialization_manifest.json").write_text(json.dumps({
        "source_scene": {"source_scene_gaussian": identity},
    }))
    path, observed = pi05_render._scene_bound_background(factory)
    assert path == source
    assert observed["sha256"] == identity["sha256"]
    assert observed["semantics"] == "frozen_raw_scene_splat_with_original_capture_objects"
    source.write_bytes(b"drift")
    with pytest.raises(ValueError, match="identity drift"):
        pi05_render._scene_bound_background(factory)
