import numpy as np

from robo.eval.observation_pipeline import ObservationPipeline, ObservationSource
from robo.rendering.harmonizer_client import IdentityEnhancer


class Source(ObservationSource):
    def get_obs(self):
        return {
            "observation/exterior_image_1_left": np.full((12, 10, 3), 25, np.uint8),
            "observation/exterior_robot_mask_1_left": np.pad(
                np.ones((6, 4), np.uint8) * 255, ((3, 3), (3, 3))),
        }


class Env:
    pass


def test_episode_scoped_stream_and_exact_core():
    enhancer = IdentityEnhancer()
    pipeline = ObservationPipeline(
        Source(), variant="harmonizer_c", env=Env(), enhancer=enhancer,
        image_keys=("observation/exterior_image_1_left",),
        restoration_options={"erode_px": 1, "dilate_px": 2})
    pipeline.reset("ep0")
    result = pipeline.get_obs()
    metadata = result.per_camera["observation/exterior_image_1_left"]
    assert metadata["robot_core_equal"] is True
    assert metadata["max_robot_core_error"] == 0
    assert enhancer.streams["ep0/observation/exterior_image_1_left"] == 1
    pipeline.reset("ep1")
    pipeline.get_obs()
    assert enhancer.streams["ep1/observation/exterior_image_1_left"] == 1
