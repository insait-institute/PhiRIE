"""Tracer replay must not invent pre-fire effects or modify the physics trace."""
import copy

import numpy as np

from physicalview.projectile_effects import particles


def test_effects_obey_timeline_and_preserve_recorded_motion():
    frames = [{'time': t, 'projectiles': {'phiview_ball_0': [t, 0, 1.],
                                        'phiview_ball_1': [0, t, 1.]}}
              for t in (.40, .41, .42)]
    hits = [{'time': .43, 'bullet': 'phiview_ball_0', 'position': [.43, 0, 1.]}]
    saved = copy.deepcopy((frames, hits))
    assert len(particles(frames, hits, .39)['means']) == 0
    visible = particles(frames, hits, .42)
    assert len(visible['means']) > 4
    assert np.isfinite(visible['means']).all()
    # A capture must render identically if future frames or contacts are absent.
    early = particles(frames[:2], [], .41)
    full = particles(frames, hits, .41)
    for key in early:
        np.testing.assert_array_equal(early[key], full[key])
    assert len(particles(frames, hits, 1.)['means']) == 0
    assert (frames, hits) == saved
