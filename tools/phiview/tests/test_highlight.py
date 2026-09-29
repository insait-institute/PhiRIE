import numpy as np
import pytest

from physicalview.highlight import highlight_objects, red_mask


def test_mask_is_red_and_outside_pixels_and_inputs_are_preserved():
    rgb = np.full((24, 24, 3), .3, dtype=np.float32)
    mask = np.zeros((24, 24), dtype=bool)
    mask[8:16, 8:16] = True
    before = mask.copy()
    out = red_mask(rgb, mask)
    assert np.all(out[mask, 0] > .8)
    assert np.all(out[mask, 1] < .1)
    assert np.all(out[6, 8:16] == 1)
    assert np.array_equal(out[:4], rgb[:4])
    assert np.array_equal(mask, before) and np.all(rgb == .3)


def test_selection_is_stronger_without_changing_picking_labels():
    labels = np.zeros((30, 30), dtype=np.uint16)
    labels[3:12, 3:12] = 1
    labels[18:27, 18:27] = 2
    before = labels.copy()
    rgb = np.full((30, 30, 3), .2, dtype=np.float32)
    out = highlight_objects(rgb, labels, 2)
    assert out[21, 21, 0] > out[6, 6, 0]
    assert np.array_equal(labels, before)


def test_empty_mask_is_identity_and_bad_shape_fails():
    rgb = np.full((10, 10, 3), .3, dtype=np.float32)
    assert np.array_equal(red_mask(rgb, np.zeros((10, 10))), rgb)
    with pytest.raises(ValueError, match='dimensions'):
        red_mask(rgb, np.zeros((8, 8)))
