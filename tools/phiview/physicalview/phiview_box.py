"""Displayed-frame bounding boxes; pure validation and known-object selection."""
import numpy as np


def pixel_box(value, shape):
    box = np.asarray(value, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or (box < 0).any() or (box > 1).any():
        raise ValueError('Selection box must have four finite coordinates inside the image')
    h, w = shape
    lo = np.minimum(box[:2], box[2:])
    hi = np.maximum(box[:2], box[2:])
    x0, y0 = np.floor(lo*[w, h]).astype(int)
    x1, y1 = np.ceil(hi*[w, h]).astype(int)
    if x1-x0 < 3 or y1-y0 < 3:
        raise ValueError('Drag a larger selection box')
    return [int(x0), int(y0), int(x1), int(y1)]


def known_object_in_box(labels, box):
    x0, y0, x1, y1 = box
    crop = labels[y0:y1, x0:x1]
    votes = np.bincount(crop.reshape(-1).astype(int))
    if len(votes) < 2:
        return None
    votes[0] = 0
    label = int(votes.argmax())
    # A tight rectangle around one known object can select it immediately.
    if votes[label] >= .45*crop.size and votes[label] >= .9*votes.sum():
        return label
    return None
