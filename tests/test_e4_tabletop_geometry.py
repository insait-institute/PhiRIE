"""Regression for added table volume ejecting preserved floor objects."""
import mujoco
import numpy as np
import pytest

from robo.rigs.pi05_rig import _table_geometry


def _contacts(table):
    spec = mujoco.MjSpec.from_string('''<mujoco><worldbody>
      <body name="floor_object" pos="0 0 .11"><freejoint/>
        <geom type="sphere" size=".03" mass=".3"/>
      </body></worldbody></mujoco>''')
    pos, size = _table_geometry(table)
    spec.worldbody.add_geom(name="tabletop", type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=pos, size=size)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return [float(c.dist) for c in data.contact], data.xpos[1].copy()


def test_slab_preserves_top_and_floor_object_without_filling_under_table():
    table = dict(cx=0, cy=0, hx=1.57, hy=1.80, top_z=.7583)
    legacy, old_pose = _contacts(table)
    assert legacy and min(legacy) < -.1  # reproduce the old penetration
    slab = {**table, "thickness_m": .02}
    corrected, new_pose = _contacts(slab)
    assert corrected == []
    np.testing.assert_array_equal(new_pose, old_pose)
    pos, size = _table_geometry(slab)
    assert pos[2] + size[2] == table["top_z"]
    assert size[:2] == [table["hx"], table["hy"]]
    assert _table_geometry(table)[1][2] == table["top_z"] / 2


@pytest.mark.parametrize("thickness", [0, -0.02, float("nan"), float("inf")])
def test_invalid_slab_thickness_is_rejected(thickness):
    with pytest.raises(ValueError, match="finite and positive"):
        _table_geometry(dict(cx=0, cy=0, hx=1, hy=1, top_z=.7,
                             thickness_m=thickness))
