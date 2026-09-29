"""Close robot geometry must remain visible with far-away inactive bodies."""
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('MUJOCO_GL') != 'egl', reason='GPU EGL renderer required')


@pytest.mark.parametrize('parked_z', [-1, -27])
def test_near_robot_link_is_visible_despite_parked_projectile(parked_z):
    mujoco = pytest.importorskip('mujoco')
    from physicalview.phiview_sim import DemoPhysics
    from physicalview.render import look_at_w2c

    sim = DemoPhysics.__new__(DemoPhysics)
    sim.model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="robot/near_link" pos=".15 0 0">
        <geom type="sphere" size=".035" rgba="1 .1 .1 1"/>
      </body>
      <body name="parked" pos="0 0 {parked_z}">
        <geom type="sphere" size=".025"/>
      </body>
    </worldbody></mujoco>''')
    sim.data = mujoco.MjData(sim.model)
    mujoco.mj_forward(sim.model, sim.data)
    sim.robot, sim.projectile, sim.renderer = True, 0, None
    w2c = look_at_w2c(np.zeros(3), np.array([1., 0, 0]))
    K = np.array([[70., 0, 64], [0, 70., 64], [0, 0, 1.]])
    try:
        image = sim.overlay(np.zeros((128, 128, 3), np.uint8), np.ones((128, 128))*100, w2c, K)
        assert sim.overlay_mask[64, 64]
        assert image[64, 64, 0] > 50
        assert sim.overlay_mask.sum() > 400
    finally:
        if sim.renderer:
            sim.renderer.close()
