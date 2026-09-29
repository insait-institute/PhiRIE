"""Unprepared image regions can become independent physical objects."""

import io
from types import SimpleNamespace

import numpy as np
import pytest

from physicalview.phiview import Demo
from physicalview.phiview_point_segment import choose_mask
from physicalview.phiview_proxy import install_proxy
from physicalview.phiview_selection import lift_mask
from physicalview.phiview_sim import DemoPhysics


def test_unknown_pixel_starts_discovery_instead_of_only_clearing_selection():
    calls = []
    d = Demo.__new__(Demo)
    d.frame_id = 7
    d.frames = {7: (np.zeros((10, 10), dtype=int), None, None, None)}
    d.click_selection = SimpleNamespace(start=lambda *args: calls.append(args))
    d.physics = SimpleNamespace(data=SimpleNamespace(time=0))
    d.selected = None
    d.audit = io.StringIO()
    d.execute({"op": "pick", "frame": 7, "x": 0.25, "y": 0.65})
    assert calls == [(7, 2, 6)]
    with pytest.raises(ValueError, match="expired"):
        d.execute({"op": "pick", "frame": 6, "x": 0.25, "y": 0.65})
    assert len(calls) == 1


def test_point_mask_must_contain_click_and_not_cover_whole_room():
    masks = np.zeros((3, 20, 20), dtype=bool)
    masks[0, :5, :5] = True
    masks[1, 7:13, 7:13] = True
    masks[2] = True
    mask, score = choose_mask(masks, [0.99, 0.8, 1.0], 9, 9)
    assert score == 0.8 and np.array_equal(mask, masks[1])
    with pytest.raises(ValueError, match="No confident object"):
        choose_mask(masks, [0.99, 0.2, 1.0], 9, 9)


def test_lift_excludes_occluded_assigned_and_invalid_gaussians():
    torch = pytest.importorskip("torch")
    points = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [9.0, 9.0, 1.0],
            [float("nan"), 0.0, 1.0],
        ]
    )
    labels = torch.tensor([0, 0, 3, 0, 0, 0])
    mask = np.ones((4, 4), dtype=bool)
    ids = lift_mask(points, labels, mask, np.ones((4, 4)), np.eye(4), np.eye(3))
    assert ids.tolist() == [0]


def test_add_proxy_preserves_existing_motion_and_reset_pose(tmp_path):
    xml = tmp_path / "source.xml"
    xml.write_text("""<mujoco><worldbody><geom type="plane" size="5 5 .1"/>
      <body name="obj_00" pos="0 0 .2"><freejoint/>
      <geom name="target" type="box" size=".1 .1 .1" mass="1"/></body>
      </worldbody></mujoco>""")
    state = SimpleNamespace(
        scene_xml=xml,
        objects={
            "obj_00": SimpleNamespace(physics=None, meta={}),
            "obj_01": SimpleNamespace(physics=None, meta={}),
        },
    )
    sim = DemoPhysics(state, tmp_path)
    qa, da = sim.addresses("obj_00")
    sim.data.qpos[qa] = 0.3
    sim.data.qvel[da] = 0.4
    original_reset = sim.initial_qpos.copy()
    points = np.random.default_rng(0).uniform(
        [0.8, -0.1, 0.1], [1.2, 0.1, 0.4], (300, 3)
    )
    proxy = install_proxy(sim, "obj_01", points, tmp_path / "new")
    assert sim.available == {"obj_00", "obj_01"}
    assert sim.enabled == {"obj_01"}
    assert sim.data.qpos[qa] == pytest.approx(0.3)
    assert sim.data.qvel[da] == pytest.approx(0.4)
    assert np.array_equal(sim.initial_qpos[: len(original_reset)], original_reset)
    assert "unmeasured" in proxy["physics"]["source"]
    sim.perturb("obj_01", "fall")
    start_z = sim.data.qpos[sim.addresses("obj_01")[0] + 2]
    for _ in range(8):
        sim.step(0.02)
    assert sim.data.qpos[sim.addresses("obj_01")[0] + 2] < start_z
    sim.reset()
    assert sim.data.qpos[qa] == 0
    assert np.allclose(sim.data.qpos[sim.addresses("obj_01")[0] :][:3], proxy["center"])
    restart_dir = tmp_path / "restart"
    restart_dir.mkdir()
    restored = DemoPhysics(state, restart_dir)
    assert restored.available == {"obj_00", "obj_01"}
    assert np.allclose(restored.initial["obj_01"][0], proxy["center"])


def test_invalid_proxy_leaves_live_model_intact(tmp_path):
    sentinel = object()
    sim = SimpleNamespace(robot=None, model=sentinel)
    with pytest.raises(ValueError, match="insufficient"):
        install_proxy(sim, "obj_00", np.zeros((2, 3)), tmp_path)
    assert sim.model is sentinel
