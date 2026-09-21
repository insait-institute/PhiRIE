"""CPU contracts for image-only controls and truthful command rejection."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from physicalview.phiview_scene import FlyCamera
from physicalview.phiview import Demo


def test_camera_diagonals_pitch_and_stall_bound():
    a = FlyCamera(np.zeros(3), 0, 0, speed=1)
    b = FlyCamera(np.zeros(3), 0, 0, speed=1)
    a.update(['w'], .1); b.update(['w', 'd'], .1)
    assert np.linalg.norm(a.position) == pytest.approx(np.linalg.norm(b.position))
    a.update([], 10, [0, 100000])
    assert -np.pi/2 < a.pitch < np.pi/2
    before = b.position.copy(); b.update(['e'], 100)
    assert np.linalg.norm(b.position-before) == pytest.approx(.1)
    w2c, K = a.matrices((1920, 1080))
    assert np.allclose(np.linalg.det(w2c[:3, :3]), 1)
    assert np.allclose(K[:2, 2], [960, 540])


def test_mouse_right_turns_toward_camera_right():
    camera = FlyCamera(np.zeros(3), .7, 0)
    right = np.array([np.sin(camera.yaw), -np.cos(camera.yaw), 0])
    camera.update([], .02, [30, 0])
    assert camera.forward() @ right > 0


def test_pick_uses_displayed_frame_and_rejects_expired():
    d = Demo.__new__(Demo)
    d.frame_id = 9
    d.frames = {7: (np.array([[0, 2], [1, 0]]), None, None, None)}
    d.scene = SimpleNamespace(names=['obj_00', 'obj_01'])
    d.click_selection = SimpleNamespace(busy=False, status={})
    d.physics = SimpleNamespace(data=SimpleNamespace(time=0))
    import io
    d.audit = io.StringIO()
    result = d.execute({'op': 'pick', 'frame': 7, 'x': .75, 'y': .25})
    assert result['selected'] == 'obj_01'
    with pytest.raises(ValueError, match='expired'):
        d.execute({'op': 'pick', 'frame': 6, 'x': .5, 'y': .5})
    with pytest.raises(ValueError, match='outside'):
        d.execute({'op': 'pick', 'frame': 7, 'x': 1, 'y': .5})


def test_invalid_controls_cannot_poison_camera():
    d = Demo.__new__(Demo); d.look = np.zeros(2)
    with pytest.raises(ValueError, match='mouse'):
        d.execute({'op': 'input', 'look': [float('nan'), 0]})
    assert np.allclose(d.look, 0)


def test_client_is_image_only_and_releases_navigation():
    html = (Path(__file__).parents[1]/'physicalview/web/phiview.html').read_text()
    assert '<img id="frame"' in html
    for forbidden in ('<canvas', 'WebGL', 'THREE.', '.ply', '.splat', 'unpkg.com', 'cdn.'):
        assert forbidden not in html
    mouse = (Path(__file__).parents[1]/'physicalview/web/phiview_mouse.js').read_text()
    assert "'blur',clearInput" in mouse and 'document.hidden' in mouse
    assert "'X-Frame-Id'" in html


def test_robot_rejects_unsupported_instructions():
    from physicalview.phiview_sim import DemoPhysics
    d = DemoPhysics.__new__(DemoPhysics)
    with pytest.raises(ValueError, match='Supported commands'):
        d.command_robot('obj_00', 'please dance')


def test_prompt_guard_rejects_cache_and_fallback(tmp_path):
    from physicalview.phiview_inpaint_guard import validate
    obj = tmp_path/'obj_00'; obj.mkdir()
    (obj/'views.json').write_text('[{}]')
    (obj/'inpainted_0.png').write_bytes(b'image')
    (tmp_path/'edit_meta.json').write_text(json.dumps({'prompt': 'wood', 'backend_final': 'qwen'}))
    (tmp_path/'inpaint_meta.json').write_text('{}')
    with pytest.raises(ValueError, match='cached'):
        validate(tmp_path, ['obj_00'], 'wood')
    (tmp_path/'inpaint_meta.json').write_text('{"obj_00/0":"lama"}')
    with pytest.raises(ValueError, match='non-prompt'):
        validate(tmp_path, ['obj_00'], 'wood')
    (tmp_path/'inpaint_meta.json').write_text('{"obj_00/0":"qwen"}')
    assert validate(tmp_path, ['obj_00'], 'wood')['verified']


def test_projectiles_activate_broadphase_and_make_contacts(tmp_path):
    from physicalview.phiview_sim import DemoPhysics
    xml = tmp_path/'source.xml'
    xml.write_text('''<mujoco><worldbody><geom type="plane" size="5 5 .1"/>
      <body name="obj_00" pos="0 0 .2"><freejoint/>
      <geom name="target" type="box" size=".1 .1 .1" mass="1"/></body>
      </worldbody></mujoco>''')
    state = SimpleNamespace(scene_xml=xml, objects={'obj_00': SimpleNamespace(physics=None)})
    sim = DemoPhysics(state, tmp_path)
    sim.shoot([.4, 0, .2], [-1, 0, 0], 3)
    for _ in range(15):
        sim.step(.02)
    assert sim.events
    assert any('target' in e['geoms'] for e in sim.events)
    sim.reset()
    assert not sim.running and np.allclose(sim.data.qvel, 0)
