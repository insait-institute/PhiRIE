"""Robot hardware and physical camera contracts for the image-only viewer."""
from __future__ import annotations

import numpy as np

ROBOTS = [
    {'id': 'droid', 'label': 'Franka + Robotiq 2F-85 (DROID)', 'pi05_compatible': True},
    {'id': 'panda', 'label': 'Franka Panda hand', 'pi05_compatible': False},
]


def validate_robot(name):
    if name not in {r['id'] for r in ROBOTS}:
        raise ValueError('Unknown robot model')
    return name


def build_robot(name, scene_xml, base, yaw):
    import mujoco
    from robo.rigs import pi05_rig as rig
    validate_robot(name)
    if name == 'droid':
        model, info = rig.build_scene_model(scene_xml, base, yaw)
        info.update(pinch_site='robot/2f85/pinch', grip_open=0., grip_closed=255.)
        return model, info
    scene = mujoco.MjSpec.from_file(str(scene_xml))
    path = rig.MENAGERIE / 'franka_emika_panda' / 'panda.xml'
    panda = mujoco.MjSpec.from_file(str(path))
    rig._absolutize_assets(panda, path)
    rig._delete_keyframes(panda)
    hand = panda.body('hand')
    hand.add_site(name='pinch', pos=[0, 0, .105], size=[.005, .005, .005])
    pos = np.array([0., -.055, .02])
    hand.add_camera(name='wrist', pos=pos,
                    quat=rig.lookat_quat(pos, [0, 0, .4], up=[0, -1, 0]), fovy=58.)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    frame = scene.worldbody.add_frame(pos=base, quat=rig.rot_to_quat_wxyz(R))
    scene.option.timestep = rig.PHYSICS_DT
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    scene.attach(panda, frame=frame, prefix='robot/')
    pos = np.asarray(base) + R @ [.05, .57, .66]
    target = np.asarray(base) + R @ [.55, 0, .10]
    scene.worldbody.add_camera(name='ext_cam', pos=pos,
                               quat=rig.lookat_quat(pos, target), fovy=68.)
    scene.option.timestep = rig.PHYSICS_DT
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model = scene.compile()
    return model, {
        'arm_joints': [f'robot/joint{i}' for i in range(1, 8)],
        'arm_actuators': [f'robot/actuator{i}' for i in range(1, 8)],
        'gripper_actuator': 'robot/actuator8',
        'gripper_driver_joint': 'robot/finger_joint1',
        'ext_cam': 'ext_cam', 'wrist_cam': 'robot/wrist',
        'home': rig.PANDA_HOME.copy(), 'pinch_site': 'robot/pinch',
        'grip_open': 255., 'grip_closed': 0.,
    }


class RobotCamera:
    def __init__(self, physics, view):
        if physics.robot is None:
            raise ValueError('Place a robot before choosing its cameras')
        self.physics = physics
        self.name = physics.robot['info'][{'exterior': 'ext_cam', 'wrist': 'wrist_cam'}[view]]

    def matrices(self, wh):
        from physicalview.render import mujoco_camera_to_w2c_K
        return mujoco_camera_to_w2c_K(self.physics.model, self.physics.data, self.name, wh)


def camera_contract(physics):
    if physics.robot is None:
        return []
    result = []
    for view in ('exterior', 'wrist'):
        cam = RobotCamera(physics, view)
        w2c, K = cam.matrices((1280, 720))
        result.append({'id': view, 'camera': cam.name, 'resolution': [1280, 720],
                       'w2c': w2c.tolist(), 'K': K.tolist(),
                       'policy_resize': [224, 224], 'preprocessing': 'resize with padding'})
    return result
