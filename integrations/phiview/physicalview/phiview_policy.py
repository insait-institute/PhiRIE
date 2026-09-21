"""Bounded asynchronous learned-policy control using the DROID observation contract."""
from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid

import numpy as np

POLICY_LABELS = {
    'scripted_ik': 'Scripted IK (no learned policy)',
    'pi05_droid_jointpos': 'π0.5 DROID',
    'droid_pi05_jointpos_with_web_and_sim': 'π0.5 DROID + web + simulation',
}


def validate_actions(value):
    actions = np.asarray(value, dtype=float)
    if actions.ndim != 2 or actions.shape[0] < 15 or actions.shape[1] < 8:
        raise ValueError('Policy must return at least 15 absolute 8D actions')
    actions = actions[:15, :8].copy()
    if not np.isfinite(actions).all() or np.any((actions[:, 7] < 0) | (actions[:, 7] > 1)):
        raise ValueError('Policy returned nonfinite actions or invalid gripper values')
    return actions


def apply_action(physics, action):
    from robo.rigs.pi05_rig import MAX_JOINT_DELTA
    r, m, d = physics.robot, physics.model, physics.data
    if r is None or r['model'] != 'droid':
        raise ValueError('π0.5 DROID requires the Franka / Robotiq rig')
    q = d.qpos[r['qadr']]
    target = q + np.clip(np.asarray(action[:7])-q, -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
    limits = m.actuator_ctrlrange[r['aids']]
    d.ctrl[r['aids']] = np.clip(target, limits[:, 0], limits[:, 1])
    d.ctrl[r['grip']] = r['info']['grip_closed' if action[7] >= .5 else 'grip_open']


class LearnedPolicy:
    def __init__(self, demo):
        self.demo = demo
        self.policy_id = 'scripted_ik'
        self.status = {'state': 'idle', 'policy': self.policy_id}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='phiview-policy')
        self.process = None
        self.future = None
        self.active = False
        self.epoch = 0
        self.actions = []
        self.ticks = 0
        self.entries = {}
        try:
            from robo.policy.registry import PolicyRegistry
            registry = PolicyRegistry.from_config_dir(Path(demo.config.repo_root)/'configs/policies')
            for name in POLICY_LABELS:
                if name != 'scripted_ik':
                    self.entries[name] = registry.get(name)
        except (ImportError, KeyError, FileNotFoundError):
            pass
        if any(c['id'] == 'pi05_droid_jointpos' and c['available'] for c in self.choices()):
            self.policy_id = 'pi05_droid_jointpos'
            self.status = {'state': 'idle', 'policy': self.policy_id}
        self.server_ready = False
        atexit.register(self.close)

    def choices(self):
        return [{'id': name, 'label': label,
                 'available': name == 'scripted_ik' or (name in self.entries and
                     Path(self.entries[name].checkpoint_path).is_dir())}
                for name, label in POLICY_LABELS.items()]

    def stop(self):
        self.active = False
        self.actions = []
        self.epoch += 1
        self.demo.physics.running = False
        if getattr(self.demo.physics, 'robot', None):
            self.demo.physics.robot_status['state'] = 'idle'
        if self.status['state'] in ('loading', 'inferencing', 'running'):
            self.status = {**self.status, 'state': 'stopped'}

    def close(self):
        self.stop()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        self.executor.shutdown(wait=False, cancel_futures=True)

    def choose(self, name):
        if not any(c['id'] == name and c['available'] for c in self.choices()):
            raise ValueError('Policy checkpoint is unavailable')
        if self.future is not None and not self.future.done():
            raise ValueError('Wait for the current model request to finish before switching policy')
        self.stop()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=5)
        self.process = None
        self.policy_id = name
        self.status = {'state': 'idle', 'policy': name}
        if self.demo.physics.robot:
            self.demo.physics.robot_status['controller'] = POLICY_LABELS[name]

    def start(self, prompt, ticks=150):
        if self.policy_id == 'scripted_ik':
            raise ValueError('Choose a learned policy')
        if self.future is not None and not self.future.done():
            raise ValueError('Wait for the current model request to finish')
        if not prompt.strip():
            raise ValueError('Enter a robot command')
        if self.demo.physics.robot is None or self.demo.physics.robot['model'] != 'droid':
            raise ValueError('π0.5 DROID requires the Franka / Robotiq rig')
        ticks = int(ticks)
        if not 15 <= ticks <= 900:
            raise ValueError('Policy horizon must be between 15 and 900 ticks')
        self.stop()
        self.prompt, self.limit, self.ticks = prompt, ticks, 0
        self.run_id = str(time.time_ns())
        self.active = True
        self.demo.physics.plan = []
        self.demo.physics.robot_status.update(controller=POLICY_LABELS[self.policy_id], state='policy_running')
        self.status = {'state': 'loading', 'policy': self.policy_id, 'ticks': 0,
                       'limit': ticks, 'run_id': self.run_id, 'success': None, 'control_hz_simulation': 15}
        self._request()

    def _ensure_server(self):
        if self.process and self.process.poll() is None:
            return
        entry = self.entries[self.policy_id]
        self.server_ready = False
        self.token = uuid.uuid4().hex
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            self.port = sock.getsockname()[1]
        directory = self.demo.out/'policies'/self.token
        directory.mkdir(parents=True)
        self.log_path = directory/'server.log'
        env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   XLA_PYTHON_CLIENT_MEM_FRACTION='.30', JAX_COMPILATION_CACHE_DIR=str(self.demo.out/'policy-cache'))
        # This process belongs only to this viewer. Never reuse an unidentified port.
        with self.log_path.open('w') as log:
            self.process = subprocess.Popen([
                str(self.demo.config.interpreter('openpi')), '-m', 'physicalview.phiview_policy_server',
                '--config', entry.training_config, '--checkpoint', str(entry.checkpoint_path),
                '--policy-id', self.policy_id, '--token', self.token, '--port', str(self.port)],
                env=env, stdout=log, stderr=subprocess.STDOUT)

    def _request(self):
        from physicalview.phiview_rigs import RobotCamera
        from openpi_client.image_tools import resize_with_pad
        from PIL import Image
        d = self.demo
        self._ensure_server()
        request = {'prompt': self.prompt}
        directory = d.out/'policies'/self.token/'runs'/self.run_id/f'observation-{self.ticks:04d}'
        directory.mkdir(parents=True)
        for view, key in [('exterior', 'exterior_image_1_left'), ('wrist', 'wrist_image_left')]:
            rgb, _, depth, w2c, K = d.scene.render(RobotCamera(d.physics, view), (1280, 720),
                'simulation', None, d.physics.available, d.physics.transforms(), False)
            rgb = d.physics.overlay(rgb, depth, w2c, K)
            small = resize_with_pad(rgb, 224, 224)
            Image.fromarray(rgb).save(directory/f'{view}.jpg')
            Image.fromarray(small).save(directory/f'{view}-224.png')
            request['observation/'+key] = small
        r = d.physics.robot
        request['observation/joint_position'] = d.physics.data.qpos[r['qadr']].astype(np.float32).copy()
        grip = d.physics.model.joint(r['info']['gripper_driver_joint']).qposadr[0]
        request['observation/gripper_position'] = np.array([np.clip(d.physics.data.qpos[grip]/.8, 0, 1)], np.float32)
        epoch, policy_id, token, process, port = self.epoch, self.policy_id, self.token, self.process, self.port
        from physicalview.phiview_rigs import camera_contract
        (directory/'request.json').write_text(json.dumps({'prompt': self.prompt, 'policy': policy_id,
            'cameras': camera_contract(d.physics),
            'joints': request['observation/joint_position'].tolist(),
            'gripper': request['observation/gripper_position'].tolist()}))

        def infer():
            from websockets.sync.client import connect
            from openpi_client import msgpack_numpy
            deadline = time.monotonic()+600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Policy server exited; see {self.log_path}')
                if epoch != self.epoch:
                    return epoch, None
                try:
                    ws = connect(f'ws://127.0.0.1:{port}', compression=None, max_size=None,
                                 ping_interval=None, open_timeout=3)
                    break
                except (OSError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f'Policy loading timed out; see {self.log_path}')
                    time.sleep(1)
            with ws:
                metadata = msgpack_numpy.unpackb(ws.recv(timeout=10))
                if metadata.get('phiview_session') != token or metadata.get('policy_id') != policy_id:
                    raise ValueError('Policy server identity mismatch')
                self.server_ready = True
                ws.send(msgpack_numpy.packb(request))
                response = ws.recv(timeout=300)
                if isinstance(response, str):
                    raise RuntimeError(response[-2000:])
                response = msgpack_numpy.unpackb(response)
                actions = validate_actions(response['actions'])
                np.save(directory/'actions.npy', actions)
                (directory/'response.json').write_text(json.dumps({
                    'metadata': metadata, 'server_timing': response.get('server_timing'),
                    'actions': actions.tolist()}, default=str))
                return epoch, actions
        self.future = self.executor.submit(infer)
        self.status.update(state='inferencing' if self.server_ready else 'loading',
                           observation=str(directory), log=str(self.log_path))
        d.dirty = True

    def advance(self):
        if not self.active:
            return False
        try:
            if self.future is not None:
                if not self.future.done():
                    self.status['state'] = 'inferencing' if self.server_ready else 'loading'
                    return True
                epoch, actions = self.future.result()
                self.future = None
                if epoch != self.epoch:
                    return True
                self.actions = list(actions)
            if self.actions:
                apply_action(self.demo.physics, self.actions.pop(0))
                self.demo.physics.running = True
                self.demo.physics.step(1/15)
                self.demo.physics.running = False
                self.ticks += 1
                self.status.update(state='running', ticks=self.ticks)
                self.demo.dirty = True
            if self.ticks >= self.limit:
                self.active = False
                self.demo.physics.robot_status.update(state='finished', success=None)
                self.status.update(state='finished', success=None,
                    outcome='Policy actions executed; task success has not been scored')
            elif not self.actions:
                self._request()
        except Exception as exc:
            self.stop()
            self.status.update(state='failed', error=str(exc))
        return True
