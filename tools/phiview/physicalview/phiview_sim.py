"""MuJoCo demo interactions. No teleport-grasp or fabricated task success."""
from __future__ import annotations

import xml.etree.ElementTree as ET
import hashlib
import copy
import numpy as np


def support_path(result_set, out):
    key = hashlib.sha256(str(result_set.splat_ply).encode()).hexdigest()[:16]
    return out / f'static-supports-{key}.xml'


def preserve_supports(result_set, out):
    """Keep existing primitive room supports when discovery replaces object IDs."""
    source = result_set.out_dir / 'sim_export' / 'scene.xml'
    if not source.exists():
        return
    root = ET.Element('mujoco')
    world = ET.SubElement(root, 'worldbody')
    for geom in ET.parse(source).getroot().findall('worldbody/geom'):
        if geom.get('type') in ('plane', 'box', 'sphere', 'capsule', 'cylinder', 'ellipsoid'):
            saved = copy.deepcopy(geom)
            saved.attrib.pop('material', None)
            world.append(saved)
    if len(world):
        ET.ElementTree(root).write(support_path(result_set, out))


class DemoPhysics:
    def __init__(self, state, out):
        import mujoco
        self.state, self.out = state, out
        if state.scene_xml is None:
            cached = support_path(state.result_set, out) if hasattr(state, 'result_set') else None
            root = ET.parse(cached) if cached and cached.exists() else ET.ElementTree(
                ET.fromstring('<mujoco><asset/><worldbody/></mujoco>'))
        else:
            root = ET.parse(state.scene_xml)
        compiler = root.find('compiler')
        if compiler is not None:
            meshdir = compiler.get('meshdir', '')
            if meshdir and not meshdir.startswith('/'):
                compiler.set('meshdir', str(state.scene_xml.parent/meshdir))
        world = root.find('worldbody')
        from physicalview.phiview_proxy import append_proxy
        for name, rec in state.objects.items():
            proxy = rec.meta.get('interactive_proxy') if hasattr(rec, 'meta') else None
            if proxy:
                append_proxy(root.getroot(), name, proxy)
        # A bounded pool of real collision projectiles; inactive ones have collisions off.
        for i in range(8):
            b = ET.SubElement(world, 'body', name=f'phiview_ball_{i}', pos=f'0 0 {-20-i}')
            ET.SubElement(b, 'freejoint')
            ET.SubElement(b, 'geom', name=f'phiview_ball_{i}', type='sphere', size='.025',
                          mass='.08', rgba='1 .35 .08 1', friction='.5 .005 .0001',
                          contype='1', conaffinity='1')
        self.xml = out/'demo_scene.xml'
        root.write(self.xml)
        self.base_xml = ET.tostring(root.getroot())
        self.variant_specs = {}
        self.model = mujoco.MjModel.from_xml_path(str(self.xml))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.robot = None
        self.robot_model = 'droid'
        self.renderer = None
        self.running = False
        self.projectile = 0
        self.enabled = set()
        self.initial = self._poses()
        self.initial_qpos = self.data.qpos.copy()
        self.plan = []
        self.plan_step = 0
        self.plan_ticks = 0
        self.robot_status = {'state': 'absent'}
        self.last_action = None
        self.events = []
        self.reset()

    def replace_variant(self, name, proposal):
        """Rebuild both collision and inertia when a registered visual proposal changes."""
        import mujoco
        import trimesh
        from scipy.spatial.transform import Rotation
        variants = dict(self.variant_specs)
        if proposal is None:
            variants.pop(name, None)
        else:
            if proposal.mesh_ply is None:
                raise ValueError('Alternative has no mesh for physical simulation')
            T = np.asarray(proposal.aligned['T'], float)
            scale = np.linalg.norm(T[:3, :3], axis=0)
            R = T[:3, :3]/scale
            if not np.allclose(R.T@R, np.eye(3), atol=1e-3) or np.linalg.det(R) < 0:
                raise ValueError('Alternative transform is not a valid positive rigid scale')
            mesh = trimesh.load(proposal.mesh_ply, force='mesh', process=False)
            mesh.vertices = np.asarray(mesh.vertices)*scale
            path = self.out/f'{name}-{proposal.source}-convex.obj'
            mesh.convex_hull.export(path)
            q = Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]
            variants[name] = (path, T[:3, 3], q)
        root = ET.fromstring(self.base_xml)
        asset = root.find('asset')
        for obj, (path, pos, quat) in variants.items():
            body = root.find(f".//body[@name='{obj}']")
            if body is None:
                raise ValueError('Alternative object has no exported body; build physics first')
            body.set('pos', ' '.join(map(str, pos))); body.set('quat', ' '.join(map(str, quat)))
            for child in list(body):
                if child.tag in ('geom', 'inertial'):
                    body.remove(child)
            meshname = f'phiview_{obj}_variant'
            ET.SubElement(asset, 'mesh', name=meshname, file=str(path))
            params = self.state.objects[obj].physics or {}
            ET.SubElement(body, 'geom', name=f'{obj}_variant_collision', type='mesh', mesh=meshname,
                          mass=str(params.get('mass_kg', .2)),
                          friction=f"{params.get('friction', .6)} .005 .0001", rgba='.6 .7 .8 1')
        # Compile before replacing the live model, so a bad mesh leaves the old model intact.
        text = ET.tostring(root).decode()
        model = mujoco.MjModel.from_xml_string(text)
        if self.renderer:
            self.renderer.close(); self.renderer = None
        self.xml.write_text(text)
        self.variant_specs = variants
        self.model, self.data = model, mujoco.MjData(model)
        mujoco.mj_forward(model, self.data)
        self.initial = self._poses(); self.initial_qpos = self.data.qpos.copy()
        self.robot = None; self.plan = []; self.running = False
        self.robot_status = {'state': 'absent'}
        self.projectile = 0
        self.reset()

    @property
    def available(self):
        return set(self._poses())

    def _poses(self):
        m, d = self.model, self.data
        return {m.body(b).name: (d.xpos[b].copy(), d.xquat[b].copy())
                for b in range(m.nbody) if m.body(b).name in self.state.objects
                and m.body_jntnum[b] == 1 and m.jnt_type[m.body_jntadr[b]] == 0}

    def transforms(self):
        from scipy.spatial.transform import Rotation
        out = {}
        for name, (p, q) in self._poses().items():
            if name not in self.enabled:
                continue
            p0, q0 = self.initial[name]
            R = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            R0 = Rotation.from_quat(q0[[1, 2, 3, 0]]).as_matrix()
            T = np.eye(4); T[:3, :3] = R@R0.T; T[:3, 3] = p-T[:3, :3]@p0
            out[name] = T
        return out

    def addresses(self, name):
        b = self.model.body(name).id
        j = self.model.body_jntadr[b]
        if j < 0 or self.model.jnt_type[j] != 0:
            raise ValueError(f'{name} has no free rigid body')
        return int(self.model.jnt_qposadr[j]), int(self.model.jnt_dofadr[j])

    def enable(self, names):
        names = set(names)
        missing = names-self.available
        if missing:
            raise ValueError(f'Collision bodies unavailable: {sorted(missing)}')
        self.enabled |= names

    def reset(self):
        import mujoco
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = 0
        self.data.time = 0
        self.data.qacc_warmstart[:] = 0
        self.data.xfrc_applied[:] = 0
        for i in range(8):
            g = self.model.geom(f'phiview_ball_{i}').id
            self.model.geom_contype[g] = self.model.geom_conaffinity[g] = 0
        self.running = False
        self.plan = []
        self.robot_status = {'state': 'idle' if self.robot else 'absent'}
        if self.robot:
            self.data.ctrl[self.robot['aids']] = self.data.qpos[self.robot['qadr']]
            self.data.ctrl[self.robot['grip']] = self.robot['info']['grip_open']
        mujoco.mj_forward(self.model, self.data)

    def parameters(self, name):
        if name not in self.available:
            return None
        b = self.model.body(name).id
        geoms = np.where(self.model.geom_bodyid == b)[0]
        return {'mass_kg': float(self.model.body_mass[b]),
                'inertia_kg_m2': self.model.body_inertia[b].tolist(),
                'friction': self.model.geom_friction[geoms[0]].tolist(),
                'estimate': self.state.objects[name].physics,
                'source': 'active MuJoCo collision model', 'enabled': name in self.enabled}

    def set_friction(self, name, value):
        value = float(value)
        if not np.isfinite(value) or not 0 <= value <= 3:
            raise ValueError('Friction must be between 0 and 3')
        bid = self.model.body(name).id
        self.model.geom_friction[self.model.geom_bodyid == bid, 0] = value

    def perturb(self, name, kind, direction=None, strength=3.):
        import mujoco
        self.enable([name])
        qa, da = self.addresses(name)
        strength = float(np.clip(float(strength), .1, 15))
        if kind == 'fall':
            self.data.qpos[qa+2] += .3
            self.data.qvel[da:da+6] = 0
        elif kind == 'friction':
            self.data.qvel[da:da+3] = [.7, 0, 0]
        elif kind == 'throw':
            v = np.asarray(direction, float)
            v /= max(np.linalg.norm(v), 1e-9)
            self.data.qvel[da:da+3] = v*strength + [0, 0, 1.]
        else:
            raise ValueError('Unknown physical interaction')
        mujoco.mj_forward(self.model, self.data)
        self.running = True
        self.last_action = {'kind': kind, 'object': name, 'time': float(self.data.time)}

    def shoot(self, origin, direction, speed=12.):
        import mujoco
        name = f'phiview_ball_{self.projectile % 8}'
        self.projectile += 1
        qa, da = self.addresses(name)
        v = np.asarray(direction, float); v /= max(np.linalg.norm(v), 1e-9)
        self.data.qpos[qa:qa+3] = np.asarray(origin)+v*.12
        self.data.qpos[qa+3:qa+7] = [1, 0, 0, 0]
        self.data.qvel[da:da+6] = 0
        self.data.qvel[da:da+3] = v*float(np.clip(speed, 1, 20))
        g = self.model.geom(name).id
        self.model.geom_contype[g] = self.model.geom_conaffinity[g] = 1
        self.enable(self.available)
        self.running = True
        mujoco.mj_forward(self.model, self.data)
        self.last_action = {'kind': 'shoot', 'projectile': name, 'speed_m_s': float(speed)}

    def step(self, seconds):
        import mujoco
        if not self.running:
            return
        m, d = self.model, self.data
        if self.plan:
            self._robot_control()
        disabled = self.available-self.enabled
        for _ in range(max(1, int(round(min(seconds, .1)/m.opt.timestep)))):
            mujoco.mj_step(m, d)
            for name in disabled:
                qa, da = self.addresses(name)
                d.qpos[qa:qa+7] = self.initial_qpos[qa:qa+7]
                d.qvel[da:da+6] = 0
            for c in d.contact:
                a = m.geom(c.geom1).name or ''; b = m.geom(c.geom2).name or ''
                if self.robot:
                    ba = m.body(int(m.geom_bodyid[c.geom1])).name or ''
                    bb = m.body(int(m.geom_bodyid[c.geom2])).name or ''
                    target = self.robot['target']
                    if ((ba.startswith('robot/') and ba != 'robot/pedestal' and bb == target) or
                        (bb.startswith('robot/') and bb != 'robot/pedestal' and ba == target)):
                        self.robot_status['target_contact_steps'] = self.robot_status.get('target_contact_steps', 0)+1
                        self.robot_status['last_target_contact'] = {'time':float(d.time),'bodies':[ba,bb], 'position':c.pos.tolist()}
                if a.startswith('phiview_ball') or b.startswith('phiview_ball'):
                    pair = sorted([a, b])
                    if not self.events or self.events[-1]['geoms'] != pair:
                        self.events.append({'time': float(d.time), 'geoms': pair})
                        self.events = self.events[-100:]
        mujoco.mj_forward(m, d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            self.running = False
            raise RuntimeError('Simulation became nonfinite; reset required')

    def add_robot(self, name, camera_position=None):
        import mujoco
        from physicalview.phiview_rigs import build_robot
        self.enable([name])
        self.reset()
        target = np.array(self.initial[name][0])
        # Put the base within Panda reach, on the object's estimated support height.
        rec = self.state.objects[name]
        lo = rec.meta.get('aabb', [[*target]])[0]
        direction = np.array([.48, .12, 0.])
        if camera_position is not None:
            direction = np.asarray(camera_position, dtype=float)-target
            direction[2] = 0.
            if not np.isfinite(direction).all() or np.linalg.norm(direction)<1e-6:
                raise ValueError('Cannot place an arm from this camera position')
        direction = direction/np.linalg.norm(direction)
        if camera_position is not None:
            # Offset to the side so the arm does not cover the selected object.
            placement = getattr(self, 'placement_options', {})
            angle = np.deg2rad(float(placement.get('angle_degrees', 50.)))
            dx, dy = direction[:2]
            direction[:2] = [dx*np.cos(angle)-dy*np.sin(angle), dx*np.sin(angle)+dy*np.cos(angle)]
        # Approach from the viewer's open side of the workspace; a fixed world
        # +X offset placed arms behind kitchen walls and inside upper cabinets.
        placement = getattr(self, 'placement_options', {})
        base = target + direction*float(placement.get('radius_m', .50))
        base[2] = float(lo[2])+float(placement.get('height_offset_m', -.04))
        yaw = float(np.arctan2(target[1]-base[1], target[0]-base[0]))
        source_xml=self.xml
        pedestal_bottom=placement.get('pedestal_bottom_z')
        if pedestal_bottom is not None:
            height=base[2]-float(pedestal_bottom)
            if not .05<=height<=1.5:raise ValueError('Pedestal height outside the supported range')
            root=ET.parse(self.xml);world=root.find('worldbody')
            stand=ET.SubElement(world,'body',name='robot/pedestal',pos=f'{base[0]} {base[1]} {float(pedestal_bottom)+height/2}')
            ET.SubElement(stand,'geom',name='robot/pedestal',type='box',size=f'.12 .12 {height/2}',
                          rgba='.25 .3 .35 1',friction='.8 .005 .0001',contype='1',conaffinity='1')
            source_xml=self.out/'robot_mount_scene.xml';root.write(source_xml)
        model, info = build_robot(self.robot_model, source_xml, base, yaw)
        if self.renderer:
            self.renderer.close(); self.renderer = None
        self.model, self.data = model, mujoco.MjData(model)
        qadr = np.array([model.joint(n).qposadr[0] for n in info['arm_joints']])
        dadr = np.array([model.joint(n).dofadr[0] for n in info['arm_joints']])
        aids = np.array([model.actuator(n).id for n in info['arm_actuators']])
        self.data.qpos[qadr] = info['home']; self.data.ctrl[aids] = info['home']
        self.robot = {'qadr': qadr, 'dadr': dadr, 'aids': aids,
                      'grip': model.actuator(info['gripper_actuator']).id,
                      'base': base.tolist(), 'target': name, 'info': info, 'model': self.robot_model}
        self.data.ctrl[self.robot['grip']] = info['grip_open']
        mujoco.mj_forward(model, self.data)
        self.initial = self._poses(); self.initial_qpos = self.data.qpos.copy()
        self.robot_status = {'state': 'idle', 'robot_model': self.robot_model, 'base': base.tolist(), 'controller': 'scripted IK with physical contacts',
                             'placement': {'method':'offset beside camera-facing direction','angle_degrees':float(placement.get('angle_degrees',50.)), 'radius_m':float(placement.get('radius_m',.5)), 'height_offset_m':float(placement.get('height_offset_m',-.04))},
                             'mount_support_verified': False}
        if pedestal_bottom is not None:
            self.robot_status['pedestal']={'bottom_z':float(pedestal_bottom),'top_z':float(base[2]),'half_width_m':.12,
                                           'source':'Inserted static MuJoCo mount; support extent still requires validation'}

    def command_robot(self, name, command, camera_position=None):
        text = command.lower().strip()
        allowed = ('reach', 'pick', 'lift', 'place', 'move', 'push')
        if not any(word in text.split() for word in allowed):
            raise ValueError('Supported commands: reach, lift, pick/place left/right, push left/right')
        if self.robot is None or self.robot['target'] != name or self.robot['model'] != self.robot_model:
            self.add_robot(name, camera_position=camera_position)
        self.enable([name])
        p = self.data.xpos[self.model.body(name).id].copy()
        direction = -1 if 'left' in text else 1
        above = p + [0, 0, .20]
        grasp = p + [0, 0, .035]
        plan = [(above, 0), (grasp, 0)]
        if 'push' in text:
            push_direction = p-np.asarray(self.robot['base'])
            push_direction[2] = 0
            push_direction /= max(np.linalg.norm(push_direction), 1e-8)
            push_direction *= direction
            plan = [(p-push_direction*.14+[0,0,.03], 1), (p+push_direction*.18+[0,0,.03], 1)]
        elif any(w in text for w in ('pick', 'lift', 'place', 'move')):
            plan += [(grasp, 1), (above, 1)]
            if any(w in text for w in ('place', 'move')):
                plan += [(above+[0, .22*direction, 0], 1),
                         (grasp+[0, .22*direction, 0], 1), (grasp+[0, .22*direction, 0], 0)]
        self.plan = plan
        self.plan_step = self.plan_ticks = 0
        self.robot_start_z = float(p[2])
        self.robot_start_position = p.copy()
        self.robot_status.update(state='running', command=command, stage=0, stages=len(plan),
                                 lifted=False, max_lift_m=0., success=None, target_contact_steps=0, max_target_displacement_m=0.)
        self.running = True

    def _robot_control(self):
        from physicalview.ik import solve_ik
        r = self.robot
        point, grip = self.plan[self.plan_step]
        # Down-facing pinch frame. Joint targets are applied through actuators only.
        result = solve_ik(self.model, self.data, r['info']['pinch_site'], point,
                          [0, 1, 0, 0], r['qadr'], r['dadr'], iters=35, pos_tol=.015, rot_tol=.15)
        q = self.data.qpos[r['qadr']]
        previous_target = self.data.ctrl[r['aids']].copy()
        self.data.ctrl[r['aids']] = previous_target + np.clip(result.q-previous_target, -.08, .08)
        self.data.ctrl[r['grip']] = r['info']['grip_closed' if grip else 'grip_open']
        self.plan_ticks += 1
        height = float(self.data.xpos[self.model.body(r['target']).id, 2])-self.robot_start_z
        self.robot_status['max_lift_m'] = max(self.robot_status['max_lift_m'], height)
        displacement = float(np.linalg.norm(self.data.xpos[self.model.body(r['target']).id]-self.robot_start_position))
        self.robot_status['max_target_displacement_m'] = max(self.robot_status['max_target_displacement_m'], displacement)
        self.robot_status['lifted'] = self.robot_status['max_lift_m'] > .05
        self.robot_status['ik_solution_error_m'] = float(result.pos_err)
        self.robot_status['ik_error_m'] = float(np.linalg.norm(self.data.site(r['info']['pinch_site']).xpos-point))
        self.robot_status['end_effector_position'] = self.data.site(r['info']['pinch_site']).xpos.tolist()
        self.robot_status['target_position'] = self.data.xpos[self.model.body(r['target']).id].tolist()
        if self.plan_ticks >= 60:
            self.plan_step += 1; self.plan_ticks = 0
            self.robot_status['stage'] = self.plan_step
            if self.plan_step == len(self.plan):
                self.plan = []
                self.robot_status.update(state='finished', success=None,
                    outcome='motion executed; inspect measured lift and contacts')

    def overlay(self, rgb, depth, w2c, K):
        import mujoco
        from physicalview.streaming import apply_free_camera, vertical_fov_deg
        self.overlay_mask = None
        if not self.robot and self.projectile == 0:
            return rgb
        h, w = rgb.shape[:2]
        m, d = self.model, self.data
        # MuJoCo scales clipping distances by scene extent. Inactive projectiles
        # parked below the room enlarge that extent and otherwise cut through
        # nearby robot links. Match the GS renderer's physical clipping range.
        m.vis.map.znear = .01 / max(float(m.stat.extent), 1e-6)
        m.vis.map.zfar = 100. / max(float(m.stat.extent), 1e-6)
        if self.renderer is None or self.renderer.width != w or self.renderer.height != h:
            if self.renderer:
                self.renderer.close()
            m.vis.global_.offwidth = max(w, m.vis.global_.offwidth)
            m.vis.global_.offheight = max(h, m.vis.global_.offheight)
            self.renderer = mujoco.Renderer(m, height=h, width=w)
        cam = apply_free_camera(mujoco.MjvCamera(), w2c, 2.)
        m.vis.global_.fovy = vertical_fov_deg(K, h)
        renderer = self.renderer
        renderer.update_scene(d, camera=cam)
        simrgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        simdepth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        seg = renderer.render().copy()
        renderer.disable_segmentation_rendering()
        visible = [i for i in range(m.ngeom)
                   if (m.body(int(m.geom_bodyid[i])).name or '').startswith('robot/')
                   or (m.geom(i).name or '').startswith('phiview_ball_')]
        mask = np.isin(seg[..., 0], visible) & ((simdepth <= depth+.015) | (depth <= 0))
        self.overlay_mask = mask
        rgb[mask] = simrgb[mask]
        return rgb
