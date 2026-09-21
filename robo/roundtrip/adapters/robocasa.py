"""Privileged native RoboCasa adapter. Never mount this module in constructor workers.

The official Gym wrapper owns observation formatting, action conversion and
native success. No DROID robot, state packing or scorer is used here.
"""
from pathlib import Path
import copy
from contextlib import contextmanager
import random
import importlib.metadata
import json
import subprocess
import numpy as np
from robo.manifest.hash import canonical_hash


def observation_hashes(obs):
    import hashlib
    return {k: (hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                if isinstance(v, np.ndarray) else canonical_hash(v)) for k,v in obs.items()}

@contextmanager
def native_rng_scope(seed):
    """Seed native legacy RNG users without perturbing other episode streams."""
    py_state=random.getstate();np_state=np.random.get_state()
    random.seed(int(seed));np.random.seed(int(seed))
    try:
        yield
    finally:
        random.setstate(py_state);np.random.set_state(np_state)

class RoboCasaAdapter:
    def __init__(self, config):
        self.config=copy.deepcopy(config); self.env=None; self.observation=None; self._source_xml=None

    @property
    def native(self):
        return self.env.unwrapped.env

    def load_instance(self):
        import gymnasium as gym
        import robocasa
        import robosuite
        import mujoco
        c=self.config
        for mod,key in ((robocasa,'robocasa_commit'),(robosuite,'robosuite_commit')):
            p=Path(mod.__file__).resolve().parents[1]
            sha=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
            if sha!=c['platform'][key]: raise ValueError(f'{key} runtime differs')
        if mujoco.__version__!=c['platform']['mujoco_version']:raise ValueError('MuJoCo runtime differs')
        instance=c['instance']
        with native_rng_scope(c['reset_seeds'][0]):
            self.env=gym.make('robocasa/'+instance['task_id'],
                robots=c['robot'], seed=c['reset_seeds'][0], split=None,
                obj_instance_split='pretrain', layout_ids=[instance['layout_id']],
                style_ids=[instance['style_id']], camera_widths=c['camera_size'][0],
                camera_heights=c['camera_size'][1], generative_textures=None,
                randomize_cameras=False)
        return self

    def reset_from_spec(self, reset):
        if self.env is None:self.load_instance()
        self._source_xml=None
        with native_rng_scope(reset['seed']):
            self.observation,_=self.env.reset(seed=int(reset['seed']))
        if self.native_success():raise ValueError('native reset is already terminal')
        return self.observation

    def step_native_action(self, action):
        action=np.asarray(action)
        if action.shape!=(12,) or not np.isfinite(action).all():raise ValueError('native policy action must be finite12D')
        from robocasa.utils.env_utils import convert_action
        if getattr(self,'scope_contact_inventory',None) is not None:
            from robo.roundtrip.scope_contacts import record_physics_contacts,substep_report
            with record_physics_contacts(self.native.sim,self.scope_contact_inventory) as contacts:
                obs,reward,done,truncated,info=self.env.step(convert_action(action))
            self.last_scope_contact_steps=substep_report(contacts,self.scope_contact_inventory)
        else:
            obs,reward,done,truncated,info=self.env.step(convert_action(action))
        self.observation=obs
        return obs,reward,bool(done or truncated),info

    def get_policy_observation(self):
        if self.observation is None:raise RuntimeError('reset required')
        return self.observation

    def refresh_observation(self):
        self.observation=self.env.unwrapped.get_observation(self.native._get_observations(force_update=True))
        return self.observation

    def native_success(self):return bool(self.native._check_success())
    def native_stage_state(self):
        if self.config.get('schema_version')==2:
            from robo.roundtrip.scorer import predicate_components
            result=predicate_components(self.native,self.config['instance']['task_id'])
            if result['native_task_success']!=self.native_success():
                raise ValueError('audited predicate components disagree with native task')
            result['generated_scorer_correspondence']='NOT_ESTABLISHED'
            return result
        return {'native_task_success':self.native_success()}

    def apply_paired_reset(self, reset, *, role='obj'):
        """After canonical/import restoration, perturb each arm's own placement.

        This is experiment reset scaffolding, never a per-step pose setter.
        The importer has already preserved the estimated reconstructed pose.
        """
        from robo.roundtrip.identity import perturb_pose, validate_delta
        import mujoco
        delta=validate_delta(reset['delta_world']);sim=self.native.sim
        obj=self.native.objects[role]
        if len(obj.joints)!=1:raise ValueError('paired reset supports one free rigid target')
        joint=obj.joints[0];pose=sim.data.get_joint_qpos(joint).copy()
        if len(pose)!=7:raise ValueError('target joint is not free')
        estimated=perturb_pose(pose,delta)
        receipt={'reset_id':reset['reset_id'],'delta_world':delta.tolist(),
                 'target_initial_pose_before':pose.tolist(),'target_initial_pose_after':estimated,
                 'pose_source':'each_arm_own_canonical_initial_pose','reference_pose_substitution':False}
        if np.array_equal(delta,np.eye(4)):
            return receipt
        state=self.get_state()
        sim.data.set_joint_qpos(joint,np.asarray(estimated))
        velocity=sim.data.get_joint_qvel(joint).copy()
        velocity[:3]=delta[:3,:3]@velocity[:3]
        sim.data.set_joint_qvel(joint,velocity)
        mask=mujoco.mjtState.mjSTATE_INTEGRATION
        integration=np.empty(mujoco.mj_stateSize(sim.model._model,mask))
        mujoco.mj_getState(sim.model._model,sim.data._data,integration,mask)
        sim.forward()
        for robot,saved in zip(self.native.robots,state['controller_state'],strict=True):
            for name,controller in robot.composite_controller.part_controllers.items():
                controller.update(force=True)
                controller.new_update=saved['parts'][name].get('new_update',True)
        self.refresh_observation()
        for name,timing in state['observable_timing'].items():
            for key,value in timing.items():setattr(self.native._observables[name],key,value)
        mujoco.mj_setState(sim.model._model,sim.data._data,integration,mask)
        return receipt


    def get_named_body_state(self):
        sim=self.native.sim
        return {sim.model.body_id2name(i) or f'body_{i}':np.r_[sim.data.body_xpos[i],sim.data.body_xquat[i]].tolist()
                for i in range(sim.model.nbody)}

    def tracked_objects(self):
        n=self.native
        return {role:np.r_[n.sim.data.body_xpos[bid],n.sim.data.body_xquat[bid]].tolist()
                for role,bid in n.obj_body_id.items()}

    def source_xml(self):
        # Recompile the original full-precision MJCF. mj_saveLastXML rounds
        # numeric parameters and is not a lossless importer identity control.
        return self._source_xml if self._source_xml is not None else self.native.model.get_xml()

    def get_state(self):
        import mujoco
        sim=self.native.sim
        mask=mujoco.mjtState.mjSTATE_INTEGRATION
        integration=np.empty(mujoco.mj_stateSize(sim.model._model,mask))
        mujoco.mj_getState(sim.model._model,sim.data._data,integration,mask)
        controller_fields=('initial_joint','initial_ref_pos','initial_ref_ori_mat',
            'goal_pos','goal_ori','goal_qpos','goal_qvel','goal_torque',
            'origin_pos','origin_ori','init_pos','init_ori','relative_ori','ori_ref',
            '_goal_update_mode','goal_vel','current_vel','last_err','summed_err',
            'saturated','last_joint_vel','new_update')
        controllers=[]
        for robot in self.native.robots:
            controllers.append({'parts':{name:{key:(getattr(c,key).tolist() if isinstance(getattr(c,key),np.ndarray) else getattr(c,key))
                for key in controller_fields if hasattr(c,key)}
                for name,c in robot.composite_controller.part_controllers.items()},
                'grippers':{name:g.current_action.tolist() for name,g in robot.gripper.items()},
                'derivative_buffers':{name:{'buf':c.derr_buf.buf.tolist(),'ptr':c.derr_buf.ptr,'size':c.derr_buf._size}
                    for name,c in robot.composite_controller.part_controllers.items() if hasattr(c,'derr_buf')}})
        return {'integration_state':integration.tolist(),'controller_state':controllers,
                'observable_timing':{name:{k:getattr(obs,k) for k in ('_time_since_last_sample','_current_delay','_sampled')}
                    for name,obs in self.native._observables.items()},
                'state_flat':sim.get_state().flatten().tolist(),'qpos':sim.data.qpos.tolist(),
                'qvel':sim.data.qvel.tolist(),'act':sim.data.act.tolist(),
                'time':float(sim.data.time),'timestep':int(self.native.timestep),
                'native_metadata':self.native.get_ep_meta(), 'object_states':self.tracked_objects()}

    def export_reference_for_evaluator(self,out):
        out=Path(out);out.mkdir(parents=True,exist_ok=False);out.chmod(0o700)
        (out/'canonical_state.json').write_text(json.dumps(self.get_state(),indent=2)+'\n')
        (out/'scene.xml').write_text(self.source_xml())
        (out/'body_bindings.json').write_text(json.dumps({k:int(v) for k,v in self.native.obj_body_id.items()},indent=2))
        return out

    def import_xml(self,xml,*,canonical_state,replaced_joint=None,estimated_pose=None,replaced_joints=None,fixture_components=None):
        # Privileged harness only. Retain original robot/distractor states, while
        # preserving the reconstructed object's estimated world pose explicitly.
        import mujoco
        if 'integration_state' not in canonical_state or 'controller_state' not in canonical_state:
            raise ValueError('canonical import requires integration and controller state')
        if replaced_joint is not None and estimated_pose is None:
            raise ValueError('reconstruction requires its own estimated pose')
        if replaced_joints is not None and (replaced_joint is not None or estimated_pose is not None):
            raise ValueError('use either one replacement pose or a named replacement mapping')
        replacements=dict(replaced_joints or {})
        if replaced_joint is not None:replacements[replaced_joint]=estimated_pose
        for joint,pose in replacements.items():
            value=np.asarray(pose,dtype=float)
            if not isinstance(joint,str) or value.shape!=(7,) or not np.isfinite(value).all() or not np.isclose(np.linalg.norm(value[3:]),1,atol=1e-6,rtol=0):
                raise ValueError('replacement pose requires named joint and finite xyz/unit wxyz')
        self.native.set_ep_meta(copy.deepcopy(canonical_state['native_metadata']))
        # Official metadata reset binds Python object/task roles to the saved
        # native scene; a seed alone is insufficient in this upstream release.
        self.native.reset()
        def topology():
            m=self.native.sim.model
            return ([(m.joint_id2name(i),int(m.jnt_type[i]),int(m.jnt_qposadr[i]),int(m.jnt_dofadr[i])) for i in range(m.njnt)],
                    [m.actuator_id2name(i) for i in range(m.nu)])
        previous_topology=topology() if replaced_joints is not None else None
        source_model=self.native.sim.model._model
        if fixture_components:
            from robo.roundtrip.fixture_scope import staged_fixture_components
            with staged_fixture_components(self.native,xml,fixture_components):
                self.native.reset_from_xml_string(xml)
                for receipt in fixture_components:
                    for name in receipt['contact_geoms']+receipt['visual_geoms']:
                        self.native.sim.model.geom_name2id(name)
        else:
            self.native.reset_from_xml_string(xml)
        if previous_topology is not None and topology()!=previous_topology:
            raise ValueError('changed/reordered dynamic topology requires explicit named integration remap')
        self._source_xml=xml
        sim=self.native.sim
        integration=np.asarray(canonical_state['integration_state'])
        if hasattr(source_model,'nbody'):
            from robo.roundtrip.adapters.integration_state import remap_integration_state
            integration=remap_integration_state(source_model,sim.model._model,integration)
        mujoco.mj_setState(sim.model._model,sim.data._data,
            integration,mujoco.mjtState.mjSTATE_INTEGRATION)
        for robot,saved in zip(self.native.robots,canonical_state['controller_state'],strict=True):
            for name,fields in saved['parts'].items():
                controller=robot.composite_controller.part_controllers[name]
                for key,value in fields.items():
                    setattr(controller,key,np.asarray(value) if isinstance(value,list) else value)
            for name,buffer in saved['derivative_buffers'].items():
                b=robot.composite_controller.part_controllers[name].derr_buf
                b.buf[:]=buffer['buf'];b.ptr=buffer['ptr'];b._size=buffer['size']
            for name,value in saved['grippers'].items():
                robot.gripper[name].current_action=np.asarray(value)

        for joint,pose in replacements.items():
            self.native.sim.data.set_joint_qpos(joint,np.asarray(pose))
        # Forward / forced sensor updates also modify qacc_warmstart. Preserve
        # the reset's integration state (including the estimated target pose)
        # across derived-cache refresh so import adds no extra solver iteration.
        integration=np.empty(mujoco.mj_stateSize(sim.model._model,mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(sim.model._model,sim.data._data,integration,mujoco.mjtState.mjSTATE_INTEGRATION)
        self.native.sim.forward()
        for robot,saved in zip(self.native.robots,canonical_state['controller_state'],strict=True):
            for name,controller in robot.composite_controller.part_controllers.items():
                controller.update(force=True)
                controller.new_update=saved['parts'][name].get('new_update',True)
        self.native.timestep=canonical_state['timestep'];self.native.cur_time=canonical_state['time']
        observation=self.refresh_observation()
        for name,timing in canonical_state['observable_timing'].items():
            for key,value in timing.items():
                setattr(self.native._observables[name],key,value)
        mujoco.mj_setState(sim.model._model,sim.data._data,integration,mujoco.mjtState.mjSTATE_INTEGRATION)
        return observation

    def render_capture(self,camera,*,width,height):
        from robosuite.utils.camera_utils import get_camera_intrinsic_matrix,get_camera_extrinsic_matrix,get_real_depth_map
        sim=self.native.sim
        name=camera.get('native_name','robot0_agentview_center')
        camid=sim.model.camera_name2id(name)
        original_pos=sim.model.cam_pos[camid].copy(); original_quat=sim.model.cam_quat[camid].copy()
        try:
            if 'T_world_from_camera' in camera:
                from scipy.spatial.transform import Rotation
                T=np.asarray(camera['T_world_from_camera'],dtype=float)
                parent=sim.model.cam_bodyid[camid];W=np.eye(4);W[:3,:3]=sim.data.body_xmat[parent].reshape(3,3);W[:3,3]=sim.data.body_xpos[parent]
                local=np.linalg.inv(W)@T@np.diag([1.,-1.,-1.,1.])
                q=Rotation.from_matrix(local[:3,:3]).as_quat()
                sim.model.cam_pos[camid]=local[:3,3];sim.model.cam_quat[camid]=q[[3,0,1,2]];sim.forward()
            rgb,depth=sim.render(width=width,height=height,camera_name=name,depth=True)
            return {'rgb':np.asarray(rgb)[::-1].copy(),'depth_m':get_real_depth_map(sim,np.asarray(depth))[::-1].copy(),
                'K':get_camera_intrinsic_matrix(sim,name,height,width),
                'T_world_from_camera':get_camera_extrinsic_matrix(sim,name),'timestamp':float(sim.data.time)}
        finally:
            sim.model.cam_pos[camid]=original_pos;sim.model.cam_quat[camid]=original_quat;sim.forward()

    def environment_lock(self):
        n=self.native
        return {'platform':self.config['platform'],'versions':{k:importlib.metadata.version(k) for k in ['robocasa','robosuite','mujoco','numpy','gymnasium']},
                'robot':self.config['robot'],'action_dim':12,'camera_names':list(self.env.unwrapped.camera_names),
                'controller':n.robots[0].composite_controller_config,'control_frequency':n.control_freq,
                'physics_timestep':float(n.sim.model.opt.timestep),'horizon':self.config['horizon'],
                'native_success_function':type(n).__name__+'._check_success',
                'policy_observation_keys':list(self.get_policy_observation())}

    def close(self):
        if self.env is not None:self.env.close();self.env=None
