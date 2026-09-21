"""Version-4 retained-native cabinet context witness, independent of target GT.

A sampled horizontal entry of the actual imported collider precedes an
unassisted release. This is a bounded geometry/physics diagnostic, not a policy
or a continuously certified motion plan. Native cabinet/door state is retained.
"""
import re
import numpy as np


def cabinet_destination(model,data,fixture_refs):
    prefix=fixture_refs['cab'];regions=[]
    for gid in range(model.ngeom):
        name=model.geom(gid).name or ''
        if re.fullmatch(re.escape(prefix)+r'_reg_level\d+',name):
            R=data.geom_xmat[gid].reshape(3,3);size=model.geom_size[gid];center=data.geom_xpos[gid]
            floor=float(center[2]-(np.abs(R)@size)[2])
            if .45<=floor<=1.50:regions.append((floor,name,gid))
    if not regions:raise ValueError('cabinet has no declared shelf region in native reachable height range')
    _,name,gid=min(regions);R=data.geom_xmat[gid].reshape(3,3);size=model.geom_size[gid];center=data.geom_xpos[gid]
    if not np.allclose(R[:,2],[0,0,1],atol=1e-8,rtol=0):raise ValueError('tilted native cabinet frame unsupported')
    geoms=[model.geom(g).name for g in range(model.ngeom) if (model.geom(g).name or '').startswith(prefix+'_') and (model.geom_contype[g] or model.geom_conaffinity[g])]
    extent=np.abs(R)@size
    return dict(task_id='PickPlaceCounterToCabinet',center_world_m=center.tolist(),bounds_world_m=[(center-extent).tolist(),(center+extent).tolist()],
        contact_geom_names=geoms,source='PRIVILEGED_RETAINED_NATIVE_DESTINATION_CONTEXT',oracle_context=True,native_target_shape_or_pose_used=False,
        interior_region=name,R_world_from_region=R.tolist(),region_halfsize_m=size.tolist(),
        region_rule='lowest native level whose floor is in [0.45,1.50] m; ties by name',
        opening_rule='native cabinet front is negative local Y; original door joint state retained')


def entry_poses(data,receipt,bounds,destination,config):
    lo,hi=np.asarray(bounds);center=(lo+hi)/2;R=np.asarray(destination['R_world_from_region']);size=np.asarray(destination['region_halfsize_m'])
    # Conservative oriented extents from the actual imported collision AABB.
    half=np.abs(R.T)@((hi-lo)/2)
    release=np.array([0.,0.,-size[2]+half[2]+config['cabinet_release_clearance_m']])
    entry=release.copy();entry[1]=-size[1]-half[1]-config['cabinet_entry_clearance_m']
    c=np.asarray(destination['center_world_m']);offset=np.asarray(receipt['position_m'])-center
    return c+R@entry+offset,c+R@release+offset,c+R@entry,half


def cabinet_access(model,data,receipt,bounds,destination,config):
    import mujoco
    from robo.roundtrip.native_context_checks import _world_bounds
    kind=mujoco.mjtState.mjSTATE_INTEGRATION;saved=np.empty(mujoco.mj_stateSize(model,kind));mujoco.mj_getState(model,data,saved,kind)
    dd=mujoco.MjData(model);mujoco.mj_setState(model,dd,saved,kind)
    joint=model.joint(receipt['free_joint_name']).id;adr=model.jnt_qposadr[joint];dof=model.jnt_dofadr[joint]
    bid=model.body(receipt['body_name']).id;target={model.geom(n).id for n in receipt['contact_geoms']};dest={model.geom(n).id for n in destination['contact_geom_names']}
    start,end,_,half=entry_poses(data,receipt,bounds,destination,config)
    size=np.asarray(destination['region_halfsize_m']);fits=bool(np.all(half<size))
    entry_hits=[];entry_trajectory=[]
    for step,position in enumerate(np.linspace(start,end,config['cabinet_entry_samples'])):
        dd.qpos[adr:adr+3]=position;dd.qvel[dof:dof+6]=0;mujoco.mj_forward(model,dd);entry_trajectory.append(position.tolist())
        for c in dd.contact[:dd.ncon]:
            if (c.geom1 in target)==(c.geom2 in target):continue
            other=c.geom2 if c.geom1 in target else c.geom1
            if c.dist < -config['max_penetration_m']:
                entry_hits.append(dict(sample=step,geom=model.geom(other).name,penetration_m=float(-c.dist)))
    # Release from the predeclared interior point even when the entry failed;
    # release evidence cannot override a blocked entry.
    dd.qpos[adr:adr+3]=end;dd.qvel[dof:dof+6]=0;dd.qacc_warmstart[dof:dof+6]=0
    contact_steps=0;counts={};trajectory=[];contacts=[]
    for _ in range(round(config['access_settle_seconds']/model.opt.timestep)):
        mujoco.mj_step(model,dd);trajectory.append(dd.xpos[bid].tolist());contacts=[]
        for c in dd.contact[:dd.ncon]:
            if (c.geom1 in target)==(c.geom2 in target):continue
            other=c.geom2 if c.geom1 in target else c.geom1;name=model.geom(other).name
            counts[name]=counts.get(name,0)+1;contacts.append(dict(geom=name,distance_m=float(c.dist),destination=other in dest))
        contact_steps+=any(c['destination'] for c in contacts)
    lo,hi=_world_bounds(model,dd,list(target));center=(lo+hi)/2;R=np.asarray(destination['R_world_from_region'])
    local=R.T@(center-np.asarray(destination['center_world_m']));inside=bool(np.all(np.abs(local)<=size))
    speed=float(np.linalg.norm(dd.qvel[dof:dof+3]));touch=any(c['destination'] for c in contacts)
    unchanged=np.empty_like(saved);mujoco.mj_getState(model,data,unchanged,kind)
    if saved.tobytes()!=unchanged.tobytes():raise AssertionError('cabinet probe mutated source state')
    return dict(passed=fits and not entry_hits and inside and touch and speed<.02,
        probe_geometry='actual_reconstructed_target_CoACD',protocol='sampled_horizontal_entry_then_unassisted_release_v4',
        unassisted=False,release_unassisted=True,entry_is_kinematic_geometric_probe=True,
        entry_passed=fits and not entry_hits,conservative_extent_fit=fits,entry_collisions=entry_hits,
        entry_trajectory_world_m=entry_trajectory,start_body_position_world_m=start.tolist(),release_body_position_world_m=end.tolist(),
        final_body_position_world_m=dd.xpos[bid].tolist(),final_collision_bounds_world_m=[lo.tolist(),hi.tolist()],
        destination_contact_steps=contact_steps,final_destination_contact=touch,terminal_speed_m_s=speed,
        final_probe_contacts=contacts,all_probe_contact_counts=counts,trajectory_world_m=trajectory,
        actual_collision_hashes=receipt['source_hashes'],source_reset_unchanged=True,
        initialization='estimated target orientation preserved; fixed negative-Y entry; zero release velocity; original native door state',
        interpretation='sampled collider entry and release witness; not policy success, continuous path certificate or whole-mesh containment')
