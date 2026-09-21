"""Public-surface fit and explicitly privileged L0 native-context probes.

No native target shape or pose is used. Destination geometry is retained oracle
context, never mislabeled public reconstruction. Corridors are sampled sphere
proxies, not IK or complete motion plans.
"""
from __future__ import annotations
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from agents.orchestrator.artifact import sha256_file


def observation_points(build_root, capture):
    root=Path(build_root);manifest=json.loads((root/'build_manifest.json').read_text())
    if manifest['capture_manifest_sha256']!=sha256_file(Path(capture)/'capture_manifest.json'):
        raise ValueError('observation build belongs to another public capture')
    path=root/'discovery/observation_points.npy'
    if manifest['source_hashes']['discovery/observation_points.npy']!=sha256_file(path):
        raise ValueError('public object observation bytes changed')
    points=np.load(path,allow_pickle=False)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<3 or not np.isfinite(points).all():
        raise ValueError('finite public observed target surface required')
    return points,{'build_manifest_sha256':sha256_file(root/'build_manifest.json'),
                   'observation_points_sha256':sha256_file(path),'source':'construction_public_TRAIN_auto_mask'}


def surface_residual(asset, observed):
    from agents.assets.s5_align import sym_score, apply_T
    import trimesh
    asset=Path(asset);mesh=trimesh.load(asset/'mesh_sim.obj',process=False,force='mesh')
    points,_=trimesh.sample.sample_surface(mesh,20000,seed=42)
    T=np.asarray(json.loads((asset/'aligned.json').read_text())['T'])
    return float(sym_score(apply_T(T,points),observed))


def _copy_named_state(source_model,source_data,model,data):
    """Copy original dynamic state by names into a diagnostic probe model."""
    for j in range(source_model.njnt):
        name=source_model.joint(j).name
        if not name:raise ValueError('probe state mapping requires named native joints')
        dst=model.joint(name).id
        if source_model.jnt_type[j]!=model.jnt_type[dst]:raise ValueError('probe changed joint type')
        kind=int(model.jnt_type[dst]);nq,nv={0:(7,6),1:(4,3),2:(1,1),3:(1,1)}[kind]
        a,b=source_model.jnt_qposadr[j],model.jnt_qposadr[dst]
        data.qpos[b:b+nq]=source_data.qpos[a:a+nq]
        a,b=source_model.jnt_dofadr[j],model.jnt_dofadr[dst]
        data.qvel[b:b+nv]=source_data.qvel[a:a+nv]
        data.qacc_warmstart[b:b+nv]=source_data.qacc_warmstart[a:a+nv]
        data.qfrc_applied[b:b+nv]=source_data.qfrc_applied[a:a+nv]
    for i in range(source_model.nu):
        name=source_model.actuator(i).name
        if not name:raise ValueError('probe requires named native actuators')
        data.ctrl[model.actuator(name).id]=source_data.ctrl[i]
    if source_model.na:
        if model.na!=source_model.na:raise ValueError('probe activation topology changed')
        data.act[:]=source_data.act
    for bid in range(1,source_model.nbody):
        name=source_model.body(bid).name
        if not name:
            if np.any(source_data.xfrc_applied[bid]) or source_model.body_mocapid[bid]>=0:
                raise ValueError('externally driven probe bodies must be named')
            continue
        dst=model.body(name).id;data.xfrc_applied[dst]=source_data.xfrc_applied[bid]
        source_mocap=source_model.body_mocapid[bid]
        if source_mocap>=0:
            dest_mocap=model.body_mocapid[dst]
            data.mocap_pos[dest_mocap]=source_data.mocap_pos[source_mocap]
            data.mocap_quat[dest_mocap]=source_data.mocap_quat[source_mocap]
    if model.neq!=source_model.neq:raise ValueError('probe changed equality topology')
    data.eq_active[:]=source_data.eq_active
    data.time=source_data.time


def _probe_model(xml,source_model,source_data,radius,*,dynamic=False):
    import mujoco
    root=ET.fromstring(xml);name='roundtrip_context_probe'
    if any(e.get('name')==name for e in root.iter()):raise ValueError('context probe name collision')
    attrs={'name':name,'pos':'0 0 5'}
    if not dynamic:attrs['mocap']='true'
    body=ET.SubElement(root.find('worldbody'),'body',**attrs)
    if dynamic:ET.SubElement(body,'freejoint',name=name+'_joint')
    ET.SubElement(body,'geom',name=name+'_geom',type='sphere',size=str(radius),mass='.001',
                  contype='1' if dynamic else '0',conaffinity='1' if dynamic else '0',friction='.4 .005 .0001')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    data=mujoco.MjData(model);_copy_named_state(source_model,source_data,model,data)
    mujoco.mj_forward(model,data)
    return model,data,model.body(name).id,model.geom(name+'_geom').id


def _world_bounds(model,data,gids):
    import mujoco
    clouds=[]
    for gid in gids:
        if model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH:
            mid=model.geom_dataid[gid];a=model.mesh_vertadr[mid];n=model.mesh_vertnum[mid]
            local=model.mesh_vert[a:a+n]
        elif model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_BOX:
            local=np.asarray([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*model.geom_size[gid]
        else:
            raise ValueError('destination bound primitive unsupported; cannot fabricate cavity')
        clouds.append(local@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid])
    if not clouds:raise ValueError('destination has no physical collision geoms')
    cloud=np.concatenate(clouds);return cloud.min(0),cloud.max(0)


def native_destination(model,data,task_id,fixture_refs):
    """L0 oracle endpoint and contact role; excludes native target metadata."""
    import mujoco
    if task_id=='PickPlaceCounterToCabinet':
        from robo.roundtrip.cabinet_context import cabinet_destination
        return cabinet_destination(model,data,fixture_refs)
    if task_id=='PickPlaceCounterToSink':
        prefix=fixture_refs['sink']
        gid=model.geom(prefix+'_reg_basin').id
        center=data.geom_xpos[gid].copy()
        extents=np.abs(data.geom_xmat[gid].reshape(3,3))@model.geom_size[gid]
        lower,upper=center-extents,center+extents
        geoms=[g for g in range(model.ngeom) if model.geom(g).name and model.geom(g).name.startswith(prefix+'_') and (model.geom_contype[g] or model.geom_conaffinity[g])]
    elif task_id=='PickPlaceSinkToCounter':
        bid=model.body('container_main').id
        descendants={bid}
        for child in range(model.nbody):
            parent=child
            while parent and parent not in descendants:parent=model.body_parentid[parent]
            if parent in descendants:descendants.add(child)
        geoms=[g for g in range(model.ngeom) if model.geom_bodyid[g] in descendants and (model.geom_contype[g] or model.geom_conaffinity[g])]
        lower,upper=_world_bounds(model,data,geoms);center=(lower+upper)/2
    else:raise ValueError('native destination family is not declared')
    return {'task_id':task_id,'center_world_m':center.tolist(),'bounds_world_m':[lower.tolist(),upper.tolist()],
            'contact_geom_names':[model.geom(g).name for g in geoms],
            'source':'PRIVILEGED_RETAINED_NATIVE_DESTINATION_CONTEXT','oracle_context':True,
            'native_target_shape_or_pose_used':False}


def context_checks(xml,model,data,receipt,task_id,fixture_refs,config):
    import mujoco
    destination=native_destination(model,data,task_id,fixture_refs)
    target_gids=[model.geom(n).id for n in receipt['contact_geoms']]
    lo,hi=_world_bounds(model,data,target_gids);target=(lo+hi)/2
    grip=model.site('gripper0_right_grip_site').id;gripper=data.site_xpos[grip].copy()
    dlow,dhigh=np.asarray(destination['bounds_world_m']);dest=np.asarray(destination['center_world_m'])
    above_target=target.copy();above_target[2]=hi[2]+config['approach_clearance_m']
    above_dest=dest.copy();above_dest[2]=dhigh[2]+config['transport_clearance_m']
    if task_id=='PickPlaceCounterToCabinet':
        if config['schema_version']!=4:raise ValueError('cabinet context requires v4')
        from robo.roundtrip.cabinet_context import entry_poses
        _,_,above_dest,_=entry_poses(data,receipt,(lo,hi),destination,config)
    radius=config['corridor_radius_m']
    pm,pd,pbid,pgid=_probe_model(xml,model,data,radius)
    mocap=pm.body_mocapid[pbid]
    # Known robot and manipulated object are endpoints, not retained obstacles.
    obstacle_names=[model.geom(g).name for g in range(model.ngeom)
        if (model.geom_contype[g] or model.geom_conaffinity[g]) and g not in target_gids
        and not (model.body(model.geom_bodyid[g]).name or '').startswith(('robot','gripper'))]
    obstacles=[pm.geom(n).id for n in obstacle_names if n]
    corridors={}
    for name,start,end in [('approach',gripper,above_target),('transport',above_target,above_dest)]:
        hits=[];minimum=None
        for index,point in enumerate(np.linspace(start,end,config['corridor_samples'])):
            pd.mocap_pos[mocap]=point;mujoco.mj_forward(pm,pd)
            for other in obstacles:
                distance=float(mujoco.mj_geomDistance(pm,pd,pgid,other,.05,None))
                if distance<.05:minimum=distance if minimum is None else min(minimum,distance)
                if distance<0:hits.append({'sample':index,'geom':pm.geom(other).name,'penetration_m':-distance})
        corridors[name]={'passed':not hits,'start_world_m':start.tolist(),'end_world_m':end.tolist(),
                         'samples':config['corridor_samples'],'radius_m':radius,'min_measured_gap_m':minimum,
                         'collisions':hits,'interpretation':'sampled swept-sphere proxy; no IK or path guarantee'}
    if task_id=='PickPlaceCounterToCabinet':
        from robo.roundtrip.cabinet_context import cabinet_access
        access=cabinet_access(model,data,receipt,(lo,hi),destination,config)
        return {'destination':destination,'corridors':corridors,'destination_access':access,
            'passed':all(c['passed'] for c in corridors.values()) and access['passed'],
            'native_task_success_accessed':False,'robot_IK':'NOT_RUN','oracle_context':True}
    # Task-conditioned witness uses the actual reconstructed target, not a
    # small surrogate sphere. Only the cloned probe state is repositioned.
    dd=mujoco.MjData(model);kind=mujoco.mjtState.mjSTATE_INTEGRATION
    saved=np.empty(mujoco.mj_stateSize(model,kind));mujoco.mj_getState(model,data,saved,kind)
    mujoco.mj_setState(model,dd,saved,kind)
    joint=model.joint(receipt['free_joint_name']).id;adr=model.jnt_qposadr[joint];dof=model.jnt_dofadr[joint]
    bid=model.body(receipt['body_name']).id
    start=data.xpos[bid].copy()
    start[:2]+=dest[:2]-target[:2]
    start[2]+=dhigh[2]+config['transport_clearance_m']-lo[2]
    dd.qpos[adr:adr+3]=start
    dd.qvel[dof:dof+6]=0
    target_contacts={model.geom(n).id for n in destination['contact_geom_names']}
    target_ids=set(target_gids)
    contact_steps=0;trajectory=[];touching=False;contact_counts={};final_contacts=[]
    for _ in range(round(config['access_settle_seconds']/model.opt.timestep)):
        mujoco.mj_step(model,dd);trajectory.append(dd.xpos[bid].tolist())
        final_contacts=[]
        for contact in dd.contact[:dd.ncon]:
            if (contact.geom1 in target_ids)==(contact.geom2 in target_ids):continue
            other=contact.geom2 if contact.geom1 in target_ids else contact.geom1
            name=model.geom(other).name
            contact_counts[name]=contact_counts.get(name,0)+1
            final_contacts.append({'geom':name,'distance_m':float(contact.dist),'destination':other in target_contacts})
        touching=any(c['destination'] for c in final_contacts);contact_steps+=touching
    flow,fhigh=_world_bounds(model,dd,target_gids);center=(flow+fhigh)/2
    speed=float(np.linalg.norm(dd.qvel[dof:dof+3]))
    inside=bool(np.all(center[:2]>=dlow[:2]) and np.all(center[:2]<=dhigh[:2]) and
                dlow[2]-.005<=flow[2]<=dhigh[2]+.005)
    access={'passed':inside and touching and speed<.02,'unassisted':True,
            'probe_geometry':'actual_reconstructed_target_CoACD',
            'initialization':'geometry-centered above retained destination; estimated orientation preserved; zero release velocity',
            'start_body_position_world_m':start.tolist(),'final_body_position_world_m':dd.xpos[bid].tolist(),
            'final_collision_bounds_world_m':[flow.tolist(),fhigh.tolist()],
            'destination_contact_steps':contact_steps,'final_destination_contact':bool(touching),
            'terminal_speed_m_s':speed,'all_probe_contact_counts':contact_counts,'final_probe_contacts':final_contacts,
            'trajectory_world_m':trajectory,'actual_collision_hashes':receipt['source_hashes'],
            'source_reset_unchanged':True,
            'interpretation':'one fixed reconstructed-target drop witness; not policy success or whole-mesh containment'}
    return {'destination':destination,'corridors':corridors,'destination_access':access,
            'passed':all(c['passed'] for c in corridors.values()) and access['passed'],
            'native_task_success_accessed':False,'robot_IK':'NOT_RUN','oracle_context':True}
