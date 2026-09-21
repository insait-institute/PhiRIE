"""Bounded simulator placement search, preserving every candidate and a no-arm control."""
import argparse,json,gc
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
from physicalview.phiview_sim import DemoPhysics
from physicalview.phiview import save_json

p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--camera',required=True)
p.add_argument('--object',required=True);p.add_argument('--out',required=True);p.add_argument('--pedestal',action='store_true');a=p.parse_args()
asset=Path(a.assets);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
objects={}
for row in json.loads((asset/'objects/objects.json').read_text()):
    name=f"obj_{row['index']:02d}";path=asset/'objects'/name/'physics.json'
    objects[name]=SimpleNamespace(meta=row,physics=json.loads(path.read_text()) if path.exists() else None)
state=SimpleNamespace(scene_xml=asset/'sim_export/scene.xml',objects=objects)
camera=np.linalg.inv(json.loads(Path(a.camera).read_text())['w2c'])[:3,3]
control=out/'control';control.mkdir(exist_ok=True);sim=DemoPhysics(state,control)
sim.enable([a.object]);sim.running=True
start=sim.data.body(a.object).xpos.copy()
for _ in range(150):sim.step(1/30)
control_end=sim.data.body(a.object).xpos.copy()
save_json(control/'result.json',{'initial':start,'final':control_end,'seconds':5.,'robot_present':False})
sim.reset();rows=[]
for angle in (-60.,-30.,0.,30.,60.,90.):
 for radius in (.35,.5,.65):
    target=np.asarray(sim.initial[a.object][0]);direction=camera-target;direction[2]=0;direction/=np.linalg.norm(direction)
    theta=np.deg2rad(angle);x,y=direction[:2];direction[:2]=[x*np.cos(theta)-y*np.sin(theta),x*np.sin(theta)+y*np.cos(theta)]
    xy=target+direction*radius;origin=np.array([xy[0],xy[1],target[2]+.35]);gid=np.array([-1],dtype=np.int32);normal=np.zeros(3)
    distance=mujoco.mj_ray(sim.model,sim.data,origin,np.array([0.,0.,-1.]),None,True,sim.model.body(a.object).id,gid,normal)
    row={'angle_degrees':angle,'radius_m':radius,'ray_distance':distance,'support_geom':sim.model.geom(gid[0]).name if gid[0]>=0 else None,'normal':normal.copy()}
    if distance<0 or normal[2]<.85:
        row['status']='no_horizontal_support';rows.append(row);continue
    support_z=origin[2]-distance
    offset=support_z-float(objects[a.object].meta['aabb'][0][2])+.002
    pedestal_bottom=None
    if a.pedestal:
        desired=float(objects[a.object].meta['aabb'][0][2])-.35
        if desired-support_z<.05:
            row['status']='support_too_high_for_requested_pedestal';rows.append(row);continue
        footprint=[]
        for dx,dy in ((-.12,-.12),(-.12,.12),(.12,-.12),(.12,.12)):
            corner=origin+np.array([dx,dy,0.]);hit=np.array([-1],dtype=np.int32);norm=np.zeros(3)
            dist=mujoco.mj_ray(sim.model,sim.data,corner,np.array([0.,0.,-1.]),None,True,sim.model.body(a.object).id,hit,norm)
            footprint.append({'distance':dist,'height':corner[2]-dist,'normal':norm.copy()})
        row['footprint_rays']=footprint
        if any(v['distance']<0 or abs(v['height']-support_z)>.03 or v['normal'][2]<.85 for v in footprint):
            row['status']='pedestal_footprint_obstructed_or_uneven';rows.append(row);continue
        pedestal_bottom=support_z;offset=-.35
    row['height_offset_m']=offset;row['support_z']=support_z
    candidate=out/f'angle{int(angle)}-r{radius:g}';candidate.mkdir(exist_ok=True)
    test=DemoPhysics(state,candidate);test.placement_options={'angle_degrees':angle,'radius_m':radius,'height_offset_m':offset}
    if pedestal_bottom is not None:test.placement_options['pedestal_bottom_z']=pedestal_bottom
    test.command_robot(a.object,'push the object',camera_position=camera)
    robot_geoms=np.array([test.model.body(test.model.geom_bodyid[g]).name.startswith('robot/') and test.model.body(test.model.geom_bodyid[g]).name!='robot/pedestal' for g in range(test.model.ngeom)])
    contact_types=test.model.geom_contype.copy();contact_affinity=test.model.geom_conaffinity.copy()
    test.model.geom_contype[robot_geoms]=0;test.model.geom_conaffinity[robot_geoms]=0
    test.plan=[]
    for _ in range(150):test.step(1/30)
    matched_control_end=test.data.body(a.object).xpos.copy()
    test.reset();test.model.geom_contype[:]=contact_types;test.model.geom_conaffinity[:]=contact_affinity
    test.command_robot(a.object,'push the object',camera_position=camera)
    def penetrations():
        values=[]
        for contact in test.data.contact:
            bodies=[test.model.body(test.model.geom_bodyid[g]).name for g in (contact.geom1,contact.geom2)]
            if sum(b.startswith('robot/') for b in bodies)==1 and a.object not in bodies and contact.dist<-.005:
                values.append(float(contact.dist))
        return values
    initial_penetrations=penetrations();worst=len(initial_penetrations)
    for _ in range(150):
        test.step(1/30);worst=max(worst,len(penetrations()))
    final=test.data.body(a.object).xpos.copy();relative=final-matched_control_end
    row.update(status='simulated',robot=test.robot_status,initial_robot_environment_penetrations=initial_penetrations,
               max_robot_environment_penetrations=worst,final_position=final,displacement_relative_to_no_arm_control=relative,
               matched_no_contact_control_final=matched_control_end,control_method='Same assembled model and integrator; robot contacts and command disabled',
               horizontal_relative_displacement=float(np.linalg.norm(relative[:2])),height_relative_to_control=float(relative[2]))
    row['candidate_contact_push']=bool(test.robot_status.get('target_contact_steps',0)>0 and np.linalg.norm(relative[:2])>.03 and abs(relative[2])<.08 and not initial_penetrations)
    rows.append(row);save_json(candidate/'result.json',row)
    save_json(out/'search.json',{'control':control_end,'candidates':rows,'boundary':'Simulator placement search, not held-out robot evaluation; candidate contact/displacement is not a grasp or task-success proof'})
    print(angle,radius,row['candidate_contact_push'],row['horizontal_relative_displacement'],row['height_relative_to_control'],worst,flush=True)
    del test;gc.collect()
save_json(out/'search.json',{'control':control_end,'candidates':rows,'boundary':'Simulator placement search, not held-out robot evaluation; candidate contact/displacement is not a grasp or task-success proof'})
