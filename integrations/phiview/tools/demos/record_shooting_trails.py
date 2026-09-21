"""Record two physical hits on the refined fb5a96b1a2 bottle.

Uses existing GT-assisted holding-capture collision geometry. Scene properties
are unmeasured estimates. Predictive aiming only chooses projectile launch
velocity; subsequent bottle motion comes exclusively from MuJoCo contacts.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from physicalview.phiview_sim import DemoPhysics

NAME='obj_19'

def physics(assets, out):
    meta=json.loads((assets/'geometry.json').read_text())
    rec=SimpleNamespace(meta={'centroid':meta['center'],'aabb':meta['aabb']},physics={'estimate':True})
    state=SimpleNamespace(scene_xml=assets/'refined-scene.xml',objects={NAME:rec})
    p=DemoPhysics(state,out)
    # Demo projectiles are explicitly configured 30 mm, 80 g collision spheres.
    root=ET.parse(p.xml)
    for g in root.findall('.//geom'):
        if g.get('name','').startswith('phiview_ball_'):g.set('size','.015')
    root.write(out/'shooting-scene.xml')
    p.model=mujoco.MjModel.from_xml_path(str(out/'shooting-scene.xml'))
    p.data=mujoco.MjData(p.model);mujoco.mj_forward(p.model,p.data)
    p.initial=p._poses();p.initial_qpos=p.data.qpos.copy();p.reset();p.enable([NAME]);p.running=True
    for _ in range(60):p.step(1/120)
    p.data.time=0
    return p

def predict(p,dt):
    d=mujoco.MjData(p.model)
    mujoco.mj_copyData(d,p.model,p.data)
    for _ in range(round(dt/p.model.opt.timestep)):mujoco.mj_step(p.model,d)
    qa,_=p.addresses(NAME)
    return d.qpos[qa:qa+3].copy()

def shoot(p,index,speed=5.):
    qa,_=p.addresses(NAME);center=p.data.qpos[qa:qa+3].copy()
    # Two distinct diagonal firing positions stay above the table.
    offset=np.array([-.62,-.36,.17]) if index==0 else np.array([.55,-.46,.14])
    origin=center+offset
    target=center+[0,0,.055 if index==0 else -.025]
    for _ in range(5):
        flight=(np.linalg.norm(target-origin)-.12)/speed
        target=predict(p,flight)+[0,0,.055 if index==0 else -.025]
        aim=target+[0,0,.5*9.81*flight**2]
    p.shoot(origin,aim-origin,speed)
    return {'projectile':f'phiview_ball_{index}','time':float(p.data.time),'origin':origin.tolist(),'aim':aim.tolist(),'speed':speed}

def rollout(assets, out, speed=5.,delay=.15):
    p=physics(assets, out);start=p.data.qpos[p.addresses(NAME)[0]:][:3].copy()
    fires=[];hits=[];seen=set();states=[];qpos=[];qvel=[]
    for tick in range(361):
        if tick in (48,48+round(delay*120)):fires.append(shoot(p,len(fires),speed))
        cs=[]
        for _ in range(5):
            p.step(1/600)
            for ci,c in enumerate(p.data.contact):
                names=[p.model.body(int(p.model.geom_bodyid[g])).name for g in (c.geom1,c.geom2)]
                balls=[n for n in names if n.startswith('phiview_ball_')]
                if NAME in names and balls:
                    force=np.zeros(6);mujoco.mj_contactForce(p.model,p.data,ci,force)
                    event={'time':float(p.data.time),'bullet':balls[0],'position':c.pos.tolist(),'normal_force':float(force[0])}
                    cs.append(event)
                    if balls[0] not in seen:seen.add(balls[0]);hits.append(event)
        pos=p.data.qpos[p.addresses(NAME)[0]:][:3].copy()
        states.append({'tick':tick,'time':float(p.data.time),'position':pos.tolist(),'displacement':float(np.linalg.norm(pos-start)),
                       'projectiles':{f['projectile']:p.data.xpos[p.model.body(f['projectile']).id].tolist() for f in fires},'contacts':cs})
        qpos.append(p.data.qpos.copy());qvel.append(p.data.qvel.copy())
    result={'speed':speed,'delay':delay,'fires':fires,'hits':hits,'max_displacement':max(s['displacement'] for s in states),'end_position':states[-1]['position'],'start_position':start.tolist()}
    print(json.dumps(result),flush=True)
    if len(seen)==2:
        (out/'shooting-trace.json').write_text(json.dumps({'summary':result,'frames':states},indent=2))
        np.savez_compressed(out/'shooting-states.npz',qpos=qpos,qvel=qvel)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--speed',default=5.,type=float)
    ap.add_argument('--delay',default=.15,type=float)
    args=ap.parse_args()
    if not 1 <= args.speed <= 20 or not .02 <= args.delay <= 1:
        ap.error('speed must be 1..20 m/s and delay .02..1 seconds')
    args.out.mkdir(parents=True,exist_ok=False)
    result=rollout(args.assets,args.out,args.speed,args.delay)
    if len(result['hits']) != 2:
        raise SystemExit('Capture failed: two distinct projectile contacts required')
