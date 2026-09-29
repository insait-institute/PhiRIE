"""Bounded physical controller probe; no fabricated successful-task label."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from physicalview.phiview_sim import DemoPhysics
from physicalview.phiview import save_json

p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--camera',required=True);p.add_argument('--object',required=True);p.add_argument('--out',required=True);p.add_argument('--command',default='push the object');p.add_argument('--radius',type=float,default=.5);p.add_argument('--angle',type=float,default=50.);p.add_argument('--height-offset',type=float,default=-.04);a=p.parse_args()
asset=Path(a.assets);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
objects={}
for row in json.loads((asset/'objects/objects.json').read_text()):
    name=f"obj_{row['index']:02d}";physical=asset/'objects'/name/'physics.json'
    objects[name]=SimpleNamespace(meta=row,physics=json.loads(physical.read_text()) if physical.exists() else None)
s=DemoPhysics(SimpleNamespace(scene_xml=asset/'sim_export/scene.xml',objects=objects),out)
record=json.loads(Path(a.camera).read_text());camera=np.linalg.inv(record['w2c'])[:3,3]
s.placement_options={'radius_m':a.radius,'angle_degrees':a.angle,'height_offset_m':a.height_offset}
s.command_robot(a.object,a.command,camera_position=camera)
trace=[]
for i in range(240):
    s.step(1/30)
    trace.append({'t':float(s.data.time),'target':s.data.body(a.object).xpos.copy(),'pinch':s.data.site('robot/2f85/pinch').xpos.copy(),'qpos':s.data.qpos.copy()})
contacts=[{'geom1':s.model.geom(c.geom1).name,'geom2':s.model.geom(c.geom2).name,'distance':float(c.dist),'position':c.pos.tolist()} for c in s.data.contact]
print('final_contact_count',len(contacts))
save_json(out/'probe.json',{'status':s.robot_status,'trace':trace,'final_contacts':contacts})
print(json.dumps(s.robot_status,indent=2))
