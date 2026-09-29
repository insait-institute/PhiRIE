"""Replay one frozen construction in the pinned native environment."""
import json
import time
import sys
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation
from robo.roundtrip.adapters.robocasa import RoboCasaAdapter
from robo.roundtrip.paired import prepare_paired_adapter
from robo.roundtrip.spec import load_spec

root = Path(sys.argv[1])
row = json.loads((root/'source.json').read_text())
config = load_spec(row['config_path'])
argv = row['argv']
object_dir = argv[argv.index('--object-dir')+1]
canonical = Path(row['bundle_dir'])
adapter = RoboCasaAdapter(config)
bundle = dict(state=json.loads((canonical/'canonical_state.json').read_text()), xml=(canonical/'scene.xml').read_text(), provenance={'reset_seed':config['reset_seeds'][0]})
receipt = prepare_paired_adapter(adapter, bundle, object_dir=object_dir)
reset = json.loads((Path(row['result_path_planned']).parent.parent/'reset_receipt.json').read_text())
# Read the exact predeclared reset entry; never reset the generated object to GT.
bank = json.loads(Path(row['reset_bank']).read_text())
entries = bank['resets'] if isinstance(bank, dict) else bank
entry = next(x for x in entries if x['reset_id']==row['reset_id'])
adapter.apply_paired_reset(entry)
(root/'import_receipt.json').write_text(json.dumps(receipt,indent=2))
sim = adapter.native.sim
model = sim.model
meshes = {}
for g in range(model.ngeom):
    typ = int(model.geom_type[g]); size = model.geom_size[g]; mesh = None
    if typ == 7:
        mid = int(model.geom_dataid[g]); va=int(model.mesh_vertadr[mid]); vn=int(model.mesh_vertnum[mid]); fa=int(model.mesh_faceadr[mid]); fn=int(model.mesh_facenum[mid])
        mesh = trimesh.Trimesh(model.mesh_vert[va:va+vn].copy(),model.mesh_face[fa:fa+fn].copy(),process=False)
    elif typ == 6: mesh = trimesh.creation.box(extents=size*2)
    elif typ == 2: mesh = trimesh.creation.icosphere(subdivisions=1,radius=size[0])
    elif typ in (3,5): mesh = trimesh.creation.cylinder(radius=size[0],height=2*size[1],sections=12)
    if mesh is None: continue
    T=np.eye(4); T[:3,:3]=Rotation.from_quat(model.geom_quat[g][[1,2,3,0]]).as_matrix(); T[:3,3]=model.geom_pos[g]; mesh.apply_transform(T)
    mesh.visual.vertex_colors=np.tile((np.clip(model.geom_rgba[g],0,1)*255).astype(np.uint8),(len(mesh.vertices),1))
    meshes.setdefault(int(model.geom_bodyid[g]),[]).append(mesh)
manifest=[]
for body, parts in meshes.items():
    path=f'body-{body}.ply'; trimesh.util.concatenate(parts).export(root/path); manifest.append(dict(body=body,path=path))
(root/'meshes.json').write_text(json.dumps(manifest))
actions=json.loads((Path(row['result_path_planned']).parent/'actions.json').read_text())
initial=adapter.get_state()
writer=imageio.get_writer(root/'native_reconstruction_replay.mp4',fps=20)
paused=False; step=0; nonce=None; finished=False
while True:
    try:
        command=json.loads((root/'command.json').read_text())
        if command['nonce']!=nonce:
            nonce=command['nonce']; paused=command['action']=='Pause'
            if command['action']=='Reset':
                adapter.import_xml(adapter.source_xml(),canonical_state=initial)
                sim=adapter.native.sim
                step=0
    except FileNotFoundError: pass
    if not paused and step<len(actions):
        adapter.step_native_action(actions[step]); step+=1
    rgb=adapter.render_capture({'native_name':'robot0_agentview_center'},width=640,height=480)['rgb']
    Image.fromarray(rgb).save(root/'frame.tmp.jpg'); (root/'frame.tmp.jpg').replace(root/'frame.jpg')
    poses={str(b):sim.data.body_xpos[b].tolist()+sim.data.body_xquat[b].tolist() for b in meshes}
    state=dict(step=step,horizon=len(actions),time=float(sim.data.time),success=adapter.native_success(),poses=poses)
    (root/'live.tmp').write_text(json.dumps(state)); (root/'live.tmp').replace(root/'live.json')
    if not finished:
        writer.append_data(rgb)
        if step==len(actions):
            writer.close(); finished=True
            (root/'demo_result.json').write_text(json.dumps(dict(step=step,source_unit=row['unit_id'],native_success=state['success'],kind='presentation_action_replay',policy_calls=0),indent=2))
    time.sleep(.04)
