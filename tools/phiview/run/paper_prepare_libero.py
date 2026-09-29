"""Audit ten distinct LIBERO layouts and preserve native XML/state for capture.
Run with openpi/.venv/bin/python (h5py); no simulator rollout occurs here.
"""
import hashlib,json,re,xml.etree.ElementTree as ET
from pathlib import Path
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/paper-capture-20260913'
DATA=Path('/group/worldcept/data/libero');ASSETS=Path('/group/worldcept/code/openpi/third_party/libero/libero/libero/assets')
chosen={}
for p in sorted((DATA/'libero_10').glob('*.hdf5')):
 match=re.match(r'((?:KITCHEN|STUDY|LIVING_ROOM)_SCENE\d+)',p.stem)
 if match:chosen.setdefault(match.group(1),p)
chosen['GOAL_TABLETOP']=DATA/'libero_goal/put_the_wine_bottle_on_the_rack_demo.hdf5'
rows=[]
for key,p in chosen.items():
 with h5py.File(p) as f:
  d=f['data/demo_0'];tree=ET.fromstring(d.attrs['model_file']);missing=[];files=[]
  for e in tree.iter():
   raw=e.get('file')
   if not raw:continue
   if '/chiliocosm/assets/' in raw:new=ASSETS/raw.split('/chiliocosm/assets/')[1]
   elif '/libero/assets/' in raw:new=ASSETS/raw.split('/libero/assets/')[1]
   elif '/robosuite/' in raw:new=ROOT/'.envs/phiview-libero/robosuite'/raw.split('/robosuite/')[1]
   else:new=Path(raw)
   new=new.resolve();e.set('file',str(new));files.append(str(new))
   if not new.exists():missing.append(str(new))
  dest=OUT/'libero-native-source'/key.lower();dest.mkdir(parents=True,exist_ok=True)
  ET.ElementTree(tree).write(dest/'source.xml');np.savez(dest/'state.npz',state=d['states'][0],init_state=d.attrs['init_state'])
  row={'dataset':'libero','scene':'libero_'+key.lower(),'layout':key,'source_hdf5':str(p),'demo':'demo_0','frame':0,'source_bytes':p.stat().st_size,'metadata':json.loads(f['data'].attrs['problem_info']),'native_xml':str(dest/'source.xml'),'state_npz':str(dest/'state.npz'),'assets':files,'missing_assets':missing,'state':'source_prepared','native_render_verified':False,'gaussian_reconstruction_complete':False}
  (dest/'source.json').write_text(json.dumps(row,indent=2));rows.append(row)
(OUT/'libero-roster.json').write_text(json.dumps(rows,indent=2));print('layouts',len(rows),'missing assets',sum(len(r['missing_assets']) for r in rows))
assert len(rows)==10
assert not any(r['missing_assets'] for r in rows)
