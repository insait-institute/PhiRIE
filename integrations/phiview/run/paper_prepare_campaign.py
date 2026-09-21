"""Create an isolated, provenance-recorded ten-scene screenshot campaign."""
import json,shutil,hashlib,subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/paper-capture-20260913'
SCENES=['c50d2d1d42','45b0dac5e3','825d228aec','7b6477cb95','578511c8a9','27dd4da69e','38d58a7a31','f9f95681fd','3e8bba0176','3864514494']
rows=[]
for sid in SCENES:
 src=Path('/group/worldcept/code/SimAny/outputs')/(sid+'_factory');dst=OUT/'assets'/(sid+'_factory')
 dst.mkdir(parents=True,exist_ok=True)
 for name in ['objects','inpaint','sim_export']:
  if (src/name).exists() and not (dst/name).exists():shutil.copytree(src/name,dst/name,symlinks=False)
 xml=dst/'sim_export/scene.xml'
 if xml.exists() and not (dst/'sim_export/relocation.json').exists():
  tree=ET.parse(xml);compiler=tree.find('compiler');changes=[]
  for e in tree.findall('.//asset/*'):
   raw=e.get('file')
   if not raw:continue
   subdir=compiler.get('texturedir' if e.tag=='texture' else 'meshdir','') if compiler is not None else ''
   candidates=[src/'sim_export'/subdir/raw,src.parent.parent/raw,src/raw]
   actual=next((p.resolve() for p in candidates if p.exists()),None)
   if actual is None:raise FileNotFoundError(f'{sid}: unresolved asset {raw}')
   try:new=dst/actual.relative_to(src)
   except ValueError:new=actual
   if not new.exists():raise FileNotFoundError(new)
   e.set('file',str(new));changes.append({'old':raw,'source':str(actual),'new':str(new)})
  tree.write(xml);(dst/'sim_export/relocation.json').write_text(json.dumps(changes,indent=2))
 metadata=[]
 for p in sorted((dst/'objects').glob('obj_*/meta.json')):
  m=json.loads(p.read_text());a=json.loads((p.parent/'aligned.json').read_text())
  metadata.append({'id':p.parent.name,'label':m['label'],'accepted':not bool(a.get('rejected')),'physics':(p.parent/'physics.json').exists()})
 rows.append({'dataset':'scannetpp','scene':sid,'result_set':sid+'_factory','source':str(src),'assets':str(dst),'objects':metadata,
              'has_clean':(dst/'inpaint/clean_background.ply').exists(),'has_sim':(dst/'sim_export/scene.xml').exists()})
 print(sid,'prepared',flush=True)
config=yaml.safe_load((ROOT/'configs/phiview.yaml').read_text());config['outputs_root']=str(OUT/'assets');config['studio_out']=str(OUT/'jobs')
(OUT/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
(OUT/'roster.json').write_text(json.dumps({'datasets':{'scannetpp':rows,'libero':{'target':10,'state':'after_scannetpp'},'behavior':{'target':10,'state':'after_libero'}},'selection':'existing scene diversity and available registered objects; camera composition survey precedes action outcomes','features':14,'sets_per_feature_per_dataset':10,'publication_review_required':True},indent=2))
