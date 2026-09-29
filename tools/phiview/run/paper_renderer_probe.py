"""Compare exact observed-camera rendering to source pixels without changing assets."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from gsplat import rasterization
from agents.core import common as C
from physicalview.phiview import save_json

p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--camera',required=True);p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
scene=Path('/data/ScanNetpp/data')/a.scene
K,W,H,_=C.load_intrinsics(scene/'dslr/nerfstudio/transforms_undistorted.json')
w2c=C.load_colmap_w2c(scene/'dslr/colmap/images.txt')[a.camera]
source=scene/'dslr/resized_undistorted_images'/a.camera
ref=np.asarray(Image.open(source).convert('RGB'))/255.
assert ref.shape==(H,W,3)
gs=C.load_gaussians(Path('/data/ScanNetppv2_gsplat/splats')/(a.scene+'.ply'))
rows=[]
for convention in ('inria_wxyz','diagnostic_xyzw'):
 for mode in ('classic','antialiased'):
  q=gs['quats'] if convention=='inria_wxyz' else gs['quats'][:,[3,0,1,2]]
  with torch.inference_mode():
   rgb,alpha,_=rasterization(means=gs['means'],quats=q,scales=gs['scales'],opacities=gs['opacities'],colors=gs['sh'],
      viewmats=torch.as_tensor(w2c,dtype=torch.float32,device='cuda')[None],Ks=torch.as_tensor(K,dtype=torch.float32,device='cuda')[None],
      width=W,height=H,sh_degree=gs['sh_degree'],packed=False,rasterize_mode=mode,near_plane=.01,far_plane=100.)
  image=rgb[0].clamp(0,1).cpu().numpy();name=convention+'-'+mode
  path=out/(name+'.png');Image.fromarray((image*255+.5).astype(np.uint8)).save(path)
  rows.append({'name':name,'psnr_against_this_observed_frame':C.psnr(ref,image),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
save_json(out/'probe.json',{'scene':a.scene,'camera':a.camera,'w2c':w2c,'K':K,'source_image':str(source),'source_resolution':[W,H],
                          'boundary':'Single observed-frame diagnostic; not held-out evaluation or permission to change quaternion convention','rows':rows})
print(json.dumps(rows,indent=2))
