"""Diagnose observed background versus generated fill; never an inpaint claim."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from physicalview.paper_capture import Capture
from physicalview.phiview import save_json
from physicalview.phiview_scene import FlyCamera, subset

p=argparse.ArgumentParser()
p.add_argument('--root',required=True)
p.add_argument('--scene',required=True)
p.add_argument('--out',required=True)
p.add_argument('--observed-origin',action='store_true')
a=p.parse_args();root=Path(a.root).resolve()
args=SimpleNamespace(config=str(root/'config.yaml'),scene=a.scene+'_factory',out=a.out,
                     width=2880,height=1920,features=None)
cap=Capture(args);d=cap.d
cap.chosen=json.loads((root/'scannetpp'/a.scene/'camera-survey.json').read_text())['selected']
pose=cap.chosen['pose']
cap.base_camera=FlyCamera(np.asarray(pose['position']),pose['yaw'],pose['pitch'],pose['fov'])
if a.observed_origin:
    from physicalview.render import look_at_w2c
    d.execute({'op':'camera','name':cap.chosen['camera']})
    center=np.asarray(d.state.objects[cap.chosen['object']].meta['centroid'])
    cap.base_camera=FlyCamera.from_w2c(look_at_w2c(d.camera.position,center),45.)
d.camera_name=cap.chosen['camera'];cap.restore()
cap.shot('diagnostic','original')
selected=cap.chosen['object']
path=root/'prompt-builds'/a.scene/'inpaint/fill_gaussians.npz'
with np.load(path,allow_pickle=False) as data:removed=data['removal_idx']
keep=torch.ones(d.scene.count,device='cuda',dtype=torch.bool)
keep[torch.as_tensor(removed,device='cuda')]=False
d.scene.prompt_backgrounds[frozenset([selected])]=subset(d.scene.raw,keep)
d.mode='clean_selected'
evidence={'method':'Remove original Gaussian rows; no generated fill or prompt model',
          'removed_rows':int(len(removed)),'removal_source':str(path),
          'removal_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'probe_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'observed_camera_origin':a.observed_origin,
          'publication_review':'pending','counted_in_campaign':False}
cap.shot('diagnostic','carved-observations',evidence)
del d.scene.prompt_backgrounds[frozenset([selected])]
cap.shot('diagnostic','existing-generated-clean',{'method':'Existing campaign clean-all fill with unselected objects restored'})
save_json(Path(a.out)/'probe.json',evidence)
