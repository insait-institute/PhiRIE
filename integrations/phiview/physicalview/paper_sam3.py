"""Fresh SAM3 detections with raw masks/scores and an unembellished overlay."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--frame',required=True);ap.add_argument('--labels',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 ckpt='/group/worldcept/hf_cache/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt'
 model=build_sam3_image_model(device='cuda',checkpoint_path=ckpt,load_from_HF=False,compile=False)
 proc=Sam3Processor(model,resolution=1008,device='cuda',confidence_threshold=.4)
 im=Image.open(a.frame).convert('RGB');im.save(out/'input.png');rgb=np.asarray(im).copy();masks={};rows=[]
 import cv2
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  state=proc.set_image(im)
  for j,label in enumerate(json.loads(a.labels)):
   proc.reset_all_prompts(state);state=proc.set_text_prompt(prompt=label,state=state)
   ms=state['masks'].cpu().numpy().reshape(-1,im.height,im.width).astype(bool)
   scores=state['scores'].float().cpu().tolist();masks[f'label_{j}']=ms
   rows.append({'label':label,'count':len(scores),'scores':scores,'mask_key':f'label_{j}'})
   color=np.array([[31,188,157],[255,185,31],[83,154,244],[212,112,194]][j%4])
   for mask in ms:
    rgb[mask]=(rgb[mask]*.82+color*.18).astype(np.uint8)
    edges=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0;rgb[edges]=color
 Image.fromarray(rgb).save(out/'automatic-detections.png');np.savez_compressed(out/'masks.npz',**masks)
 (out/'report.json').write_text(json.dumps({'frame':a.frame,'input_sha256':hashlib.sha256(Path(a.frame).read_bytes()).hexdigest(),'checkpoint':ckpt,'detections':rows,'gpu':torch.cuda.get_device_name(),'fresh_inference':True,'prompt_vocabulary_source':'scene proposal class names','scope':'one source camera; does not claim exhaustive scene discovery'},indent=2))
 if not sum(r['count'] for r in rows):raise ValueError('SAM3 detected zero objects')
if __name__=='__main__':main()
