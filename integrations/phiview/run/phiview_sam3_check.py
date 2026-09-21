import json
from pathlib import Path
import sys
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
ckpt='/group/worldcept/hf_cache/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt'
frame='/data/ScanNetpp/data/c50d2d1d42/dslr/resized_undistorted_images/DSC01593.JPG'
model=build_sam3_image_model(device='cuda',checkpoint_path=ckpt,load_from_HF=False,compile=False)
proc=Sam3Processor(model,resolution=1008,device='cuda',confidence_threshold=.4)
report={'frame':frame,'checkpoint':ckpt,'gpu':torch.cuda.get_device_name(),'detections':{}}
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 state=proc.set_image(Image.open(frame).convert('RGB'))
 for prompt in ['bottle','keyboard','computer mouse','mug','headphones']:
  proc.reset_all_prompts(state);state=proc.set_text_prompt(prompt=prompt,state=state)
  report['detections'][prompt]={'count':len(state['scores']),'scores':state['scores'].float().cpu().tolist()}
assert sum(x['count'] for x in report['detections'].values())>0
(out/'sam3-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
