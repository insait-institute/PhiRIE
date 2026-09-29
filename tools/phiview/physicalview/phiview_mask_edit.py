"""Explicitly localized Qwen editing, with saved model inputs and raw outputs.

The second image marks the target in magenta. Mask compositing alone does not
localize Qwen's edit instruction, especially when a scene contains several cups.
This runner is experimental until its actual edits pass visual review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    import torch
    from agents.edit import inpaint_qwen as q
    from physicalview.phiview_inpaint import load_qwen
    ap=argparse.ArgumentParser()
    ap.add_argument('--objects',required=True)
    ap.add_argument('--backend',choices=['qwen','sdxl','lama_sdxl'],default='qwen')
    ap.add_argument('--views',default='all')
    ap.add_argument('--prompt',required=True)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--model-side',type=int,default=1024)
    ap.add_argument('--mask-dilation',type=int,default=0)
    ap.add_argument('--localization',choices=['reference','hole'],default='reference')
    args=ap.parse_args()
    objects=json.loads((q.C.OUT/'objects/objects.json').read_text())
    objects=[o for o in objects if o['index'] in q.parse_csv(args.objects,int)]
    if args.backend=='qwen':
        pipe=load_qwen()
    else:
        import diffusers
        from diffusers import AutoPipelineForInpainting
        checkpoint='/group/worldcept/hf_cache/hub/models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1/snapshots/115134f363124c53c7d878647567d04daf26e41e'
        pipe=AutoPipelineForInpainting.from_pretrained(checkpoint,torch_dtype=torch.float16,local_files_only=True).to('cuda')
        (q.C.OUT/'inpaint/model-receipt.json').write_text(json.dumps({'model':checkpoint,'backend':'sdxl_inpaint','diffusers':diffusers.__version__,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'placement':'cuda'},indent=2))
    lama=None
    if args.backend=='lama_sdxl':
        import os
        os.environ['LAMA_MODEL']=str(Path(__file__).resolve().parents[1]/'.envs/phiview-lama/weights/big-lama.pt')
        from simple_lama_inpainting import SimpleLama
        lama=SimpleLama(device=torch.device('cuda'))
        receipt_path=q.C.OUT/'inpaint/model-receipt.json'
        model_receipt=json.loads(receipt_path.read_text())
        model_receipt.update(backend='lama_sdxl',prefill_model=os.environ['LAMA_MODEL'],prefill_sha256=hashlib.sha256(Path(os.environ['LAMA_MODEL']).read_bytes()).hexdigest(),prefill_package='simple-lama-inpainting==0.1.2')
        receipt_path.write_text(json.dumps(model_receipt,indent=2))
    meta_path=q.C.OUT/'inpaint/inpaint_meta.json'
    used=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    receipts=[]
    for obj in objects:
        od=q.C.OUT/'inpaint'/q.entry_name(obj)
        views=json.loads((od/'views.json').read_text())
        for k in (range(len(views)) if args.views=='all' else q.parse_csv(args.views,int)):
            view=views[k]
            full=np.asarray(Image.open(q.C.IMAGES_DIR/view['frame']).convert('RGB')).copy()
            mask=np.asarray(Image.open(od/f'mask_{k}.png').convert('L'))>127
            if args.mask_dilation:
                import cv2
                r=args.mask_dilation;mask=cv2.dilate(mask.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))).astype(bool)
            vv,uu=np.nonzero(mask)
            if not len(vv):raise ValueError('Empty target mask')
            h,w=mask.shape;pad=int(.65*max(np.ptp(vv),np.ptp(uu))+24)
            box=(max(0,int(uu.min())-pad),max(0,int(vv.min())-pad),min(w,int(uu.max())+pad+1),min(h,int(vv.max())+pad+1))
            x0,y0,x1,y1=box; crop=full[y0:y1,x0:x1]; m=mask[y0:y1,x0:x1]
            scale=args.model_side/max(x1-x0,y1-y0)
            size=tuple(max(64,int(v*scale)//16*16) for v in (x1-x0,y1-y0))
            original=Image.fromarray(crop).resize(size)
            marked=crop.copy();marked[m]=[255,0,255] if args.localization=='hole' else (.15*marked[m]+.85*np.array([255,0,255])).astype(np.uint8)
            reference=Image.fromarray(marked).resize(size)
            prefix=od/f'localized_{args.backend}_{args.localization}_{args.model_side}_d{args.mask_dilation}_{k}'
            original.save(str(prefix)+'_input.png');reference.save(str(prefix)+'_reference.png')
            prompt=('Edit Image 1. Image 2 is a localization reference: the bright magenta overlay marks exactly the target '+str(obj['label'])+' to erase. Completely remove that marked object from Image 1, including its silhouette and every fragment. Fill its original location with the continuation of the surrounding background. Do not move or replace the object. Keep all other objects in their original positions. Return the edited Image 1 without any magenta marks. '+args.prompt)
            if args.localization=='hole':
                prompt=('Fill the solid bright magenta hole in this photograph. Replace EVERY magenta pixel with a seamless continuation of the empty background wall and supporting surface around the hole. The magenta area is missing background, not an object to restore. Do not add any object or text there. Keep the surrounding photograph unchanged. '+args.prompt)
            seed=args.seed+k
            if args.backend in ('sdxl','lama_sdxl'):
                prompt=args.prompt
                if lama is not None:
                    prefill=lama(Image.fromarray(full),Image.fromarray((mask*255).astype(np.uint8)))
                    prefill=prefill.crop((0,0,full.shape[1],full.shape[0]));prefill.save(str(prefix)+'_prefill.png')
                    original=prefill.crop(box).resize(size)
                model_mask=Image.fromarray((m*255).astype(np.uint8)).resize(size,resample=Image.Resampling.NEAREST)
                model_mask.save(str(prefix)+'_mask.png')
                result=pipe(image=original,mask_image=model_mask,prompt=prompt,
                    negative_prompt='cup, mug, bottle, object, container, hole, cavity, magenta, drawing, text',
                    height=size[1],width=size[0],strength=.25 if lama is not None else 1.,num_inference_steps=40,guidance_scale=7.5,
                    generator=torch.Generator('cpu').manual_seed(seed)).images[0]
            else:
                result=pipe(image=[original,reference] if args.localization=='reference' else reference,prompt=prompt,
                    negative_prompt='remaining target object, relocated object, duplicate object, magenta overlay',
                    height=size[1],width=size[0],num_inference_steps=40,true_cfg_scale=4.,
                    generator=torch.Generator('cpu').manual_seed(seed)).images[0]
            result.save(str(prefix)+'_raw.png')
            # Persist the compositing region and actual edit without retouching.
            final=q.paste(full,box,result,m)
            output=od/f'inpainted_{k}.png';Image.fromarray(final).save(output)
            used[f'{q.entry_name(obj)}/{k}']=args.backend
            receipt={'object':q.entry_name(obj),'view':k,'frame':view['frame'],'prompt':prompt,
                'backend':args.backend,'seed':seed,'steps':40,'cfg':7.5 if args.backend!='qwen' else 4.,'strength':.25 if lama is not None else 1.,'box':box,'input_size':size,'source_crop_size':[x1-x0,y1-y0],
                'model_input_resampled':True,'source_image_resolution_claim_unchanged':True,
                'localization':args.localization,'mask_dilation_pixels':args.mask_dilation,'output':str(output),
                'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'visual_review':'pending'}
            receipts.append(receipt)
            Path(str(prefix)+'_receipt.json').write_text(json.dumps(receipt,indent=2))
            print(f'[mask-edit] {q.entry_name(obj)} view {k} finished',flush=True)
    (q.C.OUT/'inpaint/inpaint_meta.json').write_text(json.dumps(used,indent=2))
    (q.C.OUT/'inpaint/localized-edit-receipt.json').write_text(json.dumps(receipts,indent=2))

if __name__=='__main__':main()
