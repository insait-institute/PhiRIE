"""Evaluator-only heldout views of a sealed static source GS; no constructor call."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robo.roundtrip.gaussian_build import sha, save
from robo.roundtrip.capture import validate_public_capture, validate_camera


def sealed_views(config):
    build=Path(config['source_build']); capture=Path(config['public_capture']); vault=Path(config['private_capture'])
    if sha(build/'gs_build_manifest.json')!=config['source_build_manifest_sha256']:
        raise ValueError('source GS build changed after sealing')
    manifest=json.loads((build/'gs_build_manifest.json').read_text())
    public=validate_public_capture(capture)
    if manifest['status']!='BUILT' or manifest['source_gaussian']!='MEASURED':
        raise ValueError('successful source GS required')
    digest=sha(capture/'capture_manifest.json')
    if digest!=manifest['capture_manifest_sha256'] or digest!=config['public_capture_manifest_sha256']:
        raise ValueError('source GS belongs to a different capture')
    receipt_path=vault/'capture_receipt.json'
    if sha(receipt_path)!=config['private_capture_receipt_sha256']:
        raise ValueError('heldout capture receipt changed')
    receipt=json.loads(receipt_path.read_text())
    if receipt['public_manifest_sha256']!=digest or receipt['capture_id']!=public['capture_id']:
        raise ValueError('heldout views belong to a different static capture')
    cameras=vault/'test/cameras.jsonl'
    if sha(cameras)!=receipt['private_files']['test/cameras.jsonl']:
        raise ValueError('heldout camera bytes changed')
    rows=[json.loads(l) for l in cameras.read_text().splitlines()]
    if len(rows)!=receipt['counts']['test'] or len(rows)!=config['planned_views'] or len(rows)!=2:
        raise ValueError('retain exactly the two original heldout views')
    train={r['frame_id'] for r in map(json.loads,(capture/'train/cameras.jsonl').read_text().splitlines())}
    if len({r['frame_id'] for r in rows})!=2 or train.intersection(r['frame_id'] for r in rows):
        raise ValueError('evaluation/initialization view leakage')
    for row in rows:
        name=f"test/rgb/{row['frame_id']}.png"
        if (row['rgb']!=name or (vault/name).is_symlink()
                or sha(vault/name)!=row['rgb_sha256'] or row['rgb_sha256']!=receipt['private_files'][name]):
            raise ValueError('heldout image lineage changed')
        validate_camera(row['K'],row['T_world_from_camera'],width=public['width'],height=public['height'])
    ply=build/'pilot/scene.ply'
    if sha(ply)!=manifest['phases']['pilot']['scene_ply_sha256'] or sha(ply)!=config['source_ply_sha256']:
        raise ValueError('frozen Gaussian bytes changed')
    return public,rows,ply


def evaluate(config,out):
    from agents.core import common
    from robo.eval import fidelity_metrics
    import torch
    started=time.monotonic();public,rows,ply=sealed_views(config)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    pred_dir=out/'render';gt_dir=out/'reference';pred_dir.mkdir();gt_dir.mkdir()
    gs=common.load_gaussians(ply,device='cuda');records=[]
    for row in rows:
        begin=time.monotonic()
        rgb,_,_=common.render_view(gs,np.linalg.inv(row['T_world_from_camera']),np.asarray(row['K']),public['width'],public['height'])
        if rgb.shape!=(public['height'],public['width'],3) or not np.isfinite(rgb).all():
            raise ValueError('invalid full-frame Gaussian render')
        name=row['frame_id']+'.png';pred=pred_dir/name;gt=gt_dir/name
        # Match the existing E2 renderer's declared PNG quantization exactly.
        Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).save(pred)
        original=Path(config['private_capture'])/row['rgb'];shutil.copyfile(original,gt)
        records.append({'frame_id':row['frame_id'],'camera':row,'pred_rgb':str(pred),'pred_sha256':sha(pred),
                        'reference_rgb':str(gt),'reference_sha256':sha(gt),'original_reference':str(original),
                        'render_and_save_s':time.monotonic()-begin})
    render={'kind':'source_scene_GS_static_DEV_heldout_v1','config':config,'planned_views':2,'rendered_views':len(records),
            'records':records,'renderer':'agents.core.common.render_view','renderer_source_sha256':sha(common.__file__),
            'scale':1.0,'quantization':'clip(0,1)*255 uint8 PNG','background':None,
            'evaluator_alignment':'NONE','exposure_matching':'NONE','training_invoked':False,
            'native_simulator_imported':False,'policy_arm':False,'gpu':torch.cuda.get_device_name(0)}
    save(out/'render_manifest.json',render)
    lpips=fidelity_metrics.LPIPSEvaluator('cuda')
    if lpips.model is None:raise RuntimeError('pinned LPIPS unavailable: '+str(lpips.error))
    result=fidelity_metrics.evaluate_images(pred_dir,gt_dir,lpips_evaluator=lpips,strict=True,_return_details=True)
    if result['n_images']!=2 or any(not isinstance(result[k],float) or not math.isfinite(result[k]) for k in ('psnr','ssim','lpips')):
        raise ValueError('complete finite heldout metrics required')
    result.update(scope='source_scene_GS_static_capture',tier='DEV',planned_views=2,
        source_render_manifest=str(out/'render_manifest.json'),source_render_sha256=sha(out/'render_manifest.json'),
        metric_implementation_sha256=sha(fidelity_metrics.__file__),lpips_provenance=lpips.provenance,
        independent_from_training=True,wall_s=time.monotonic()-started,paper_ready=False,policy_arm=False)
    save(out/'fidelity_metrics.json',result)
    # Contact sheet is separate from metric inputs; no overlay/crop enters evaluation.
    sheet=Image.new('RGB',(1920,800),'white');draw=ImageDraw.Draw(sheet)
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,20) if Path(font_path).exists() else ImageFont.load_default()
    for j,row in enumerate(records):
        target=np.asarray(Image.open(row['reference_rgb']).convert('RGB'));pred=np.asarray(Image.open(row['pred_rgb']).convert('RGB'))
        error=np.clip(np.abs(target.astype(int)-pred.astype(int))*4,0,255).astype(np.uint8)
        for i,(label,array) in enumerate([('Original heldout RGB',target),('Frozen source GS',pred),('Absolute RGB difference x4',error)]):
            y=j*400;draw.text((i*640+12,y+8),row['frame_id']+' | '+label,fill='black',font=font)
            sheet.paste(Image.fromarray(array).resize((640,360),Image.Resampling.LANCZOS),(i*640,y+35))
    sheet.save(out/'contact_sheet.png')
    sealed_views(config)  # no source/config/view mutation during evaluation
    save(out/'evaluator_receipt.json',{'status':'PASS','source_ply_sha256':sha(ply),
        'render_manifest_sha256':sha(out/'render_manifest.json'),'metrics_sha256':sha(out/'fidelity_metrics.json'),
        'contact_sheet_sha256':sha(out/'contact_sheet.png'),'training_or_tuning':False,'policy_claim':False})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    print(json.dumps(evaluate(json.loads(a.config.read_text()),a.out),allow_nan=False))


if __name__=='__main__':main()
