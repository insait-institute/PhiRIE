"""Resumable real-model construction -> PhiView capture, one scene per process."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from types import SimpleNamespace
from physicalview.config import load_config
from physicalview import pipeline
from physicalview.phiview import save_json


def run_command(argv,env,cwd,out,label,timeout=14400):
    out.mkdir(parents=True,exist_ok=True)
    receipt=out/(label+'.json'); log=out/(label+'.log')
    if receipt.exists() or log.exists():
        history=out/'attempts'/str(time.time_ns());history.mkdir(parents=True)
        for previous in (receipt,log):
            if previous.exists():shutil.move(str(previous),history/previous.name)
    record={'argv':list(map(str,argv)),'cwd':str(cwd),'env':env,'started':time.time(),'status':'running'}
    save_json(receipt,record)
    combined=dict(os.environ);combined.update(env)
    # PhysicalView adapter modules must remain importable with stage-specific envs.
    combined['PYTHONPATH']=str(Path(__file__).resolve().parents[1])+':'+combined.get('PYTHONPATH','')
    combined['HF_HOME']='/group/worldcept/hf_cache';combined['TORCH_HOME']='/group/worldcept/torch_hub_cache'
    try:
        with log.open('w') as f:
            result=subprocess.run(list(map(str,argv)),cwd=cwd,env=combined,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
        record.update(exit_code=result.returncode,status='succeeded' if result.returncode==0 else 'failed')
        if result.returncode:raise RuntimeError(f'{label}: exit {result.returncode}; see {log}')
    except Exception as e:
        record.update(status='failed',error=str(e));raise
    finally:
        record['ended']=time.time();save_json(receipt,record)


def stage(spec,logs,label):
    run_command(spec.argv,spec.env,spec.cwd,logs,label)


def inpaint(ctx,names,prompt,logs):
    ids=[int(n[4:]) for n in names]
    specs=pipeline.inpaint(ctx,SimpleNamespace(kind='object',object_ids=ids),prompt,'qwen_image_edit',refine_iters=1000)
    prepared=all((ctx.out_dir/'inpaint'/n/'removal_idx.npy').exists() and (ctx.out_dir/'inpaint'/n/'mask_0.png').exists() for n in names)
    if prepared:specs=specs[2:]
    for name in names:
        for p in (ctx.out_dir/'inpaint'/name).glob('inpainted_*.png'):p.unlink()
    for i,spec in enumerate(specs):
        if 'agents.edit.inpaint_qwen' in spec.argv:
            spec.argv[spec.argv.index('agents.edit.inpaint_qwen')]='physicalview.phiview_inpaint'
            spec.env['SIMANY_REQUIRE_QWEN']='1'
            # The paper campaign reserves an H200. Keep the model resident;
            # the adapter checks for 70 GiB free before loading it. An explicit
            # override still permits CPU offload on a smaller GPU.
            spec.env['PHIVIEW_QWEN_PLACEMENT']=os.environ.get('PHIVIEW_QWEN_PLACEMENT','cuda')
        if 'agents.edit.inpaint_fill' in spec.argv:
            run_command([ctx.config.interpreter('studio'),'-m','physicalview.phiview_inpaint_guard','--root',ctx.out_dir/'inpaint','--objects',','.join(names),'--prompt',prompt],{},ctx.config.package_root,logs,'prompt-guard')
        stage(spec,logs,f'{i:02d}-{spec.tags.get("step","stage")}')
    if not (ctx.out_dir/'inpaint/clean_background.ply').exists():raise ValueError('No completed Gaussian inpaint output')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--index',type=int,required=True)
    ap.add_argument('--dataset',default='scannetpp',choices=['scannetpp','libero','behavior'])
    args=ap.parse_args();root=Path(args.root).resolve()
    config_path=root/('config.yaml' if args.dataset=='scannetpp' else f'{args.dataset}-config.yaml')
    cfg=load_config(config_path)
    if args.dataset=='scannetpp':
        row=json.loads((root/'roster.json').read_text())['datasets']['scannetpp'][args.index]
    else:
        row=json.loads((root/f'{args.dataset}-rows'/f'{args.index}.json').read_text())
    sid=row['scene'];asset=Path(row['assets']);out=root/args.dataset/sid;out.mkdir(parents=True,exist_ok=True)
    status={'scene':sid,'state':'running','started':time.time(),'job_id':os.environ.get('SLURM_JOB_ID')}
    save_json(out/'campaign-status.json',status)
    try:
        ctx=pipeline.StageContext(cfg,sid,asset,auto=row.get('auto',False),scene_dir=cfg.scannetpp_root/'data'/sid)
        names=[r['id'] for r in row['objects'] if r['accepted']]
        if not (asset/'sim_export/scene.xml').exists():
            if not all((asset/'objects'/n/'object.urdf').exists() for n in names):
                stage(pipeline.physics(ctx,None),out/'build-logs','physics')
            stage(pipeline.export_mjcf(ctx,'room',True),out/'build-logs','export-room')
        if not (asset/'inpaint/clean_background.ply').exists() or not (asset/'inpaint/paper-clean-receipt.json').exists():
            inpaint(ctx,names,'Remove the masked foreground object completely. Reconstruct the empty supporting surface with consistent texture, perspective, and lighting. Preserve the surrounding scene.',out/'clean-all-logs')
            save_json(asset/'inpaint/paper-clean-receipt.json',{'objects':names,'backend':'Qwen-Image-Edit-2511','fresh_execution':True,'fitting_iterations':1000,'visual_review':'pending'})
        studio=cfg.interpreter('studio')
        capture_cmd=[studio,'-m','physicalview.paper_capture','--config',config_path,'--scene',row['result_set'],'--out',out]
        from physicalview.paper_capture import SURVEY_VERSION
        survey_file=out/'camera-survey.json'
        if not survey_file.exists() or json.loads(survey_file.read_text()).get('version')!=SURVEY_VERSION:
            if survey_file.exists():
                history=out/'survey-attempts'/str(time.time_ns());history.mkdir(parents=True)
                shutil.move(str(survey_file),history/survey_file.name)
            run_command(capture_cmd+['--survey-only'],{},cfg.package_root,out/'capture-logs','survey')
        chosen=json.loads((out/'camera-survey.json').read_text())['selected'];name=chosen['object'];label=chosen['label']
        prompt=f'Remove the {label} completely. Reconstruct the empty supporting surface with matching material, texture, perspective, and lighting. Preserve every other object and all unmasked parts of the scene.'
        pwork=root/'prompt-builds'/sid
        prompt_record=out/'prompt-source.json'
        if prompt_record.exists():
            previous=json.loads(prompt_record.read_text())
            if previous.get('objects')!=[name] or previous.get('prompt')!=prompt:
                history=out/'prompt-attempts'/str(time.time_ns());history.mkdir(parents=True)
                shutil.move(str(prompt_record),history/prompt_record.name)
        if not (out/'prompt-source.json').exists():
            pwork.mkdir(parents=True,exist_ok=True)
            for folder in ('objects','inpaint'):
                if not (pwork/folder).exists():shutil.copytree(asset/folder,pwork/folder)
            pctx=pipeline.StageContext(cfg,sid,pwork,auto=ctx.auto,scene_dir=ctx.scene_dir)
            inpaint(pctx,[name],prompt,out/'prompt-logs')
            save_json(out/'prompt-source.json',{'objects':[name],'prompt':prompt,'path':pwork/'inpaint/clean_background.ply','receipt':pwork/'inpaint/prompt-receipt.json','fitting_iterations':1000,'fresh_model_run':True})
        if not (out/'fresh-sam3/report.json').exists():
            labels=sorted(set(r['label'] for r in row['objects']))
            run_command([cfg.interpreter('sam3'),'-m','physicalview.paper_sam3','--frame',ctx.resolved_images_dir()/chosen['camera'],'--labels',json.dumps(labels),'--out',out/'fresh-sam3'],{},cfg.package_root,out/'capture-logs','fresh-sam3')
        run_command(capture_cmd,{},cfg.package_root,out/'capture-logs','features')
        counts=json.loads((out/'capture-complete.json').read_text())['counts']
        status.update(state='captured' if counts['captured']==counts['total'] else 'partial',counts=counts,publication_review='pending')
    except Exception as e:
        status.update(state='failed',error=str(e),traceback=traceback.format_exc());raise
    finally:status['ended']=time.time();save_json(out/'campaign-status.json',status)

if __name__=='__main__':main()
