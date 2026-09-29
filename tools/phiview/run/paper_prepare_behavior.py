"""CPU source preparation, isolated from GPU feature captures and their counts."""
import argparse,json,subprocess,shutil,time
from pathlib import Path
from physicalview.config import load_config
from physicalview.phiview import save_json

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--indices',nargs='+',type=int,default=list(range(10)));a=p.parse_args()
root=Path(a.root).resolve();cfg=load_config(root/'behavior-config.yaml');work=root/'behavior-source-multishard-preflight'
rows=json.loads((root/'behavior-roster.json').read_text());work.mkdir(parents=True,exist_ok=True)
history=work/'attempts'/str(time.time_ns())
if (work/'preparation-status.json').exists():
    history.mkdir(parents=True);shutil.copy2(work/'preparation-status.json',history/'preparation-status.json')
records=[]
for index in a.indices:
    row=rows[index];sid=row['scene'];scene=work/'data'/sid;log=work/(sid+'.log')
    record={'scene':sid,'source_only':True,'feature_groups':0,'started':time.time()}
    try:
        if not (scene/'paper-source-resolution.json').is_file():
            history.mkdir(parents=True,exist_ok=True)
            if scene.exists():shutil.move(str(scene),history/sid)
            if log.exists():shutil.move(str(log),history/log.name)
            with log.open('w') as stream:
                result=subprocess.run([str(cfg.interpreter('main')),'-m','physicalview.paper_behavior','--task',row['task'],
                    '--scene-name',sid,'--root',str(work),'--episodes','40','--max-frames','120','--tasks-config',str(root/'behavior-tasks.yaml')],
                    stdout=stream,stderr=subprocess.STDOUT,timeout=600)
            if result.returncode:raise RuntimeError(f'Extraction failed: {log}')
        pose=json.loads((scene/'pose-selection.json').read_text())
        resolution=json.loads((scene/'paper-source-resolution.json').read_text())
        if pose['selected_camera_frames']<8 or resolution['upscale']!=1:raise RuntimeError('Source contract failed')
        images=list((scene/'dslr/resized_undistorted_images').glob('*.jpg'))
        if len(images)!=min(pose['selected_camera_frames'],120):raise RuntimeError('RGB count differs from selected poses')
        destination=cfg.scannetpp_root/'data'/sid
        if destination.exists():
            if not (destination/'paper-source-resolution.json').is_file():raise RuntimeError(f'Partial existing destination needs review: {destination}')
            if (destination/'pose-selection.json').read_bytes()!=(scene/'pose-selection.json').read_bytes():
                if (cfg.splats_root/(sid+'.ply')).exists():raise RuntimeError('Source replacement would invalidate an existing trained Gaussian scene')
                history.mkdir(parents=True,exist_ok=True);shutil.move(str(destination),history/('previous-destination-'+sid))
        if not destination.exists():
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(scene,destination)
        record.update(status='source_prepared',selected_episode=pose['selected_episode'],views=len(images),resolution=[resolution['width'],resolution['height']],destination=str(destination))
    except Exception as exc:
        record.update(status='failed',error=str(exc))
    record['ended']=time.time();records.append(record);save_json(work/'preparation-status.json',records)
    print(json.dumps(record),flush=True)
