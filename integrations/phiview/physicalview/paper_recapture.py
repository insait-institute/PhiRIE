"""Recapture existing constructed scenes with current renderer; archive prior frames."""
import argparse,json,os,time
from pathlib import Path
from physicalview.config import load_config
from physicalview.paper_capture import FEATURES
from physicalview.paper_campaign import run_command
from physicalview.phiview import save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--index',required=True,type=int)
    a=p.parse_args();root=Path(a.root).resolve();config=root/'config.yaml';cfg=load_config(config)
    row=json.loads((root/'roster.json').read_text())['datasets']['scannetpp'][a.index]
    out=root/'scannetpp'/row['scene'];status={'state':'running','job_id':os.environ.get('SLURM_JOB_ID'),
        'started':time.time(),'reason':'Antialiased rendering and independent room overview; construction assets unchanged'}
    save_json(out/'recapture-status.json',status)
    try:
        run_command([cfg.interpreter('studio'),'-m','physicalview.paper_capture','--config',config,
            '--scene',row['result_set'],'--out',out,'--features',*FEATURES],{},cfg.package_root,out/'recapture-logs','features')
        data=json.loads((out/'capture-complete.json').read_text())
        if data['counts']['captured']!=len(FEATURES):raise RuntimeError('Incomplete feature recapture')
        for key in FEATURES:
            for filename in data['features'][key]['frames']:
                frame=out/filename;side=json.loads(frame.with_suffix('.json').read_text())
                if not frame.is_file() or side.get('rasterize_mode')!='antialiased':
                    raise RuntimeError(f'Missing new-renderer frame: {frame}')
        status.update(state='captured',publication_review='pending',counts=data['counts'])
    except Exception as e:
        status.update(state='failed',error=str(e));raise
    finally:
        status['ended']=time.time();save_json(out/'recapture-status.json',status)
        from physicalview.paper_pack import refresh
        refresh(root)


if __name__=='__main__':main()
