"""Gate dataset transitions on every required feature, rather than Slurm exit alone."""
import argparse
import json
from pathlib import Path
from physicalview.paper_capture import FEATURES


def require_complete(root, dataset):
    root=Path(root)
    if dataset=='scannetpp':
        rows=json.loads((root/'roster.json').read_text())['datasets'][dataset]
    else:
        rows=json.loads((root/f'{dataset}-roster.json').read_text())
    missing=[]
    if len(rows)!=10:raise ValueError(f'{dataset}: expected 10 scene sources, found {len(rows)}')
    for row in rows:
        scene=row['scene'];p=root/dataset/scene/'capture-complete.json'
        data=json.loads(p.read_text()) if p.exists() else {}
        for key in FEATURES:
            feature=data.get('features',{}).get(key,{})
            if feature.get('status')!='captured' or not feature.get('frames'):
                missing.append(f'{scene}/{key}')
            else:
                for frame in feature['frames']:
                    image=root/dataset/scene/frame
                    if not image.is_file() or not image.with_suffix('.json').is_file():
                        missing.append(f'{scene}/{key}:missing frame/sidecar');break
    if missing:raise RuntimeError(f'{dataset} has {len(missing)} incomplete scene-feature groups: '+', '.join(missing[:20]))
    return {'dataset':dataset,'scenes':10,'captured_groups':140,'publication_quality_verified':False}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--dataset',required=True,choices=['scannetpp','libero','behavior']);a=ap.parse_args()
    print(json.dumps(require_complete(a.root,a.dataset)))
if __name__=='__main__':main()
