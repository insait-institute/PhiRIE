"""Build a review gallery and editable SVG panels without modifying scene pixels."""
import argparse,base64,csv,html,json,math,time
from pathlib import Path
from PIL import Image
from physicalview.paper_capture import FEATURES
from physicalview.phiview import save_json


def panel(folder,feature,paths,sidecar):
    # Image data remains lossless; text/caption is a separate editable vector layer.
    if feature=='i_fall_friction':
        picks=[]
        for action in ('fall','friction'):
            group=[p for p in paths if p.stem.startswith(action+'-')]
            picks.extend(group if len(group)<=3 else [group[0],group[len(group)//2],group[-1]])
    elif feature=='m_navigation':picks=paths
    else:picks=paths if len(paths)<=3 else [paths[0],paths[len(paths)//2],paths[-1]]
    cell_w=960;cell_h=640;gap=24;top=70;bottom=75
    columns=min(4 if feature=='m_navigation' else 3,len(picks));rows=math.ceil(len(picks)/columns)
    width=cell_w*columns+gap*(columns-1)
    height=top+rows*(cell_h+bottom)
    parameter=feature=='e_physical_parameters'
    if parameter:width+=520
    doc=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="12" y="42" font-family="DejaVu Sans,sans-serif" font-size="28">{html.escape(FEATURES[feature])}</text>']
    for i,p in enumerate(picks):
        x=(i%columns)*(cell_w+gap);y=top+(i//columns)*(cell_h+bottom)
        im=Image.open(p);w,h=im.size;dh=min(cell_h,cell_w*h/w);dw=dh*w/h
        encoded=base64.b64encode(p.read_bytes()).decode()
        doc += [f'<image x="{x}" y="{y}" width="{dw}" height="{dh}" href="data:image/png;base64,{encoded}"/>',
                f'<text x="{x+12}" y="{y+cell_h+32}" font-family="DejaVu Sans,sans-serif" font-size="24">{html.escape(p.stem)}</text>']
    if parameter:
        data=json.loads((folder/'parameters.json').read_text());x=cell_w+32
        lines=[('Selected object',sidecar['state']['selected']),('Mass (kg)',f"{data['mass_kg']:.4g}"),('Sliding friction',f"{data['friction'][0]:.4g}"),('Inertia (kg m²)',', '.join(f'{v:.3g}' for v in data['inertia_kg_m2'])),('Source','Active MuJoCo model')]
        for i,(label,value) in enumerate(lines):
            y=top+45+i*105
            doc += [f'<text x="{x}" y="{y}" font-family="DejaVu Sans,sans-serif" font-size="20" fill="#64748b">{html.escape(label)}</text>',f'<text x="{x}" y="{y+35}" font-family="DejaVu Sans,sans-serif" font-size="25">{html.escape(str(value))}</text>']
    doc.append('</svg>');(folder/'panel.svg').write_text('\n'.join(doc))
    # Lightweight review derivative; publication SVG still embeds original PNGs.
    from PIL import ImageDraw
    scale=1280/width;preview=Image.new('RGB',(1280,round(height*scale)),'white');draw=ImageDraw.Draw(preview)
    draw.text((8,8),FEATURES[feature],fill='black')
    for i,p in enumerate(picks):
        im=Image.open(p).convert('RGB');im.thumbnail((round(cell_w*scale),round(cell_h*scale)))
        x=round((i%columns)*(cell_w+gap)*scale);y=round((top+(i//columns)*(cell_h+bottom))*scale)
        preview.paste(im,(x,y));draw.text((x+4,y+round(cell_h*scale)+5),p.stem,fill='black')
    if parameter:
        for i,(label,value) in enumerate(lines):
            x=round((cell_w+32)*scale);y=round((top+45+i*105)*scale)
            draw.text((x,y),label,fill='#64748b');draw.text((x,y+20),str(value),fill='black')
    preview.save(folder/'review-preview.jpg',quality=90)
    (folder/'panel-layout.json').write_text(json.dumps({'frames':[p.name for p in picks],
        'columns':columns,'rows':rows,'original_pngs_embedded':True,
        'review_preview_only':'review-preview.jpg'},indent=2))


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args(argv);root=Path(a.root).resolve();rows=[];cards=[]
    import fcntl
    lock=(root/'.paper-pack.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    for dataset in ['scannetpp','libero','behavior']:
        for scene in sorted((root/dataset).glob('*')):
            if not scene.is_dir():continue
            file=scene/'features.json';features=json.loads(file.read_text()) if file.exists() else {}
            for key,title in FEATURES.items():
                row=features.get(key,{'status':'not_run','frames':[]})
                expected=row.get('frames',[])
                paths=[scene/p for p in expected if (scene/p).is_file() and (scene/p).with_suffix('.json').is_file()]
                status=row['status']
                if status=='captured' and (not paths or len(paths)!=len(expected)):
                    status='missing_artifacts'
                review_path=scene/key/'review.json'
                review=json.loads(review_path.read_text()) if review_path.exists() else {'status':'pending'}
                rows.append({'dataset':dataset,'scene':scene.name,'feature':key,'status':status,'pngs':len(paths),'review':review['status'],'error':row.get('error','')})
                if paths and len(paths)==len(expected) and (key!='e_physical_parameters' or (scene/key/'parameters.json').is_file()):
                    side=json.loads(paths[0].with_suffix('.json').read_text());panel(scene/key,key,paths,side)
                    link=(scene/key/'panel.svg').relative_to(root)
                    preview=(scene/key/'review-preview.jpg').relative_to(root)
                    cards.append(f'<article data-dataset="{dataset}" data-feature="{key}"><h2>{dataset} / {scene.name} / {key}</h2><p>{html.escape(title)} · {status} · publication review: {html.escape(review["status"])}</p><a href="{link}"><img loading="lazy" src="{preview}"></a><p><a href="{link}">Editable SVG with lossless source PNGs</a> · <a href="{scene.relative_to(root)}/features.json">Evidence</a> · <a href="{scene.relative_to(root)}/{key}/">Lossless PNG and sidecars</a></p></article>')
    save_json(root/'coverage.json',rows)
    summary={'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             'requested_groups':len(rows),'datasets':{},'quality_status':'Capture and publication review are separate; pending/needs_work groups are not paper-approved.'}
    for dataset in ('scannetpp','libero','behavior'):
        subset=[r for r in rows if r['dataset']==dataset];scenes=set(r['scene'] for r in subset)
        summary['datasets'][dataset]={'captured_groups':sum(r['status']=='captured' for r in subset),
            'feature_pngs':sum(r['pngs'] for r in subset),
            'approved_groups':sum(r['status']=='captured' and r['review']=='approved' for r in subset),
            'complete_scenes':sum(all(r['status']=='captured' for r in subset if r['scene']==scene) for scene in scenes)}
    save_json(root/'progress-summary.json',summary)
    with (root/'coverage.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','scene','feature','status','pngs','review','error']);w.writeheader();w.writerows(rows)
    doc='''<!doctype html><meta charset="utf-8"><title>PhiView paper figure review</title><style>body{font-family:system-ui;margin:40px;background:#f4f6f9;color:#152238}header{position:sticky;top:0;background:white;padding:16px;z-index:2}article{background:white;padding:20px;margin:24px 0;border-radius:10px}h2{font-size:18px}img{width:100%;height:auto}select{padding:8px;margin:8px}p{color:#475569}</style><header><h1>PhiView — paper figure review</h1><p>Actual renders and simulation states. Captured does not mean approved for publication. See evidence for source resolution and model/GT provenance.</p><select id="dataset"><option value="">All datasets</option><option>scannetpp</option><option>libero</option><option>behavior</option></select><select id="feature"><option value="">All features</option>'''
    doc+=''.join(f'<option value="{k}">{html.escape(v)}</option>' for k,v in FEATURES.items())+'</select></header>'+''.join(cards)
    doc+='''<script>function filter(){document.querySelectorAll('article').forEach(a=>a.hidden=(dataset.value&&a.dataset.dataset!==dataset.value)||(feature.value&&a.dataset.feature!==feature.value))}dataset.onchange=feature.onchange=filter;</script>'''
    (root/'gallery.html').write_text(doc)
    print(json.dumps({'scene_feature_rows':len(rows),'captured':sum(r['status']=='captured' for r in rows),'review_approved':sum(r['review']=='approved' for r in rows)}))


def refresh(root):
    """Update review artifacts without masking the capture process exit status."""
    try:main(['--root',str(root)])
    except Exception as exc:
        print(f'Paper gallery refresh failed (capture evidence retained): {exc}',flush=True)

if __name__=='__main__':main()
