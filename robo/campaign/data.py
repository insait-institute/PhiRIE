"""Dataset inventory, scene-family splits, and outcome-independent preview gallery."""
from __future__ import annotations
import html
from pathlib import Path
from .core import digest, load, rows, save, receipt, safe_id

# Capabilities are not claims that task/policy adapters have been validated.
DATASETS = {
    'robocasa': ('native_tasks', 'mujoco', 'https://robocasa.ai/'),
    'behavior': ('native_tasks', 'omnigibson', 'https://behavior.stanford.edu/'),
    'replicacad': ('rearrangement', 'habitat', 'https://aihabitat.org/datasets/replica_cad/'),
    'hssd': ('scene_reconstruction', 'habitat', 'https://3dlg-hcvc.github.io/hssd/'),
    'hypersim': ('offline_appearance', 'none', 'https://github.com/apple/ml-hypersim'),
    'scannetpp': ('real_scan', 'none', 'https://scannetpp.ml/'),
    'replica': ('real_scan', 'habitat', 'https://github.com/facebookresearch/Replica-Dataset'),
    'hm3d': ('real_scan', 'habitat', 'https://aihabitat.org/datasets/hm3d/'),
    'procthor': ('scene_reconstruction', 'ai2thor', 'https://procthor.allenai.org/'),
    'interiorverse': ('offline_appearance', 'none', 'https://interiorverse.github.io/'),
    'droid': ('real_capture', 'none', 'https://droid-dataset.github.io/'),
    'phone': ('real_capture', 'none', 'local-consented-recordings'),
}


def validate_inventory(records):
    seen = set(); families = {}; assets = {}
    for r in records:
        if r['dataset'] not in DATASETS:
            raise ValueError('unknown dataset')
        safe_id(r['scene_id']); safe_id(r['group_id'])
        key = (r['dataset'], r['release'], r['scene_id'])
        if key in seen:
            raise ValueError(f'duplicate scene: {key}')
        seen.add(key)
        if r['split'] not in ('train', 'dev', 'test', 'demo'):
            raise ValueError('split must be declared before method outcomes')
        family = (r['dataset'], r['group_id'])
        if family in families and families[family] != r['split']:
            raise ValueError(f'scene family leaks across splits: {family}')
        families[family] = r['split']
        # Cross-dataset captures of the same underlying scene must be grouped too.
        if r.get('physical_scene_id'):
            g = r['physical_scene_id']
            if g in assets and assets[g] != r['split']:
                raise ValueError('physical scene reused across data sources/splits')
            assets[g] = r['split']
        if not r.get('license_checked') or not r.get('source_index'):
            raise ValueError('license review and upstream scene-index source required')
        if any(k in r for k in ('success', 'method_score', 'lpips', 'psnr')):
            raise ValueError('outcomes cannot enter scene selection')
        if r.get('policy_ready') and not all(r.get(k) for k in
                ('task_ids', 'policy_receipt', 'native_import_gate')):
            raise ValueError('renderable scene is not automatically policy-ready')
    return records


def freeze(inventory, out, *, counts=None, seed=2027):
    records = validate_inventory(rows(inventory))
    counts = counts or {}
    for dataset,requested in counts.items():
        if dataset not in DATASETS or not isinstance(requested,dict):raise ValueError('unknown dataset count request')
        for split,n in requested.items():
            if split not in ('train','dev','test','demo') or type(n) is not int or n<0:raise ValueError('invalid split quota')
            available=sum(r['dataset']==dataset and r['split']==split for r in records)
            if available<n:raise ValueError(f'insufficient inventory for {dataset}/{split}: {available} < {n}')
    selected = []
    for dataset in sorted({r['dataset'] for r in records}):
        for split in ('train', 'dev', 'test', 'demo'):
            candidates = [r for r in records if r['dataset'] == dataset and r['split'] == split]
            # Stable hash order, never visual quality or downstream success ranking.
            candidates.sort(key=lambda r: digest([seed, dataset, r['group_id'], r['scene_id']]))
            n = counts.get(dataset, {}).get(split, len(candidates))
            if not isinstance(n, int) or n < 0 or len(candidates) < n:
                raise ValueError(f'insufficient inventory for {dataset}/{split}: {len(candidates)} < {n}')
            selected.extend(candidates[:n])
    root = Path(out); root.mkdir(parents=True, exist_ok=False)
    save(root/'scenes.jsonl', selected, jsonl=True)
    save(root/'manifest.json', {'schema': 1, 'source': receipt(inventory), 'seed': seed,
        'counts': counts, 'selected': len(selected), 'scene_hash': digest(selected),
        'selection_uses_outcomes': False})
    return selected


def catalog(out):
    """Write an inventory template, not invented scene IDs or download promises."""
    root = Path(out); root.mkdir(parents=True, exist_ok=False)
    save(root/'datasets.json', {k: {'capability': v[0], 'backend': v[1], 'source': v[2]}
                               for k, v in DATASETS.items()})
    save(root/'inventory_schema.json', {'required': ['dataset', 'release', 'scene_id',
         'group_id', 'split', 'license_checked', 'source_index'],
         'optional': ['physical_scene_id', 'thumbnail', 'room_type', 'style', 'task_ids',
                      'policy_ready', 'policy_receipt', 'native_import_gate',
                      'external_api_permitted', 'pretraining_overlap'],
         'no_fake_ids': True})


def gallery(inventory, out):
    records = validate_inventory(rows(inventory))
    root = Path(out); root.mkdir(parents=True, exist_ok=False)
    from PIL import Image
    cards = []
    for i, r in enumerate(records):
        p = r.get('thumbnail')
        if not p:
            continue
        image = Image.open(p).convert('RGB'); image.thumbnail((640, 400))
        name = f'{i:05d}.jpg'; image.save(root/name, quality=88)
        label = html.escape(f"{r['dataset']} / {r['scene_id']} / {r['split']}")
        cards.append(f'<figure><img src="{name}"><figcaption>{label}</figcaption></figure>')
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Source scene gallery</title>'
        '<style>body{font:18px sans-serif;background:#fafafa}main{display:grid;grid-template-columns:repeat(3,1fr)}'
        'figure{margin:12px}img{width:100%}figcaption{padding:10px}</style>'
        '<h1>Source-only scene gallery</h1><p>DEMO curation is separate from TEST sampling.</p><main>'
        + ''.join(cards) + '</main>')
    save(root/'source.json', receipt(inventory))
