"""Frozen Chorus encoding and deterministic open-vocabulary Gaussian instances.

Chorus is attributed prior work. Our wrapper preserves source Gaussian identity;
connected components are a baseline, not a claim to solve arbitrary instances.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .core import digest, local_model, receipt, save


def encode(config, inputs, out):
    from .models import source
    src = source(config)
    import chorus
    model = local_model(config)
    # Disable filtering explicitly: feature row i must remain Gaussian row i.
    encoder = chorus.load('chorus_3dgs', checkpoint=model, outputs=('lang', 'dino'),
                          return_numpy=True, outlier_filter=False)
    encoded = encoder.encode(inputs['gaussians'])
    ids = np.load(inputs['gaussian_ids'], allow_pickle=False)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids): raise ValueError('unique persistent Gaussian IDs required')
    artifacts = {}
    for key in ('lang', 'dino'):
        feat = np.asarray(encoded.features[key])
        if feat.ndim != 2 or len(feat) != len(ids) or not np.isfinite(feat).all():
            raise ValueError('Chorus output does not align to the unfiltered Gaussian source')
        path = Path(out)/(key+'.npy'); np.save(path, feat); artifacts[key] = path
    np.save(Path(out)/'gaussian_ids.npy', ids); artifacts['ids'] = Path(out)/'gaussian_ids.npy'
    save(Path(out)/'encoding.json', {'source': src, 'input': receipt(inputs['gaussians']),
         'ids': receipt(inputs['gaussian_ids']), 'outlier_filter': False,
         'checkpoint_manifest': receipt(config['checkpoint_manifest']),
         'text_space': config['text_space'], 'pretraining_overlap': config.get('pretraining_overlap', 'UNKNOWN')})
    artifacts['receipt'] = Path(out)/'encoding.json'
    return artifacts


def normalized(x):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or not np.isfinite(x).all(): raise ValueError('finite feature matrix required')
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm <= 1e-12): raise ValueError('zero feature vector')
    return x/norm


def instances(coords, features, query, ids, *, threshold=.25, radius=.03,
              edge_cosine=.8, min_size=30, k_neighbors=24):
    """Sparse k-neighbor grouping. Distinct components of one text remain separate."""
    from scipy.spatial import cKDTree
    xyz = np.asarray(coords, np.float64); f = normalized(features); q = normalized(np.asarray(query).reshape(1, -1))[0]
    ids = np.asarray(ids)
    if xyz.shape != (len(f), 3) or ids.shape != (len(f),) or len(np.unique(ids)) != len(ids):
        raise ValueError('coordinate/feature/ID alignment mismatch')
    if not np.isfinite(xyz).all() or radius <= 0 or min_size < 1 or k_neighbors < 1:
        raise ValueError('invalid geometry/grouping parameters')
    score = f@q; idx = np.flatnonzero(score >= threshold)
    if not len(idx): return []
    tree = cKDTree(xyz[idx]); parent = np.arange(len(idx))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    distances, neighbors = tree.query(xyz[idx], k=min(k_neighbors+1, len(idx)), distance_upper_bound=radius)
    distances = np.asarray(distances).reshape(len(idx), -1); neighbors = np.asarray(neighbors).reshape(len(idx), -1)
    for i in range(len(idx)):
        for j, d in zip(neighbors[i], distances[i]):
            if j >= len(idx) or j == i or not np.isfinite(d): continue
            if float(f[idx[i]]@f[idx[j]]) >= edge_cosine:
                a, b = find(i), find(int(j)); parent[max(a,b)] = min(a,b)
    groups = {}
    for i in range(len(idx)): groups.setdefault(find(i), []).append(int(idx[i]))
    result = []
    for members in groups.values():
        if len(members) < min_size: continue
        source_ids = sorted(ids[members].tolist())
        result.append({'instance_id': 'inst-'+digest(source_ids)[:20], 'source_gaussian_ids': source_ids,
            'mean_text_similarity': float(score[members].mean()), 'center_m': xyz[members].mean(axis=0).tolist(),
            'aabb_m': [xyz[members].min(axis=0).tolist(), xyz[members].max(axis=0).tolist()]})
    return sorted(result, key=lambda r: r['instance_id'])


def group(config, inputs, out):
    # Query embedding is computed with the matching frozen SigLIP2 text model.
    from .core import load
    if config.get('coordinates_unit') != 'm':
        raise ValueError('declare observation-derived metric coordinates before grouping')
    meta = load(inputs['text_metadata'])
    if meta['text_space'] != config['text_space']: raise ValueError('text/scene feature spaces differ')
    values = [np.load(inputs[k], allow_pickle=False) for k in ('coords', 'features', 'query', 'gaussian_ids')]
    groups = instances(*values, **config.get('grouping', {}))
    save(Path(out)/'instances.json', {'instances': groups, 'query': meta['query'],
         'text_space': meta['text_space'], 'source': {k: receipt(v) for k,v in inputs.items()},
         'empty_query_is_failure': len(groups) == 0, 'new_category_manipulation_claim': False})
    return {'instances': Path(out)/'instances.json'}


def text_query(config, inputs, out):
    """Matching local SigLIP2 text model; never substitute a different CLIP space."""
    import torch
    from transformers import AutoModel,AutoProcessor
    model_path=local_model(config)
    if not isinstance(config.get('query'),str) or not config['query'].strip():raise ValueError('nonempty text query')
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    model=AutoModel.from_pretrained(model_path,local_files_only=True).to(config.get('device','cuda')).eval()
    tokens=processor(text=[config['query']],padding='max_length',return_tensors='pt')
    tokens={k:v.to(model.device) for k,v in tokens.items()}
    with torch.no_grad():feat=model.get_text_features(**tokens)
    if hasattr(feat,'pooler_output'):feat=feat.pooler_output
    feat=normalized(feat.float().cpu().numpy())
    np.save(Path(out)/'query.npy',feat[0])
    save(Path(out)/'text.json',{'query':config['query'],'text_space':config['text_space'],
         'checkpoint_manifest':receipt(config['checkpoint_manifest'])})
    return {'query':Path(out)/'query.npy','metadata':Path(out)/'text.json'}
