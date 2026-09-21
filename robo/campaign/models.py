"""Real generator/edit-model adapters. Invoke in each model's pinned environment.

Outputs are canonical proposals, NOT registered simulation assets. Registration,
CoACD and native import remain owned by the existing PhiRoom pipeline.
"""
from __future__ import annotations
import base64
import fcntl
import io
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import numpy as np
from PIL import Image
from .core import checked_path, digest, load, local_model, receipt, save, source_identity


_PIPELINES = {}
_VERIFIED = {}

def frozen_model(config):
    key=(str(config['checkpoint']), receipt(config['checkpoint_manifest'])['sha256'])
    if key not in _VERIFIED: _VERIFIED[key]=local_model(config)
    return _VERIFIED[key]

def source(config):
    actual = source_identity(config['source_root'])
    if actual['commit'] != config['source_commit']:
        raise ValueError('third-party source revision differs')
    sys.path.insert(0, actual['root'])
    return actual


def generate(config, inputs, out):
    """TRELLIS v1, TRELLIS.2 or official ReconViaGen v0.2 API, no GUI imports."""
    import torch
    import trimesh
    out = Path(out); backend = config['backend']; src = source(config)
    model = frozen_model(config)
    images = [Image.open(inputs[name]).convert('RGBA') for name in config['images']]
    if not images: raise ValueError('at least one image required')
    if backend != 'reconviagen' and len(images) != 1:
        raise ValueError('single-view generators receive exactly the declared anchor')
    seed = int(config.get('seed', 42)); torch.manual_seed(seed)
    start = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    kwargs = dict(config.get('inference', {}))
    if {'seed', 'image', 'formats', 'preprocess_image'} & set(kwargs):
        raise ValueError('reserved inference arguments')
    if backend == 'trellis2':
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        import o_voxel
        key=(backend,model)
        if key not in _PIPELINES:
            pipe=Trellis2ImageTo3DPipeline.from_pretrained(model);pipe.cuda();_PIPELINES[key]=pipe
        pipe=_PIPELINES[key]
        mesh = pipe.run(images[0], seed=seed, **kwargs)[0]
        if config.get('pbr_export', True):
            glb = o_voxel.postprocess.to_glb(vertices=mesh.vertices, faces=mesh.faces,
                attr_volume=mesh.attrs, coords=mesh.coords, attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size, aabb=[[-.5, -.5, -.5], [.5, .5, .5]],
                decimation_target=int(config.get('decimation_target', 1000000)),
                texture_size=int(config.get('texture_size', 2048)), remesh=True,
                remesh_band=1, remesh_project=0, verbose=False)
            glb.export(out/'visual.glb', extension_webp=True)
        gs = None
    elif backend in ('trellis', 'reconviagen'):
        if backend == 'trellis':
            from trellis.pipelines import TrellisImageTo3DPipeline
            key=(backend,model)
            if key not in _PIPELINES:
                pipe=TrellisImageTo3DPipeline.from_pretrained(model);pipe.cuda();_PIPELINES[key]=pipe
            pipe=_PIPELINES[key]
            result = pipe.run(images[0], seed=seed, formats=['mesh', 'gaussian'], **kwargs)
        else:
            from trellis.pipelines import TrellisVGGTTo3DPipeline
            key=(backend,model)
            if key not in _PIPELINES:
                pipe=TrellisVGGTTo3DPipeline.from_pretrained(model)
                pipe._device=torch.device('cuda');pipe.low_vram=bool(config.get('low_vram',True))
                pipe.birefnet_model.cuda()
                if not pipe.low_vram:
                    for module in pipe.models.values():module.to(pipe._device)
                    pipe.VGGT_model.to(pipe._device)
                _PIPELINES[key]=pipe
            pipe=_PIPELINES[key]
            prepared=[pipe.preprocess_image(im) for im in images]
            result, _, _ = pipe.run(image=prepared, seed=seed, formats=['mesh', 'gaussian'],
                                   preprocess_image=False, **kwargs)
        mesh = result['mesh'][0]; gs = result['gaussian'][0]
        if config.get('pbr_export', True):
            from trellis.utils import postprocessing_utils
            glb = postprocessing_utils.to_glb(gs, mesh, simplify=.95,
                texture_size=int(config.get('texture_size', 2048)), verbose=False)
            glb.export(out/'visual.glb')
    else:
        raise ValueError('unsupported generator; RVG-v0.5 must have its own admitted adapter, never alias v0.2')
    vertices = mesh.vertices.detach().cpu().numpy(); faces = mesh.faces.detach().cpu().numpy()
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all() or not len(faces):
        raise ValueError('generator produced invalid geometry')
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(out/'canonical.obj')
    artifacts = {'mesh': out/'canonical.obj'}
    if (out/'visual.glb').exists(): artifacts['visual'] = out/'visual.glb'
    if gs is not None:
        gs.save_ply(str(out/'gaussians.ply'), transform=None); artifacts['gaussians'] = out/'gaussians.ply'
    save(out/'proposal.json', {'backend': backend, 'seed': seed, 'source': src,
        'checkpoint_manifest': receipt(config['checkpoint_manifest']),
        'input_images': [receipt(inputs[n]) for n in config['images']],
        'inference': kwargs, 'wall_s': time.monotonic()-start,
        'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
        'canonical_units': 'generator units; registration required', 'registration': 'NOT_RUN',
        'physics': 'NOT_RUN', 'artifacts': {k: receipt(v) for k,v in artifacts.items()}})
    artifacts['proposal'] = out/'proposal.json'
    return artifacts


def masked_composite(original, generated, mask):
    a, b = np.asarray(original), np.asarray(generated)
    m = np.asarray(mask)
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3 or m.shape != a.shape[:2]:
        raise ValueError('exact input/output/mask grid required; do not silently resize provider output')
    m = m.astype(bool); out = a.copy(); out[m] = b[m]
    if not np.array_equal(out[~m], a[~m]): raise AssertionError('outside-mask preservation failed')
    return out


def reserve_remote(config, request_id):
    """Atomic shared call/cost reservation; no automatic billable retries.

    USD reservation is an operator-approved conservative bound, not measured billing.
    Real provider usage must be reconciled after the run. Failed calls remain reserved.
    """
    policy_path = Path(config['approval_file']).absolute(); policy = load(policy_path)
    if policy.get('approved') is not True or policy.get('data_upload_permitted') is not True:
        raise PermissionError('remote image upload and spending approval required')
    if config['model_id'] not in policy.get('models', []): raise PermissionError('model not approved')
    cost = float(policy['reserved_usd_per_call'])
    if cost <= 0 or int(policy['max_calls']) < 1: raise ValueError('positive remote cost/call bounds required')
    ledger = Path(policy['ledger']).absolute(); ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open('a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX); f.seek(0)
        previous = [json.loads(s) for s in f if s.strip()]
        if any(x['request_id'] == request_id for x in previous):
            raise RuntimeError('request already reserved; inspect previous result rather than paying twice')
        if len(previous) >= int(policy['max_calls']) or sum(x['reserved_usd'] for x in previous)+cost > float(policy['max_usd']):
            raise PermissionError('remote campaign budget exhausted')
        f.write(json.dumps({'request_id': request_id, 'model_id': config['model_id'],
                           'reserved_usd': cost, 'approval_sha': receipt(policy_path)['sha256']})+'\n')
        f.flush(); os.fsync(f.fileno())
    return policy


def gemini_edit(config, image, mask_image, prompt, request_id):
    key = os.environ.get(config.get('api_key_env', 'GEMINI_API_KEY'))
    if not key: raise PermissionError('API key absent; never paste credentials in config')
    reserve_remote(config, request_id)
    def part(im):
        f = io.BytesIO(); im.save(f, format='PNG')
        return {'inlineData': {'mimeType': 'image/png', 'data': base64.b64encode(f.getvalue()).decode()}}
    model_id = config['model_id']
    if not model_id.replace('-', '').replace('.', '').isalnum(): raise ValueError('invalid model identifier')
    body = {'contents': [{'parts': [{'text': prompt}, part(image), part(mask_image)]}],
            'generationConfig': {'responseModalities': ['TEXT', 'IMAGE']}}
    req = urllib.request.Request(f'https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'x-goog-api-key': key}, method='POST')
    with urllib.request.urlopen(req, timeout=float(config.get('timeout_s', 180))) as response:
        data = json.load(response)
    candidates = data.get('candidates', [])
    images = [p['inlineData'] for c in candidates for p in c.get('content', {}).get('parts', [])
              if not p.get('thought',False) and 'inlineData' in p and p['inlineData'].get('mimeType', '').startswith('image/')]
    if len(images) != 1: raise RuntimeError('provider did not return exactly one image')
    im = Image.open(io.BytesIO(base64.b64decode(images[0]['data']))).convert('RGB')
    return im, {'model_version': data.get('modelVersion'), 'usage': data.get('usageMetadata'),
                'request_seed_supported': False, 'automatic_retries': 0}


def inpaint(config, inputs, out):
    """Object removal/background completion, NOT online observation harmonization."""
    out = Path(out)
    image = Image.open(inputs['rgb']).convert('RGB'); mask = np.asarray(Image.open(inputs['mask']).convert('L')) >= 128
    if mask.shape != (image.height, image.width) or not mask.any(): raise ValueError('nonempty aligned mask required')
    backend = config['backend']; start = time.monotonic(); meta = {}
    prompt = config.get('prompt', 'Remove the foreground object inside the white mask. Complete the background using surrounding surfaces. Keep camera, room layout and all unmasked objects unchanged. The second image is a binary edit mask, not content to insert.')
    mask_image = Image.fromarray(mask.astype('uint8')*255)
    if backend == 'telea':
        import cv2
        generated = Image.fromarray(cv2.inpaint(np.asarray(image), mask.astype('uint8')*255,
                                               float(config.get('radius', 3)), cv2.INPAINT_TELEA))
    elif backend in ('qwen_edit', 'sdxl_inpaint'):
        import torch
        model = frozen_model(config)
        rng = torch.Generator(device='cuda').manual_seed(int(config.get('seed', 42)))
        if backend == 'qwen_edit':
            from diffusers import QwenImageEditPlusPipeline
            key=(backend,model)
            if key not in _PIPELINES: _PIPELINES[key]=QwenImageEditPlusPipeline.from_pretrained(model,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
            pipe=_PIPELINES[key]
            generated = pipe(image=[image, mask_image], prompt=prompt, generator=rng,
                true_cfg_scale=float(config.get('true_cfg_scale', 4)), negative_prompt=' ',
                num_inference_steps=int(config.get('steps', 40)), guidance_scale=1.0,
                num_images_per_prompt=1).images[0]
            meta['mask_interface'] = 'instruction + second reference mask; not native mask conditioning'
        else:
            from diffusers import AutoPipelineForInpainting
            key=(backend,model)
            if key not in _PIPELINES: _PIPELINES[key]=AutoPipelineForInpainting.from_pretrained(model,torch_dtype=torch.float16,local_files_only=True).to('cuda')
            pipe=_PIPELINES[key]
            generated = pipe(prompt=prompt, image=image, mask_image=mask_image, generator=rng,
                num_inference_steps=int(config.get('steps', 40)), height=image.height, width=image.width).images[0]
        meta['checkpoint_manifest'] = receipt(config['checkpoint_manifest'])
    elif backend == 'gemini':
        if config.get('scene_upload_allowed') is not True: raise PermissionError('scene-level external upload permission absent')
        generated, meta = gemini_edit(config, image, mask_image, prompt,
                                     digest([receipt(inputs['rgb']), receipt(inputs['mask']), config, str(out)]))
    else:
        raise ValueError('unsupported editor; add separately versioned official adapter')
    generated = generated.convert('RGB'); generated.save(out/'provider_raw.png')
    if generated.size != image.size:
        # An image editor can return a different size. Explicit canvas policy is part of treatment.
        if config.get('resize_policy') != 'declared_bilinear':
            raise ValueError('provider changed canvas size; raw output preserved, no silent resampling')
        meta['original_provider_size'] = list(generated.size)
        generated = generated.resize(image.size, Image.Resampling.BILINEAR)
    final = masked_composite(image, generated, mask)
    Image.fromarray(final).save(out/'completed.png')
    save(out/'inpaint.json', {'backend': backend, 'prompt': prompt, 'mask': receipt(inputs['mask']),
        'rgb': receipt(inputs['rgb']), 'seed': config.get('seed'), 'wall_s': time.monotonic()-start,
        'outside_mask_exact': True, 'raw': receipt(out/'provider_raw.png'),
        'composited': receipt(out/'completed.png'), 'meta': meta,
        'physics_changed': False, 'temporal_consistency': 'NOT_ESTABLISHED'})
    return {'image': out/'completed.png', 'raw': out/'provider_raw.png', 'receipt': out/'inpaint.json'}
