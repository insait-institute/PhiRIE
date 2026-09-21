"""Inpainting stage 3 (venv, GPU): erase each object from its related views.

Qwen-Image-Edit-2509 via QwenImageEditPlusPipeline (the class the checkpoint
declares; it has no mask input - the paste() step enforces the mask by
compositing only inside it, so pixels outside the removal mask stay
bit-identical). Falls back to LaMa (needs LAMA_MODEL or a cached
big-lama.pt). Writes obj_XX/inpainted_{k}.png (full frames) +
inpaint_meta.json recording the backend per view.
"""
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from PIL import Image, ImageFilter

from agents.core import common as C

CROP_PAD = 0.35     # relative bbox padding around the mask
CROP_MAX = 1024     # longest crop side fed to the model
STEPS = 30


def load_qwen():
    import torch
    from diffusers import QwenImageEditPlusPipeline
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    if total > 90:
        pipe.enable_model_cpu_offload()
    else:
        # 48GB GPUs: layer-by-layer streaming, slow but fits
        pipe.enable_sequential_cpu_offload()
    return pipe


def load_lama():
    # CPU on purpose: when Qwen fails mid-run its weights may still hold the
    # GPU, and LaMa at 1MP is seconds on CPU anyway
    from simple_lama_inpainting import SimpleLama
    return SimpleLama(device="cpu")


def qwen_feasible():
    import torch
    if C.env("QWEN") == "0":
        return False
    if not torch.cuda.is_available() or torch.__version__ < "2.5":
        return False  # Qwen2.5-VL attention needs enable_gqa (torch>=2.5)
    if C.env("QWEN") == "force":
        return True  # sequential offload path fits any modern GPU
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    return total > 90


def paste(full, crop_box, patch, mask_crop):
    u0, v0, u1, v1 = crop_box
    patch = np.asarray(patch.resize((u1 - u0, v1 - v0)))
    region = full[v0:v1, u0:u1].astype(np.float32)
    a = np.asarray(
        Image.fromarray((mask_crop * 255).astype(np.uint8))
        .filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)[..., None] / 255.0
    full[v0:v1, u0:u1] = (a * patch + (1 - a) * region).astype(np.uint8)
    return full


def free_qwen(pipe):
    import torch
    try:
        del pipe
    except Exception:
        pass
    import gc
    gc.collect()
    torch.cuda.empty_cache()


PUBLIC_ALGORITHM = {'crop_pad': CROP_PAD, 'crop_max': CROP_MAX,
                    'blur_radius': 2, 'outside_mask': 'byte_exact',
                    'backend': 'lama_cpu', 'seed': 0}


def erase_masked_view(full, mask, model):
    """Existing crop/LaMa/paste recipe with explicit outside-mask preservation.

    Model exceptions propagate. An empty mask is a non-invocation, never an
    identity image reported as enhanced. The input arrays are not mutated.
    """
    full, mask = np.asarray(full), np.asarray(mask)
    if (full.dtype != np.uint8 or full.ndim != 3 or full.shape[2] != 3
            or mask.dtype != np.bool_ or mask.shape != full.shape[:2]):
        raise ValueError('erase input requires uint8 RGB and a same-size boolean mask')
    vv, uu = np.nonzero(mask)
    if not len(vv):
        return None
    H, W = mask.shape
    pad = int(CROP_PAD * max(np.ptp(vv), np.ptp(uu)) + 16)
    v0, v1 = max(int(vv.min()) - pad, 0), min(int(vv.max()) + pad, H)
    u0, u1 = max(int(uu.min()) - pad, 0), min(int(uu.max()) + pad, W)
    crop = Image.fromarray(full[v0:v1, u0:u1])
    msk_c = mask[v0:v1, u0:u1]
    scale = min(CROP_MAX / max(crop.size), 1.0)
    size = (max(int(crop.width * scale) // 8 * 8, 64),
            max(int(crop.height * scale) // 8 * 8, 64))
    prediction = model(crop.resize(size), Image.fromarray(
        (msk_c * 255).astype(np.uint8)).resize(size).convert('L'))
    if not isinstance(prediction, Image.Image) or prediction.mode != 'RGB' or prediction.size != size:
        raise ValueError('LaMa returned a malformed RGB prediction')
    output = paste(full.copy(), (u0, v0, u1, v1), prediction, msk_c)
    # Legacy Gaussian-blurred alpha also changed the exterior boundary band.
    # The declared strict treatment clips that band to the actual removal mask.
    output[~mask] = full[~mask]
    return output


def erase_public_views(mask_bundle, output, model):
    """Consume an authenticated mask bundle; retain every planned view/status."""
    from run.icra2027.e3_fresh_generation_contract import checked_identity
    from robo.eval.agentic_ablation import sha256_file
    output = Path(output)
    seen = set()
    for row in mask_bundle['rows']:
        key = (row['object_slot'], row['view_index'])
        if (key in seen or type(key[1]) is not int or key[1] < 0
                or not key[0].startswith('obj_') or '/' in key[0] or '\\' in key[0]
                or row['frame'] not in mask_bundle['train_images']
                or row['status'] not in {'MASK_READY', 'EMPTY_PROJECTED_MASK'}):
            raise ValueError('mask view roster/identity/status differs')
        seen.add(key)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    object_states = {row['object_slot']: row for row in mask_bundle['result']['objects']}
    for row in mask_bundle['rows']:
        record = dict(row, erasure_status='NOT_RUN', backend=None, output_identity=None,
                      failure=None, outside_mask_equal=None)
        if object_states[row['object_slot']].get('plane_status') == 'NO_PLANE':
            record['erasure_status'] = 'NOT_APPLICABLE_NO_PLANE'
            rows.append(record)
            continue
        if row['status'] == 'EMPTY_PROJECTED_MASK':
            if row['mask_identity'] is not None or row['projected_pixels'] != 0:
                raise ValueError('empty projected mask has an inference artifact')
            record['erasure_status'] = 'EMPTY_MASK'
            rows.append(record)
            continue
        image_path = checked_identity(mask_bundle['train_images'][row['frame']])
        mask_path = checked_identity(row['mask_identity'])
        with Image.open(image_path) as image:
            full = np.asarray(image.convert('RGB')).copy()
        with Image.open(mask_path) as image:
            if image.mode != 'L':
                raise ValueError('removal mask must be an L-mode binary image')
            raw = np.asarray(image)
            if not np.isin(raw, [0, 255]).all():
                raise ValueError('removal mask must be binary')
            mask = raw > 127
        if int(mask.sum()) != row['final_pixels'] or not mask.any():
            raise ValueError('ready mask pixel count changed or is empty')
        try:
            result = erase_masked_view(full, mask, model)
            path = output / row['object_slot'] / f"inpainted_{row['view_index']}.png"
            path.parent.mkdir(exist_ok=True)
            with path.open('xb') as stream:
                Image.fromarray(result).save(stream, format='PNG')
            record.update(erasure_status='ERASED', backend='lama_cpu',
                outside_mask_equal=bool(np.array_equal(result[~mask], full[~mask])),
                output_identity={'path': str(path), 'bytes': path.stat().st_size,
                                 'sha256': sha256_file(path)})
        except Exception as error:
            record.update(erasure_status='ENHANCER_FAILED', backend='lama_cpu',
                failure={'type': type(error).__name__, 'reason': str(error)})
        rows.append(record)
    return {'schema_version': 1, 'scope': 'automatic_train_only_erasure',
            'algorithm': PUBLIC_ALGORITHM, 'planned_objects': len(mask_bundle['objects']),
            'objects': mask_bundle['result']['objects'], 'planned_views': len(rows),
            'erased_views': sum(r['erasure_status'] == 'ERASED' for r in rows),
            'enhancer_failed_views': sum(r['erasure_status'] == 'ENHANCER_FAILED' for r in rows),
            'empty_mask_views': sum(r['erasure_status'] == 'EMPTY_MASK' for r in rows),
            'no_plane_views': sum(r['erasure_status'] == 'NOT_APPLICABLE_NO_PLANE' for r in rows),
            'rows': rows, 'paper_ready': False, 'cleaned_background_created': False}


def public_runtime():
    """Pin installed eraser bytes using the existing package closure helper."""
    import importlib.metadata
    import PIL
    from agents.recon.droid_extract import runtime_manifest, input_identity
    result = runtime_manifest(['torch', 'numpy', 'simple_lama_inpainting'])
    dist = importlib.metadata.distribution('Pillow')
    files = sorted(dist.files or [], key=str)
    records = [p for p in files if str(p).endswith('.dist-info/RECORD')]
    if len(records) != 1:
        raise ValueError('Pillow requires one installed RECORD')
    result['packages']['Pillow'] = {'version': dist.version,
        'entrypoint': input_identity(PIL.__file__),
        'record': input_identity(dist.locate_file(records[0])),
        'files': [input_identity(dist.locate_file(p).resolve()) for p in files
                  if str(p).endswith('.py') or '.so' in p.name]}
    return result


def run_public(context_path, contract_path):
    """Strict, E0-bound LaMa CPU erasure; old mask/preparation seals stay intact."""
    import sys
    import yaml
    import torch
    from robo.eval import agentic_ablation as e3
    from robo.eval import e3_factory_materializer as materializer
    from run.icra2027.e3_fresh_generation_contract import checked_identity
    from agents.edit.inpaint_masks import validate_public_masks, _identity
    context_path = Path(context_path).absolute()
    context = yaml.safe_load(context_path.read_text())
    expected = {'schema_version', 'scope', 'freeze_id', 'scene_id', 'mask_seal',
                'backend', 'checkpoint', 'runtime', 'algorithm'}
    if (set(context) != expected or context['schema_version'] != 1
            or context['scope'] != 'automatic_train_only_erasure'
            or context['backend'] != 'lama_cpu' or context['algorithm'] != PUBLIC_ALGORITHM):
        raise ValueError('strict erasure requires declared LaMa CPU algorithm; fallback is prohibited')
    root = e3.REPOSITORY_ROOT / 'outputs/icra2027' / context['freeze_id']
    e3._validate_cli_execution(contract_path, context['freeze_id'], root, config_paths=[context_path])
    context_identity, contract_identity = _identity(context_path), _identity(contract_path)
    e3._require_scene_id(context['scene_id'])
    seal = checked_identity(context['mask_seal'])
    if seal.name != 'seal.json':
        raise ValueError('mask seal name differs')
    bundle = validate_public_masks(seal.parent)
    if (bundle['context']['scene_id'] != context['scene_id']
            or bundle['seal_identity'] != context['mask_seal']):
        raise ValueError('mask and erasure scene differ')
    checkpoint = checked_identity(context['checkpoint'])
    if public_runtime() != context['runtime']:
        raise ValueError('erasure runtime differs from frozen package bytes')
    destination = e3.checked_repo_path(root / 'fidelity/erasure' / context['scene_id'],
                                       'erasure output', must_exist=False)
    claim = destination.with_name(destination.name + '.claim.json')
    failure_path = destination.with_name(destination.name + '.failure.json')
    partial_path = destination.with_name(destination.name + '.failed_partial')
    if any(p.exists() or p.is_symlink() for p in (destination, claim, failure_path, partial_path)):
        raise FileExistsError('immutable erasure output/attempt already exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    materializer._write_inside(claim, materializer._json_bytes({
        'context_sha256': e3.sha256_file(context_path), 'pid': os.getpid()}))
    image_root = Path(next(iter(bundle['train_images'].values()))['path']).parent
    from run.icra2027.e3_auto_discovery_pilot import enforce_read_boundary
    def guard(event, arguments):
        if event == 'socket.connect':
            raise ValueError('strict erasure forbids downloads/network access')
        enforce_read_boundary(event, arguments, forbidden_scene=image_root.parent.parent,
            image_root=image_root, allowed_images=set(bundle['train_images']))
    sys.addaudithook(guard)
    result = None
    try:
        with e3._atomic_directory(destination) as staging:
            try:
                # Explicit checkpoint/device overrides ambient legacy fallback settings.
                os.environ['LAMA_MODEL'] = str(checkpoint)
                torch.set_num_threads(4)
                torch.manual_seed(0)
                states = {r['object_slot']: r for r in bundle['result']['objects']}
                applicable = any(r['status'] == 'MASK_READY' and
                    states[r['object_slot']].get('plane_status') != 'NO_PLANE' for r in bundle['rows'])
                model = load_lama() if applicable else None
                result = erase_public_views(bundle, staging / 'views', model)
                for row in result['rows']:
                    if row['output_identity']:
                        relative = Path(row['output_identity']['path']).relative_to(staging)
                        row['output_identity']['path'] = str(destination / relative)
                result.update(status='FAILED' if result['enhancer_failed_views'] else 'COMPLETE',
                    source_commit=json.loads(Path(contract_path).read_text())['code']['commit'],
                    context_path=str(context_path), context_sha256=e3.sha256_file(context_path),
                    context_identity=context_identity, contract_identity=contract_identity,
                    mask_seal=context['mask_seal'], checkpoint=context['checkpoint'],
                    contract_sha256=json.loads(Path(contract_path).read_text())['contract_sha256'],
                    freeze_id=context['freeze_id'], scene_id=context['scene_id'])
                checked_identity(context['checkpoint'])
                checked_identity(context['mask_seal'])
                for ref in bundle['train_images'].values():
                    checked_identity(ref)
                for row in bundle['rows']:
                    if row['mask_identity']:
                        checked_identity(row['mask_identity'])
                checked_identity(context_identity)
                checked_identity(contract_identity)
                if public_runtime() != context['runtime']:
                    raise ValueError('erasure runtime changed during inference')
                if not any((staging / 'views').iterdir()):
                    (staging / 'views').rmdir()
                materializer._write_inside(staging / 'erasure.json', materializer._json_bytes(result))
                members = {str(p.relative_to(staging)): e3.sha256_file(p)
                           for p in staging.rglob('*') if p.is_file()}
                materializer._write_inside(staging / 'seal.json', materializer._json_bytes({'schema_version': 1, 'members': members}))
            except Exception:
                staging.rename(partial_path)
                raise
        return result
    except Exception as error:
        if result and partial_path.exists():
            for row in result['rows']:
                if row['output_identity']:
                    path = Path(row['output_identity']['path'])
                    if path.is_relative_to(destination):
                        row['output_identity']['path'] = str(partial_path / path.relative_to(destination))
        materializer._write_inside(failure_path,
            materializer._json_bytes({'status': 'FAILED', 'error_type': type(error).__name__,
                'error': str(error), 'paper_ready': False, 'backend': 'lama_cpu',
                'fallback_attempted': False, 'context_identity': context_identity,
                'contract_identity': contract_identity, 'planned_objects': len(bundle['objects']),
                'objects': bundle['result']['objects'], 'planned_views': len(bundle['rows']),
                'planned_rows': bundle['rows'], 'completed_rows': result['rows'] if result else [],
                'partial_output': str(partial_path)}))
        raise


def validate_public_erasure(directory):
    """Authenticate a completed erasure handoff without reopening inference."""
    from agents.edit.inpaint_masks import _sealed, _checked, _bound_contract, validate_public_masks
    from robo.eval import agentic_ablation as e3
    directory, seal = _sealed(directory)
    result = json.loads((directory / 'erasure.json').read_text())
    context_path = _checked(result['context_identity'])
    contract_path = _checked(result['contract_identity'])
    context, contract = _bound_contract(context_path, contract_path, producer_commit=result['source_commit'])
    if (result['status'] != 'COMPLETE' or result['scope'] != 'automatic_train_only_erasure'
            or result['schema_version'] != 1 or context['scope'] != result['scope']
            or context['backend'] != 'lama_cpu' or context['algorithm'] != PUBLIC_ALGORITHM
            or result['algorithm'] != PUBLIC_ALGORITHM or result['paper_ready'] is not False
            or result['cleaned_background_created'] is not False
            or result['contract_sha256'] != contract['contract_sha256']
            or result['context_path'] != str(context_path)
            or result['context_sha256'] != result['context_identity']['sha256']
            or result['freeze_id'] != context['freeze_id'] or result['scene_id'] != context['scene_id']
            or result['checkpoint'] != context['checkpoint'] or result['mask_seal'] != context['mask_seal']
            or directory != e3.REPOSITORY_ROOT / 'outputs/icra2027' / context['freeze_id'] / 'fidelity/erasure' / context['scene_id']):
        raise ValueError('erasure source/treatment/terminal status differs')
    source = _checked(context['mask_seal'])
    masks = validate_public_masks(source.parent)
    if masks['seal_identity'] != context['mask_seal'] or masks['context']['scene_id'] != context['scene_id']:
        raise ValueError('erasure mask source differs')
    expected = {(r['object_slot'], r['view_index']): r for r in masks['rows']}
    rows = result['rows']
    if (result['planned_objects'] != len(masks['objects']) or result['objects'] != masks['result']['objects']
            or result['planned_views'] != len(expected) or len(rows) != len(expected)
            or {(r['object_slot'], r['view_index']) for r in rows} != set(expected)):
        raise ValueError('erasure dropped or duplicated a planned object/view')
    states = {r['object_slot']: r for r in result['objects']}
    members = {'erasure.json'}
    for row in rows:
        original = expected[(row['object_slot'], row['view_index'])]
        if any(row[k] != v for k, v in original.items()):
            raise ValueError('erasure changed an upstream mask/view field')
        if states[row['object_slot']].get('plane_status') == 'NO_PLANE':
            status = 'NOT_APPLICABLE_NO_PLANE'
        elif original['status'] == 'EMPTY_PROJECTED_MASK':
            status = 'EMPTY_MASK'
        else:
            status = 'ERASED'
        if row['erasure_status'] != status or row['failure'] is not None:
            raise ValueError('incomplete/failed erasure cannot become a fill target')
        if status != 'ERASED':
            if any(row[k] is not None for k in ('backend', 'output_identity', 'outside_mask_equal')):
                raise ValueError('non-invoked erasure has an enhancement artifact')
            continue
        path = _checked(row['output_identity'])
        if path != directory / 'views' / row['object_slot'] / f"inpainted_{row['view_index']}.png":
            raise ValueError('erasure artifact path differs')
        with Image.open(_checked(masks['train_images'][row['frame']])) as image:
            full = np.asarray(image.convert('RGB'))
        with Image.open(_checked(row['mask_identity'])) as image:
            mask = np.asarray(image) > 127
        with Image.open(path) as image:
            if image.mode != 'RGB': raise ValueError('erasure output is not RGB')
            actual = np.asarray(image)
        if (actual.shape != full.shape or not np.array_equal(actual[~mask], full[~mask])
                or row['outside_mask_equal'] is not True or row['backend'] != 'lama_cpu'):
            raise ValueError('erasure changed source pixels outside the mask or backend')
        members.add(str(path.relative_to(directory)))
    for field, status in [('erased_views', 'ERASED'), ('empty_mask_views', 'EMPTY_MASK'),
                          ('enhancer_failed_views', 'ENHANCER_FAILED'), ('no_plane_views', 'NOT_APPLICABLE_NO_PLANE')]:
        if result[field] != sum(r['erasure_status'] == status for r in rows):
            raise ValueError('erasure denominator counts differ')
    if set(json.loads((directory / 'seal.json').read_text())['members']) != members:
        raise ValueError('erasure contains undeclared output products')
    return {'directory': str(directory), 'seal_identity': seal, 'context': context,
            'context_identity': result['context_identity'], 'contract_identity': result['contract_identity'],
            'result': result, 'mask_bundle': masks}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-context')
    parser.add_argument('--contract-manifest')
    parser.add_argument('--runtime', action='store_true')
    args = parser.parse_args(argv)
    if args.runtime:
        print('E2_ERASURE_RUNTIME_JSON=' + json.dumps(public_runtime())); return
    if bool(args.public_context) != bool(args.contract_manifest):
        parser.error('public context and E0 contract required together')
    if args.public_context:
        result = run_public(args.public_context, args.contract_manifest)
        print(json.dumps(result, indent=2))
        if result['status'] == 'FAILED': raise SystemExit(1)
        return
    backend, pipe, lama = "lama", None, None
    if qwen_feasible():
        try:
            pipe = load_qwen()
            backend = "qwen"
        except Exception:
            print("[iq] Qwen load FAILED, falling back to LaMa:")
            traceback.print_exc()
            free_qwen(pipe)
            pipe = None
            if C.env("REQUIRE_QWEN") == "1":
                raise
    else:
        print("[iq] Qwen infeasible on this GPU/torch; using LaMa")
    if backend == "lama":
        lama = load_lama()

    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    shard = C.env("SHARD")  # "i/N" -> objects where pos%N==i
    if shard:
        i, n = map(int, shard.split("/"))
        objects = [m for k, m in enumerate(objects) if k % n == i]
        print(f"[iq] shard {shard}: {len(objects)} objects")
    meta_all = {}
    for m in objects:
        odir = C.OUT / "inpaint" / f"obj_{m['index']:02d}"
        # no plane -> nothing will be filled; skip the expensive erase
        if not (odir / "views.json").exists() or not (odir / "plane.json").exists():
            continue
        views = json.loads((odir / "views.json").read_text())
        for k, view in enumerate(views):
            opath = odir / f"inpainted_{k}.png"
            if opath.exists():
                print(f"[iq] obj_{m['index']:02d} view{k}: cached")
                continue
            mpath = odir / f"mask_{k}.png"
            if not mpath.exists():
                continue
            full = np.asarray(Image.open(
                C.IMAGES_DIR / view["frame"]).convert("RGB")).copy()
            mask = np.asarray(Image.open(mpath)) > 127
            vv, uu = np.nonzero(mask)
            if len(vv) == 0:
                continue
            H, W = mask.shape
            pad = int(CROP_PAD * max(np.ptp(vv), np.ptp(uu)) + 16)
            v0, v1 = max(vv.min() - pad, 0), min(vv.max() + pad, H)
            u0, u1 = max(uu.min() - pad, 0), min(uu.max() + pad, W)
            img_c = Image.fromarray(full[v0:v1, u0:u1])
            msk_c = mask[v0:v1, u0:u1]
            sc = min(CROP_MAX / max(img_c.size), 1.0)
            size = (max(int(img_c.width * sc) // 8 * 8, 64),
                    max(int(img_c.height * sc) // 8 * 8, 64))
            img_r = img_c.resize(size)
            msk_r = Image.fromarray((msk_c * 255).astype(np.uint8)).resize(size)

            used = backend
            if backend == "qwen":
                try:
                    res = pipe(
                        image=img_r,
                        prompt=(f"remove the {m['label']} completely from the "
                                "scene; show the empty flat surface behind "
                                "it, seamlessly continuing the table top and "
                                "background, photorealistic, same lighting"),
                        negative_prompt=" ",
                        num_inference_steps=STEPS,
                        true_cfg_scale=4.0).images[0]
                except Exception:
                    print(f"[iq] qwen FAILED on obj_{m['index']:02d} view{k}, "
                          "switching to LaMa for the rest:")
                    traceback.print_exc()
                    free_qwen(pipe)
                    pipe, backend = None, "lama"
                    if lama is None:
                        lama = load_lama()
                    used, res = "lama", lama(img_r, msk_r.convert("L"))
            else:
                res = lama(img_r, msk_r.convert("L"))

            full = paste(full, (u0, v0, u1, v1), res, msk_c)
            Image.fromarray(full).save(opath)
            meta_all[f"obj_{m['index']:02d}/{k}"] = used
            print(f"[iq] obj_{m['index']:02d} view{k} "
                  f"{view['frame']}: inpainted ({used})")
    suffix = f"_{shard.replace('/', 'of')}" if shard else ""
    C.save_json(C.OUT / "inpaint" / f"inpaint_meta{suffix}.json", meta_all)


if __name__ == "__main__":
    main()
