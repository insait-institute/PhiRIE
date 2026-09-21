"""Inpainting stage 2 (sam3 env, GPU): image-evidence masks for removal.

The projected GT mask misses parts the scan never captured (transparent
bottles, thin structures) - inpainting with an incomplete mask leaves object
ghosts. Per related view: SAM3 with the class prompt, take instances with
IoU > MIN_IOU against the projected mask, and store union(projected, SAM3)
dilated - the final 2D removal mask for the Qwen inpainting stage.

Usage: inpaint_masks.py --images-dir D --out-dir SIMANY_OUT
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image, ImageFilter

from agents.core import common as _common
PROMPT_MAP = {
    "plastic bottle": "bottle", "glass bottle": "bottle",
    "water bottle": "bottle", "cup": "mug", "mouse": "computer mouse",
    "carboard box": "cardboard box", "storage box": "box",
}
MIN_IOU = 0.15
DILATE = 9



PUBLIC_ALGORITHM = {'min_iou': MIN_IOU, 'dilate': DILATE, 'confidence': 0.4, 'resolution': 1008}


def _merge_mask(projected, candidates):
    projected = np.asarray(projected)
    candidates = np.asarray(candidates)
    if (projected.ndim != 2 or projected.dtype != bool or candidates.ndim != 3
            or candidates.shape[1:] != projected.shape or not np.isfinite(candidates).all()
            or not np.isin(candidates, [0, 1]).all()):
        raise ValueError('SAM3/projected masks have invalid shape or values')
    final = projected.copy()
    selected = 0
    for mask in candidates.astype(bool):
        inter = np.logical_and(mask, projected).sum()
        if inter / max(np.logical_or(mask, projected).sum(), 1) > MIN_IOU:
            final |= mask
            selected += 1
    final = np.asarray(Image.fromarray((final * 255).astype(np.uint8))
        .filter(ImageFilter.MaxFilter(DILATE))) > 127
    return final, selected


def _identity(path):
    from run.icra2027.e3_auto_discovery_pilot import identity
    value = identity(path)
    return {key: value[key] for key in ('path', 'bytes', 'sha256')}


def _checked(record):
    from run.icra2027.e3_fresh_generation_contract import checked_identity
    return checked_identity(record)


def _sealed(directory):
    """Verify every exact member before a consumer reads source products."""
    from robo.eval import agentic_ablation as e3
    directory = e3.checked_repo_path(directory, 'public mask stage', kind='dir')
    seal_path = e3.checked_repo_path(directory/'seal.json', 'public mask seal', kind='file')
    seal = json.loads(seal_path.read_text())
    if set(seal) != {'schema_version', 'members'} or seal['schema_version'] != 1:
        raise ValueError('public mask/preparation seal schema differs')
    children = list(directory.rglob('*'))
    if any(p.is_symlink() for p in children):
        raise ValueError('public sealed output contains a symlink')
    actual = {str(p.relative_to(directory)) for p in children if p.is_file() and p != seal_path}
    expected_dirs = {str(parent) for name in seal['members'] for parent in Path(name).parents if str(parent) != '.'}
    if {str(p.relative_to(directory)) for p in children if p.is_dir()} != expected_dirs:
        raise ValueError('public sealed directory roster differs')
    if actual != set(seal['members']):
        raise ValueError('public mask/preparation file roster differs')
    for name, digest in seal['members'].items():
        path = e3.checked_repo_path(directory/name, 'public sealed member', kind='file')
        if not path.is_relative_to(directory) or e3.sha256_file(path) != digest:
            raise ValueError('public mask/preparation member changed')
    return directory, _identity(seal_path)


def _bound_contract(context_path, contract_path, *, producer_commit=None):
    from robo.manifest.hash import canonical_hash
    context = json.loads(context_path.read_text()) if context_path.suffix == '.json' else __import__('yaml').safe_load(context_path.read_text())
    contract = json.loads(contract_path.read_text())
    payload = {k: v for k, v in contract.items() if k not in {'created_utc', 'environment', 'contract_sha256'}}
    resources = [r for r in contract['resource_inventory'] if r.get('resolved_path') == str(context_path)
                 and r.get('hash_method') == 'content_sha256']
    if (canonical_hash(payload) != contract['contract_sha256'] or contract['code']['dirty'] is not False
            or contract['freeze_id'] != context['freeze_id']
            or producer_commit is not None and contract['code']['commit'] != producer_commit
            or len(resources) != 1 or resources[0]['sha256'] != _identity(context_path)['sha256']):
        raise ValueError('public source E0/context/commit binding differs')
    return context, contract


def _preparation(spec, scene_id):
    """Consume immutable preparation without reopening its no-overwrite runner."""
    from agents.edit import inpaint_prepare as prep
    from robo.eval import agentic_ablation as e3
    directory, seal = _sealed(spec['directory'])
    if seal != {key: spec['seal'][key] for key in seal}:
        raise ValueError('public preparation seal differs from frozen context')
    context_path, contract_path = _checked(spec['context']), _checked(spec['contract'])
    context, _ = _bound_contract(context_path, contract_path, producer_commit=spec['producer_commit'])
    if (context['scope'] != 'automatic_train_only_removal_preparation' or context['scene_id'] != scene_id
            or context['policy_id'] != 'A4' or context['seed'] != 0 or context['algorithm'] != prep.PUBLIC_ALGORITHM
            or directory != e3.REPOSITORY_ROOT/'outputs/icra2027'/context['freeze_id']/'fidelity/removal'/scene_id/'inpaint'):
        raise ValueError('public preparation context scope/output differs')
    result = json.loads((directory/'public_prepare.json').read_text())
    manifest_path = _checked(context['materialization_manifest'])
    factory = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    validation = prep._validate_source_factory(factory, manifest, scene_id)
    if (result['context_sha256'] != _identity(context_path)['sha256']
            or result['materialization_sha256'] != _identity(manifest_path)['sha256']
            or result['source_factory'] != str(factory) or result['source_validation'] != validation
            or result['paper_ready'] is not False or result['cleaned_background_created'] is not False
            or result['official_test_images_read'] != 0 or result['seed'] != 0
            or result['algorithm'] != prep.PUBLIC_ALGORITHM):
        raise ValueError('public preparation result identity or semantics differ')
    generation = __import__('yaml').safe_load(_checked(context['generation_config']).read_text())
    source = Path(generation['source_pilot'])
    boundary_path = source/'input_manifest.json'
    if _identity(boundary_path)['sha256'] != generation['source_discovery_hashes']['input_manifest.json']:
        raise ValueError('original TRAIN boundary changed')
    boundary = json.loads(boundary_path.read_text())
    training = result['training_frames']
    split = json.loads(_checked(boundary['metadata']['train_test_lists.json']).read_text())
    if (training != boundary['boundary']['training_frames'] or len(training) != len(set(training))
            or training != result['gaussian_provenance']['training_frames']
            or result['gaussian_provenance']['status'] != 'FRESH_OFFICIAL_TRAIN_ONLY'
            or not set(training) <= set(split['train']) or set(training) & set(split['test'])
            or set(training) != set(boundary['input_images'])):
        raise ValueError('public mask input is not the authenticated TRAIN roster')
    images = boundary['input_images']
    objects = json.loads((factory/'objects/objects.json').read_text())
    ids = [r['automatic_instance_id'] for r in objects]
    statuses = result['objects']
    if (len(set(ids)) != len(ids) or result['planned_objects'] != len(ids)
            or [r['automatic_instance_id'] for r in statuses] != ids
            or any('gt_object_id' in r or r['index'] != r['automatic_instance_id'] or r['instance_namespace'] != 'automatic' for r in objects)):
        raise ValueError('public mask planned automatic population differs')
    jobs = []
    for obj, status in zip(objects, statuses):
        slot = f"obj_{obj['automatic_instance_id']:02d}"
        if status['object_slot'] != slot:
            raise ValueError('public preparation object slot differs')
        aligned = json.loads((factory/'objects'/slot/'aligned.json').read_text())
        if aligned['terminal_action'] != status['terminal_action']:
            raise ValueError('public preparation changed a constructor decision')
        if status['terminal_action'] != 'accept':
            if status['terminal_action'] not in {'reject', 'abstain'} or (directory/slot).exists():
                raise ValueError('nonaccepted public job has preparation artifacts')
            continue
        if aligned.get('rejected'):
            raise ValueError('public preparation changed an accepted constructor decision')
        projected = np.load(directory/slot/'proj_masks.npz', allow_pickle=False)
        frames = projected['frames'].tolist(); masks = projected['masks']
        views = json.loads((directory/slot/'views.json').read_text())
        if (set(projected.files) != {'frames', 'masks'} or masks.dtype != bool or masks.ndim != 3
                or len(masks) != len(frames) or len(frames) != len(set(frames))
                or frames != status['selected_frames'] or frames != [v['frame'] for v in views]
                or any(f not in training for f in frames)
                or [int(m.sum()) for m in masks] != status['projected_mask_pixels']):
            raise ValueError('public projected mask/view roster differs')
        for index, frame in enumerate(frames):
            image = _checked(images[frame])
            if image.name != frame:
                raise ValueError('public TRAIN RGB filename differs')
            with Image.open(image) as rgb:
                if rgb.size != tuple(reversed(masks[index].shape)):
                    raise ValueError('public TRAIN RGB/projected mask resolution differs')
            jobs.append(dict(object_slot=slot, automatic_instance_id=obj['automatic_instance_id'],
                view_index=index, frame=frame, projected_pixels=int(masks[index].sum()),
                projected=masks[index].copy(), label=obj['label']))
    return dict(directory=str(directory), report=result, seal_identity=seal, context_identity=_identity(context_path),
        contract_identity=_identity(contract_path), source_factory=str(factory), objects=objects,
        train_images=images, jobs=jobs)


def _mask_context(context_path, contract_path, *, executing):
    from robo.eval import agentic_ablation as e3
    context_path, contract_path = Path(context_path).absolute(), Path(contract_path).absolute()
    context, contract = _bound_contract(context_path, contract_path)
    if (set(context) != {'schema_version', 'scope', 'freeze_id', 'scene_id', 'preparation', 'sam3_source',
                         'sam3_checkpoint', 'python', 'runtime', 'seed', 'algorithm'}
            or context['schema_version'] != 1 or context['scope'] != 'automatic_train_only_removal_masks'
            or type(context['seed']) is not int or context['seed'] != 0 or context['algorithm'] != PUBLIC_ALGORITHM):
        raise ValueError('public mask algorithm/context differs')
    e3._require_scene_id(context['scene_id'])
    output = e3.REPOSITORY_ROOT/'outputs/icra2027'/context['freeze_id']/'fidelity/removal_masks'/context['scene_id']
    if executing:
        e3._validate_cli_execution(contract_path, context['freeze_id'], output, config_paths=[context_path])
    preparation = _preparation(context['preparation'], context['scene_id'])
    if Path(preparation['directory']) == output or output in Path(preparation['directory']).parents:
        raise ValueError('public masks must have a separate new output')
    return context, contract, output, preparation


def _validate_model(context):
    import sys
    from robo.eval.fidelity_replacements import _tree_inventory
    from run.icra2027.e3_trellis_generation_pilot import runtime_identity, targeted_runtime_identity
    if str(Path(sys.executable).absolute()) != context['python']:
        raise ValueError('public SAM3 interpreter differs')
    _checked(context['sam3_checkpoint'])
    source = Path(context['sam3_source']['path'])
    if _tree_inventory(source, label='public SAM3 source')['tree_sha256'] != context['sam3_source']['tree_sha256']:
        raise ValueError('public SAM3 source tree changed')
    observed = {'packages_sha256': runtime_identity(context['python'])[1],
                'targeted_bytes': targeted_runtime_identity(context['python'])}
    if observed != context['runtime']:
        raise ValueError('public SAM3 runtime differs')


def _public_processor(context):
    import sys
    import sam3
    if not Path(sam3.__file__).resolve().is_relative_to(Path(context['sam3_source']['path']).resolve()):
        raise ValueError('imported SAM3 source differs from pinned source')
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(device='cuda', checkpoint_path=context['sam3_checkpoint']['path'],
                                  load_from_HF=False, compile=False)
    model.eval()
    return Sam3Processor(model, resolution=1008, device='cuda', confidence_threshold=0.4)


def _run_jobs(context, preparation, output, rows):
    active = [j for j in preparation['jobs'] if j['projected_pixels'] > 0]
    for job in preparation['jobs']:
        if not job['projected_pixels']:
            rows.append({k: job[k] for k in ('object_slot', 'automatic_instance_id', 'view_index', 'frame', 'projected_pixels')}
                        | dict(status='EMPTY_PROJECTED_MASK', mask_identity=None, final_pixels=0))
    if not active:
        return 0
    proc = _public_processor(context)
    by_frame = {}
    for job in active:
        by_frame.setdefault(job['frame'], []).append(job)
    for frame, items in by_frame.items():
        with Image.open(_checked(preparation['train_images'][frame])) as source:
            image = source.convert('RGB')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            state = proc.set_image(image)
            cache = {}
            for job in items:
                prompt = PROMPT_MAP.get(job['label'].strip().lower(), job['label'].strip().lower())
                if not prompt:
                    raise ValueError('public automatic label cannot yield an empty SAM3 prompt')
                if prompt not in cache:
                    proc.reset_all_prompts(state)
                    state = proc.set_text_prompt(prompt=prompt, state=state)
                    cache[prompt] = state['masks'].squeeze(1).cpu().numpy().copy()
                final, selected = _merge_mask(job['projected'], cache[prompt])
                path = output/job['object_slot']/f"mask_{job['view_index']}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((final*255).astype(np.uint8)).save(path)
                rows.append({k: job[k] for k in ('object_slot', 'automatic_instance_id', 'view_index', 'frame', 'projected_pixels')}
                    | dict(status='MASK_READY', mask_identity=_identity(path), final_pixels=int(final.sum()),
                           sam_candidates=len(cache[prompt]), selected_sam_candidates=selected))
    return 1


def run_public(context_path, contract_path):
    import sys
    from robo.eval import agentic_ablation as e3
    from robo.eval import e3_factory_materializer as materializer
    from run.icra2027.e3_auto_discovery_pilot import enforce_read_boundary
    context, contract, output, preparation = _mask_context(context_path, contract_path, executing=True)
    failure = output.with_name(output.name+'_failure.json')
    partial = output.with_name(output.name+'_failed_partial')
    claim = output.with_name(output.name+'_claim.json')
    if any(p.exists() or p.is_symlink() for p in (output, failure, partial, claim)):
        raise FileExistsError('public mask output/claim already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    materializer._write_inside(claim, materializer._json_bytes({'context': _identity(context_path)}))
    sys.dont_write_bytecode = True
    sys.path.insert(0, context['sam3_source']['path'])
    os.environ.update(HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    image_root = Path(next(iter(preparation['train_images'].values()))['path']).parent
    def guard(event, arguments):
        if event == 'socket.connect':
            raise ValueError('public masks forbid network access')
        enforce_read_boundary(event, arguments, forbidden_scene=image_root.parent.parent,
            image_root=image_root, allowed_images=set(preparation['train_images']))
    sys.addaudithook(guard)
    torch.manual_seed(0); np.random.seed(0)
    rows = []
    try:
        _validate_model(context)
        with e3._atomic_directory(output) as staging:
            try:
                loads = _run_jobs(context, preparation, staging, rows)
                # Re-authenticate bytes after inference; a mid-run drift cannot be sealed.
                _validate_model(context)
                final_context, final_contract, final_output, final_preparation = _mask_context(
                    context_path, contract_path, executing=True)
                if (final_context != context or final_contract != contract or final_output != output
                        or {k:v for k,v in final_preparation.items() if k != 'jobs'}
                        != {k:v for k,v in preparation.items() if k != 'jobs'}
                        or len(final_preparation['jobs']) != len(preparation['jobs'])
                        or any({k:v for k,v in a.items() if k != 'projected'}
                               != {k:v for k,v in b.items() if k != 'projected'}
                               or not np.array_equal(a['projected'], b['projected'])
                               for a,b in zip(final_preparation['jobs'], preparation['jobs']))):
                    raise ValueError('public mask inputs changed during inference')
                # Paths describe final immutable members, never ephemeral staging names.
                for row in rows:
                    if row['mask_identity'] is not None:
                        row['mask_identity']['path'] = str(output/Path(row['mask_identity']['path']).relative_to(staging))
                rows.sort(key=lambda r: (r['automatic_instance_id'], r['view_index']))
                result = dict(schema_version=1, scope='automatic_train_only_removal_masks',
                    status='COMPLETE' if loads else 'NO_APPLICABLE_VIEWS', context_identity=_identity(context_path),
                    contract_identity=_identity(contract_path), producer_commit=contract['code']['commit'],
                    planned_objects=len(preparation['objects']), planned_views=len(preparation['jobs']),
                    objects=preparation['report']['objects'], rows=rows, model_loads=loads,
                    seed=0, algorithm=PUBLIC_ALGORITHM, paper_ready=False, cleaned_background_created=False,
                    official_test_images_read=0)
                materializer._write_inside(staging/'public_masks.json', materializer._json_bytes(result))
                members = {str(p.relative_to(staging)): e3.sha256_file(p) for p in staging.rglob('*') if p.is_file()}
                materializer._write_inside(staging/'seal.json', materializer._json_bytes({'schema_version':1,'members':members}))
            except Exception:
                for row in rows:
                    if row['mask_identity'] is not None:
                        old_path = Path(row['mask_identity']['path'])
                        base = staging if old_path.is_relative_to(staging) else output
                        row['mask_identity']['path'] = str(partial/old_path.relative_to(base))
                staging.rename(partial)
                raise
        return result
    except Exception as error:
        planned = [{k:v for k,v in j.items() if k != 'projected'} for j in preparation['jobs']]
        materializer._write_inside(failure, materializer._json_bytes(dict(status='FAILED',
            classification='mask_stage_failure',error_type=type(error).__name__,error=str(error),
            context_identity=_identity(context_path),planned_rows=planned,completed_rows=rows,
            partial_output=str(partial),paper_ready=False)))
        raise


def validate_public_masks(directory):
    """Read-only authenticated mask handoff for a separately frozen erase stage."""
    directory, seal = _sealed(directory)
    result = json.loads((directory/'public_masks.json').read_text())
    context_path, contract_path = _checked(result['context_identity']), _checked(result['contract_identity'])
    context, contract, expected, preparation = _mask_context(context_path, contract_path, executing=False)
    if (directory != expected or result['producer_commit'] != contract['code']['commit']
            or result['scope'] != 'automatic_train_only_removal_masks' or result['schema_version'] != 1
            or result['planned_objects'] != len(preparation['objects'])
            or result['objects'] != preparation['report']['objects'] or result['planned_views'] != len(preparation['jobs'])
            or result['paper_ready'] is not False or result['cleaned_background_created'] is not False
            or result['official_test_images_read'] != 0 or result['seed'] != 0 or result['algorithm'] != PUBLIC_ALGORITHM):
        raise ValueError('public mask output population/source differs')
    expected_rows = {(j['object_slot'], j['view_index']): j for j in preparation['jobs']}
    rows = result['rows']
    if len(rows) != len(expected_rows) or {(r['object_slot'],r['view_index']) for r in rows} != set(expected_rows):
        raise ValueError('public mask result dropped or duplicated a planned view')
    for row in rows:
        job = expected_rows[(row['object_slot'],row['view_index'])]
        if any(row[k] != job[k] for k in ('automatic_instance_id','frame','projected_pixels')):
            raise ValueError('public mask result changed a planned view identity')
        if job['projected_pixels'] == 0:
            if row['status'] != 'EMPTY_PROJECTED_MASK' or row['mask_identity'] is not None or row['final_pixels'] != 0:
                raise ValueError('empty projection was silently replaced')
        else:
            path = _checked(row['mask_identity'])
            if row['status'] != 'MASK_READY' or path != directory/row['object_slot']/f"mask_{row['view_index']}.png":
                raise ValueError('public mask member destination differs')
            with Image.open(path) as image:
                mask = np.asarray(image)
            if (mask.shape != job['projected'].shape or mask.dtype != np.uint8 or not np.isin(mask,[0,255]).all()
                    or int((mask>0).sum()) != row['final_pixels'] or np.any(job['projected'] & (mask==0))):
                raise ValueError('public output mask dropped projected pixels or changed encoding')
    expected_members = {'public_masks.json'} | {str(Path(r['mask_identity']['path']).relative_to(directory))
        for r in rows if r['mask_identity'] is not None}
    if set(json.loads((directory/'seal.json').read_text())['members']) != expected_members:
        raise ValueError('public mask output contains undeclared products')
    active = any(j['projected_pixels'] for j in preparation['jobs'])
    if result['model_loads'] != int(active) or result['status'] != ('COMPLETE' if active else 'NO_APPLICABLE_VIEWS'):
        raise ValueError('public mask model invocation/terminal status differs')
    return dict(directory=str(directory),seal_identity=seal,context=context,context_identity=_identity(context_path),
        contract_identity=_identity(contract_path),preparation={k:v for k,v in preparation.items() if k!='jobs'},
        source_factory=preparation['source_factory'],objects=preparation['objects'],train_images=preparation['train_images'],
        rows=rows,result=result)

def _legacy_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(device="cuda", checkpoint_path=_common.resolve_sam3_ckpt(),
                                   load_from_HF=False, compile=False)
    proc = Sam3Processor(model, resolution=1008, device="cuda",
                         confidence_threshold=0.4)

    objects = json.loads((out / "objects" / "objects.json").read_text())
    jobs = []  # (frame, obj_meta, view_index)
    for m in objects:
        odir = out / "inpaint" / f"obj_{m['index']:02d}"
        if not (odir / "proj_masks.npz").exists():
            continue
        pmz = np.load(odir / "proj_masks.npz")
        for k, frame in enumerate(pmz["frames"]):
            jobs.append((str(frame), m, k))

    by_frame = {}
    for frame, m, k in jobs:
        by_frame.setdefault(frame, []).append((m, k))

    for frame, items in by_frame.items():
        image = Image.open(Path(args.images_dir) / frame).convert("RGB")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(image)
            cache = {}
            for m, k in items:
                odir = out / "inpaint" / f"obj_{m['index']:02d}"
                pmz = np.load(odir / "proj_masks.npz")
                proj = pmz["masks"][k]
                prompt = PROMPT_MAP.get(m["label"].strip().lower(),
                                        m["label"].strip().lower())
                if prompt not in cache:
                    proc.reset_all_prompts(state)
                    state = proc.set_text_prompt(prompt=prompt, state=state)
                    cache[prompt] = state["masks"].squeeze(1).cpu().numpy()
                final = _merge_mask(proj, cache[prompt])[0]
                fpath = odir / f"mask_{k}.png"
                Image.fromarray((final * 255).astype(np.uint8)).save(fpath)
                print(f"[im] obj_{m['index']:02d} view{k} {frame}: "
                      f"proj {int(proj.sum())}px -> final {int(final.sum())}px")
    print("[im] done")


def main(argv=None):
    import sys
    values = sys.argv[1:] if argv is None else argv
    if '--public-context' not in values and '--contract-manifest' not in values:
        return _legacy_main()
    parser = argparse.ArgumentParser(description='Strict TRAIN-only SAM3 removal masks')
    parser.add_argument('--public-context', required=True)
    parser.add_argument('--contract-manifest', required=True)
    args = parser.parse_args(values)
    return run_public(args.public_context, args.contract_manifest)


if __name__ == "__main__":
    main()
