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


def main(argv=None):
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
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
