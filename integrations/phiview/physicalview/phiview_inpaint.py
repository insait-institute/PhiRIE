"""Prompted edit adapter using the available Qwen 2511 checkpoint."""
import os
from pathlib import Path


def load_qwen():
    from agents.edit import inpaint_qwen
    import json
    import torch
    import diffusers
    from diffusers import QwenImageEditPlusPipeline
    root = os.environ.get('PHIVIEW_QWEN_MODEL', 'Qwen/Qwen-Image-Edit-2511')
    revision = os.environ.get('PHIVIEW_QWEN_REVISION', '6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9')
    placement = os.environ.get('PHIVIEW_QWEN_PLACEMENT', 'cpu_offload')
    if placement not in ('cpu_offload', 'cuda'):
        raise ValueError('Unknown Qwen placement')
    options = {}
    if not Path(root).is_dir():
        options['revision'] = revision
    if placement == 'cuda':
        free, total = torch.cuda.mem_get_info()
        if free < 70 * 2**30:
            raise RuntimeError('Direct Qwen placement requires at least 70 GiB free GPU memory')
        options['device_map'] = 'cuda'
    pipe = QwenImageEditPlusPipeline.from_pretrained(root, torch_dtype=torch.bfloat16,
                                                    local_files_only=True, **options)
    if not getattr(pipe.transformer.config, 'zero_cond_t', False):
        raise RuntimeError('Qwen 2511 requires a Diffusers build with zero_cond_t support')
    if placement == 'cpu_offload':
        pipe.enable_model_cpu_offload()
    (inpaint_qwen.C.OUT/'inpaint'/'model-receipt.json').write_text(json.dumps({
        'model': str(root), 'revision': revision, 'diffusers': diffusers.__version__, 'zero_cond_t': True,
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'placement': placement}, indent=2))
    print(f'[PhiView] Prompt model: {root}', flush=True)
    return pipe


def main():
    from agents.edit import inpaint_qwen
    inpaint_qwen.load_qwen = load_qwen
    inpaint_qwen.main()


if __name__ == '__main__':
    main()
