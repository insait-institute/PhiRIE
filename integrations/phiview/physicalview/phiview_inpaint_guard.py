"""Reject cached/no-op or fallback edits before presenting a prompted inpaint."""
import json
from pathlib import Path


def validate(root, objects, prompt):
    root = Path(root)
    meta = json.loads((root/'edit_meta.json').read_text())
    views = json.loads((root/'inpaint_meta.json').read_text())
    if meta.get('prompt') != prompt or meta.get('backend_final') != 'qwen':
        raise ValueError('The requested Qwen prompt was not executed successfully')
    for name in objects:
        rows = json.loads((root/name/'views.json').read_text())
        if not rows:
            raise ValueError(f'No inpainting views for {name}')
        for i in range(len(rows)):
            if views.get(f'{name}/{i}') != 'qwen':
                raise ValueError(f'{name} view {i} was cached, skipped or used a non-prompt backend')
            if not (root/name/f'inpainted_{i}.png').is_file():
                raise ValueError('Edited image is missing')
    return {'prompt': prompt, 'objects': objects, 'backend': 'qwen', 'verified': True}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--objects', required=True)
    ap.add_argument('--prompt', required=True)
    args = ap.parse_args()
    result = validate(args.root, args.objects.split(','), args.prompt)
    (Path(args.root)/'prompt-receipt.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
