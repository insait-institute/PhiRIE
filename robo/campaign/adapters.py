"""Dispatch scientific components; external/native integrations fail closed."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
from .core import checked_path, load, receipt, save


def execute(task, inputs, out, runtime):
    kind = task['kind']; config = task['params']; out = Path(out)
    if kind == 'rgb_video_reconstruction':
        from .video import run
        return run(config,inputs,out)
    if kind == 'asset_bridge':
        from .assets import bridge
        return bridge(config,inputs,out)
    if kind == 'generator':
        from .models import generate
        return generate(config, inputs, out)
    if kind == 'inpaint':
        from .models import inpaint
        return inpaint(config, inputs, out)
    if kind == 'text_query':
        from .semantics import text_query
        return text_query(config,inputs,out)
    if kind == 'chorus':
        from .semantics import encode
        return encode(config, inputs, out)
    if kind == 'semantic_instances':
        from .semantics import group
        return group(config, inputs, out)
    if kind == 'harmonizer_sequence':
        from .harmonizer import run
        return run(config,inputs,out)
    if kind == 'gaussian_color_cache':
        from .color_cache import run
        return run(config,inputs,out)
    if kind == 'state_visual':
        from .visual import run
        return run(config, inputs, out)
    if kind == 'simfoundry':
        from .models import source
        src = source(config)
        if config.get('scene_upload_allowed') is not True: raise PermissionError('scene-level API data-sharing approval absent')
        approval = load(config['approval_file'])
        if approval.get('approved') is not True or approval.get('data_upload_permitted') is not True or approval.get('provider_side_budget_cap_verified') is not True:
            raise PermissionError('SimFoundry cloud stages need approved data sharing and provider-side spend cap')
        scene = 'simany-'+task['id']; output = Path(src['root'])/'Data'/scene
        if output.exists(): raise FileExistsError('official output exists; bind it explicitly instead of rerunning')
        cmd = ['bash','scripts/pipeline/A_reconstruction/run.sh','--scene-name',scene,
               '--video-fpath',inputs['video']]
        cmd += list(config.get('extra_args', []))
        with (out/'official.log').open('x') as log:
            subprocess.run(cmd,cwd=src['root'],check=True,stdout=log,stderr=subprocess.STDOUT,
                           timeout=float(config.get('timeout_s',7200)))
        path = output/'s14_og/reconstructed_og_scene.json'
        if not path.is_file(): raise RuntimeError('official SimFoundry output missing')
        save(out/'official.json',{'method':'SimFoundry official reconstruction','source':src,
            'command':cmd,'scene_descriptor':receipt(path),'policy_comparison':'NOT_RUN',
            'acquisition':config['acquisition'],'manual_minutes':config.get('manual_minutes',0)})
        return {'scene':path,'receipt':out/'official.json'}
    if kind == 'polaris_bundle':
        # Official PolaRiS uses a scan/composition workflow, not a single automatic
        # video-to-scene command. Validate its actual supplied output and effort.
        from .models import source
        src = source(config); root = Path(config['bundle_root']).absolute()
        required = [root/'scene.usda',root/'initial_conditions.json']
        if not all(p.is_file() for p in required) or not (root/'assets').is_dir():
            raise FileNotFoundError('official PolaRiS composed scene/conditions/assets are required')
        if 'manual_minutes' not in config or not config.get('acquisition'):
            raise ValueError('PolaRiS human effort and extra scans must be declared')
        from pxr import Usd
        stage = Usd.Stage.Open(str(required[0]))
        if stage is None: raise ValueError('PolaRiS USD cannot be opened')
        # Import gate is structural only. It does not replace native policy or scorer checks.
        save(out/'official.json',{'method':'PolaRiS official composed bundle','source':src,
            'files':[receipt(p) for p in sorted(root.rglob('*')) if p.is_file()],
            'manual_minutes':config['manual_minutes'],'acquisition':config['acquisition'],
            'policy_comparison':'NOT_RUN','automated_constructor_claim':False})
        return {'scene':required[0],'conditions':required[1],'receipt':out/'official.json'}
    if kind == 'command':
        # Integration bridge to existing capture/build/native harness, not a new
        # robot-loop implementation. argv substitutions never pass through shell.
        for key in ('same_engine_gate','rubric_gate'):
            if config.get('purpose') == 'native_policy':
                checked_path(config[key])
                gate = load(config[key]['path'])
                if gate.get('passed') is not True: raise ValueError(f'{key} failed')
        substitutions = {**inputs,'out':str(out)}
        argv = [str(x).format_map(substitutions) for x in config['argv']]
        if not argv: raise ValueError('empty command')
        env = os.environ.copy()
        for k,v in config.get('env',{}).items():
            if any(s in k.upper() for s in ('TOKEN','SECRET','KEY')): raise ValueError('do not put credentials in task manifests')
            env[k] = str(v)
        with (out/'command.log').open('x') as log:
            subprocess.run(argv,cwd=config.get('cwd',runtime['source_root']),env=env,check=True,
                           stdout=log,stderr=subprocess.STDOUT,timeout=float(config.get('timeout_s',14400)))
        path = Path(str(config['output_manifest']).format_map(substitutions))
        produced = load(path)
        if not isinstance(produced,dict) or not produced.get('artifacts'):
            raise ValueError('integration must publish named artifact path/sha256 receipts')
        artifacts = {name:checked_path(item) for name,item in produced['artifacts'].items()}
        artifacts['integration_receipt'] = path
        return artifacts
    raise ValueError('unsupported kind')
