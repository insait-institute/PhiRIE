#!/usr/bin/env python3
"""Concrete SR1 empty-root launcher for the SR2 SAM3/TRELLIS constructor.

Run this inside an existing Slurm allocation. Only declared source modules,
Python dependencies, checkpoints, driver libraries and allocated GPU device
nodes are mounted; the native simulator, its assets, vault and host home are
absent. CPU observed-surface fusion is an explicit method, never a GPU fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from robo.roundtrip.capture import constructor_command, slurm_device_minors


CODE_DIRECTORIES = ("agents/core", "agents/assets", "agents/orchestrator", "models")
CODE_FILES = ("agents/__init__.py", "robo/__init__.py", "robo/eval/__init__.py",
    "robo/eval/fidelity_replacements.py", "robo/sim/__init__.py", "robo/sim/s7_sim.py",
    "robo/roundtrip/build.py", "robo/roundtrip/capture.py", "robo/roundtrip/shared_candidates.py",
    "robo/roundtrip/asset_metadata.py", "robo/roundtrip/workspace.py", "robo/roundtrip/workspace_inventory.py", "robo/roundtrip/workspace_components.py",
    "run/icra2027/e3_trellis_generation_pilot.py", "run/icra2027/e3_auto_discovery_pilot.py",
    "run/icra2027/e3_generator_backend.py", "agents/discover/derive_mesh_from_splat.py")


def allocated_devices(environ):
    indices = slurm_device_minors(environ)
    if len(indices) != 1:
        raise ValueError("first-slice constructor requires exactly one allocated GPU")
    return [f"/dev/nvidia{next(iter(indices))}", "/dev/nvidiactl", "/dev/nvidia-uvm",
            *(["/dev/nvidia-uvm-tools"] if Path("/dev/nvidia-uvm-tools").exists() else [])]


def cpu_stage(config):
    if sum(k in config for k in ('shared_candidates','observed_surface','observed_workspace','workspace_inventory'))>1:raise ValueError('constructor stages are mutually exclusive')
    phase=config.get('shared_candidates',{}).get('phase','all')
    if phase not in ('all','rvg','select'):raise ValueError('unsupported shared candidate stage')
    return 'observed_surface' in config or 'observed_workspace' in config or phase=='select'


def runtime_mounts(code, config, config_path):
    """Explicit trusted runtime closure; never mount a repository ancestor."""
    code = Path(code).resolve(strict=True)
    mounts = {}
    for directory in CODE_DIRECTORIES:
        # Individual .py files exclude data, cache and native assets even if a
        # future producer accidentally adds an artifact under a code directory.
        for path in sorted((code / directory).rglob("*.py")):
            if path.is_symlink():
                raise ValueError("constructor source may not be a symlink")
            mounts[str(path)] = "/code/" + str(path.relative_to(code))
    for name in CODE_FILES:
        path = code / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"constructor source missing or symlink: {name}")
        mounts[str(path)] = "/code/" + name
    for python in {config["sam3_python"], config["trellis"]["python"]}:
        env_root = Path(python).absolute().parent.parent
        if not (env_root / "lib/python3.11/site-packages").is_dir():
            raise ValueError("expected explicit Python3.11 constructor environment")
        for native in ("robocasa", "robosuite", "omnigibson"):
            if (env_root / "lib/python3.11/site-packages" / native).exists():
                raise ValueError(f"constructor environment contains native reference API: {native}")
        mounts[str(env_root)] = str(env_root)
    for source in [config["sam3_source"]["path"], config["sam3_checkpoint"]["path"],
                   *(item["path"] for item in config["trellis"]["models"].values())]:
        mounts[str(Path(source))] = str(Path(source))
    if 'observed_workspace' in config:
        observed=config['observed_workspace'];inventory=Path(observed['inventory_root']).resolve(strict=True)
        for name,digest in observed['inventory_files'].items():
            path=inventory/name
            if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('unsafe or changed observed workspace inventory')
            mounts[str(path)]='/inventory/'+name
        physics=Path(observed['physics_prior_path']).resolve(strict=True)
        if hashlib.sha256(physics.read_bytes()).hexdigest()!=observed['physics_prior_sha256']:raise ValueError('workspace physical prior changed')
        mounts[str(physics)]='/workspace_physics.json'
    if "shared_candidates" in config:
        shared = config["shared_candidates"]
        b0 = Path(shared["b0_build"]).resolve(strict=True)
        manifest = json.loads((b0/"build_manifest.json").read_text())
        for name in [*manifest["source_hashes"], "build_manifest.json"]:
            path = b0/name
            if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink() or not path.is_file():
                raise ValueError("unsafe shared B0 artifact closure")
            if name != "build_manifest.json" and hashlib.sha256(path.read_bytes()).hexdigest()!=manifest["source_hashes"][name]:
                raise ValueError("shared B0 bytes changed")
            mounts[str(path)] = "/b0/" + name
        if shared.get('phase')=='select':
            source=Path(shared['rvg_source']).resolve(strict=True)
            receipt_path=source/'rvg_receipt.json'
            if hashlib.sha256(receipt_path.read_bytes()).hexdigest()!=shared['rvg_receipt_sha256']:
                raise ValueError('cached RVG receipt changed')
            receipt=json.loads(receipt_path.read_text())
            if receipt['b0_manifest_sha256']!=hashlib.sha256((b0/'build_manifest.json').read_bytes()).hexdigest():
                raise ValueError('cached RVG belongs to another B0')
            for name in [*receipt['artifacts'],'rvg_receipt.json']:
                path=source/name
                if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.is_file():
                    raise ValueError('unsafe cached RVG artifact')
                if name!='rvg_receipt.json' and hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['artifacts'][name]:
                    raise ValueError('cached RVG artifact changed')
                mounts[str(path)]='/rvg/'+name
        for spec in shared["rvg"]["models"].values():
            mounts[str(Path(spec["path"]))] = str(Path(spec["path"]))
        policy = Path(shared["policy_config"]).resolve(strict=True)
        if policy.is_symlink() or hashlib.sha256(policy.read_bytes()).hexdigest()!=shared["policy_config_sha256"]:
            raise ValueError("shared controller config changed")
        mounts[str(policy)] = "/policies.yaml"
    # All paths are system runtime libraries, not /usr, /opt, host /home or /data.
    for path in ("/usr/lib", "/usr/lib64", "/opt/nvidia-driver/lib", "/etc/ld.so.cache"):
        if Path(path).exists():
            mounts[path] = path
    mounts[str(Path(config_path).resolve(strict=True))] = "/build_config.json"
    return mounts


WORKER = r'''
import json, os, pathlib, runpy, subprocess, sys, yaml
config=yaml.safe_load(pathlib.Path('/build_config.json').read_text())
env={'PYTHONPATH':'/code','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1',
     'SIMANY_NO_GT':'1','SIMANY_AUTO':'1','SIMANY_MESH_SRC':'derived',
     'SIMANY_OUT':'/output/construction','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
     'HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','CUDA_VISIBLE_DEVICES':'0',
     'LD_LIBRARY_PATH':'/opt/nvidia-driver/lib','OMP_NUM_THREADS':'4',
     'OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
     'TORCH_EXTENSIONS_DIR':'/output/runtime_cache/torch_extensions',
     'TRITON_CACHE_DIR':'/output/runtime_cache/triton',
     'XDG_CACHE_HOME':'/output/runtime_cache/xdg','CUDA_CACHE_PATH':'/output/runtime_cache/cuda'}
os.environ.update(env)
sys.path.insert(0,'/code')
if sys.argv[1]=='preflight' and ('observed_surface' in config or 'observed_workspace' in config or config.get('shared_candidates',{}).get('phase')=='select'):
    import importlib.util
    for module in ('robocasa','robosuite','omnigibson'):
        assert importlib.util.find_spec(module) is None, 'native API exposed: '+module
    assert not list(pathlib.Path('/dev').glob('nvidia*')), 'CPU control exposes GPU'
    from agents.discover.derive_mesh_from_splat import new_tsdf_volume
    from agents.assets.s6_physics import coacd_parts
    new_tsdf_volume(.002,.006)
    pathlib.Path('/output/runtime_preflight.json').write_text(json.dumps({'status':'PASS','method':'CPU_observed_or_shared_selection','gpu_devices':[], 'native_api_unavailable':True}))
elif sys.argv[1]=='preflight':
    results=[]
    check="""import importlib.util,json,os,torch
assert torch.cuda.is_available(), 'isolated CUDA unavailable'
assert torch.cuda.device_count()==1, 'more than allocated GPU exposed'
for module in ('robocasa','robosuite','omnigibson'):
 assert importlib.util.find_spec(module) is None, 'native API exposed: '+module
value=torch.arange(8,device='cuda').sum().item()
print(json.dumps({'python':__import__('sys').executable,'torch':torch.__version__,
 'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'device_count':torch.cuda.device_count(),
 'cuda_sum':value,'native_api_unavailable':True}))
"""
    for python in (config['sam3_python'],config['trellis']['python']):
        result=subprocess.run([python,'-c',check],text=True,capture_output=True)
        results.append({'python':python,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        if result.returncode:
            pathlib.Path('/output/runtime_preflight.json').write_text(json.dumps({'status':'FAIL','checks':results},indent=2))
            print(result.stderr);sys.exit(result.returncode)
    # Exercise exact producer import closure in both isolated environments.
    imports=[(config['sam3_python'],"import sys;sys.path.insert(0,"+repr(config['sam3_source']['path'])+");from robo.roundtrip.build import segment;from sam3.model_builder import build_sam3_image_model;from sam3.model.sam3_image_processor import Sam3Processor;print('SAM3 exact imports PASS')"),
             (config['trellis']['python'],"from agents.orchestrator.runtime import align_and_probe;from agents.assets.s6_physics import coacd_parts;import models.s4_trellis;from trellis.pipelines import TrellisImageTo3DPipeline;from trellis.models.structured_latent_vae import decoder_mesh;from trellis.representations.mesh import cube2mesh;import open3d,spconv.pytorch,nvdiffrast.torch;print('TRELLIS registration exact imports PASS')")]
    os.environ['SIMANY_TRELLIS_DIR']=config['trellis']['models']['trellis_source']['path']
    for python,command in imports:
        result=subprocess.run([python,'-c',command],text=True,capture_output=True)
        results.append({'python':python,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        if result.returncode:
            pathlib.Path('/output/runtime_preflight.json').write_text(json.dumps({'status':'FAIL','checks':results},indent=2))
            print(result.stderr);sys.exit(result.returncode)
    pathlib.Path('/output/runtime_preflight.json').write_text(json.dumps({'status':'PASS','checks':results},indent=2))
    print('isolated GPU/runtime preflight PASS')
elif 'observed_workspace' in config:
    pathlib.Path('/output/workspace_components_config.json').write_text(json.dumps(config['observed_workspace']))
    result=subprocess.run([config['trellis']['python'],'-m','robo.roundtrip.workspace_components',
        '--inventory','/inventory','--capture','/capture','--physics','/workspace_physics.json',
        '--config','/output/workspace_components_config.json','--out','/output/components'])
    sys.exit(result.returncode)
elif 'observed_surface' in config:
    pathlib.Path('/output/observed_config.json').write_text(json.dumps(config['observed_surface']))
    result=subprocess.run([config['trellis']['python'],'-m','robo.roundtrip.shared_candidates','--phase','observed',
        '--b0-build','/b0','--capture','/capture','--out','/output/observed','--config','/output/observed_config.json'])
    sys.exit(result.returncode)
elif 'shared_candidates' in config:
    shared=config['shared_candidates']
    pathlib.Path('/output/rvg_config.json').write_text(json.dumps(shared['rvg'],indent=2))
    commands=[(shared['rvg']['python'],['--phase','rvg','--config','/output/rvg_config.json','--out','/output/rvg']),
              (config['trellis']['python'],['--phase','select','--config','/policies.yaml','--rvg','/output/rvg',
               '--source-commit',config['source_commit'],'--out','/output/selection'])]
    phase=shared.get('phase','all')
    if phase=='rvg':commands=commands[:1]
    elif phase=='select':
        commands=commands[1:]
        commands[0][1][commands[0][1].index('--rvg')+1]='/rvg'
    elif phase!='all':raise ValueError('unsupported shared candidate stage')
    for python,args in commands:
        result=subprocess.run([python,'-m','robo.roundtrip.shared_candidates','--b0-build','/b0','--capture','/capture',*args])
        if result.returncode:sys.exit(result.returncode)
else:
    sys.argv=['robo.roundtrip.build','--config','/build_config.json','--capture','/capture','--out','/output','--phase','run']
    runpy.run_module('robo.roundtrip.build',run_name='__main__')
'''


def launch_command(*, code, config, config_path, capture, output, bubblewrap, forbidden_roots, preflight=False):
    mounts = runtime_mounts(code, config, config_path)
    command = constructor_command(bubblewrap=bubblewrap, public_capture=capture, output=output,
        runtime_mounts=mounts, forbidden_roots=forbidden_roots,
        device_paths=[] if cpu_stage(config) else allocated_devices(os.environ),
        argv=[config["trellis"]["python"], "-c", WORKER, "preflight" if preflight else "build"])
    # Debian's merged-/usr loader links. These point only to RO system library mounts.
    index = command.index("--chdir")
    command[index:index] = ["--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64"]
    return command, mounts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--forbidden", type=Path, action="append", required=True)
    parser.add_argument("--bubblewrap", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(args.config.read_text())
    if args.receipt.exists():
        raise FileExistsError("new immutable launcher receipt required")
    args.out.mkdir(parents=True, exist_ok=True)
    command, mounts = launch_command(code=args.code, config=config, config_path=args.config,
        capture=args.capture, output=args.out, bubblewrap=args.bubblewrap,
        forbidden_roots=args.forbidden, preflight=args.preflight)
    code_hashes = {target: hashlib.sha256(Path(source).read_bytes()).hexdigest()
                   for source, target in mounts.items() if target.startswith("/code/")}
    receipt = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bubblewrap_sha256": hashlib.sha256(args.bubblewrap.read_bytes()).hexdigest(),
        "devices": [] if cpu_stage(config) else allocated_devices(os.environ), "runtime_mounts": mounts,
        "source_hashes": code_hashes, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "forbidden_roots": [str(path.resolve()) for path in args.forbidden],
        "public_capture": str(args.capture.resolve()), "output": str(args.out.resolve()),
        "network": "unshared", "unsandboxed_fallback": False, "preflight": args.preflight}
    # Preserve the concrete launch contract even when bwrap itself fails.
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    result = subprocess.run(command)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
