#!/usr/bin/env python3
"""Pin the fixed TRELLIS.2 pilot using the existing TRAIN discovery declaration."""
import argparse
import os
from pathlib import Path
import sys
import yaml

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from robo.eval.fidelity_replacements import _tree_inventory
from run.icra2027.e3_trellis_generation_pilot import identity, runtime_identity, planned_jobs
from run.icra2027.e3_generator_backend import validate_config


def build(freeze_id, output_dir, *, evidence_root, python):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    root = Path(evidence_root).resolve()
    source = CODE/'configs/experiments/icra2027/trellis_cohort/38d58a7a31.yaml'
    old = yaml.safe_load(source.read_text())
    config = {key: value for key, value in old.items()
              if key.startswith('source_') or key in {'schema_version', 'seed', 'paper_ready'}}
    resources = dict(trellis2_source=root/'third_party/TRELLIS2',
                     trellis2_snapshot=root/'checkpoints/trellis2-4b',
                     dinov3_snapshot=root/'checkpoints/dinov3-vitl16')
    models = {}
    for name, path in resources.items():
        record = _tree_inventory(path, label=name)
        models[name] = dict(path=str(path), tree_sha256=record['tree_sha256'],
                            file_count=record['file_count'])
    prefix = root/'.cache/icra2027/e2-replacements/trellis-image-large-25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/ss_dec_conv3d_16l8_fp16'
    for name, suffix in [('ss_decoder_config','.json'), ('ss_decoder_weight','.safetensors')]:
        models[name] = identity(Path(str(prefix)+suffix))
    _, runtime_hash = runtime_identity(python)
    import json
    gpu_smoke_path = root/'outputs/trellis2-setup/gpu_import_smoke.json'
    gpu_smoke = json.loads(gpu_smoke_path.read_text())
    if (gpu_smoke.get('passed') is not True or gpu_smoke.get('pipeline_class_import') != 'PASS'
            or gpu_smoke.get('flash_attention_cuda_kernel') != 'PASS'
            or gpu_smoke.get('cumesh_cuda_kernel') != 'PASS'):
        raise ValueError('TRELLIS.2 environment CUDA smoke must pass before pilot configuration')
    obs = root/'outputs/icra2027/20260905-33bd974-v1/agentic/observations/38d58a7a31'
    config.update(generator='trellis2', native_gaussian=False,
                  study_scope='trellis2_mesh_engineering', pipeline_type='512',
                  upstream_commit='75fbf0183001ed9876c8dbb35de6b68552ee08bd',
                  upstream_model_revision='af44b45f2e35a493886929c6d786e563ec68364d',
                  dinov3_revision='ea8dc2863c51be0a264bab82070e3e8836b02d51',
                  freeze_id=freeze_id, output_scene_id='38d58a7a31',
                  python=python, models=models, runtime_sha256=runtime_hash,
                  gpu_environment_smoke=identity(gpu_smoke_path),
                  contract_resource_id='e3_trellis2_config',
                  observation_source=dict(manifest=identity(obs/'manifest.json'),
                      seal=identity(obs/'seal.json'),
                      e0=identity(obs.parents[2]/'contract/freeze_manifest.json'),
                      source_code_commit='0f6b7080de23e926bcab2f3072c84381573370df',
                      source_freeze_id='20260905-33bd974-v1'))
    validate_config(config)
    jobs, _, _, _ = planned_jobs(config, Path('unused'))
    if len(jobs)!=15 or sum(row['prepared'] for row in jobs)!=15:
        raise ValueError('fixed pilot must retain all 15 planned/prepared objects')
    freeze = yaml.safe_load((CODE/'configs/experiments/icra2027/e3_fresh_trellis_pilot_freeze.yaml').read_text())
    config_path = output_dir/'pilot.yaml'
    freeze.update(freeze_id=freeze_id, input_roots=[dict(id='e3_trellis2_config',
        path=str(config_path.resolve()), kind='experiment_config', required=True)],
        hardware=dict(accelerator='cpu', gpu_required=False,
            purpose='Contract for independent TRELLIS.2 15-object mesh/PBR pilot; native Gaussian unavailable'))
    output_dir.mkdir(parents=True)
    for name, data in [('pilot.yaml', config), ('freeze.yaml', freeze)]:
        with (output_dir/name).open('x') as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
    return config_path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--freeze-id', required=True)
    p.add_argument('--out', required=True)
    code = Path(__file__).resolve().parents[2]
    p.add_argument('--evidence-root', default=os.environ.get('SIMANY_EVIDENCE_ROOT', str(code)))
    p.add_argument('--python', default=os.environ.get('SIMANY_TRELLIS2_PY', str(code / '.envs/trellis2/bin/python')))
    a=p.parse_args()
    print(build(a.freeze_id, a.out, evidence_root=a.evidence_root, python=a.python))


if __name__=='__main__':
    main()
