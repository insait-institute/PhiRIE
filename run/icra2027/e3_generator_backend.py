"""Explicit generator capabilities for the existing initial-proposal runner."""
from pathlib import Path
import os

from run.icra2027.e3_auto_discovery_pilot import PilotError

TRELLIS2_OUTPUTS = ('trellis2_mesh.ply', 'mesh_sim.ply', 'mesh_sim.obj',
                    'trellis2_pbr.npz', 'trellis2_conditioning.png')


def generator(config):
    name = config.get('generator', 'trellis')
    if name not in {'trellis', 'trellis2'}:
        raise PilotError('unknown initial generator: ' + str(name))
    return name


def validate_config(config):
    if generator(config) != 'trellis2':
        return
    if config.get('raw_reuse'):
        raise PilotError('TRELLIS.2 cannot reuse TRELLIS v1 proposal receipts')
    if config.get('pipeline_type') not in {'512', '1024', '1024_cascade', '1536_cascade'}:
        raise PilotError('explicit supported TRELLIS.2 pipeline_type required')
    if config.get('study_scope') != 'trellis2_mesh_engineering':
        raise PilotError('TRELLIS.2 requires a separate mesh engineering study')
    if config.get('native_gaussian') is not False:
        raise PilotError('TRELLIS.2 has no native Gaussian output')
    required = {'trellis2_source', 'trellis2_snapshot', 'dinov3_snapshot',
                'ss_decoder_config', 'ss_decoder_weight'}
    if set(config.get('models', {})) != required:
        raise PilotError('TRELLIS.2 needs exact local source/checkpoint closure')
    if len(config.get('upstream_commit', '')) != 40:
        raise PilotError('TRELLIS.2 upstream commit must be pinned')
    for spec in config['models'].values():
        if not Path(spec['path']).is_absolute():
            raise PilotError('TRELLIS.2 resources must use absolute local paths')


def environment(config, code, out=None):
    """No inherited v1 model variables can select the new backend implicitly."""
    validate_config(config)
    models = config['models']
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(code), SIMANY_ROOT=str(code), HF_HUB_OFFLINE='1',
               HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TRANSFORMERS_OFFLINE='1')
    env.pop('PYTHONHOME', None)
    env['GIT_OPTIONAL_LOCKS'] = '0'
    if generator(config) == 'trellis2':
        env.update(SIMANY_TRELLIS2_DIR=models['trellis2_source']['path'],
                   SIMANY_TRELLIS2_MODEL=models['trellis2_snapshot']['path'],
                   SIMANY_DINOV3_MODEL=models['dinov3_snapshot']['path'],
                   SIMANY_SS_DECODER=str(Path(models['ss_decoder_weight']['path']).with_suffix('')),
                   SIMANY_TRELLIS2_SOURCE_COMMIT=config['upstream_commit'],
                   SIMANY_TRELLIS2_PIPELINE_TYPE=config['pipeline_type'],
                   SIMANY_TRELLIS2_SEED=str(config['seed']))
    else:
        env.update(SIMANY_TRELLIS_DIR=models['trellis_source']['path'],
                   SIMANY_TRELLIS_MODEL=models['trellis_snapshot']['path'],
                   SIMANY_DINOV2_REPO=models['dinov2_source']['path'],
                   TORCH_HOME=str(Path(models['dinov2_source']['path']).parent.parent))
    if out is not None:
        env.update(SIMANY_OUT=str(out/'construction'), SIMANY_AUTO='1',
                   SIMANY_MESH_SRC='derived', SIMANY_GENERATION_RECORDS=str(out/'producer_records'))
        if generator(config) == 'trellis2':
            cache = out/'runtime_cache'
            env.update(TORCH_EXTENSIONS_DIR=str(cache/'torch_extensions'),
                       TRITON_CACHE_DIR=str(cache/'triton'),
                       TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
                       NUMBA_CACHE_DIR=str(cache/'numba'),
                       FLEX_GEMM_AUTOTUNE_CACHE_PATH=str(cache/'flexgemm/autotune_cache.json'))
    return env


def audit_trellis2(config_path, root, *, write_report=True):
    """Independent artifact checks, using the original runner's population binding."""
    import json
    import numpy as np
    import trimesh
    from PIL import Image
    from run.icra2027 import e3_trellis_generation_pilot as runner
    c, commit = runner.context(config_path, root)
    if generator(c) != 'trellis2':
        raise PilotError('this artifact audit is for explicit TRELLIS.2 proposals')
    out = runner.output_directory(c, root)
    manifest = json.loads((out/'input_manifest.json').read_text())
    runner.validate_source_binding(c, out, manifest)
    runner.validate_staged(out, manifest)
    pool = json.loads((out/'proposal_pool.json').read_text())
    if (pool.get('generator') != 'trellis2' or pool['code_commit'] != commit
            or pool['freeze_id'] != c['freeze_id']
            or pool['input_manifest_sha256'] != runner.sha(out/'input_manifest.json')
            or pool['proposal_records_sha256'] != runner.sha(out/'proposal_records.jsonl')
            or pool['rows'] != [json.loads(line) for line in (out/'proposal_records.jsonl').read_text().splitlines()]
            or pool['rows'] != runner.collect_records(out, manifest, pool['exit_code'])
            or pool['planned_jobs'] != len(manifest['jobs'])):
        raise PilotError('TRELLIS.2 pool/proposal closure differs')
    checked = []
    for row in pool['rows']:
        if row['status'] != 'available':
            continue
        for name, spec in row['artifacts'].items():
            path = Path(spec['path'])
            path.resolve().relative_to(out.resolve())
            if path.is_symlink() or runner.sha(path) != spec['sha256']:
                raise PilotError('TRELLIS.2 artifact hash differs: '+name)
            if name.endswith(('.ply', '.obj')):
                mesh = trimesh.load(path, process=False)
                if (not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces)
                        or len(mesh.vertices) < 3 or not np.isfinite(mesh.vertices).all()
                        or mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices)):
                    raise PilotError('TRELLIS.2 invalid mesh: '+name)
            elif name.endswith('.npz'):
                with np.load(path, allow_pickle=False) as data:
                    if not {'attrs', 'coords', 'voxel_size'}.issubset(data.files):
                        raise PilotError('TRELLIS.2 native material fields missing')
                    for field in data.files:
                        value = data[field]
                        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                            raise PilotError('TRELLIS.2 nonfinite native material')
            elif name.endswith('.png'):
                with Image.open(path) as image:
                    image.verify()
            checked.append(spec)
    report = dict(schema_version=1, generator='trellis2', code_commit=commit,
                  freeze_id=c['freeze_id'], config_sha256=runner.sha(config_path),
                  input_manifest_sha256=runner.sha(out/'input_manifest.json'),
                  proposal_pool_sha256=runner.sha(out/'proposal_pool.json'),
                  planned_jobs=pool['planned_jobs'],
                  available_jobs=sum(r['status']=='available' for r in pool['rows']),
                  failed_or_unavailable_jobs=sum(r['status']!='available' for r in pool['rows']),
                  artifact_count=len(checked), native_gaussian=False,
                  full_twin_ready=False, paper_ready=False, artifact_checks='PASS')
    if write_report:
        runner.write_new(out/'postrun_audit.json', report)
        print(json.dumps(report, sort_keys=True))
    return report
