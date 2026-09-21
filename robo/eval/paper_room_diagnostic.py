"""Publish a sealed common-view diagnostic without recomputing image metrics."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from robo.manifest.hash import git_snapshot

PRODUCER = '11866350bf3a12160f922b89db2fda9bf3a06fe3'
METHODS = ['Input scene Gaussian, reconstruction ceiling', 'Factorized composite, automatic discovery']


def checked(ref):
    path = Path(ref['path'])
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError('room diagnostic source uses an alias')
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
        raise ValueError('room diagnostic source hash differs')
    return path


def validate_rows(table, coverage):
    """Retain the original denominator and display only canonical paired means."""
    if (table.get('validation', {}).get('valid') is not True or table.get('lpips_backend_error')
            or coverage.get('quality_comparison_paired') is not True
            or coverage.get('planned_scenes') != 2 or coverage.get('planned_objects') != 17
            or coverage.get('planned_views_per_method') != 16
            or coverage.get('paper_ready') is not False
            or coverage.get('full_e3_gt_access') is not False):
        raise ValueError('room diagnostic scope or metric validation differs')
    for method, available in zip(METHODS, [16, 8]):
        source = [r for r in table['rows'] if r['method'] == method]
        cells = [r for r in coverage['rows'] if r['method'] == method]
        if (len(source) != 1 or len(cells) != 2
                or sum(r['planned_views'] for r in cells) != 16
                or sum(r['available_views'] for r in cells) != available
                or sum(r['analysis_views'] for r in cells) != 8):
            raise ValueError('room diagnostic planned/available/paired denominator differs')
        row = source[0]
        if row['unit'] != 'room' or row['n_scenes'] != 1 or row['n_images'] != 8:
            raise ValueError('room diagnostic quality must use the same one-scene eight-view support')
        if row.get('metric_samples') != dict(psnr=8, ssim=8, lpips=8):
            raise ValueError('room diagnostic per-metric sample counts differ')
        if any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) for k in ('psnr', 'ssim', 'lpips')):
            raise ValueError('room diagnostic contains missing/nonfinite quality')
    # Preserve every source row and its index for exact numeric claim lineage.
    return table


def validate_metric_identity(table, manifest, freeze_id, metric_source_commit):
    expected = dict(code_commit=metric_source_commit, freeze_id=freeze_id,
                    manifest_path=manifest['path'], manifest_sha256=manifest['sha256'])
    if (table.get('freeze_id') != freeze_id or table.get('manifest_path') != manifest['path']
            or table.get('manifest_sha256') != manifest['sha256'] or table.get('provenance') != expected
            or table.get('paper_ready') is not False):
        raise ValueError('room diagnostic table-to-manifest or source binding differs')


def validate_source(spec, path, table, seal_path, seal):
    code = Path(spec['producer']['path']).resolve(strict=True)
    snapshot = git_snapshot(code)
    if spec['producer']['commit'] != PRODUCER or snapshot['commit'] != PRODUCER or snapshot['dirty']:
        raise ValueError('room diagnostic producer source differs')
    config = checked(spec['config']); contract = checked(spec['contract'])
    if path != seal_path.parent/'table/fidelity_table.json' or seal_path.name != 'seal.json':
        raise ValueError('room diagnostic metric source path differs')
    # Import only the exact original producer. Its context replays E0 and source
    # hashes; collect_scenes replays the original render producers and all pixels.
    # No renderer, LPIPS model or metric evaluator is invoked here.
    program = '''import json,sys
from pathlib import Path
from run.icra2027 import e2_public_factorized as p
c,e,out=p.context(sys.argv[1],sys.argv[2]);d=out/'metrics'
if d/'seal.json'!=Path(sys.argv[3]):raise ValueError('original metric output differs')
p._sealed(d)
results,seals=p.collect_scenes(c,e,out)
manifest,rows=p.make_manifest(c,e,results)
if manifest!=json.loads((d/'fidelity_manifest.json').read_text()):raise ValueError('paired manifest changed')
coverage=json.loads((d/'coverage.json').read_text())
if (coverage['rows']!=rows or coverage['scene_seals']!=seals or not c.get('paired_protocol')
    or coverage['paired_protocol']!=c['paired_protocol']):raise ValueError('paired coverage changed')
seal=json.loads((d/'seal.json').read_text())
receipt=json.loads((d/'receipt.json').read_text())
expected=[c['execution']['python']['metrics'],'-m','robo.eval.fidelity_metrics','--manifest',str(d/'fidelity_manifest.json'),'--out',str(d/'table'),'--lpips-device','cuda','--bootstrap-samples','2000','--bootstrap-seed','42']
if (receipt['status']!='PASS' or receipt['command']!=expected or receipt['metric_source']!=c['metric_source']
    or receipt['table']!=p._identity(d/'table/fidelity_table.json') or receipt['coverage']!=p._identity(d/'coverage.json')):
    raise ValueError('original canonical metric receipt differs')
print(json.dumps(dict(coverage=coverage,table=json.loads((d/'table/fidelity_table.json').read_text()),seal=seal,protocol=c['paired_protocol'],contract_sha256=e['contract_sha256'],manifest=p._identity(d/'fidelity_manifest.json'),freeze_id=c['freeze_id'],metric_source_commit=c['metric_source']['code_commit'])))
'''
    environment = dict(os.environ, PYTHONPATH=str(code), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    result = json.loads(subprocess.check_output([spec['producer']['python'], '-c', program,
        str(config), str(contract), str(seal_path)], cwd=code, env=environment, text=True))
    if result['table'] != table or result['seal'] != seal or git_snapshot(code) != snapshot:
        raise ValueError('room diagnostic original source/output changed')
    checked(spec['config']); checked(spec['contract'])
    validate_metric_identity(table, result['manifest'], result['freeze_id'], result['metric_source_commit'])
    return validate_rows(table, result['coverage']), result['coverage'], {'producer_commit': PRODUCER,
        'contract_sha256': result['contract_sha256'], 'paired_protocol': result['protocol'],
        'canonical_metric_recomputed': False}
