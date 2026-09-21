"""Authenticate frozen full E2 appearance for canonical paper formatting.

This consumer rechecks sealed pixels/metadata and replays the original manifest
builder. It never renders images, loads LPIPS, or recomputes image metrics.
"""
import json
import math
import os
from pathlib import Path
import subprocess

from robo.manifest.hash import git_snapshot
from robo.eval.paper_room_diagnostic import checked, validate_metric_identity, METHODS

PRODUCER = '7a1f85b1414645c5416f8c824c57e738220c40f8'
SCOPE = 'full_public_factorized_official_test_evaluation'
MEMBERS = {'coverage.json', 'metrics.log', 'fidelity_manifest.json', 'receipt.json',
           'table/fidelity_table.csv', 'table/fidelity_table.json'}


def require(condition, message):
    if not condition:
        raise ValueError('full appearance paper source: ' + message)


def validate_rows(table, coverage, manifest, protocol):
    """Validate membership, paired support, leakage and metric denominators."""
    require(table['validation']['valid'] is True and table['lpips_backend_error'] is None
            and table['paper_ready'] is False and coverage['paper_ready'] is False
            and coverage['scope'] == SCOPE and coverage['conditional_quality'] is True
            and coverage['quality_comparison_paired'] is True and coverage['full_e3_gt_access'] is False
            and coverage['planned_scenes'] == 50 and coverage['planned_objects'] == 1871
            and coverage['planned_views_per_method'] == 400, 'scope or promotion differs')
    roster = protocol['scene_ids']
    require(len(roster) == len(set(roster)) == 50 and set(coverage['scene_seals']) == set(roster)
            and protocol['planned_scenes'] == 50 and protocol['planned_objects'] == 1871
            and protocol['planned_views'] == 400 and protocol['views_per_scene'] == 8,
            'predeclared scene denominator differs')
    cells = {(r['scene_id'], r['method']): r for r in coverage['rows']}
    require(len(cells) == len(coverage['rows']) == 100
            and set(cells) == {(s, m) for s in roster for m in METHODS},
            'omitted, duplicated or substituted coverage cell')
    expected_common = []
    for sid in roster:
        raw, composite = (cells[sid, m] for m in METHODS)
        frames = protocol['evaluation_frames'][sid]
        require(len(frames) == len(set(frames)) == 8, 'camera roster differs')
        for row in (raw, composite):
            require(row['planned_views'] == 8 and row['planned_objects'] == protocol['population'][sid]
                    and row['background_status'] in {'COMPLETE', 'NO_REMOVAL', 'BLOCKED_UNFILLABLE_ACCEPTED'}
                    and row['unchanged_source_background'] == (row['background_status'] == 'NO_REMOVAL'),
                    'scene population or unchanged background label differs')
            require(row['background_status'] != 'NO_REMOVAL' or row['accepted_objects'] == 0,
                    'zero-object background promoted to successful removal')
        require(all(raw[k] == composite[k] for k in
                    ('accepted_objects', 'background_status', 'source_preparation', 'unchanged_source_background')),
                'paired construction identity differs')
        blocked = composite['background_status'] == 'BLOCKED_UNFILLABLE_ACCEPTED'
        n = 0 if blocked else 8
        require(raw['available_views'] == 8 and raw['status'] == 'COMPLETE' and raw['missing_views'] == []
                and composite['available_views'] == n
                and composite['status'] == ('BLOCKED_UNFILLABLE_ACCEPTED' if blocked else 'COMPLETE')
                and composite['missing_views'] == (frames if blocked else [])
                and raw['analysis_views'] == composite['analysis_views'] == n,
                'availability, TRAIN blocker or conditional support differs')
        if n:
            expected_common.extend((sid, frame) for frame in frames)
    common = [(v['scene_id'], v['view_id']) for v in coverage['common_eligible_views']]
    require(len(common) == len(set(common)) and common == expected_common,
            'common support is not the complete available intersection')
    n_views, n_scenes = len(common), len({s for s, _ in common})
    require(n_views > 0, 'no appearance comparison exists')
    records_by_method = {}
    for method in METHODS:
        records = manifest['room_methods'][method]['records']
        views = [(r['scene_id'], v['view_id']) for r in records for v in r['views']]
        require(views == common and len(records) == n_scenes, 'metric manifest uses unequal or selected support')
        for r in records:
            require(r['n_views'] == len(r['views']) == 8 and r['freeze_id'] == manifest['freeze_id']
                    and {Path(v).stem for v in r['optimization_input_frames']}.isdisjoint(
                        Path(v['view_id']).stem for v in r['views'])
                    and r['evaluation_frames'] == [Path(v['render_path']).name for v in r['views']],
                    'TRAIN/TEST leakage or metric frame identity differs')
        records_by_method[method] = {(r['scene_id'], v['view_id']): v for r in records for v in r['views']}
        rows = [r for r in table['rows'] if r['method'] == method]
        require(len(rows) == 1, 'metric row omitted or duplicated')
        row = rows[0]
        require(row['unit'] == 'room' and row['n_scenes'] == n_scenes and row['n_images'] == n_views
                and row['metric_samples'] == dict.fromkeys(('psnr', 'ssim', 'lpips'), n_views),
                'metric sample denominator differs')
        for key in ('psnr', 'ssim', 'lpips'):
            metric = row['metrics'][key]
            require(type(row[key]) in (int, float) and math.isfinite(row[key])
                    and metric['value'] == row[key] and metric['n'] == n_views
                    and metric['bootstrap']['unit'] == 'scene'
                    and metric['bootstrap']['n_units'] == n_scenes
                    and metric['bootstrap']['samples'] == 2000
                    and metric['source_artifact_hash'] == row['metric_source_hashes'][key],
                    'missing, nonfinite or rebound metric')
    require(all(records_by_method[METHODS[0]][key]['gt_sha256'] ==
                records_by_method[METHODS[1]][key]['gt_sha256'] for key in common),
            'paired reference pixels differ')
    measured = [r for r in table['rows'] if r['method'] in METHODS]
    require(len(table['rows']) == table['rows_count'] == 8
            and all(r['n_images'] == 0 and all(r.get(k) is None for k in
                    ('psnr', 'ssim', 'lpips', 'cd_cm', 'f1_20', 'collapses'))
                    for r in table['rows'] if r not in measured), 'unmeasured method promoted')
    require(table['validation']['generator_leakage_count'] == 0
            and table['validation']['registration_evaluation_hash_collision_count'] == 0
            and table['validation']['lpips_provenance_complete'] is True,
            'canonical leakage or LPIPS validation failed')
    # All source rows and their indices survive for exact claim-ledger pointers.
    return table


# Run in the exact original consumer checkout. The publication checks all frozen
# output bytes plus metadata identities, but does not repeat construction or the
# expensive original raw-source geometric validator.
PROGRAM = r'''
import json,sys
from pathlib import Path
from run.icra2027 import e2_public_factorized as p
c,e,out=p.context(sys.argv[1],sys.argv[2]); d=out/'metrics'
if d/'seal.json'!=Path(sys.argv[3]):raise ValueError('metric output differs')
p._sealed(d)
coverage=json.loads((d/'coverage.json').read_text()); results={}; seals={}
protocol,inventory=p.full_inventory(c)
bank=json.loads(p._checked(protocol['camera_bank']).read_text())
for sid in c['scene_ids']:
 directory,seal=p._sealed(out/'scenes'/sid)
 r=json.loads((directory/'result.json').read_text())
 if (r['scene_id']!=sid or r['scope']!=p.FULL_SCOPE or r['freeze_id']!=c['freeze_id']
     or r['code_commit']!=e['code']['commit'] or r['config_identity']!=p._identity(Path(sys.argv[1]))
     or r['contract_identity']!=p._identity(Path(sys.argv[2]))
     or r['source_fill']!=c['fill_seals'].get(sid) or r['planned_objects']!=c['planned_objects'][sid]
     or r['planned_views']!=8 or r['paper_ready'] is not False
     or r['full_e3_gt_access'] is not False or r['construction_modified'] is not False):
  raise ValueError('sealed scene metadata identity differs')
 plan=r['raw_source']['plan']
 if ([v['frame'] for v in plan['evaluation_images']]!=protocol['evaluation_frames'][sid]
     or plan['camera_bank']['sha256']!=protocol['camera_bank']['sha256']
     or plan['original_camera_plan']!=bank['scenes'][c['scene_ids'].index(sid)]['plan']):
  raise ValueError('scene camera roster differs')
 original=json.loads(p._checked(plan['original_camera_plan']).read_text())
 if any(plan[k]!=original[k] for k in ('evaluation_images','optimization_input_frames','calibration')):
  raise ValueError('original camera matrices or TRAIN split changed')
 for name in ('raw_views','composite_views'):
  for v in r[name]:
   for role in ('render','gt'):
    image=Path(v[role+'_path'])
    if not image.is_relative_to(directory) or json.loads((directory/'seal.json').read_text())['members'].get(str(image.relative_to(directory)))!=v[role+'_sha256']:
     raise ValueError('view not bound by checked scene seal')
 results[sid]=r; seals[sid]=seal
manifest,rows=p.make_manifest(c,e,results)
if (manifest!=json.loads((d/'fidelity_manifest.json').read_text())
    or coverage['rows']!=rows or coverage['scene_seals']!=seals):
 raise ValueError('canonical manifest or coverage changed')
receipt=json.loads((d/'receipt.json').read_text())
expected=[c['execution']['python']['metrics'],'-m','robo.eval.fidelity_metrics','--manifest',str(d/'fidelity_manifest.json'),'--out',str(d/'table'),'--lpips-device','cuda','--bootstrap-samples','2000','--bootstrap-seed','42']
if (receipt['status']!='PASS' or receipt['command']!=expected or receipt['metric_source']!=c['metric_source']
    or receipt['table']!=p._identity(d/'table/fidelity_table.json') or receipt['coverage']!=p._identity(d/'coverage.json')
    or receipt['paper_ready'] is not False):raise ValueError('canonical metric receipt differs')
table=json.loads((d/'table/fidelity_table.json').read_text())
for role in ('backbone_checkpoint','linear_checkpoint'):
 ref=table['lpips_provenance'][role]; ref={**ref,'bytes':ref['size_bytes'],'path':str(Path(c['metric_source']['code_root'])/ref['path'])}
 p._checked(ref)
print(json.dumps(dict(coverage=coverage,table=table,seal=json.loads((d/'seal.json').read_text()),
 protocol=protocol,contract_sha256=e['contract_sha256'],manifest=manifest,manifest_identity=p._identity(d/'fidelity_manifest.json'),
 freeze_id=c['freeze_id'],metric_source_commit=c['metric_source']['code_commit'])))
'''


def validate_source(spec, path, table, seal_path, seal):
    """Return original table, original coverage, and provenance for paper_pipeline."""
    code = Path(spec['producer']['path']).resolve(strict=True)
    snapshot = git_snapshot(code)
    require(spec['producer']['commit'] == PRODUCER and snapshot['commit'] == PRODUCER
            and not snapshot['dirty'], 'original producer source differs')
    require(checked(spec) == path and checked(spec['completion_audit']) == seal_path
            and path == seal_path.parent/'table/fidelity_table.json' and seal_path.name == 'seal.json'
            and seal['schema_version'] == 1 and set(seal['members']) == MEMBERS,
            'entrypoint or exact metric member roster differs')
    config, contract = checked(spec['config']), checked(spec['contract'])
    environment = dict(os.environ, PYTHONPATH=str(code), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    result = json.loads(subprocess.check_output([spec['producer']['python'], '-c', PROGRAM,
        str(config), str(contract), str(seal_path)], cwd=code, env=environment, text=True))
    require(result['table'] == table and result['seal'] == seal and git_snapshot(code) == snapshot,
            'source/output changed during publication check')
    for ref in (spec, spec['completion_audit'], spec['config'], spec['contract']):
        checked(ref)
    validate_metric_identity(table, result['manifest_identity'], result['freeze_id'], result['metric_source_commit'])
    validate_rows(table, result['coverage'], result['manifest'], result['protocol'])
    return table, result['coverage'], dict(producer_commit=PRODUCER,
        contract_sha256=result['contract_sha256'], full_protocol=result['manifest']['provenance']['full_protocol'],
        canonical_metric_recomputed=False, paired_delta_inference='NOT_RUN',
        verification_scope='Original-source context and manifest replay; all sealed metric and scene pixels verified; no construction or image metric recomputation')
