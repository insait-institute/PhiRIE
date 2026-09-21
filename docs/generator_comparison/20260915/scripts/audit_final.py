"""Read-only artifact validation and final available-cohort aggregation."""
from pathlib import Path
import argparse, collections, csv, datetime, hashlib, json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, default=Path(__file__).parent)
R = parser.parse_args().run
S = json.loads((R / 'results/summary.json').read_text())
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
assert len(S['rows']) == 128
assert collections.Counter(x['status'] for x in S['rows']) == {'EVALUATED': 116, 'MISSING_INPUT': 12}
assert S['accounting']['active_gpu_jobs'] == 0
assert S['accounting']['reserved_gpu_hours'] == 0
assert S['accounting']['spent_gpu_hours'] <= 8
assert len(S['common_support']) == 10
verified = {}
checks = 0

def check(receipt):
    global checks
    p = Path(receipt['path'])
    if str(p) not in verified:
        digest = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(block)
        verified[str(p)] = {'bytes': p.stat().st_size, 'sha256': digest.hexdigest()}
    actual = verified[str(p)]
    assert actual['sha256'] == receipt['sha256'], p
    size = receipt.get('bytes', receipt.get('size_bytes'))
    if size is not None:
        assert actual['bytes'] == size, p
    assert actual['bytes'] > 0, p
    checks += 1
    return p

records = []
for row in S['rows']:
    if row['status'] == 'MISSING_INPUT':
        continue
    gen = json.loads(check(row['generation']).read_text())
    ev = json.loads(check(row['evaluation']).read_text())
    assert gen['status'] == 'GENERATION_COMPLETE' and gen['seed'] == 0
    assert gen['backend'] == row['backend']
    assert ev['status'] == 'EVALUATED' and ev['slot_id'] == row['slot_id']
    assert ev['registered'] and ev['exported'] and ev['collision_import_valid']
    assert ev['registration_gt_read'] is False and ev['evaluator_alignment'] == 'NONE'
    assert ev['scale_baked_once'] is True
    assert ev['appearance_metrics'] is None
    assert ev['generation']['sha256'] == row['generation']['sha256']
    check(ev['generation'])
    for receipt in gen['artifacts'].values():
        check(receipt)
    for receipt in gen['input_images']:
        check(receipt)
    assert 5 <= len(gen['input_images']) <= 12 if row['backend'] == 'reconviagen' else len(gen['input_images']) == 1
    assert len({x['sha256'] for x in gen['input_images']}) == len(gen['input_images'])
    for name in ['config', 'render', 'urdf', 'world_surface']:
        check(ev[name])
    if ev.get('evaluation_surface'):
        check(ev['evaluation_surface'])
    if row['backend'] in ('sam3d', 'reconviagen'):
        assert 'gaussians' in gen['artifacts']
    points = np.load(ev['world_surface']['path'])
    assert points.ndim == 2 and points.shape[1] == 3 and len(points) > 0
    assert np.isfinite(points).all()
    with np.load(gen['artifacts']['native_arrays']['path']) as arrays:
        vertices, faces = arrays['vertices'], arrays['faces']
        assert len(vertices) > 0 and len(faces) > 0 and vertices.shape[1] == faces.shape[1] == 3
        assert np.isfinite(vertices).all() and faces.min() >= 0 and faces.max() < len(vertices)
    records.append({'slot_id': row['slot_id'], 'backend': row['backend'], 'generation': row['generation'],
                    'evaluation': row['evaluation'], 'stable': ev['stability']['stable'],
                    'independent_metrics': ev['metrics'] is not None})

audit = {'checked_utc': now, 'status': 'PASS', 'verified_slots': len(records),
         'artifact_receipts_verified': checks, 'unique_files_verified': len(verified),
         'checks': ['original generation/evaluation/input/artifact SHA256 and byte receipts',
                    'finite native meshes and world surfaces, valid native face indices',
                    'registration isolated from evaluator and scale baked once',
                    'export and collision import records', 'seed 0 and single declared proposal',
                    '5-12 distinct conditioning images for all 29 multi-view ReconViaGen objects',
                    '29 SAM and 29 ReconViaGen production Gaussian files present',
                    '128 planned slots retained; 116 evaluated and 12 missing input',
                    'all task allocations terminal and cumulative GPU cost within allowance'],
         'limitations': ['Appearance quality is unmeasured; Gaussian files were not appearance-scored.',
                         'EVALUATED includes unstable and geometrically poor results.'],
         'records': records}
(R / 'audits/completed-four-backend-artifacts.json').write_text(json.dumps(audit, indent=2) + '\n')

rows = []
for b in S['backends']:
    assert b['generated'] == b['registered'] == b['exported'] == b['collision_import_valid'] == b['stability_tested'] == 29
    assert b['measured'] == b['common_support'] == 10
    rows.append({k: b[k] for k in ['backend', 'display_name', 'planned', 'generated', 'registered', 'exported',
                                  'collision_import_valid', 'stability_tested', 'stable', 'common_support']}
                | b['common_support_means'])
final = {'checked_utc': now, 'status': 'COMPLETE_ON_AVAILABLE_COHORT',
         'dataset': 'ScanNet++ historically observed DEV follow-up', 'seed': 0,
         'planned_slots': 128, 'evaluated_slots': 116, 'missing_input_slots': 12,
         'missing_input_objects': sorted({x['object_key'] for x in S['rows'] if x['status'] == 'MISSING_INPUT'}),
         'pending_or_running_jobs': 0, 'unresolved_runnable_slots': 0,
         'common_independent_geometry_objects': S['common_support'], 'rows': rows,
         'gpu_hours_spent': S['accounting']['spent_gpu_hours'], 'gpu_hours_reserved': 0,
         'interpretation': 'Descriptive comparison on 10 independently matched objects. The other 19 generated objects per backend have no independent geometry match. Scientific failures are retained without quality-based reruns. Appearance is unmeasured; the full 32-object denominator remains partial because three original inputs are unavailable.'}
(R / 'results/completed_four_backend_comparison.json').write_text(json.dumps(final, indent=2) + '\n')
with (R / 'results/completed_four_backend_comparison.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
failure = json.loads((R / 'results/failure_reconciliation.json').read_text())
failure.update(checked_utc=now, status='ALL_RUNNABLE_WORK_COMPLETED', completed_four_backend_slots=116,
               current_sam_production=[], pending_or_running_jobs=0, unresolved_runnable_slots=0)
failure['sam_cancelled_allocation'] = {'job': '901688', 'state': 'CANCELLED by 0',
    'cause': 'SIGTERM recorded; no further cause in scheduler metadata',
    'preserved_generations': 28, 'preserved_evaluations': 27,
    'resolved_by': [{'job': '902359', 'action': 'evaluate saved a1020 generation only', 'state': 'COMPLETED'},
                    {'job': '902360', 'action': 'generate and evaluate never-started a1070 only', 'state': 'COMPLETED'}],
    'node': 'hala', 'arrays': False, 'completed_generations_repeated': 0}
(R / 'results/failure_reconciliation.json').write_text(json.dumps(failure, indent=2) + '\n')
print(json.dumps({k: v for k, v in audit.items() if k != 'records'}, indent=2))
print(json.dumps(final, indent=2))
