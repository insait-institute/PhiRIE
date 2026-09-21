"""Descriptive scope diagnostics from immutable native traces, without rerollout.

These measurements explain motion and coverage; they are not new grasp/lift
success rubrics, contact-force reconstructions, or absolute pose-error metrics.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def summarize_trace(rows, initial):
    if not rows:raise ValueError('empty trace cannot establish physical outcome')
    ticks=[r['tick'] for r in rows]
    if ticks != list(range(len(rows))):raise ValueError('trace must retain contiguous crash prefix from tick zero')
    objects={}
    for name,start in initial['object_states'].items():
        xyz=np.asarray([r['objects'][name][:3] for r in rows],dtype=float)
        start=np.asarray(start[:3],dtype=float)
        if not np.isfinite(xyz).all() or not np.isfinite(start).all():raise ValueError('nonfinite trajectory')
        delta=xyz-start
        objects[name]={'initial_xyz_m':start.tolist(),'final_xyz_m':xyz[-1].tolist(),
            'first_step_displacement_m':float(np.linalg.norm(delta[0])),
            'max_displacement_from_initial_m':float(np.linalg.norm(delta,axis=1).max()),
            'max_xy_displacement_from_initial_m':float(np.linalg.norm(delta[:,:2],axis=1).max()),
            'max_z_above_initial_m':float(delta[:,2].max()),'min_z_relative_initial_m':float(delta[:,2].min()),
            'final_displacement_m':float(np.linalg.norm(delta[-1]))}
    predicates=[r['native_predicates'] for r in rows]
    distances=[p['gripper_distance_m'] for p in predicates if p.get('gripper_distance_m') is not None]
    return {'recorded_ticks':len(rows),'duration_s':rows[-1]['simulation_time_s']-initial['time'],
        'objects':objects,'min_gripper_to_native_origin_m':min(distances) if distances else None,
        'inside_sink_tick_count':sum(p.get('object_origin_inside_sink') is True for p in predicates),
        'native_success_tick_count':sum(p.get('native_task_success') is True for p in predicates),
        'final_native_predicates':predicates[-1],
        'contact_pairs':'NOT_RECORDED','grasp_success':None,'lift_success':None,
        'absolute_pose_error':None,'interpretation_scope':'within-arm displacement only; raw native body-origin predicate retained'}


def generate(pilot,out):
    pilot=Path(pilot);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    summary=json.loads((pilot/'scope_pilot_summary.json').read_text());arms={}
    for row in summary['rows']:
        label=row.get('label') or row.get('method')
        if label is None:raise ValueError('summary row lacks declared arm label')
        ep=pilot/label/'episode';result=json.loads((ep/'result.json').read_text())
        with gzip.open(ep/'trace.json.gz','rt') as f:trace=json.load(f)
        initial=json.loads((ep/'initial_state.json').read_text())
        arms[label]={'episode':str(ep.resolve()),'source_hashes':{n:sha(ep/n) for n in ['trace.json.gz','initial_state.json','result.json','continuous.mp4']},
            'result':{k:result.get(k) for k in ['executed','error','success','policy_engine','source_commit']},
            'motion':summarize_trace(trace,initial)}
    record={'schema_version':1,'kind':'descriptive_posthoc_dev_scope_diagnosis','source_summary':str(pilot/'scope_pilot_summary.json'),
        'source_summary_sha256':sha(pilot/'scope_pilot_summary.json'),'arms':arms,'simulation_executed':False,
        'limitations':['No contact ledger or gripper-force samples were recorded; contact/grasp mechanism remains unresolved.',
        'No new success thresholds. Within-arm motion is descriptive, not a new lift/grasp headline metric.',
        'Generated and native body origins need not coincide; absolute correspondence remains unestablished.']}
    (out/'scope_trace_diagnostics.json').write_text(json.dumps(record,indent=2)+'\n');return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',required=True);p.add_argument('--out',required=True);a=p.parse_args();generate(a.pilot,a.out)
if __name__=='__main__':main()
