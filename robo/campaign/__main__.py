"""python -m robo.campaign --help"""
from __future__ import annotations
import argparse
import json
from . import data, runner
from .core import load


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='op',required=True)
    q=sub.add_parser('catalog');q.add_argument('--out',required=True)
    q=sub.add_parser('freeze');q.add_argument('--inventory',required=True);q.add_argument('--out',required=True);q.add_argument('--counts');q.add_argument('--seed',type=int,default=2027)
    q=sub.add_parser('gallery');q.add_argument('--inventory',required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('matrix');q.add_argument('--scenes',required=True);q.add_argument('--models',required=True);q.add_argument('--out',required=True);q.add_argument('--stage',action='append');q.add_argument('--max-tasks',type=int,default=20000)
    q=sub.add_parser('prepare');q.add_argument('--tasks',required=True);q.add_argument('--runtime',required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('launch');q.add_argument('--bundle',required=True);q.add_argument('--max-jobs',type=int,default=1);q.add_argument('--submit',action='store_true')
    q=sub.add_parser('worker');q.add_argument('--bundle',required=True);q.add_argument('--task',required=True)
    q=sub.add_parser('worker-batch');q.add_argument('--bundle',required=True);q.add_argument('--task',action='append',required=True)
    q=sub.add_parser('collect');q.add_argument('--bundle',required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('train-visual');q.add_argument('--config',required=True);q.add_argument('--out',required=True)
    a=p.parse_args()
    if a.op=='catalog': result=data.catalog(a.out)
    elif a.op=='freeze': result={'scenes':len(data.freeze(a.inventory,a.out,counts=load(a.counts) if a.counts else None,seed=a.seed))}
    elif a.op=='gallery': result=data.gallery(a.inventory,a.out)
    elif a.op=='matrix':
        from .matrix import build
        result=build(a.scenes,a.models,a.out,stages=a.stage,max_tasks=a.max_tasks)
    elif a.op=='prepare': result=runner.prepare(a.tasks,a.runtime,a.out)
    elif a.op=='launch': result=runner.launch(a.bundle,submit=a.submit,max_jobs=a.max_jobs)
    elif a.op=='worker': result=runner.worker(a.bundle,a.task)
    elif a.op=='worker-batch': result=runner.worker_batch(a.bundle,a.task)
    elif a.op=='collect': result=runner.collect(a.bundle,a.out)
    else:
        from .learned import fit
        result=fit(a.config,a.out)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
