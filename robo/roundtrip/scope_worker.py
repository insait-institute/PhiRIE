"""Seal one scope dependency then invoke the existing canonical paired runner."""
import argparse
import json
from pathlib import Path
from robo.roundtrip.scope_bundle import seal_bundle
from robo.roundtrip.matrix import save_new


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['config','canonical-reference','canonical-manifest','reset-bank','out','host','port','baseline-config','baseline-episode','target-dir','destination-build-manifest']:
        p.add_argument('--'+n,required=True)
    p.add_argument('--destination-candidate-pool');p.add_argument('--destination-rvg-receipt');p.add_argument('--destination-repair-receipt')
    a=p.parse_args(argv);out=Path(a.out)
    record=seal_bundle(a.baseline_episode,a.baseline_config,a.target_dir,a.destination_build_manifest,a.destination_candidate_pool,a.destination_rvg_receipt,
        **({'destination_repair_receipt':a.destination_repair_receipt} if a.destination_repair_receipt else {}))
    bundle=out.parent/'scope_bundle.json';save_new(bundle,record)
    from robo.roundtrip.paired import main as paired_main
    command=[]
    for k in ['config','canonical_reference','canonical_manifest','reset_bank','out','host','port','baseline_config']:
        command+=['--'+k.replace('_','-'),str(getattr(a,k))]
    command+=['--scope-bundle',str(bundle),'--reference-episode',a.baseline_episode]
    return paired_main(command)

if __name__=='__main__':raise SystemExit(main())
