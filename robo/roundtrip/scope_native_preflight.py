"""Native scope reset/handle smoke before loading a policy engine; zero episodes."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np


def main(argv=None):
    from robo.roundtrip.adapters.robocasa import RoboCasaAdapter
    from robo.roundtrip.paired import prepare_scope_adapter
    from robo.roundtrip.scope_bundle import validate_bundle
    from robo.roundtrip.identity import file_hash
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scope-bundle',required=True)
    p.add_argument('--canonical-reference',required=True);p.add_argument('--out',required=True)
    a=p.parse_args(argv);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();adapter=None
    record=json.loads(Path(a.scope_bundle).read_text());baseline=Path(record['baseline_config'])
    config=json.loads(baseline.read_text());config.update(scope='L1_target_destination',replacement_scope='target_destination')
    validate_bundle(record,config,record['baseline_episode'],baseline)
    reference=Path(a.canonical_reference);state=json.loads((reference/'canonical_state.json').read_text())
    bundle={'xml':(reference/'scene.xml').read_text(),'state':state,'provenance':{'reset_seed':0,'preflight_only':True}}
    try:
        adapter=RoboCasaAdapter(config)
        baseline_import=json.loads((Path(record['baseline_episode']).parent/'import_receipt.json').read_text())
        manifest=prepare_scope_adapter(adapter,bundle,record,baseline_import)
        observed=adapter.get_state();keep=np.ones(len(observed['qpos']),bool)
        for joint in manifest['replaced_joints']:
            jid=adapter.native.sim.model.joint_name2id(joint);adr=int(adapter.native.sim.model.jnt_qposadr[jid]);keep[adr:adr+7]=False
        unchanged=np.array_equal(np.asarray(observed['qpos'])[keep],np.asarray(state['qpos'])[keep])
        (out/'scope_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        result={'status':'PASS' if unchanged else 'FAIL','unreplaced_qpos_byte_equal':unchanged,
                'native_fixture_handles':'BOUND_AND_COMPILED','native_policy_episodes':0,'policy_invoked':False,
                'elapsed_s':time.monotonic()-start,'scope_bundle_sha256':file_hash(a.scope_bundle)}
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
        return 0 if unchanged else 2
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps({'type':type(exc).__name__,'reason':str(exc),'native_policy_episodes':0,'elapsed_s':time.monotonic()-start},indent=2)+'\n');raise
    finally:
        if adapter is not None:adapter.close()


if __name__=='__main__':raise SystemExit(main())
