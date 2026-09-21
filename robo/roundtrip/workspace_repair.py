"""Explicit one-action collision repair of a sealed observed DEV component."""
import argparse
import json
from pathlib import Path
import time

from robo.roundtrip.identity import file_hash
from robo.roundtrip.scope_bundle import artifact_hashes
from robo.roundtrip.scope_verification import regenerate_collision
from robo.roundtrip.build import _write


def repair(component, out, *, receipt_sha256):
    component, out = Path(component), Path(out)
    receipt_path = component/'workspace_component_receipt.json'
    if file_hash(receipt_path) != receipt_sha256:
        raise ValueError('sealed source component receipt differs')
    source = json.loads(receipt_path.read_text())
    if (source.get('method') != 'OBSERVED_WORKSPACE_COMPONENT_TSDF' or
            source.get('status') != 'BUILT' or source.get('role') not in
            ('source_support', 'destination_support', 'obstacle')):
        raise ValueError('only sealed observed workspace components are admissible')
    if any(Path(n).is_absolute() or '..' in Path(n).parts or file_hash(component/n)!=h
           for n,h in source['artifact_hashes'].items()) or any(
           source['artifact_hashes'].get(n)!=h for n,h in artifact_hashes(component).items()):
        raise ValueError('source artifact closure differs')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(schema_version=1, tier='DEV',
        method='OBSERVED_WORKSPACE_COMPONENT_TSDF_EXPLICIT_COLLISION_REPAIR',
        source_component=str(component.resolve()), source_receipt_sha256=receipt_sha256,
        source_artifact_hashes=source['artifact_hashes'],
        action_id=source['cluster_id']+'-collision-regeneration-001',
        parent_proposal_id=source['cluster_id'], max_calls=1, actual_calls=1,
        heldout_access=False, native_asset_access=False, L2_READY=False,
        dependencies_invalidated=['native_import','contact_handles','stability','scope_admission'],
        original_proposal_status='PRESERVED', native_policy='NOT_RUN')
    try:
        details = regenerate_collision(component, out/'object')
        # The copied producer receipt describes the old artifact; keep it under
        # an explicitly historical name, never make it appear to certify repair.
        copied = out/'object/workspace_component_receipt.json'
        copied.rename(out/'object/parent_workspace_component_receipt.json')
        result.update(status='REPAIRED_NOT_ADMITTED', object_dir=str((out/'object').resolve()),
                      action=details)
    except Exception as exc:
        result.update(status='REPAIR_FAILED', error=f'{type(exc).__name__}: {exc}',
                      partial_output=str((out/'object').resolve()))
    result['wall_s'] = time.monotonic()-started
    _write(out/'workspace_repair_receipt.json', result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--component',required=True);p.add_argument('--out',required=True)
    p.add_argument('--receipt-sha256',required=True)
    a=p.parse_args(); repair(a.component,a.out,receipt_sha256=a.receipt_sha256)

if __name__=='__main__':main()
