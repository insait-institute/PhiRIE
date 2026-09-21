"""Actual DEV four-role native integration; no complete L2 or policy claim."""
import argparse
import json
from pathlib import Path
from robo.roundtrip.identity import file_hash, validate_canonical_instance
from robo.roundtrip.scope_bundle import artifact_hashes
from robo.roundtrip.scope import check_scope_identity_physics
from robo.roundtrip.workspace_import import import_observed_workspace
from robo.roundtrip.scope_verification import measure_imported_scope,CONFIG
from robo.roundtrip.build import _write


def run(config,out):
    out=Path(out)
    if config['tier']!='DEV' or config['identity_steps']!=200:raise ValueError('fixed DEV engineering protocol required')
    for name,digest in config['input_files'].items():
        if file_hash(name)!=digest:raise ValueError('frozen engineering input changed')
    entities=config['entities']
    for e in entities:
        if artifact_hashes(e['object_dir'])!=e['artifact_hashes']:raise ValueError('constructed artifact changed before native access')
    coverage=json.loads(Path(config['coverage']).read_text())
    # Privileged native roles are opened only AFTER constructor closure checks.
    binding=json.loads(Path(config['binding']).read_text());bundle=Path(binding['bundle_dir'])
    if binding['canonical_instance_id']!=config['canonical_instance_id'] or binding['capture_manifest_sha256']!=coverage['capture_manifest_sha256']:raise ValueError('canonical identity differs')
    validate_canonical_instance(json.loads(Path(binding['canonical_manifest']).read_text()),bundle)
    xml=(bundle/'scene.xml').read_text();state=json.loads((bundle/'canonical_state.json').read_text())
    out.mkdir(parents=True,exist_ok=False)
    try:
        controls=[{k:e[k] for k in ('body_name','role','component_kind')} for e in entities if e.get('component_kind')]
        _,identity=check_scope_identity_physics(xml,entities=controls,integration_state=state['integration_state'],steps=200)
        _write(out/'import_identity.json',identity)
        if identity['runtime_identity']!='PASS':raise ValueError('new support role identity failed')
        imported,manifest=import_observed_workspace(xml,base_entities=entities[:2],support_entities=entities[2:],coverage=coverage,bounds=coverage['workspace_bounds_world_m'])
        (out/'scene.xml').write_text(imported);_write(out/'scope_manifest.json',manifest)
        probe,_=measure_imported_scope(xml,state,imported,manifest,CONFIG)
        _write(out/'initialization_evidence.json',probe)
        result=dict(status='EXECUTED_ENGINEERING',native_import=True,identity='PASS',finite=probe['finite'],
            initialization_gate=probe['passed'],reason_codes=probe['reason_codes'],native_policy_episodes=0,
            actual_physics_steps=probe['settle_steps'],max_penetration_m=probe['max_penetration_m'],
            controller_decision=None,L2_READY=False,scope='DEV_partial_observed_workspace',
            complete_obstacle_replacement=False,source_support='observed bottom surface replacing basin direct geometry; unseen basin walls not recovered')
    except Exception as exc:
        result=dict(status='ENGINEERING_FAILED',error=f'{type(exc).__name__}: {exc}',native_policy_episodes=0,L2_READY=False)
    _write(out/'result.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.out)
if __name__=='__main__':main()
