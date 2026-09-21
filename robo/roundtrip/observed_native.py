"""Validate frozen TRAIN-only observed-surface controls before native import."""
from pathlib import Path
import json
from robo.roundtrip.identity import file_hash
from robo.roundtrip.scope_bundle import artifact_hashes


def validate_observed_control(receipt_path,object_dir,b0_build,capture,config):
    from robo.roundtrip.shared_candidates import checked_b0
    receipt_path=Path(receipt_path);directory=Path(object_dir).resolve()
    if receipt_path.resolve()!=directory/'observed_surface_receipt.json':raise ValueError('observed receipt must bind the imported object directory')
    receipt=json.loads(receipt_path.read_text());b0_build=Path(b0_build)
    if (receipt.get('schema_version')!=2 or receipt.get('method')!='OBSERVED_SURFACE_TSDF' or
        receipt.get('status')!='BUILT' or receipt.get('scope')!='L0_target_only' or receipt.get('object_role')!='target' or
        receipt.get('heldout_access') is not False or receipt.get('config')!={'voxel_m':.002,'sdf_trunc_m':.006}):
        raise ValueError('observed control requires frozen TRAIN-only TSDF receipt')
    manifest,discovery=checked_b0(b0_build,capture)
    for key in ['canonical_instance_id','cohort_id']:
        if receipt.get(key)!=config[key] or manifest.get(key)!=config[key]:raise ValueError('observed control canonical/cohort differs')
    if (receipt['b0_manifest_sha256']!=file_hash(b0_build/'build_manifest.json') or
        receipt['capture_manifest_sha256']!=manifest['capture_manifest_sha256']):
        raise ValueError('observed control B0/capture closure differs')
    candidates={r['candidate_id']:r for r in json.loads((b0_build/'discovery/candidates.json').read_text())}
    expected=[dict(candidate_id=cid,frame_id=candidates[cid]['frame_id'],mask_sha256=candidates[cid]['mask_sha256']) for cid in discovery['accepted_candidates']]
    if receipt['source_views']!=expected:raise ValueError('observed control changed frozen TRAIN view/mask roster')
    for name,digest in receipt['artifact_hashes'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or file_hash(directory/path)!=digest:
            raise ValueError('observed control artifact closure changed')
    for name,digest in artifact_hashes(directory).items():
        if receipt['artifact_hashes'].get(name)!=digest:raise ValueError('imported control artifact is not sealed')
    if file_hash(directory/'physics.json')!=file_hash(b0_build/'construction/objects/obj_00/physics.json'):
        raise ValueError('observed control changed the B0 physical-prior bytes')
    return dict(method='OBSERVED_SURFACE_TSDF',receipt_path=str(receipt_path.resolve()),receipt_sha256=file_hash(receipt_path),
        b0_manifest_sha256=receipt['b0_manifest_sha256'],capture_manifest_sha256=receipt['capture_manifest_sha256'],
        heldout_access=False,artifact_hashes=receipt['artifact_hashes'],
        estimated_alignment='frozen TSDF local frame from public observation mean; no GT recentering',
        physical_prior='byte-identical B0 mass/friction configuration',completion=receipt['completion'])
