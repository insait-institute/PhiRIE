import copy
import json
from pathlib import Path
import pytest
from robo.roundtrip.observed_native import validate_observed_control
from robo.roundtrip.scope_bundle import artifact_hashes
from robo.roundtrip.identity import file_hash


def test_observed_native_requires_exact_source_geometry_and_physics(tmp_path,monkeypatch):
    import robo.roundtrip.shared_candidates as shared
    b0=tmp_path/'b0';obj=tmp_path/'observed';obj.mkdir();(b0/'construction/objects/obj_00').mkdir(parents=True);(b0/'discovery').mkdir()
    for name in ['aligned.json','physics.json','mesh_sim.obj','collision/part_00.obj']:
        p=obj/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
    (b0/'construction/objects/obj_00/physics.json').write_bytes((obj/'physics.json').read_bytes())
    config={'canonical_instance_id':'native-a','cohort_id':'dev'}
    manifest=dict(config,capture_manifest_sha256='a'*64);(b0/'build_manifest.json').write_text(json.dumps(manifest))
    candidates=[dict(candidate_id='c0',frame_id='f0',mask_sha256='b'*64)];(b0/'discovery/candidates.json').write_text(json.dumps(candidates))
    monkeypatch.setattr(shared,'checked_b0',lambda b,c:(manifest,{'accepted_candidates':['c0']}))
    receipt=dict(config,schema_version=2,method='OBSERVED_SURFACE_TSDF',status='BUILT',scope='L0_target_only',object_role='target',heldout_access=False,
        config={'voxel_m':.002,'sdf_trunc_m':.006},b0_manifest_sha256=file_hash(b0/'build_manifest.json'),capture_manifest_sha256='a'*64,source_views=candidates,
        artifact_hashes=artifact_hashes(obj),completion='partial visual, convex collision approximation')
    path=obj/'observed_surface_receipt.json';path.write_text(json.dumps(receipt))
    assert validate_observed_control(path,obj,b0,tmp_path/'capture',config)['heldout_access'] is False
    changed=copy.deepcopy(receipt);changed['heldout_access']=True;path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='TRAIN-only'):validate_observed_control(path,obj,b0,tmp_path/'capture',config)
    path.write_text(json.dumps(receipt));(obj/'mesh_sim.obj').write_text('changed after freeze')
    with pytest.raises(ValueError,match='closure changed'):validate_observed_control(path,obj,b0,tmp_path/'capture',config)
