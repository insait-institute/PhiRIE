import json
import numpy as np
import pytest
from PIL import Image
from robo.roundtrip import shared_candidates as module


def test_tsdf_real_fusion_and_no_gt_center(tmp_path, monkeypatch):
    pytest.importorskip("open3d")
    b0=tmp_path/"b0";capture=tmp_path/"capture";out=tmp_path/"out"
    (b0/"discovery").mkdir(parents=True);(b0/"construction/objects/obj_00").mkdir(parents=True);capture.mkdir()
    mask=b0/"discovery/c.png";Image.fromarray(np.full((32,32),255,dtype=np.uint8)).save(mask)
    Image.fromarray(np.full((32,32,3),128,dtype=np.uint8)).save(capture/"rgb.png")
    np.save(capture/"depth.npy",np.ones((32,32),dtype=np.float32))
    np.save(b0/"discovery/observation_points.npy",np.array([[.11,.02,1.],[.09,-.02,1.]]))
    (b0/"discovery/candidates.json").write_text(json.dumps([dict(candidate_id="c",frame_id="f",mask_sha256=module._sha(mask))]))
    (b0/"construction/objects/obj_00/physics.json").write_text('{"mass":0.1}')
    (b0/"build_manifest.json").write_text('{}')
    monkeypatch.setattr(module,"checked_b0",lambda *a: ({"canonical_instance_id":"native-test","cohort_id":"dev"},{"accepted_candidates":["c"],"capture_manifest_sha256":"abc"}))
    monkeypatch.setattr(module,"read_train",lambda *a: ({},[dict(frame_id="f",rgb="rgb.png",depth_m="depth.npy",K=[[100,0,16],[0,100,16],[0,0,1]],T_world_from_camera=np.eye(4).tolist())]))
    import agents.assets.s6_physics as physics
    monkeypatch.setattr(physics,"coacd_parts",lambda *a,**kw: ["mock-convex"])
    result=module.observed_surface(b0,capture,out,{"voxel_m":.002,"sdf_trunc_m":.006})
    assert result["visual_mesh_faces"]>4 and not result["heldout_access"]
    assert np.allclose(np.array(json.loads((out/"aligned.json").read_text())["T"])[:3,3],[.1,0,1])
    assert result["policy_outcomes"]=="NOT_RUN"
    with pytest.raises(FileExistsError):module.observed_surface(b0,capture,out,{"voxel_m":.002,"sdf_trunc_m":.006})
    with pytest.raises(ValueError,match="frozen DEV"):module.observed_surface(b0,capture,tmp_path/"other",{"voxel_m":.004,"sdf_trunc_m":.012})
    Image.fromarray(np.zeros((32,32),dtype=np.uint8)).save(mask)
    with pytest.raises(ValueError,match="mask changed"):module.observed_surface(b0,capture,tmp_path/"changed",{"voxel_m":.002,"sdf_trunc_m":.006})


def test_invalid_tsdf_grid():
    pytest.importorskip("open3d")
    from agents.discover.derive_mesh_from_splat import new_tsdf_volume
    with pytest.raises(ValueError):new_tsdf_volume(.006,.002)
