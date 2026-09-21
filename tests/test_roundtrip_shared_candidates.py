import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from robo.roundtrip import shared_candidates as shared


@pytest.fixture
def pool(tmp_path, monkeypatch):
    b0=tmp_path/"b0"; obj=b0/"construction/objects/obj_00"
    (obj/"collision").mkdir(parents=True)
    (b0/"registration").mkdir(); (b0/"discovery").mkdir()
    for name in ("mesh_sim.ply","mesh_sim.obj"): (obj/name).write_text("fixed mesh")
    (obj/"collision/part_00.obj").write_text("exact production collision")
    (obj/"physics.json").write_text('{"mass_kg":0.3,"friction":0.5}')
    reg=dict(T=np.eye(4).tolist(),scale=1.,world_dims_m=[1,1,1])
    (obj/"aligned.json").write_text(json.dumps(dict(T=reg["T"],scale=1.,world_dims=[1,1,1])))
    evidence=dict(schema_frame_unit_valid=True,scale_ratio_vs_observation=1.,
        symmetric_clipped_registration_residual_m=.02,observation_point_count=300,
        visible_fraction=.01,support_gap_m=0.,support_overlap_fraction=1.,
        initial_penetration_m=0.,collision_valid=True,usable_convex_parts=1,
        settle_drift_m=0.,settle_sunk=False,settle_stable=True,missing_evidence=[])
    (b0/"registration/evidence.json").write_text(json.dumps(dict(raw_values=evidence)))
    (b0/"registration/registration.json").write_text(json.dumps(reg))
    (b0/"build_manifest.json").write_text("{}")
    np.save(b0/"discovery/observation_points.npy",np.ones((300,3)))
    discovery=dict(capture_manifest_sha256="a"*64,anchor_mask_pixel_fraction=.01,
                   observation_sha256="b"*64,target_prompt="mug")
    monkeypatch.setattr(shared,"checked_b0",lambda *a: ({},discovery))
    rvg=tmp_path/"rvg";rvg.mkdir()
    rr=dict(status="tool_failure",reason="injected generator failure",b0_manifest_sha256=shared._sha(b0/"build_manifest.json"))
    (rvg/"rvg_receipt.json").write_text(json.dumps(rr))
    policy=yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/experiments/icra2027/agentic_policies.yaml").read_text())
    return b0,rvg,reg,evidence,policy


def test_b0_reuses_collision_and_b3_does_not_abstain_on_low_confidence(pool,tmp_path):
    b0,rvg,reg,evidence,policy=pool
    calls=[]
    def aligner(*args,**kw):
        calls.append(kw);T=np.eye(4);T[0,3]=.01
        return dict(alignment=dict(reg,T=T.tolist()),evidence=evidence,wall_s=.1)
    def collision(path,**kw):
        (path/"collision").mkdir();(path/"collision/part_00.obj").write_text("new collision")
        return ["part_00.obj"]
    out=tmp_path/"selected"
    result=shared.select_pool(b0,tmp_path/"capture",rvg,out,policy,"c"*40,aligner=aligner,collision=collision)
    assert len(calls)==1 and calls[0]["signed_source_up"] is True
    assert result["retry_candidate"]["proposal_id"] != result["initial_candidates"][0]["proposal_id"]
    assert result["outcomes"][0]["native_method"]=="B0_FIXED_NATIVE"
    assert result["outcomes"][3]["native_method"]=="B3_AGENT_NATIVE"
    assert result["outcomes"][3]["terminal_action"]=="accept"
    b0dir=Path(result["outcomes"][0]["object_dir"])
    assert (b0dir/"collision/part_00.obj").read_bytes()==(b0/"construction/objects/obj_00/collision/part_00.obj").read_bytes()
    assert result["failures"][0]["reason"]=="injected generator failure"
    with pytest.raises(FileExistsError):
        shared.select_pool(b0,tmp_path/"capture",rvg,out,policy,"c"*40,aligner=aligner,collision=collision)


def test_identical_retry_transform_not_promoted_as_new_candidate(pool,tmp_path):
    b0,rvg,reg,evidence,policy=pool
    result=shared.select_pool(b0,tmp_path/"capture",rvg,tmp_path/"selected",policy,"c"*40,
        aligner=lambda *a,**kw: dict(alignment=reg,evidence=evidence,wall_s=.1),
        collision=lambda *a,**kw: pytest.fail("B0 collision must be reused"))
    assert result["retry_candidate"] is None
    assert result["failures"][-1]["status"]=="no_new_transform"
    assert len({r["object_dir"] for r in result["outcomes"]})==1


def test_other_instance_rvg_rejected(pool,tmp_path):
    b0,rvg,reg,evidence,policy=pool
    (rvg/"rvg_receipt.json").write_text(json.dumps(dict(status="available",b0_manifest_sha256="f"*64)))
    with pytest.raises(ValueError,match="different canonical"):
        shared.select_pool(b0,tmp_path/"capture",rvg,tmp_path/"selected",policy,"c"*40)


def test_capture_and_role_identify_different_dependency_proposals():
    discovery={'capture_manifest_sha256':'a'*64}
    assert shared.proposal_prefix({},discovery)=='a'*20
    assert shared.proposal_prefix({'object_role':'receptacle'},discovery)!='a'*20
    import hashlib
    assert shared.proposal_prefix({'object_role':'support'},discovery)==hashlib.sha256(('a'*64+'|support').encode()).hexdigest()[:20]
    assert len({shared.proposal_prefix({'object_role':role},discovery) for role in ('target','receptacle','support')})==3
    with pytest.raises(ValueError,match='unsupported'):
        shared.proposal_prefix({'object_role':'hidden_native_asset'},discovery)
