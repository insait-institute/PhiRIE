"""Full canonical multi-scene inventories export rejected and empty scenes."""
import copy
import json

import numpy as np
import pytest
import yaml

from robo.eval import agentic_ablation as e3
from robo.eval import e3_factory_materializer as materializer
from tests.test_agentic_automatic_population import population
from tests.test_e3_factory_materializer import _write_json, _reseal_inventory


@pytest.fixture
def multi_factory_input(tmp_path, monkeypatch):
    from agents.core import common
    from plyfile import PlyData, PlyElement

    def forbidden(*args, **kwargs):
        raise AssertionError("GT/implicit instance loader must not run")
    monkeypatch.setattr(common, "load_gt_instances", forbidden)
    monkeypatch.setattr(common, "load_instances", forbidden)
    monkeypatch.setattr(e3, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(materializer, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(materializer, "_require_clean_code_snapshot", lambda: {
        "commit": "d"*40, "dirty": False, "status": [], "code_root": str(materializer.CODE_ROOT)})
    jobs, contract = population.__wrapped__(tmp_path)
    descriptors = {}
    for source in jobs["automatic_sources"]:
        from pathlib import Path
        directory = Path(source["discovery_directory"])
        construction = directory/"construction"
        count = source["planned_jobs"]
        base = np.array([[0,0,0], [.1,0,0], [0,.1,0], [0,0,.1]])
        vertices = np.vstack([base+[i,0,0] for i in range(max(count, 1))])
        base_faces = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]])
        faces = np.vstack([base_faces+4*i for i in range(max(count, 1))])
        ply_vertices = np.array([tuple(v) for v in vertices], dtype=[("x","f4"),("y","f4"),("z","f4")])
        ply_faces = np.array([(face,) for face in faces], dtype=[("vertex_indices","i4",(3,))])
        PlyData([PlyElement.describe(ply_vertices,"vertex"), PlyElement.describe(ply_faces,"face")], text=True).write(construction/"derived_mesh.ply")
        np.savez(construction/"auto_instances.npz", labels=np.asarray(["unprepared"]*count, dtype="U16"),
                 scores=np.full(count, .8), **{f"vert_idx_{i}": np.arange(4*i,4*i+4) for i in range(count)})
        index = {str(path.relative_to(construction)): {
            "path":str(path), "bytes":path.stat().st_size, "sha256":e3.sha256_file(path)}
            for path in construction.rglob("*") if path.is_file()}
        _write_json(directory/"output_hashes.json", index)
        complete = json.loads((directory/"all_jobs_manifest.json").read_text())
        complete["output_hashes_sha256"] = e3.sha256_file(directory/"output_hashes.json")
        _write_json(directory/"all_jobs_manifest.json", complete)
        postrun = json.loads((directory/"postrun_audit.json").read_text())
        postrun.update(output_hashes_sha256=complete["output_hashes_sha256"],
                       all_jobs_manifest_sha256=e3.sha256_file(directory/"all_jobs_manifest.json"))
        _write_json(directory/"postrun_audit.json", postrun)
        source["discovery_hashes"] = {name:e3.sha256_file(directory/name) for name in source["discovery_hashes"]}
        descriptor = tmp_path/(source["scene_id"]+"-descriptor.json")
        _write_json(descriptor, {"schema_version":1, **{key:source[key] for key in (
            "scene_id", "discovery_directory", "discovery_hashes")}})
        descriptors[source["scene_id"]] = descriptor
    out = tmp_path/"agentic"
    audit = e3.run_inventory(jobs, contract, "freeze", out)
    policies = yaml.safe_load((materializer.CODE_ROOT/"configs/experiments/icra2027/agentic_automatic_policies.yaml").read_text())
    for scene in audit["scene_roster"]:
        e3.run_observe("freeze", out, scene)
        e3.run_control(policies, contract, "freeze", out, scene)
    return tmp_path, out, descriptors, audit


def test_canonical_multiscene_export_retains_empty_and_rejected_scenes(multi_factory_input):
    root, out, descriptors, audit = multi_factory_input
    assert audit["counts"] == {"scenes":2, "jobs":3, "policy_object_rows":15}
    for scene, count in zip(audit["scene_roster"], [3, 0], strict=True):
        for policy, action in [("A0", "rejected"), ("A4", "abstained")]:
            destination = root/f"materialized-{scene}-{policy}"
            result = materializer.materialize_factory_variant(e3_root=out, scene_id=scene, policy_id=policy,
                out=destination, automatic_scene_contract=descriptors[scene])
            assert result["roster"]["job_count"] == count
            assert result["roster"][action+"_count"] == count
            assert result["roster"]["accepted_count"] == 0
            assert len(json.loads((destination/"objects/objects.json").read_text())) == count
            assert not list(destination.rglob("*.urdf"))
            materializer.validate_materialized_factory(destination, expected_scene_id=scene, expected_policy_id=policy)
    assert not (out/"aggregate_seal.json").exists()


@pytest.mark.parametrize("mutation", ["swap_entries", "missing_empty", "missing_requested", "drop_empty_roster",
                                      "reorder_roster", "counts", "entry_freeze", "entry_job", "malformed_job", "global_provenance"])
def test_multiscene_audit_drift_rejected_before_export(multi_factory_input, mutation):
    root, out, descriptors, original = multi_factory_input
    audit = copy.deepcopy(original)
    first, second = audit["scene_roster"]
    if mutation == "swap_entries":
        audit["scene_audits"][first], audit["scene_audits"][second] = audit["scene_audits"][second], audit["scene_audits"][first]
    elif mutation == "missing_empty": audit["scene_audits"].pop(second)
    elif mutation == "missing_requested": audit["scene_audits"].pop(first)
    elif mutation == "drop_empty_roster": audit["scene_roster"].pop()
    elif mutation == "reorder_roster": audit["scene_roster"].reverse()
    elif mutation == "counts": audit["counts"]["scenes"] = 1
    elif mutation == "entry_freeze": audit["scene_audits"][first]["freeze_id"] = "other"
    elif mutation == "entry_job": audit["scene_audits"][first]["jobs"][0]["source_job_id"] = second+":auto:1000"
    elif mutation == "malformed_job": audit["scene_audits"][first]["jobs"][0] = "not-a-job-record"
    else: audit["source_gaussian_training_provenance"] = "UNKNOWN"
    _write_json(out/"input_inventory/inventory_audit.json", audit)
    _reseal_inventory(out)
    with pytest.raises(ValueError, match="automatic"):
        materializer.materialize_factory_variant(e3_root=out, scene_id=first, policy_id="A0",
            out=root/"bad", automatic_scene_contract=descriptors[first])
    assert not (root/"bad").exists()


def test_multiscene_descriptor_cannot_be_swapped(multi_factory_input):
    root, out, descriptors, audit = multi_factory_input
    first, second = audit["scene_roster"]
    with pytest.raises(ValueError, match="scope/scene/hash"):
        materializer.materialize_factory_variant(e3_root=out, scene_id=first, policy_id="A4",
            out=root/"bad", automatic_scene_contract=descriptors[second])
    assert not (root/"bad").exists()
