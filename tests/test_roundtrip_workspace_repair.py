import json
import pytest
from robo.roundtrip.workspace_repair import repair
from robo.roundtrip.identity import file_hash
from robo.roundtrip.scope_bundle import artifact_hashes


def source(tmp_path):
    import trimesh
    p=tmp_path/'source';(p/'collision').mkdir(parents=True)
    for n in ('aligned.json','physics.json'): (p/n).write_text('{}')
    mesh=trimesh.creation.box();mesh.export(p/'mesh_sim.obj')
    # A volumetric surface with one omitted triangle is invalid collision,
    # but convex regeneration must create a genuine new closed artifact.
    broken=trimesh.Trimesh(vertices=mesh.vertices,faces=mesh.faces[:-1],process=False)
    broken.export(p/'collision/part_00.obj')
    (p/'workspace_component_receipt.json').write_text(json.dumps(dict(
        method='OBSERVED_WORKSPACE_COMPONENT_TSDF',status='BUILT',role='source_support',
        cluster_id='source_support-cluster-0000',artifact_hashes=artifact_hashes(p))))
    return p,file_hash(p/'workspace_component_receipt.json')


def test_explicit_new_artifact_preserves_parent_and_noncollision(tmp_path):
    p,sha=source(tmp_path);before=artifact_hashes(p)
    r=repair(p,tmp_path/'repair',receipt_sha256=sha)
    assert r['status']=='REPAIRED_NOT_ADMITTED' and not r['L2_READY']
    assert artifact_hashes(p)==before
    after=artifact_hashes(tmp_path/'repair/object')
    assert after['collision/part_00.obj']!=before['collision/part_00.obj']
    assert all(after[k]==before[k] for k in ('aligned.json','physics.json','mesh_sim.obj'))
    assert not (tmp_path/'repair/object/workspace_component_receipt.json').exists()
    with pytest.raises(FileExistsError):repair(p,tmp_path/'repair',receipt_sha256=sha)


def test_tamper_rejected_before_output(tmp_path):
    p,sha=source(tmp_path);(p/'physics.json').write_text('{"mass":999}')
    with pytest.raises(ValueError,match='closure'):repair(p,tmp_path/'repair',receipt_sha256=sha)
    assert not (tmp_path/'repair').exists()
