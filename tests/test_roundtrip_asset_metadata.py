import json
import numpy as np
import pytest
import trimesh
from robo.roundtrip.asset_metadata import exported_dimensions,correct_legacy_orphan_dimensions


@pytest.fixture
def legacy(tmp_path):
    root=tmp_path/'source';root.mkdir();mesh=trimesh.creation.box([.1,.2,.3]);mesh.export(root/'mesh_sim.obj')
    mesh.vertices=np.vstack([mesh.vertices,[[1.,2.,3.],[4.,5.,6.]]]);mesh.export(root/'mesh_sim.ply')
    aligned=dict(T=np.eye(4).tolist(),scale=1.,world_dims=np.ptp(mesh.vertices,axis=0).tolist())
    (root/'aligned.json').write_text(json.dumps(aligned));(root/'physics.json').write_text('{}')
    return root


def test_orphans_only_change_proven_dimension_metadata(legacy,tmp_path):
    before=(legacy/'aligned.json').read_bytes();receipt=correct_legacy_orphan_dimensions(legacy,tmp_path/'corrected')
    assert receipt['orphan_vertices']==2 and receipt['changed_files']==['aligned.json']
    assert receipt['changed_fields']==['world_dims'] and receipt['controller_action'] is False
    assert (legacy/'aligned.json').read_bytes()==before
    np.testing.assert_allclose(receipt['new_world_dims'],[.1,.2,.3])
    with pytest.raises(FileExistsError):correct_legacy_orphan_dimensions(legacy,tmp_path/'corrected')


@pytest.mark.parametrize('bad',['referenced_surface','stored_dims','no_orphans','scale'])
def test_correction_fails_closed_for_other_causes(legacy,tmp_path,bad):
    if bad in ('referenced_surface','no_orphans'):
        mesh=trimesh.load(legacy/'mesh_sim.ply',process=False)
        if bad=='referenced_surface':mesh.vertices[0,0]+=.01
        else:mesh.remove_unreferenced_vertices()
        mesh.export(legacy/'mesh_sim.ply')
    elif bad=='stored_dims':
        p=legacy/'aligned.json';d=json.loads(p.read_text());d['world_dims'][0]*=100;p.write_text(json.dumps(d))
    else:
        with pytest.raises(ValueError):exported_dimensions(legacy,-1)
        return
    with pytest.raises(ValueError):correct_legacy_orphan_dimensions(legacy,tmp_path/'wrong')
    assert not (tmp_path/'wrong').exists()


def test_render_override_binds_old_geometry_to_metadata_copy(legacy,tmp_path):
    from robo.roundtrip.fidelity_native import metadata_render_methods
    proof=correct_legacy_orphan_dimensions(legacy,tmp_path/'corrected');path=tmp_path/'receipt.json';path.write_text(json.dumps(proof))
    row=dict(method='B0_FIXED_NATIVE',object_dir=str(legacy));frozen={row['method']:proof['parent_hashes']}
    bound,overrides=metadata_render_methods([row],frozen,[path])
    assert bound[0]['object_dir']==str(tmp_path/'corrected') and row['object_dir']==str(legacy)
    assert overrides[0]['evaluated_artifact_hashes']==proof['parent_hashes']
    assert overrides[0]['import_artifact_hashes']==proof['child_hashes']
    with pytest.raises(ValueError,match='ambiguous'):metadata_render_methods([row],frozen,[path,path])
    with pytest.raises(ValueError,match='unused'):metadata_render_methods([],{},[path])


@pytest.mark.parametrize('bad',['mesh','physics','transform','scale','dimensions','unknown_file'])
def test_render_correction_rejects_mutations_even_resealed(legacy,tmp_path,bad):
    from robo.roundtrip.asset_metadata import validate_metadata_correction
    import hashlib
    proof=correct_legacy_orphan_dimensions(legacy,tmp_path/'corrected');child=tmp_path/'corrected'
    if bad in ('transform','scale','dimensions'):
        p=child/'aligned.json';d=json.loads(p.read_text())
        if bad=='transform':d['T'][0][3]+=.01
        elif bad=='scale':d['scale']*=100
        else:d['world_dims'][0]*=100
        p.write_text(json.dumps(d))
    else:
        p=child/({'mesh':'mesh_sim.obj','physics':'physics.json','unknown_file':'secret.json'}[bad]);p.write_text('changed')
    proof['child_hashes']={str(p.relative_to(child)):hashlib.sha256(p.read_bytes()).hexdigest() for p in child.rglob('*') if p.is_file()}
    path=tmp_path/'receipt.json';path.write_text(json.dumps(proof))
    with pytest.raises(ValueError):validate_metadata_correction(path)
