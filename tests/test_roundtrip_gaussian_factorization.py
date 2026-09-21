import numpy as np
import pytest
import json
from pathlib import Path
from robo.roundtrip.gaussian_factorization import checked_file, observed_vertex_indices
from robo.roundtrip.gaussian_build import sha
from tests.test_roundtrip_gaussian_build import capture


def test_existing_auto_mapping_and_far_rejection():
    points=np.stack([np.arange(12)*.01,np.zeros(12),np.zeros(12)],axis=1)
    vertices=np.concatenate([points,points+10])
    assert np.array_equal(observed_vertex_indices(vertices,points),np.arange(12))
    with pytest.raises(ValueError,match='fewer than ten'):
        observed_vertex_indices(vertices,points+100)


def test_nonfinite_mesh_rejected():
    with pytest.raises(ValueError,match='nonfinite'):
        observed_vertex_indices(np.full((20,3),np.nan),np.zeros((20,3)))


def test_object_source_hash_and_path_boundary(tmp_path):
    path=tmp_path/'observed.npy';path.write_bytes(b'observed')
    manifest={'source_hashes':{'observed.npy':sha(path)}}
    assert checked_file(tmp_path,manifest,'observed.npy')==path
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='unbound'):checked_file(tmp_path,manifest,'observed.npy')
    with pytest.raises(ValueError,match='unbound'):checked_file(tmp_path,manifest,'../private.xml')


def test_symlink_source_rejected(tmp_path):
    path=tmp_path/'data';path.write_bytes(b'x');(tmp_path/'link').symlink_to(path)
    with pytest.raises(ValueError,match='unbound'):
        checked_file(tmp_path,{'source_hashes':{'link':sha(path)}},'link')


def test_actual_synthetic_tsdf_and_existing_removal(capture,tmp_path,monkeypatch):
    pytest.importorskip('open3d')
    import trimesh
    from plyfile import PlyData,PlyElement
    from robo.roundtrip.gaussian_build import capture_rows
    from robo.roundtrip import gaussian_factorization as module
    public,rows=capture_rows(capture)
    source=tmp_path/'object';source.mkdir();gs=tmp_path/'gs';gs.mkdir()
    (source/'build_manifest.json').write_text('{}');(gs/'gs_build_manifest.json').write_text('{}')
    xyz=np.array([[x,y,2.] for x in np.linspace(-.2,.2,20) for y in np.linspace(-.2,.2,20)])
    vertices=np.empty(len(xyz),dtype=[(k,'f4') for k in 'xyz'])
    for i,k in enumerate('xyz'):vertices[k]=xyz[:,i]
    ply=gs/'source.ply';PlyData([PlyElement.describe(vertices,'vertex')]).write(str(ply))
    trimesh.creation.box([.4,.4,.01]).export(source/'trellis_mesh.ply')
    (source/'trellis_gs.ply').write_bytes(ply.read_bytes())
    T=np.eye(4);T[2,3]=2
    (source/'aligned.json').write_text(json.dumps({'T':T.tolist()}))
    paths={'construction/objects/obj_00/'+key:source/key
           for key in ['trellis_mesh.ply','trellis_gs.ply','aligned.json']}
    # Fixture source identity only; real TSDF, mesh raycasting, seeded asset
    # sampling, plane fit and removal producer run without renderer doubles.
    monkeypatch.setattr(module,'source_inputs',lambda *a:(public,rows,ply,
        {'target_prompt':'synthetic_object','object_id':'observed_0','adapter_mean_sam_score':.9},xyz,paths))
    result=module.prepare(capture,gs,source,tmp_path/'out')
    assert result['preparation']['planned_objects']==1
    assert result['observed_mesh_faces']>0 and result['target_mesh_vertices']>=10
    assert (tmp_path/'out/preparation/removal_union_idx.npy').is_file()
    assert result['clean_background']=='NOT_RUN'
    with pytest.raises(FileExistsError):module.prepare(capture,gs,source,tmp_path/'out')
    # The tiny 16px synthetic camera's mapped surface has subpixel holes.
    # Provide declared synthetic masks to exercise erasure independently;
    # preserve the preceding real TSDF/removal test and reseal this fixture.
    mask_path=tmp_path/'out/preparation/obj_1000/proj_masks.npz'
    with np.load(mask_path) as packed:frames=packed['frames'];shape=packed['masks'].shape
    masks=np.zeros(shape,bool);masks[:,6:10,6:10]=True
    np.savez_compressed(mask_path,frames=frames,masks=masks)
    receipt_path=tmp_path/'out/factorization_preparation.json'
    receipt=json.loads(receipt_path.read_text())
    receipt['output_files']['preparation/obj_1000/proj_masks.npz']=sha(mask_path)
    receipt_path.write_text(json.dumps(receipt))
    from PIL import Image
    checkpoint=tmp_path/'test-model';checkpoint.write_bytes(b'synthetic test model identity')
    def model(image,mask):return Image.new('RGB',image.size,(128,0,128))
    erased=module.erase(capture,gs,source,tmp_path/'out',sha(tmp_path/'out/factorization_preparation.json'),
                        checkpoint,sha(checkpoint),tmp_path/'erased',model=model)
    assert erased['planned_views']==erased['erased_views']>0
    assert all(row['outside_mask_equal'] for row in erased['rows'])
    erasure_path=tmp_path/'erased/native_erasure.json'
    assert module.sealed_erasure(tmp_path/'erased',sha(erasure_path),sha(receipt_path))['status']=='COMPLETE'
    from agents.edit import inpaint_fill
    def fake_fill(public):
        staged=json.loads((public['factory']/'objects/objects.json').read_text())
        assert np.allclose(staged[0]['aabb'],[xyz.min(axis=0),xyz.max(axis=0)])
        assert 'aabb' not in json.loads((tmp_path/'out/factory/objects/objects.json').read_text())[0]
        (public['output']/'clean_background.ply').write_bytes(b'synthetic stage handoff')
        return {'filled_objects':1,'iterations':1500}
    monkeypatch.setattr(inpaint_fill,'_fill',fake_fill)
    monkeypatch.setattr(inpaint_fill,'_validate_fill_products',lambda *a:None)
    filled=module.fill(capture,gs,source,tmp_path/'out',sha(receipt_path),
                      tmp_path/'erased',sha(erasure_path),tmp_path/'fill')
    assert filled['status']=='BUILT' and filled['object_motion']=='NOT_RUN'
    with pytest.raises(FileExistsError):
        module.fill(capture,gs,source,tmp_path/'out',sha(receipt_path),
                    tmp_path/'erased',sha(erasure_path),tmp_path/'fill')
    image_path=tmp_path/'erased'/erased['rows'][0]['erased_rgb']
    image_path.write_bytes(b'changed artifact')
    with pytest.raises(ValueError,match='output bytes changed'):
        module.sealed_erasure(tmp_path/'erased',sha(erasure_path),sha(receipt_path))
    def failed_model(*args):raise RuntimeError('injected enhancer failure')
    failed=module.erase(capture,gs,source,tmp_path/'out',sha(tmp_path/'out/factorization_preparation.json'),
                        checkpoint,sha(checkpoint),tmp_path/'failed',model=failed_model)
    assert failed['planned_views']==erased['planned_views'] and failed['erased_views']==0
    assert all(row['status']=='ENHANCER_FAILED' and row['erased_rgb'] is None for row in failed['rows'])
    with pytest.raises(ValueError,match='incomplete erasure'):
        module.sealed_erasure(tmp_path/'failed',sha(tmp_path/'failed/native_erasure.json'),sha(receipt_path))
