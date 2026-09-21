import json
import numpy as np
import pytest
from robo.roundtrip.fidelity_native import estimated_surface, visual_surface


def test_fidelity_split_binding_preserves_dev_and_rejects_cross_split():
    from robo.roundtrip.fidelity_native import fidelity_tier
    assert fidelity_tier({'tier':'DEV'},{'dataset_split':'development'})=='DEV'
    assert fidelity_tier({'tier':'TEST'},{'dataset_split':'test'})=='TEST'
    for build,canonical in [({'tier':'DEV'},{'dataset_split':'test'}),
                            ({'tier':'TEST'},{'dataset_split':'development'}),
                            ({'tier':'TEST'},{'dataset_split':'unknown'})]:
        with pytest.raises(ValueError,match='split differs'):
            fidelity_tier(build,canonical)


def test_context_geometry_consumes_only_bound_accepted_artifacts(tmp_path):
    from robo.roundtrip.fidelity_native import context_geometry_rows, file_hash
    from robo.roundtrip.system_verification import asset_identity
    obj=tmp_path/'asset';obj.mkdir();(obj/'collision').mkdir()
    for n in ('aligned.json','physics.json','mesh_sim.obj'):(obj/n).write_text(n)
    bundle=tmp_path/'native';bundle.mkdir()
    for n in ('scene.xml','canonical_state.json'):(bundle/n).write_text(n)
    b0=tmp_path/'b0.json';b0.write_text('b0');pool=tmp_path/'pool.json';pool.write_text('pool')
    binding=dict(canonical_instance_id='one',capture_manifest_sha256='capture',bundle_dir=str(bundle))
    m=dict(method='B4',accepted=True,selected_asset=str(obj),selected_asset_identity=asset_identity(obj),
        capture_manifest_sha256='capture',candidate_pool_sha256=file_hash(pool),
        public_observation_provenance=dict(build_manifest_sha256=file_hash(b0)),
        source_hashes={str(p):file_hash(p) for p in bundle.iterdir()})
    manifest=tmp_path/'context.json';manifest.write_text(json.dumps(m))
    row=dict(canonical_instance_id='one',controller_method='B4_ROOM_REPAIR_NATIVE',accepted=True,
        terminal_status='READY',object_dir=str(obj),build_manifest=str(manifest),build_manifest_sha256=file_hash(manifest))
    path=tmp_path/'bindings.jsonl';path.write_text(json.dumps(row)+'\n')
    assert context_geometry_rows(path,binding,b0,pool)[0][0]['object_dir']==str(obj)
    (obj/'mesh_sim.obj').write_text('changed')
    with pytest.raises(ValueError,match='asset changed'):context_geometry_rows(path,binding,b0,pool)
    (obj/'mesh_sim.obj').write_text('mesh_sim.obj')
    m['accepted']=False;manifest.write_text(json.dumps(m));row.update(accepted=False,terminal_status='ABSTAINED',build_manifest_sha256=file_hash(manifest));path.write_text(json.dumps(row)+'\n')
    assert context_geometry_rows(path,binding,b0,pool)[0]==[]
    m['capture_manifest_sha256']='another';manifest.write_text(json.dumps(m));row['build_manifest_sha256']=file_hash(manifest);path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='lineage differs'):context_geometry_rows(path,binding,b0,pool)


def test_alignment_is_applied_once(tmp_path):
    trimesh=pytest.importorskip('trimesh');m=trimesh.creation.box([1,2,3]);m.export(tmp_path/'mesh_sim.obj')
    T=np.diag([.1,.1,.1,1.]);T[:3,3]=[1,2,3]
    (tmp_path/'aligned.json').write_text(json.dumps({'T':T.tolist(),'scale':.1}))
    out=estimated_surface(tmp_path)
    assert np.allclose(out.bounds,[[.95,1.9,2.85],[1.05,2.1,3.15]])
    (tmp_path/'aligned.json').write_text(json.dumps({'T':T.tolist(),'scale':1}))
    with pytest.raises(ValueError,match='scale disagrees'):estimated_surface(tmp_path)


def test_native_compiled_visual_surface_frame_and_children(tmp_path):
    mujoco=pytest.importorskip('mujoco');trimesh=pytest.importorskip('trimesh')
    mesh=trimesh.creation.box([.2,.3,.4]);mesh.apply_translation([.5,0,0]);mesh.export(tmp_path/'a.obj')
    xml=f'''<mujoco><asset><mesh name="m" file="{tmp_path/'a.obj'}"/></asset><worldbody><body name="obj_main" pos="1 2 3"><geom type="mesh" mesh="m" group="1"/><geom type="box" size="5 5 5" group="0"/><body pos="0 1 0"><geom type="mesh" mesh="m" group="1"/></body></body></worldbody></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    out,names=visual_surface(model,data,1)
    assert len(names)==2
    assert np.allclose(out.bounds,[[1.4,1.85,2.8],[1.6,3.15,3.2]],atol=1e-7)
    model.geom_group[:]=0
    with pytest.raises(ValueError,match='no declared visual'):visual_surface(model,data,1)


@pytest.mark.parametrize('extra',[{}, {'reference_receipt':'ref','candidate_pool':'pool'}, {'reference_receipt':'ref','context_bindings':'contexts'}])
def test_observed_geometry_requires_independent_cached_surface(tmp_path,monkeypatch,extra):
    pytest.importorskip('mujoco')
    import robo.roundtrip.shared_candidates as shared
    from robo.roundtrip.fidelity_native import evaluate
    monkeypatch.setattr(shared,'checked_b0',lambda b,c:({'canonical_instance_id':'native-one'},{}))
    binding=tmp_path/'binding.json';binding.write_text(json.dumps(dict(canonical_instance_id='native-one',capture_public='capture')))
    with pytest.raises(ValueError,match='independent control arm'):
        evaluate(binding,tmp_path/'b0',tmp_path/'out',observed_control='observed_receipt',**extra)
    assert not (tmp_path/'out').exists()
