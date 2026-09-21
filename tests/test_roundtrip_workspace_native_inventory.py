import numpy as np
import pytest
import mujoco
from robo.roundtrip.workspace_native_inventory import collision_inventory


def test_compiled_rotated_box_plane_and_explicit_pair_are_inventory_only():
    xml='''<mujoco><worldbody><geom name="floor" type="plane" size="2 2 .1"/><body pos="1 2 3" euler="0 0 90"><geom name="inside" type="box" size=".1 .2 .3"/><geom name="far" type="sphere" pos="5 0 0" size=".1" contype="0" conaffinity="0"/></body><geom name="visual" type="sphere" pos="0 0 10" size=".1" contype="0" conaffinity="0"/></worldbody><contact><pair geom1="inside" geom2="far"/></contact></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d);before=d.qpos.copy()
    rows=collision_inventory(m,d,[[.7,1.7,2.6],[1.3,2.3,3.4]]);by={r['native_geom']:r for r in rows}
    assert set(by)=={'floor','inside','far'} and by['far']['explicit_contact_pair']
    assert by['inside']['contained_in_declared_workspace'] and not by['far']['intersects_declared_workspace']
    assert not by['floor']['intersects_declared_workspace'] and by['floor']['canonical_world_aabb_m'] is None
    assert np.allclose(by['inside']['canonical_world_aabb_m'],[[.8,1.9,2.7],[1.2,2.1,3.3]])
    assert np.array_equal(before,d.qpos)
    with pytest.raises(ValueError):collision_inventory(m,d,[[0,0,0],[0,0,0]])


def test_constructor_tampering_fails_before_native_binding_is_opened(tmp_path):
    import json
    from robo.roundtrip.identity import file_hash
    from robo.roundtrip.workspace_native_inventory import audit
    inv=tmp_path/'inventory.json';inv.write_text('{}');d=tmp_path/'support';d.mkdir();(d/'mesh_sim.obj').write_text('changed')
    c=tmp_path/'workspace_components.json';c.write_text(json.dumps(dict(tier='DEV',method='OBSERVED_WORKSPACE_COMPONENT_TSDF',inventory_manifest_sha256=file_hash(inv),rows=[dict(status='BUILT',cluster_id='support',artifact_hashes={'mesh_sim.obj':'0'*64})])))
    with pytest.raises(ValueError,match='before native audit'):audit(c,inv,tmp_path/'native_binding_MUST_NOT_OPEN.json',tmp_path/'out')
    assert not (tmp_path/'out').exists()
