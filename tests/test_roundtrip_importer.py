"""Native importer engineering controls; these are not reconstruction outcomes."""
import json
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robo.roundtrip.importers.robocasa import (
    decompose_alignment, import_identity, import_reconstructed_object, rebind_native_object,
)


XML = '''<mujoco><option timestep="0.002"/><asset>
<material name="original_mat" rgba="0.1 0.3 0.8 1"/></asset><worldbody>
<geom name="floor" type="plane" size="5 5 .1"/>
<body name="obj_main" pos="3 2 1" quat=".9238795325 0 0 .3826834324">
<freejoint name="obj_joint"/><inertial pos=".01 -.02 .03" mass=".8" diaginertia=".01 .02 .025"/>
<geom name="obj_old_collision" type="box" size=".1 .2 .3" friction=".7 .02 .001" material="original_mat"/>
<body name="obj_child" pos=".02 .03 .04"><geom name="obj_old_visual" type="sphere" size=".02" contype="0" conaffinity="0"/></body>
<site name="obj_site" pos=".1 .2 .3"/></body>
<body name="robot" pos="0 0 1"><geom name="robot_geom" type="sphere" size=".1"/></body>
</worldbody></mujoco>'''


@pytest.fixture
def factory(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box([.05, .08, .12])
    mesh.apply_translation([.12, -.07, .03])  # deliberately off-center/asymmetric
    mesh.export(tmp_path / "mesh_sim.obj")
    (tmp_path / "collision").mkdir()
    mesh.export(tmp_path / "collision" / "part_00.obj")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [21, -37, 53], degrees=True).as_matrix() * 1.7
    transform[:3, 3] = [.41, -.25, 1.13]
    (tmp_path / "aligned.json").write_text(json.dumps({"T": transform.tolist(), "scale": 1.7}))
    (tmp_path / "physics.json").write_text(json.dumps({"mass_kg": .3, "friction": .5, "source": "frozen_prior"}))
    return tmp_path


def imported(factory, **kwargs):
    return import_reconstructed_object(XML, body_name="obj_main", object_dir=factory, object_id="observed_0", **kwargs)


def test_identity_preserves_compiled_physics_and_continuous_settle():
    mujoco = pytest.importorskip("mujoco")
    output, receipt = import_identity(XML, body_name="obj_main")
    assert receipt["body_xml_equal"]
    original, copied = [mujoco.MjModel.from_xml_string(x) for x in (XML, output)]
    for field in ("body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat",
                  "geom_pos", "geom_quat", "geom_size", "geom_friction", "geom_contype",
                  "geom_conaffinity", "geom_solimp", "geom_solref", "geom_matid", "mat_rgba"):
        np.testing.assert_array_equal(getattr(original, field), getattr(copied, field))
    states = [mujoco.MjData(model) for model in (original, copied)]
    for _ in range(100):
        for model, data in zip((original, copied), states):
            mujoco.mj_step(model, data)
        np.testing.assert_array_equal(states[0].qpos, states[1].qpos)
        np.testing.assert_array_equal(states[0].qvel, states[1].qvel)


def test_offcenter_scale_quaternion_and_no_native_geometry(factory):
    mujoco = pytest.importorskip("mujoco")
    output, receipt = imported(factory)
    model = mujoco.MjModel.from_xml_string(output)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = model.body("obj_main").id
    alignment = json.loads((factory / "aligned.json").read_text())
    transform = np.asarray(alignment["T"])
    np.testing.assert_allclose(data.xpos[body], transform[:3, 3])
    np.testing.assert_allclose(data.xmat[body].reshape(3, 3), transform[:3, :3] / 1.7)
    np.testing.assert_allclose(model.body_ipos[body], np.array([.12, -.07, .03]) * 1.7)
    np.testing.assert_allclose(data.xipos[body], transform[:3, :3] @ [.12, -.07, .03] + transform[:3, 3])
    assert model.body_mass[body] == .3
    assert receipt["scale_applications"] == 1
    assert receipt["removed_original_geoms"] == ["obj_old_collision", "obj_old_visual"]
    for name in receipt["removed_original_geoms"]:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) == -1
    assert model.geom("robot_geom").id >= 0
    assert model.geom("floor").id >= 0
    # MuJoCo recenters meshes internally; use its compiled mesh transform to
    # verify actual world vertices, not a misleading mesh bounding-box check.
    geom = model.geom(receipt["contact_geoms"][0]).id
    mesh_id = model.geom_dataid[geom]
    start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    vertices = model.mesh_vert[start:start + count]
    world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    source = pytest.importorskip("trimesh").load(factory / "mesh_sim.obj", process=False).vertices
    expected = source @ transform[:3, :3].T + transform[:3, 3]
    from scipy.spatial import cKDTree
    assert cKDTree(expected).query(world)[0].max() < 1e-6


@pytest.mark.parametrize("case", ["double_scale", "anisotropic", "reflection", "nan", "rejected"])
def test_bad_alignment_rejected(factory, case):
    path = factory / "aligned.json"
    data = json.loads(path.read_text())
    if case == "double_scale":
        data["scale"] *= 100
    elif case == "anisotropic":
        data["T"][0][0] *= 2
    elif case == "reflection":
        for row in data["T"][:3]:
            row[0] *= -1
    elif case == "nan":
        data["T"][0][0] = float("nan")
    else:
        data["rejected"] = "evidence unavailable"
    with pytest.raises(ValueError):
        decompose_alignment(data)


def test_same_alignment_is_not_100x_centimeter_scale(factory):
    _, receipt = imported(factory)
    low, high = np.array(receipt["bounds_local_m"])
    np.testing.assert_allclose(high - low, np.array([.05, .08, .12]) * 1.7)
    assert not np.allclose(high - low, np.array([.05, .08, .12]) * 170)


@pytest.mark.parametrize("case", ["missing_collision", "missing_texture", "negative_mass", "articulated_role", "inflation"])
def test_missing_or_unsupported_assets_fail(factory, case):
    args = {}
    if case == "missing_collision":
        (factory / "collision" / "part_00.obj").unlink()
    elif case == "missing_texture":
        args["texture_path"] = factory / "absent.png"
    elif case == "negative_mass":
        (factory / "physics.json").write_text('{"mass_kg":-1,"friction":0.5}')
    elif case == "articulated_role":
        args["role"] = "articulated_door"
    else:
        args["contact_prior"] = {"minimum_thickness_m": .001}
    with pytest.raises((ValueError, FileNotFoundError)):
        imported(factory, **args)


def test_old_external_contact_dependency_rejected(factory):
    xml = XML.replace("</mujoco>", '<contact><pair geom1="obj_old_collision" geom2="floor"/></contact></mujoco>')
    with pytest.raises(ValueError, match="external contact"):
        import_reconstructed_object(xml, body_name="obj_main", object_dir=factory, object_id="x")


def test_native_binding_uses_reconstructed_geometry(factory):
    _, receipt = imported(factory)
    class Object:
        root_body = "obj_main"
        naming_prefix = "obj_"
        _regions = {"bbox": {"elem": ET.Element("geom", size="100 100 100")}}
        @property
        def contact_geoms(self):
            return [self.naming_prefix + x for x in self._contact_geoms]
    model = SimpleNamespace(geom_name2id=lambda n: receipt["contact_geoms"].index(n)
        if n in receipt["contact_geoms"] else receipt["visual_geoms"].index(n), body_name2id=lambda n: 7)
    obj = Object()
    env = SimpleNamespace(objects={"obj": obj}, obj_body_id={"obj": 999}, sim=SimpleNamespace(model=model))
    binding = rebind_native_object(env, object_name="obj", receipt=receipt)
    assert env.obj_body_id["obj"] == 7
    assert obj.contact_geoms == receipt["contact_geoms"]
    assert binding["target_bounds_source"] == "reconstructed_mesh"
    assert np.max(np.fromstring(obj._regions["bbox"]["elem"].get("size"), sep=" ")) < .11


def test_no_estimated_pose_restore_and_no_robot_change(factory):
    output, receipt = imported(factory)
    before, after = ET.fromstring(XML), ET.fromstring(output)
    assert ET.tostring(before.find("./worldbody/body[@name='robot']")) == ET.tostring(after.find("./worldbody/body[@name='robot']"))
    assert receipt["position_m"] != [3, 2, 1]
    assert receipt["position_m"] == [.41, -.25, 1.13]
