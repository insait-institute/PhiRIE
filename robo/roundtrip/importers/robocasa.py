"""Bounded rigid-target import into the existing native RoboCasa MJCF.

This evaluator-side adapter consumes the existing factory's aligned.json,
physics.json, mesh_sim.obj and CoACD parts. It never resets an estimated pose
to a reference pose. Native room context is retained and explicitly reported.
Construction must already have been isolated and frozen before calling it.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def _numbers(values):
    return " ".join(format(float(x), ".17g") for x in values)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decompose_alignment(aligned):
    """Validate factory Sim(3); return scale, world position, wxyz quaternion."""
    if aligned.get("rejected"):
        raise ValueError("cannot import a rejected construction")
    transform = np.asarray(aligned["T"], dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("alignment T must be a finite 4x4 Sim(3)")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-12, rtol=0):
        raise ValueError("alignment has invalid homogeneous row")
    scale = float(np.cbrt(np.linalg.det(transform[:3, :3])))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and right-handed")
    rotation = transform[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0):
        raise ValueError("alignment must have isotropic scale, no shear")
    if not np.isclose(float(aligned["scale"]), scale, atol=1e-10, rtol=1e-8):
        raise ValueError("aligned scale disagrees with T (possible double scaling)")
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return scale, transform[:3, 3].copy(), xyzw[[3, 0, 1, 2]]


def _body(root, name):
    matches = [body for body in root.iter("body") if body.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one native body {name!r}")
    parents = {child: parent for parent in root.iter() for child in parent}
    return matches[0], parents[matches[0]]


def _replace(root, parent, old, new):
    """Common U1 / reconstructed replacement and serialization path."""
    index = list(parent).index(old)
    parent.remove(old)
    parent.insert(index, new)
    return ET.tostring(root, encoding="unicode")


def _static_parent_world(root, parent):
    """Evaluator-only coordinate conversion; never used to estimate an asset."""
    parents = {child: node for node in root.iter() for child in node}
    chain = []
    while parent.tag != "worldbody":
        if parent.tag != "body" or parent.find("joint") is not None or parent.find("freejoint") is not None:
            raise ValueError("reconstructed static body requires a static parent chain")
        chain.append(parent)
        parent = parents[parent]
    transform = np.eye(4)
    for body in reversed(chain):
        if any(k in body.attrib for k in ("euler", "axisangle", "xyaxes", "zaxis")):
            raise ValueError("static parent orientation must be explicit wxyz quaternion")
        position = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        quaternion = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
        if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite([*position, *quaternion]).all():
            raise ValueError("invalid static parent pose")
        local = np.eye(4)
        local[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        local[:3, 3] = position
        transform = transform @ local
    return transform


def import_identity(xml: str, *, body_name: str):
    """Privileged U1: detach/copy/import all native target bytes and attributes.

    This is the lossless importer control, not reconstructed-arm evidence.
    Runtime frame/state/action identity must additionally be tested by SR0.
    """
    root = ET.fromstring(xml)
    body, parent = _body(root, body_name)
    imported = copy.deepcopy(body)
    output = _replace(root, parent, body, imported)
    return output, {
        "control": "U1", "privileged_native_asset_access": True,
        "body_name": body_name, "body_xml_equal": ET.tostring(body) == ET.tostring(imported),
        "source_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "imported_xml_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "runtime_identity": "NOT_RUN",
    }


def import_reconstructed_object(
    xml: str, *, body_name: str, object_dir: str | Path, object_id: str,
    role: str = "target", rgba=(0.65, 0.65, 0.65, 1.0),
    contact_prior: dict | None = None, texture_path: str | Path | None = None,
):
    """Replace one rigid target/context body; return XML and bound receipt.

    The canonical mesh is NOT recentered. Scale is applied only by MJCF mesh
    scale; body translation/rotation come ONLY from aligned.json. Inertia is
    the existing s6 mass/dimension box prior, centered on reconstructed bounds.
    Uniform color is an explicit appearance diagnostic unless a texture is
    supplied. Container cavities remain unions of supplied convex parts; an
    independent entry/contact diagnostic must establish opening preservation.
    """
    import trimesh

    if role not in {"target", "receptacle", "support", "obstacle"}:
        raise ValueError("unsupported reconstructed role")
    directory = Path(object_dir).resolve(strict=True)
    aligned_path, physics_path = directory / "aligned.json", directory / "physics.json"
    aligned = json.loads(aligned_path.read_text())
    physics = json.loads(physics_path.read_text())
    scale, position, quaternion = decompose_alignment(aligned)
    visual = directory / "mesh_sim.obj"
    collision = sorted((directory / "collision").glob("part_*.obj"))
    if not visual.is_file() or not collision:
        raise ValueError("constructed visual mesh and collision parts are required")
    mesh = trimesh.load(visual, process=False, force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError("visual mesh has invalid vertices")
    lower, upper = vertices.min(0) * scale, vertices.max(0) * scale
    dimensions, center = upper - lower, (upper + lower) / 2
    if np.any(dimensions <= 0):
        raise ValueError("reconstructed mesh must be three-dimensional")
    if "world_dims" in aligned and not np.allclose(
            np.asarray(aligned["world_dims"], dtype=float), dimensions, rtol=1e-5, atol=1e-7):
        raise ValueError("aligned world_dims disagrees with scaled mesh (frame/scale mismatch)")
    mass, friction = float(physics["mass_kg"]), float(physics["friction"])
    if not np.isfinite([mass, friction]).all() or mass <= 0 or friction < 0:
        raise ValueError("physical prior needs positive mass and nonnegative friction")
    inertia = mass / 12 * (np.sum(dimensions ** 2) - dimensions ** 2)
    color = np.asarray(rgba, dtype=float)
    if color.shape != (4,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
        raise ValueError("rgba must be four finite values in [0,1]")
    # Frozen global defaults, never inferred from hidden native physical data.
    contact = {"solref": [0.02, 1.0], "solimp": [0.9, 0.95, 0.001],
               "torsional_friction": 0.005, "rolling_friction": 0.0001,
               "margin_m": 0.0, "minimum_thickness_m": 0.0}
    if contact_prior:
        unknown = set(contact_prior) - set(contact)
        if unknown:
            raise ValueError(f"unknown contact prior fields: {sorted(unknown)}")
        contact.update(contact_prior)
    if float(contact["minimum_thickness_m"]) != 0:
        raise ValueError("mesh inflation is not implemented; cannot silently apply thickness")
    if len(contact["solref"]) != 2 or len(contact["solimp"]) != 3:
        raise ValueError("invalid contact parameter shape")
    vals = [*contact["solref"], *contact["solimp"], contact["torsional_friction"],
            contact["rolling_friction"], contact["margin_m"]]
    if not np.isfinite(vals).all() or min(vals) < 0:
        raise ValueError("contact parameters must be finite and nonnegative")
    root = ET.fromstring(xml)
    original, parent = _body(root, body_name)
    if original.get("mocap", "false") == "true":
        raise ValueError("mocap body replacement needs a separate state mapping")
    joints = list(original.iter("joint")) + list(original.iter("freejoint"))
    free = bool(joints)
    if free and (len(joints) != 1 or (joints[0].tag == "joint" and joints[0].get("type") != "free")
                 or joints[0] not in list(original)):
        raise ValueError("replacement supports static or one root free joint, no articulation")
    if role == "target" and not free:
        raise ValueError("manipulated target must retain a free joint")
    if free and parent.tag != "worldbody":
        raise ValueError("free body must be directly world-parented")
    parent_world = _static_parent_world(root, parent)
    world_pose = np.eye(4)
    world_pose[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    world_pose[:3, 3] = position
    local_pose = np.linalg.inv(parent_world) @ world_pose
    local_q = Rotation.from_matrix(local_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
    original_geoms = {g.get("name") for g in original.iter("geom") if g.get("name")}
    old_sites = {s.get("name") for s in original.iter("site") if s.get("name")}
    # Reject dependencies that would otherwise retain stale geometry semantics.
    for section in ("contact", "equality", "tendon", "sensor"):
        for element in root.findall(f"./{section}//*"):
            if any(value in original_geoms | old_sites for value in element.attrib.values()):
                raise ValueError(f"external {section} references removed target geometry/site")
            if section == "equality" and body_name in element.attrib.values():
                raise ValueError("target equality constraint would retain an original weld")
    name = body_name + "__roundtrip"
    if any(e.get("name", "").startswith(name) for e in root.iter()):
        raise ValueError("target already imported or importer name collision")
    replacement = ET.Element("body", name=body_name, pos=_numbers(local_pose[:3, 3]), quat=_numbers(local_q))
    if free:
        replacement.append(copy.deepcopy(joints[0]))
    ET.SubElement(replacement, "inertial", pos=_numbers(center), mass=str(mass),
                  diaginertia=_numbers(inertia))
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    asset_paths = [aligned_path, physics_path, visual, *collision]
    material = name + "_material"
    material_attrs = {"name": material, "rgba": _numbers(color),
                      "specular": "0.5", "shininess": "0.5", "reflectance": "0"}
    if texture_path is not None:
        texture = Path(texture_path).resolve(strict=True)
        if not texture.is_file():
            raise ValueError("declared texture is missing")
        asset_paths.append(texture)
        ET.SubElement(asset, "texture", name=name + "_texture", type="2d", file=str(texture))
        material_attrs["texture"] = name + "_texture"
    ET.SubElement(asset, "material", **material_attrs)
    geom_names = []
    for index, path in enumerate([visual, *collision]):
        if index:
            part = trimesh.load(path, process=False, force="mesh")
            if not part.is_watertight or not part.is_convex or part.volume <= 0:
                raise ValueError(f"collision part is not a closed convex volume: {path.name}")
        mesh_name = name + f"_mesh_{index}"
        geom_name = name + f"_geom_{index}"
        ET.SubElement(asset, "mesh", name=mesh_name, file=str(path), scale=_numbers([scale] * 3),
                      refpos="0 0 0", refquat="1 0 0 0")
        attrs = dict(name=geom_name, type="mesh", mesh=mesh_name, pos="0 0 0", quat="1 0 0 0",
                     group="1" if index == 0 else "0", contype="0" if index == 0 else "1",
                     conaffinity="0" if index == 0 else "1", mass="0", density="0",
                     rgba=_numbers(color) if index == 0 else "0 0 0 0", condim="3",
                     friction=_numbers([friction, contact["torsional_friction"], contact["rolling_friction"]]),
                     solref=_numbers(contact["solref"]), solimp=_numbers(contact["solimp"]),
                     margin=str(contact["margin_m"]), gap="0", priority="0")
        if index == 0:
            attrs["material"] = material
        ET.SubElement(replacement, "geom", **attrs)
        geom_names.append(geom_name)
    output = _replace(root, parent, original, replacement)
    receipt = {
        "object_id": object_id, "body_name": body_name,
        "scope": "target_only" if role == "target" else "single_rigid_context_entity",
        "role": role, "quaternion_convention": "wxyz", "units": "meters",
        "body_semantics": "free" if free else "static",
        "free_joint_name": joints[0].get("name") if free else None,
        "parent_world_transform": parent_world.tolist(),
        "body_local_position_m": local_pose[:3, 3].tolist(), "body_local_quaternion_wxyz": local_q.tolist(),
        "canonical_frame": "factory mesh vertices; no recentering", "scale_applications": 1,
        "scale": scale, "position_m": position.tolist(), "quaternion_wxyz": quaternion.tolist(),
        "bounds_local_m": [lower.tolist(), upper.tolist()], "com_local_m": center.tolist(),
        "mass_kg": mass, "inertia_kg_m2": inertia.tolist(),
        "inertia_prior": "s6_uniform_box_on_reconstructed_bounds", "physics_prior": physics,
        "contact_prior": contact, "geometry_inflation_m": 0.0,
        "visual_geoms": geom_names[:1], "contact_geoms": geom_names[1:],
        "removed_original_geoms": sorted(original_geoms), "removed_original_sites": sorted(old_sites),
        "retained_native_context": "all entities except this receipt's removed subtree; scope inventory required",
        "native_fixture_metadata": "not rebound by XML import; caller must bind native role handles",
        "appearance": "textured_mesh" if texture_path is not None else "uniform_color_diagnostic",
        "source_hashes": {str(path): _sha(path) for path in asset_paths},
        "source_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "imported_xml_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "native_scorer_binding": "PENDING",
        "scorer_frame_correspondence": "NOT_ESTABLISHED; generated geometry has no inferred native-origin correspondence",
        "opening_preservation": "NOT_RUN" if role == "receptacle" else "NOT_APPLICABLE",
    }
    return output, receipt


def rebind_native_object(env, *, object_name: str, receipt: dict):
    """Refresh the existing MJCFObject's contact lists and reconstructed bbox.

    Call AFTER reset_from_xml_string and native reference setup, before scoring.
    This retains the native class and rubric; no reference dimension is read.
    """
    obj = env.objects[object_name]
    if obj.root_body != receipt["body_name"]:
        raise ValueError("native role binding body differs from imported body")
    prefix = obj.naming_prefix
    def local(names):
        if not all(name.startswith(prefix) for name in names):
            raise ValueError("replacement geom does not respect native naming prefix")
        return [name[len(prefix):] for name in names]
    for name in receipt["contact_geoms"] + receipt["visual_geoms"]:
        env.sim.model.geom_name2id(name)  # fail before mutation if compiled geometry is absent
    lower, upper = np.asarray(receipt["bounds_local_m"], dtype=float)
    if np.any(upper <= lower):
        raise ValueError("replacement native bbox is invalid")
    center, half = (lower + upper) / 2, (upper - lower) / 2
    bbox = ET.Element("geom", name=prefix + "reg_bbox", type="box",
                      pos=_numbers(center), size=_numbers(half))
    obj._regions = {"bbox": {"elem": bbox, "p0": lower.copy(),
        "px": np.array([upper[0], lower[1], lower[2]]),
        "py": np.array([lower[0], upper[1], lower[2]]),
        "pz": np.array([lower[0], lower[1], upper[2]])}}
    obj._contact_geoms = local(receipt["contact_geoms"])
    obj._visual_geoms = local(receipt["visual_geoms"])
    obj._sites = []
    obj._bodies = local([receipt["body_name"]])
    env.obj_body_id[object_name] = env.sim.model.body_name2id(receipt["body_name"])
    return {"object_name": object_name, "native_scorer_binding": "BOUND",
            "target_bounds_source": "reconstructed_mesh", "native_class": type(obj).__name__,
            "contact_geoms": list(obj.contact_geoms),
            "role": receipt.get("role", "target"),
            "bounds_and_radius_source": "reconstructed_mesh_bounds",
            "absolute_scorer_correspondence": "NOT_ESTABLISHED",
            "retained_fixture_metadata": "oracle room context"}
