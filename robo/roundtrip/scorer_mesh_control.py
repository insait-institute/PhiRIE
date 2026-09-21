"""Exact known local-frame reexpression of a generated rigid mesh import."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from robo.roundtrip.identity import validate_delta


def pose(element):
    if any(k in element.attrib for k in ['euler','axisangle','xyaxes','zaxis','fromto']):
        raise ValueError('frame control requires explicit pos/quat representation')
    T=np.eye(4);T[:3,3]=np.fromstring(element.get('pos','0 0 0'),sep=' ')
    q=np.fromstring(element.get('quat','1 0 0 0'),sep=' ')
    T[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    return T


def set_pose(element,T):
    def nums(a):return ' '.join(format(float(x),'.17g') for x in a)
    q=Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]]
    element.set('pos',nums(T[:3,3]));element.set('quat',nums(q))


def reexpress_xml(xml,body_name,old_from_new):
    """Change body coordinates and compensate all attached geometry/inertia.

    The transform is synthetically known; it is never inferred from native GT.
    Only a world-child rigid free body is supported. Mesh asset bytes are intact.
    """
    B=validate_delta(old_from_new);root=ET.fromstring(xml)
    body=root.find(f'worldbody/body[@name="{body_name}"]')
    if body is None:raise ValueError('frame control requires world-child target')
    joints=list(body.findall('joint'))+list(body.findall('freejoint'))
    if len(joints)!=1 or (joints[0].tag=='joint' and joints[0].get('type')!='free'):
        raise ValueError('frame control requires one free joint')
    inverse=np.linalg.inv(B)
    set_pose(body,pose(body)@B)
    for child in body:
        if child.tag in ['geom','site','body','inertial']:set_pose(child,inverse@pose(child))
        elif child.tag not in ['joint','freejoint']:raise ValueError('unsupported rigid body child')
    return ET.tostring(root,encoding='unicode'),inverse


def compare_compiled(xml,body_name,data,old_from_new):
    """Physical world surface and inertial COM check; no physics stepping."""
    import mujoco
    from robo.roundtrip.fidelity_native import visual_surface
    original=mujoco.MjModel.from_xml_string(xml);a=mujoco.MjData(original)
    a.qpos[:]=data.qpos;a.qvel[:]=data.qvel;mujoco.mj_forward(original,a)
    new_xml,inverse=reexpress_xml(xml,body_name,old_from_new)
    changed=mujoco.MjModel.from_xml_string(new_xml);b=mujoco.MjData(changed)
    b.qpos[:]=data.qpos;b.qvel[:]=data.qvel
    bid=mujoco.mj_name2id(original,mujoco.mjtObj.mjOBJ_BODY,body_name)
    newbid=mujoco.mj_name2id(changed,mujoco.mjtObj.mjOBJ_BODY,body_name)
    j=int(changed.body_jntadr[newbid]);qadr=int(changed.jnt_qposadr[j])
    T=np.eye(4);T[:3,3]=a.xpos[bid];T[:3,:3]=a.xmat[bid].reshape(3,3)
    newT=T@validate_delta(old_from_new);q=Rotation.from_matrix(newT[:3,:3]).as_quat()[[3,0,1,2]]
    b.qpos[qadr:qadr+7]=np.r_[newT[:3,3],q];mujoco.mj_forward(changed,b)
    ma,an=visual_surface(original,a,bid);mb,bn=visual_surface(changed,b,newbid)
    if an!=bn:raise ValueError('frame control changed visual geometry roster')
    vertices_error=float(np.max(abs(ma.vertices-mb.vertices)))
    com_error=float(np.max(abs(a.xipos[bid]-b.xipos[newbid])))
    # Include contact geometry world transforms as well as visible surface.
    geoms=[i for i in range(original.ngeom) if original.geom_bodyid[i]==bid]
    geom_pos_error=float(np.max(abs(a.geom_xpos[geoms]-b.geom_xpos[geoms])))
    geom_rot_error=float(np.max(abs(a.geom_xmat[geoms]-b.geom_xmat[geoms])))
    return dict(visual_world_vertices_max_abs_m=vertices_error,COM_max_abs_m=com_error,
        attached_geom_world_position_max_abs_m=geom_pos_error,attached_geom_rotation_max_abs=geom_rot_error,
        passed=max(vertices_error,geom_pos_error)<1e-7 and max(com_error,geom_rot_error)<1e-12,
        compiled_mesh_roundoff_tolerance_m=1e-7,rigid_frame_tolerance=1e-12,
        known_body_from_scorer=inverse.tolist(),mesh_assets_modified=False),b,inverse
