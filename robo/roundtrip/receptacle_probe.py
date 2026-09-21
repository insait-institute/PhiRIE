"""Independent unassisted cavity-entry diagnostic on imported collision.

Probe start and interior z interval must be fixed from constructor-visible
observations; this evaluator does not look up native container geometry. A failed
probe is diagnostic evidence, never an automatic switch to retained native mesh.
"""
from __future__ import annotations

import hashlib
import numpy as np


def probe_entry(xml, *, contact_geoms, start_world_m, interior_world_z_m,
                radius_m=0.005, seconds=1.5, timestep_s=0.002):
    import mujoco
    import xml.etree.ElementTree as ET

    start, interior = np.asarray(start_world_m, float), np.asarray(interior_world_z_m, float)
    if start.shape != (3,) or interior.shape != (2,) or not np.isfinite([*start, *interior, radius_m, seconds, timestep_s]).all():
        raise ValueError('finite probe geometry and timing required')
    if not 0 < radius_m < .1 or not 0 < timestep_s <= .01 or not 0 < seconds <= 10:
        raise ValueError('probe outside bounded diagnostic parameters')
    if round(seconds / timestep_s) < 1:
        raise ValueError('probe requires at least one integration step')
    if not interior[0] < interior[1] < start[2] - radius_m:
        raise ValueError('start must be above declared interior')
    root = ET.fromstring(xml)
    if root.find('./worldbody/body[@name="roundtrip_entry_probe"]') is not None:
        raise ValueError('probe name collision')
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.set('timestep', str(timestep_s))
    option.set('gravity', '0 0 -9.81')
    world = root.find('worldbody')
    if world is None:
        raise ValueError('worldbody missing')
    body = ET.SubElement(world, 'body', name='roundtrip_entry_probe', pos=' '.join(map(str, start)))
    ET.SubElement(body, 'freejoint', name='roundtrip_entry_probe_joint')
    ET.SubElement(body, 'geom', name='roundtrip_entry_probe_geom', type='sphere',
                  size=str(radius_m), mass='.001', friction='.4 .005 .0001')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    probe_bid = model.body('roundtrip_entry_probe').id
    probe_gid = model.geom('roundtrip_entry_probe_geom').id
    contacts = {model.geom(name).id for name in contact_geoms}
    if not contacts:
        raise ValueError('declared receptacle contact geoms required')
    contact_steps, entered_steps, max_depth = 0, 0, 0.0
    history = []
    for _ in range(round(seconds / timestep_s)):
        mujoco.mj_step(model, data)
        p = data.xpos[probe_bid].copy()
        touching = any((c.geom1 == probe_gid and c.geom2 in contacts) or
                       (c.geom2 == probe_gid and c.geom1 in contacts) for c in data.contact[:data.ncon])
        inside = interior[0] <= p[2] <= interior[1]
        contact_steps += bool(touching)
        entered_steps += bool(inside)
        history.append(p.tolist())
        max_depth = max(max_depth, float(start[2] - p[2]))
    final = np.asarray(history[-1])
    final_inside = bool(interior[0] <= final[2] <= interior[1])
    # Entry, retained bottom contact and low terminal speed are independently
    # visible; no success flag is attached to the manipulation task itself.
    jid = model.joint('roundtrip_entry_probe_joint').id
    adr = model.jnt_dofadr[jid]
    speed = float(np.linalg.norm(data.qvel[adr:adr + 3]))
    return {'schema_version': 1, 'diagnostic': 'unassisted_collision_cavity_entry',
            'source_xml_sha256': hashlib.sha256(xml.encode()).hexdigest(),
            'mujoco_version': mujoco.__version__, 'start_world_m': start.tolist(),
            'interior_world_z_m': interior.tolist(), 'radius_m': radius_m,
            'timestep_s': timestep_s, 'steps': len(history), 'contact_steps': contact_steps,
            'entered_steps': entered_steps, 'final_world_m': final.tolist(),
            'terminal_speed_m_s': speed, 'max_drop_m': max_depth,
            'terminal_receptacle_contact': bool(touching),
            'passed': final_inside and touching and entered_steps > 0 and speed < .02,
            'trajectory_world_m': history, 'native_task_success': None,
            'limitations': 'Single predeclared entry path; does not prove general containment or task success'}
