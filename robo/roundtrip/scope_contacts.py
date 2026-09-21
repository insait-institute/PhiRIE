"""Read compiled native contacts without advancing or modifying physics.

Direct samples are instantaneous. The optional native step context records every
completed integration substep. Geometry membership comes from import receipts.
"""
from __future__ import annotations
import numpy as np


def bind_contact_inventory(model,entities,workspace_bounds_world_m=None):
    import mujoco
    roles={}
    for entity in entities:
        for name in entity['contact_geoms']:
            gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)
            if gid<0 or gid in roles:raise ValueError('missing or duplicate reconstructed contact geometry')
            roles[gid]=entity['object_id']
    bounds=None if workspace_bounds_world_m is None else np.asarray(workspace_bounds_world_m,float)
    if bounds is not None and (bounds.shape!=(2,3) or not np.isfinite(bounds).all() or np.any(bounds[1]<=bounds[0])):
        raise ValueError('invalid declared contact workspace')
    return {'geom_roles':roles,'workspace_bounds_world_m':None if bounds is None else bounds.tolist(),
        'geom_names':{i:mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(model.ngeom)},
        'sampling':'post-policy-tick contacts only; no substep completeness claim'}


def sample_contacts(model,data,inventory,*,include_all=False):
    import mujoco
    bounds=inventory['workspace_bounds_world_m'];bounds=None if bounds is None else np.asarray(bounds)
    rows=[]
    for index in range(data.ncon):
        c=data.contact[index];pair=[int(c.geom1),int(c.geom2)]
        roles=[inventory['geom_roles'].get(g) for g in pair]
        if not include_all and all(r is None for r in roles):continue
        force=np.zeros(6);mujoco.mj_contactForce(model,data,index,force)
        point=np.asarray(c.pos)
        rows.append({'geom_names':[inventory['geom_names'][g] for g in pair],
            'object_ids':roles,'geometry_classification':['reconstructed' if r is not None else 'retained_reference' for r in roles],
            'contact_position_world_m':point.tolist(),'distance_m':float(c.dist),
            'normal_force_N':float(force[0]),'contact_frame_force_torque':force.tolist(),
            'touches_retained_reference':any(r is None for r in roles),
            'outside_declared_workspace':None if bounds is None else bool(np.any(point<bounds[0]) or np.any(point>bounds[1]))})
    return {'sampling':inventory['sampling'],'contacts':rows,'reconstructed_contact_count':sum(any(x is not None for x in r['object_ids']) for r in rows),'all_native_contacts_included':include_all,
        'retained_context_contact_count':sum(r['touches_retained_reference'] for r in rows),
        'outside_workspace_contact_count':None if bounds is None else sum(r['outside_declared_workspace'] for r in rows)}


def bind_adapter_contacts(adapter,manifest):
    adapter.scope_contact_inventory=bind_contact_inventory(adapter.native.sim.model._model,
        manifest['entities'],manifest.get('workspace_bounds_world_m'))
    return adapter.scope_contact_inventory


from contextlib import contextmanager

@contextmanager
def record_physics_contacts(sim,inventory):
    """Observe every pinned MjSim step/step2 completion within one policy action.

    All native contact pairs are retained, including robot/context pairs where
    neither geom was reconstructed. No mj_forward, sensor or control mutation.
    """
    rows=[];original={};had={}
    def wrapper(name,call):
        def sampled(*args,**kwargs):
            result=call(*args,**kwargs)
            sample=sample_contacts(sim.model._model,sim.data._data,inventory,include_all=True)
            sample['sampling']='post-native-physics-step contact state'
            rows.append({'physics_step':len(rows),'simulation_time_s':float(sim.data.time),**sample})
            return result
        return sampled
    try:
        for name in ('step','step2'):
            had[name]=name in vars(sim);original[name]=getattr(sim,name)
            setattr(sim,name,wrapper(name,original[name]))
        yield rows
    finally:
        for name,call in original.items():
            if had[name]:setattr(sim,name,call)
            else:delattr(sim,name)


def substep_report(rows,inventory):
    if not rows:raise ValueError('native scope action performed no observed physics step')
    return {'sampling':'every completed native MjSim.step/step2 within policy action',
        'physics_steps':len(rows),'contact_steps':rows,'workspace_bounds_world_m':inventory['workspace_bounds_world_m'],
        'any_retained_context_contact':any(r['retained_context_contact_count'] for r in rows),
        'any_outside_workspace_contact':None if inventory['workspace_bounds_world_m'] is None else any(r['outside_workspace_contact_count'] for r in rows),
        'continuous_time_between_integrator_steps':'not observed; native discrete physics-step contact states are complete'}
