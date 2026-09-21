"""Preserve native integration state when fixed geometry changes body count."""
import numpy as np


def remap_integration_state(source_model, target_model, state):
    """Only fixed-body external-force slots may change; dynamic topology is frozen.

    Removed bodies must have zero applied external force. No solver-derived
    matrices, contacts, or accelerations are copied from the old geometry.
    """
    import mujoco
    mask=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.asarray(state,dtype=float)
    if state.shape!=(mujoco.mj_stateSize(source_model,mask),) or not np.isfinite(state).all():
        raise ValueError('canonical integration vector does not match source native model')
    def names(model,kind,count):
        return [mujoco.mj_id2name(model,kind,i) for i in range(count)]
    for kind,count in [(mujoco.mjtObj.mjOBJ_JOINT,'njnt'),(mujoco.mjtObj.mjOBJ_ACTUATOR,'nu'),(mujoco.mjtObj.mjOBJ_EQUALITY,'neq')]:
        if names(source_model,kind,getattr(source_model,count))!=names(target_model,kind,getattr(target_model,count)):
            raise ValueError('changed/reordered dynamic topology requires explicit named integration remap')
    for key in ['jnt_type','jnt_qposadr','jnt_dofadr']:
        if not np.array_equal(getattr(source_model,key),getattr(target_model,key)):
            raise ValueError('changed/reordered dynamic topology requires explicit named integration remap')
    bodykind=mujoco.mjtObj.mjOBJ_BODY
    srcnames=names(source_model,bodykind,source_model.nbody)
    dstnames=names(target_model,bodykind,target_model.nbody)
    def mocaps(model,roster):
        return [roster[int(np.flatnonzero(model.body_mocapid==i)[0])] for i in range(model.nmocap)]
    if mocaps(source_model,srcnames)!=mocaps(target_model,dstnames):
        raise ValueError('changed mocap identity is unsupported')
    source=mujoco.MjData(source_model)
    mujoco.mj_setState(source_model,source,state,mask)
    result=mujoco.MjData(target_model)
    # MuJoCo's individual state bits include all integration fields, including
    # warmstart, activation, applied generalized force, equality and plugin state.
    for bit in [1<<i for i in range(int(mujoco.mjtState.mjNSTATE))]:
        if not int(mask)&bit or bit==int(mujoco.mjtState.mjSTATE_XFRC_APPLIED):continue
        n=mujoco.mj_stateSize(source_model,bit)
        if n!=mujoco.mj_stateSize(target_model,bit):
            raise ValueError('non-body integration state dimension changed')
        value=np.empty(n);mujoco.mj_getState(source_model,source,value,bit)
        mujoco.mj_setState(target_model,result,value,bit)
    if srcnames==dstnames:
        result.xfrc_applied[:]=source.xfrc_applied
    else:
        if len(set(x for x in srcnames if x is not None))!=sum(x is not None for x in srcnames) or len(set(x for x in dstnames if x is not None))!=sum(x is not None for x in dstnames):
            raise ValueError('body-force remap requires unique named bodies')
        target_ids={name:i for i,name in enumerate(dstnames) if name is not None}
        for i,name in enumerate(srcnames):
            if name in target_ids:result.xfrc_applied[target_ids[name]]=source.xfrc_applied[i]
            elif np.any(source.xfrc_applied[i]!=0):
                raise ValueError('removed fixed body has nonzero external force')
    value=np.empty(mujoco.mj_stateSize(target_model,mask))
    mujoco.mj_getState(target_model,result,value,mask)
    return value
