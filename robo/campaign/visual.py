"""3D state-conditioned visual correction, without changing simulated geometry.

V1: depth/instance validated temporal correspondence, bounded image residuals,
protected robot/contact/silhouette pixels, and online causal history. This is a
prototype to evaluate against the official Harmonizer, not a certified renderer.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from .core import load, receipt, save


def transform_points(T, xyz):
    T = np.asarray(T, float)
    if T.shape != (4,4) or not np.isfinite(T).all() or not np.allclose(T[3], [0,0,0,1]):
        raise ValueError('finite homogeneous rigid transform required')
    if not np.allclose(T[:3,:3].T@T[:3,:3], np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(T[:3,:3]), 1, atol=1e-5):
        raise ValueError('motion must be SE(3); bake scale into geometry once')
    return xyz@T[:3,:3].T + T[:3,3]


def backward_warp(previous_rgb, previous_depth, previous_ids, depth, ids, K, previous_K,
                  T_world_camera, previous_T_world_camera, object_poses, previous_object_poses,
                  *, depth_atol=.005, depth_rtol=.01):
    """Current pixel -> previous pixel using camera and per-object rigid motion.

    Depth is positive camera-Z in metres. ID 0 is static background; -1 unknown.
    Nearest sampling never blends different instances. Invalid/disoccluded points
    are returned as invalid and must NOT be temporally averaged.
    """
    d = np.asarray(depth, float); ident = np.asarray(ids)
    old = np.asarray(previous_rgb); pd = np.asarray(previous_depth, float); pi = np.asarray(previous_ids)
    if d.ndim != 2 or ident.shape != d.shape or old.shape != (*pd.shape,3) or pi.shape != pd.shape:
        raise ValueError('depth, ID and image grids differ')
    h,w = d.shape; yy,xx = np.indices((h,w)); K = np.asarray(K,float); pK = np.asarray(previous_K,float)
    if K.shape != (3,3) or pK.shape != (3,3) or abs(np.linalg.det(K)) < 1e-12 or abs(np.linalg.det(pK)) < 1e-12:
        raise ValueError('invertible intrinsics required')
    rays = np.stack([xx,yy,np.ones_like(xx)], -1).reshape(-1,3)@np.linalg.inv(K).T
    good = np.isfinite(d.reshape(-1)) & (d.reshape(-1)>0) & (ident.reshape(-1)>=0)
    xyz = rays*np.where(good,d.reshape(-1),0)[:,None]
    world = transform_points(T_world_camera, xyz); old_world = world.copy(); flat = ident.reshape(-1)
    for oid in np.unique(flat[good]):
        if oid == 0: continue
        mask = flat == oid
        if str(oid) not in object_poses or str(oid) not in previous_object_poses:
            good[mask] = False; continue
        local = transform_points(np.linalg.inv(object_poses[str(oid)]),world[mask])
        old_world[mask] = transform_points(previous_object_poses[str(oid)],local)
    camera = transform_points(np.linalg.inv(previous_T_world_camera), old_world)
    z = camera[:,2]; good &= z>0
    pixels = camera@pK.T; uv = pixels[:,:2]/np.where(z>0,z,1)[:,None]
    finite = np.isfinite(uv).all(axis=1); good &= finite
    uv = np.where(finite[:,None],uv,0); u,v = np.rint(uv).astype(np.int64).T
    good &= (u>=0)&(v>=0)&(u<pd.shape[1])&(v<pd.shape[0])
    safe_u = np.clip(u,0,pd.shape[1]-1); safe_v = np.clip(v,0,pd.shape[0]-1)
    zd = pd[safe_v,safe_u]
    good &= np.isfinite(zd)&(zd>0)&(np.abs(zd-z) <= depth_atol+depth_rtol*np.maximum(zd,0))
    good &= pi[safe_v,safe_u] == flat
    warped = old[safe_v,safe_u].reshape(h,w,3).copy(); valid = good.reshape(h,w)
    warped[~valid] = 0
    return warped, valid


def protected_mask(ids, robot_mask, contact_mask, *, edge_radius=1):
    from scipy.ndimage import binary_dilation
    ident = np.asarray(ids); robot = np.asarray(robot_mask,bool); contact = np.asarray(contact_mask,bool)
    if robot.shape != ident.shape or contact.shape != ident.shape: raise ValueError('mask dimensions differ')
    edge = np.zeros(ident.shape, bool)
    edge[1:] |= ident[1:] != ident[:-1]; edge[:-1] |= ident[1:] != ident[:-1]
    edge[:,1:] |= ident[:,1:] != ident[:,:-1]; edge[:,:-1] |= ident[:,1:] != ident[:,:-1]
    if edge_radius > 0: edge = binary_dilation(edge,iterations=edge_radius)
    return robot | contact | edge | (ident<0)


def compose(raw, enhanced, protect, confidence, *, strength=.5, max_residual=32,
            warped=None, valid=None, temporal_weight=.25):
    raw = np.asarray(raw); enhanced = np.asarray(enhanced)
    p = np.asarray(protect,bool); c = np.asarray(confidence,float)
    if raw.dtype != np.uint8 or enhanced.dtype != np.uint8 or raw.shape != enhanced.shape or raw.shape != (*p.shape,3) or c.shape != p.shape:
        raise ValueError('aligned uint8 RGB, protection and confidence required')
    if not np.isfinite(c).all() or np.any((c<0)|(c>1)) or not 0<=strength<=1 or not 0<=temporal_weight<=1 or not 0<=max_residual<=255:
        raise ValueError('invalid bounded correction settings')
    corrected = enhanced.astype(float)
    if warped is not None:
        if np.shape(warped)!=raw.shape or np.shape(valid)!=p.shape: raise ValueError('warp/valid grid mismatch')
        weight = temporal_weight*np.asarray(valid,bool)[...,None]
        corrected = (1-weight)*corrected + weight*np.asarray(warped,float)
    residual = np.clip(corrected-raw.astype(float), -max_residual,max_residual)
    result = np.rint(np.clip(raw.astype(float)+strength*c[...,None]*residual,0,255)).astype(np.uint8)
    result[p] = raw[p]
    return result


def run(config, inputs, out):
    state = load(inputs['state']); raw = np.asarray(Image.open(inputs['rgb']).convert('RGB'))
    enhanced = np.asarray(Image.open(inputs['enhanced']).convert('RGB'))
    fields = np.load(inputs['buffers'],allow_pickle=False)
    if state['frame_index'] < 0 or state.get('sim_state_sha256') is None:
        raise ValueError('current simulator-state identity required')
    protect = protected_mask(fields['ids'],fields['robot_mask'],fields['contact_mask'],
                             edge_radius=int(config.get('edge_radius',1)))
    warp = valid = None
    if 'previous_state' in inputs:
        prev = load(inputs['previous_state'])
        if prev['stream_id'] != state['stream_id'] or prev['frame_index'] != state['frame_index']-1:
            raise ValueError('cross-camera/episode or out-of-order temporal history')
        prev_fields = np.load(inputs['previous_buffers'],allow_pickle=False)
        warp,valid = backward_warp(np.asarray(Image.open(inputs['previous_rgb']).convert('RGB')),
            prev_fields['depth'],prev_fields['ids'],fields['depth'],fields['ids'],
            state['K'],prev['K'],state['T_world_camera'],prev['T_world_camera'],
            state['object_poses'],prev['object_poses'],**config.get('warp',{}))
    image = compose(raw,enhanced,protect,fields['confidence'],warped=warp,valid=valid,**config.get('compose',{}))
    if 'learned' in config:
        from .learned import predict
        image=predict(config['learned'],raw,enhanced,fields,protect,warp,valid)
    Image.fromarray(image).save(Path(out)/'policy_rgb.png')
    Image.fromarray(protect.astype('uint8')*255).save(Path(out)/'protected.png')
    save(Path(out)/'visual.json', {'state': state, 'inputs': {k:receipt(v) for k,v in inputs.items()},
        'protected_equal': bool(np.array_equal(image[protect],raw[protect])),
        'valid_warp_pixels': int(valid.sum()) if valid is not None else 0,
        'physics_changed':False,'unprotected_geometry_preserved':'NOT_GUARANTEED',
        'method':'StateResidual16' if 'learned' in config else 'state-conditioned bounded residual prototype',
        'learned_checkpoint':receipt(config['learned']['checkpoint_manifest']) if 'learned' in config else None})
    return {'rgb':Path(out)/'policy_rgb.png','protection':Path(out)/'protected.png','receipt':Path(out)/'visual.json'}
