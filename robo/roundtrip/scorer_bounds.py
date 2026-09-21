"""Evaluator-only bounds for native body-origin predicate dependence.

Bounds cover a compiled visual mesh's world AABB, without choosing a favorable
point or changing the official scorer. They do not establish physical frame
correspondence and cannot alone admit a held-out population claim.
"""
from itertools import product
import numpy as np


def corners(vertices):
    v=np.asarray(vertices,dtype=float)
    if v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all():
        raise ValueError('finite physical visual surface required')
    lo=v.min(0);hi=v.max(0)
    return np.array(list(product(*zip(lo,hi)))),lo,hi


def distance_bounds(points, location, dimensions=3):
    _,lo,hi=corners(points);p=np.asarray(location)[:dimensions]
    lo=lo[:dimensions];hi=hi[:dimensions]
    minimum=float(np.linalg.norm(np.maximum(np.maximum(lo-p,p-hi),0)))
    maximum=float(np.linalg.norm(np.maximum(abs(lo-p),abs(hi-p))))
    return minimum,maximum


def sink_bounds(points,regions):
    points,_,_=corners(points);all_inside=False;all_disjoint=True
    for region in regions.values():
        p0,px,py,pz=np.asarray(region,dtype=float)
        inside=True;disjoint=False
        for end in [px,py,pz]:
            axis=end-p0
            if np.linalg.norm(axis)==0:raise ValueError('degenerate native fixture region')
            values=points@axis;low=float(p0@axis);high=float(end@axis)
            inside &= bool(values.min()>=low and values.max()<=high)
            disjoint |= bool(values.max()<low or values.min()>high)
        all_inside |= inside;all_disjoint &= disjoint
    if not regions:raise ValueError('native fixture regions missing')
    return 'always_true' if all_inside else 'always_false' if all_disjoint else 'ambiguous'


def combine_bounds(parts):
    if any(p=='always_false' for p in parts):return 'invariant_failure_within_declared_envelope'
    if all(p=='always_true' for p in parts):return 'invariant_success_within_declared_envelope'
    return 'origin_sensitive_or_unresolved_within_declared_envelope'


def audit_bounds(env,task_id,vertices):
    points,lo,hi=corners(vertices)
    origin=np.asarray(env.sim.data.body_xpos[env.obj_body_id['obj']])
    eef=np.asarray(env.sim.data.site_xpos[env.robots[0].eef_site_id['right']])
    dmin,dmax=distance_bounds(points,eef)
    far='always_true' if dmin>.25 else 'always_false' if dmax<=.25 else 'ambiguous'
    parts=[far]
    details={}
    if task_id=='PickPlaceCounterToSink':
        region=env.sink.get_int_sites(relative=False)
        inside=sink_bounds(points,region);parts.append(inside)
        details={'sink_origin_inside_bound':inside,'native_sink_regions':{k:np.asarray(v).tolist() for k,v in region.items()}}
    elif task_id=='PickPlaceSinkToCounter':
        center=env.sim.data.body_xpos[env.obj_body_id['container']]
        rmin,rmax=distance_bounds(points,center,2);threshold=float(env.objects['container'].horizontal_radius*.7)
        xy='always_true' if rmax<threshold else 'always_false' if rmin>=threshold else 'ambiguous'
        target_contact=bool(env.check_contact(env.objects['obj'],env.objects['container']))
        support_contact=bool(env.check_contact(env.objects['container'],env.counter))
        parts += [xy,'always_true' if target_contact and support_contact else 'always_false']
        details={'receptacle_xy_distance_bounds_m':[rmin,rmax],'native_receptacle_threshold_m':threshold,
                 'target_receptacle_contact':target_contact,'receptacle_counter_contact':support_contact}
    else:raise ValueError('unsupported native scorer family')
    inside_envelope=bool(np.all(origin>=lo) and np.all(origin<=hi))
    return dict(envelope='world_AABB_of_compiled_visual_mesh_vertices',world_envelope_m=[lo.tolist(),hi.tolist()],
        raw_origin_within_envelope=inside_envelope,raw_origin_world_m=origin.tolist(),gripper_world_m=eef.tolist(),
        gripper_distance_bounds_m=[dmin,dmax],native_gripper_threshold_m=.25,gripper_far_bound=far,
        bound_conclusion=combine_bounds(parts) if inside_envelope else 'raw_origin_outside_physical_envelope_UNQUALIFIED',
        correspondence='NOT_ESTABLISHED',scope='bounded final-state diagnostic, not population admission',**details)
