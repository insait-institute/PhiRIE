"""Privileged scorer frame diagnostics; never modifies physics or thresholds.

An exact local-frame transform is usable for representation controls. A generated
shape does not automatically establish a correspondence to the native body frame.
Its raw native predicate remains explicitly binding-specific.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from robo.roundtrip.identity import validate_delta


class _View:
    def __init__(self, source, **overrides):
        self._source=source;self.__dict__.update(overrides)
    def __getattr__(self,name):return getattr(self._source,name)


def scorer_view(env, body_from_scorer):
    """Read-only position/orientation view for fixed, justified scorer frames."""
    positions=np.asarray(env.sim.data.body_xpos).copy()
    quaternions=np.asarray(env.sim.data.body_xquat).copy()
    for role,transform in body_from_scorer.items():
        T=validate_delta(transform);bid=env.obj_body_id[role]
        R=Rotation.from_quat(quaternions[bid][[1,2,3,0]]).as_matrix()
        positions[bid]+=R@T[:3,3]
        q=Rotation.from_matrix(R@T[:3,:3]).as_quat();quaternions[bid]=q[[3,0,1,2]]
    data=_View(env.sim.data,body_xpos=positions,body_xquat=quaternions)
    return _View(env,sim=_View(env.sim,data=data))


def predicate_components(env, task_id, *, body_from_scorer=None):
    from robocasa.utils import object_utils as OU
    target=env if body_from_scorer is None else scorer_view(env,body_from_scorer)
    far=bool(OU.gripper_obj_far(target))
    pos=target.sim.data.body_xpos[target.obj_body_id['obj']]
    gripper=target.sim.data.site_xpos[target.robots[0].eef_site_id['right']]
    result={'gripper_distance_m':float(np.linalg.norm(pos-gripper)),
            'gripper_far':far,'gripper_threshold_m':.25,
            'frame_status':'raw_native_body_origin' if body_from_scorer is None else 'explicit_fixed_scorer_frame',
            'whole_mesh_containment_certified':False}
    if task_id=='PickPlaceCounterToSink':
        inside=bool(OU.obj_inside_of(target,'obj',target.sink,partial_check=True))
        result.update(object_origin_inside_sink=inside,native_task_success=inside and far)
    elif task_id=='PickPlaceCounterToCabinet':
        # Upstream uses the FULL bbox and default th=.05. Axes are not
        # normalized, so th must not be described as a uniform 5cm margin.
        inside=bool(OU.obj_inside_of(target,'obj',target.cab))
        result.update(object_bbox_inside_cabinet=inside,native_containment_parameter_th=.05,
                      containment_parameter_units='upstream unnormalized dot-product formula',
                      native_task_success=inside and far)
    elif task_id=='PickPlaceSinkToCounter':
        inside=bool(OU.check_obj_in_receptacle(target,'obj','container'))
        contact=bool(target.check_contact(target.objects['container'],target.counter))
        result.update(object_in_receptacle=inside,receptacle_on_counter=contact,
                      native_task_success=inside and contact and far)
    else:raise ValueError('native predicate family not audited')
    return result


def run_metamorphic_check():
    """Real MuJoCo primitive geometry and pinned native predicate helpers.

    This is a representation-control fixture, not a native room episode or a
    proof of correspondence for generated assets. No renderer/GPU is required.
    """
    import mujoco
    from types import SimpleNamespace as NS
    from robocasa.models.objects.objects import MJCFObject
    from robocasa.models.fixtures import Fixture
    from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToSink
    obj=object.__new__(MJCFObject);obj._name='obj'
    fixture=object.__new__(Fixture)
    fixture.get_int_sites=lambda relative=False:{'inside':(np.array([-.2,-.2,-.2]),np.array([.2,-.2,-.2]),np.array([-.2,.2,-.2]),np.array([-.2,-.2,.2]))}
    angle=np.pi/3;R=Rotation.from_euler('z',angle).as_matrix();t=np.array([.1,0,0])
    inverse=np.eye(4);inverse[:3,:3]=R.T;inverse[:3,3]=-R.T@t
    q=Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
    qi=Rotation.from_matrix(R.T).as_quat()[[3,0,1,2]]
    def nums(v):return ' '.join(format(float(x),'.17g') for x in v)
    def make(shifted):
        pos=t if shifted else np.zeros(3);quat=q if shifted else [1,0,0,0]
        local=inverse[:3,3] if shifted else np.zeros(3);localq=qi if shifted else [1,0,0,0]
        xml=f'<mujoco><worldbody><site name="gripper" pos=".3 0 0"/><body name="obj" pos="{nums(pos)}" quat="{nums(quat)}"><freejoint/><inertial pos="{nums(local)}" quat="{nums(localq)}" mass="1" diaginertia=".001 .002 .003"/><geom type="box" size=".02 .03 .04" pos="{nums(local)}" quat="{nums(localq)}"/></body></worldbody></mujoco>'
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        data=NS(body_xpos=d.xpos,body_xquat=d.xquat,site_xpos=d.site_xpos)
        env=NS(sim=NS(data=data),obj_body_id={'obj':1},objects={'obj':obj},sink=fixture,
               robots=[NS(eef_site_id={'right':0})],get_fixture=lambda f:f)
        return env,m,d
    a,am,ad=make(False);b,bm,bd=make(True)
    raw_a=predicate_components(a,'PickPlaceCounterToSink');raw_b=predicate_components(b,'PickPlaceCounterToSink')
    bound=predicate_components(b,'PickPlaceCounterToSink',body_from_scorer={'obj':inverse.tolist()})
    original_predicate=bool(PickPlaceCounterToSink._check_success(a))
    bound_predicate=bool(PickPlaceCounterToSink._check_success(scorer_view(b,{'obj':inverse.tolist()})))
    geometry_error=float(np.max(np.abs(ad.geom_xpos-bd.geom_xpos)))
    orientation_error=float(np.max(np.abs(ad.geom_xmat-bd.geom_xmat)))
    com_error=float(np.max(np.abs(ad.xipos-bd.xipos)))
    return {'schema_version':2,'fixture_kind':'real_mujoco_primitive_and_pinned_native_scorer',
        'real_native_room_episodes':0,'mujoco_version':mujoco.__version__,
        'physical_geometry_max_abs_m':geometry_error,'physical_orientation_max_abs':orientation_error,
        'physical_COM_max_abs_m':com_error,'raw_original':raw_a,'raw_reexpressed':raw_b,
        'bound_reexpressed':bound,'body_from_scorer':inverse.tolist(),
        'origin_dependency_detected':raw_a['native_task_success']!=raw_b['native_task_success'],
        'passed':geometry_error<1e-14 and orientation_error<1e-14 and com_error<1e-14 and
                 original_predicate==bound_predicate and raw_a['native_task_success']==bound['native_task_success'],
        'thresholds_unchanged':True,'physics_modified_by_scorer':False,
        'generated_asset_correspondence':'NOT_ESTABLISHED; raw DEV success is binding-specific',
        'strong_generated_task_retention_gate':'NOT_RUN'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    args=parser.parse_args(argv);report=run_metamorphic_check()
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2));return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
