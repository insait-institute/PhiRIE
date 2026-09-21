"""Evaluator-only conservative native inventory AFTER observed component freeze.

This audit cannot certify obstacle absence or feed native geometry/identities back
into the constructor. No native state is advanced and no generated asset is moved.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from robo.roundtrip.identity import file_hash,validate_canonical_instance
from robo.roundtrip.build import _write


def collision_inventory(model,data,bounds):
    import mujoco
    bounds=np.asarray(bounds,float)
    if bounds.shape!=(2,3) or not np.isfinite(bounds).all() or np.any(bounds[1]<=bounds[0]):raise ValueError('invalid frozen workspace bounds')
    center=bounds.mean(0);half=(bounds[1]-bounds[0])/2
    explicit=set(map(int,model.pair_geom1))|set(map(int,model.pair_geom2));rows=[]
    for g in range(model.ngeom):
        if not(model.geom_contype[g] or model.geom_conaffinity[g] or g in explicit):continue
        R=np.asarray(data.geom_xmat[g]).reshape(3,3);position=np.asarray(data.geom_xpos[g]);kind=int(model.geom_type[g])
        if kind==int(mujoco.mjtGeom.mjGEOM_PLANE):
            intersects=bool(abs(np.dot(center-position,R[:,2]))<=np.dot(half,np.abs(R[:,2])))
            box=None;contained=False
        else:
            local=np.asarray(model.geom_aabb[g]);world_center=R@local[:3]+position;extent=np.abs(R)@local[3:]
            box=np.array([world_center-extent,world_center+extent]);intersects=bool(np.all(box[1]>=bounds[0]) and np.all(box[0]<=bounds[1]));contained=bool(np.all(box[0]>=bounds[0]) and np.all(box[1]<=bounds[1]))
        body=int(model.geom_bodyid[g]);ancestors=[]
        while body:
            ancestors.append(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body));body=int(model.body_parentid[body])
        rows.append(dict(native_geom=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g),native_body_ancestry=ancestors,
            geom_type=kind,explicit_contact_pair=g in explicit,collision_masks=[int(model.geom_contype[g]),int(model.geom_conaffinity[g])],
            canonical_world_aabb_m=None if box is None else box.tolist(),intersects_declared_workspace=intersects,
            contained_in_declared_workspace=contained,retained_reference_in_observed_component_only_stage=True,
            constructor_role_correspondence='UNRESOLVED',actual_policy_contact='NOT_RUN'))
    return rows


def audit(components_path,inventory_path,binding_path,out):
    """Check the entire immutable constructor closure before opening native data."""
    components_path=Path(components_path);inventory_path=Path(inventory_path);binding_path=Path(binding_path);out=Path(out)
    c=json.loads(components_path.read_text());inv=json.loads(inventory_path.read_text())
    if c.get('tier')!='DEV' or c.get('method')!='OBSERVED_WORKSPACE_COMPONENT_TSDF' or c['inventory_manifest_sha256']!=file_hash(inventory_path):raise ValueError('frozen observed component provenance differs')
    for row in c['rows']:
        if row['status']!='BUILT':continue
        directory=components_path.parent/row['cluster_id']
        for name,digest in row['artifact_hashes'].items():
            p=Path(name)
            if p.is_absolute() or '..' in p.parts or file_hash(directory/p)!=digest:raise ValueError('constructed component changed before native audit')
    b=json.loads(binding_path.read_text())
    if b['canonical_instance_id']!=c['canonical_instance_id'] or b['capture_manifest_sha256']!=c['capture_manifest_sha256']:raise ValueError('native/capture identity differs')
    manifest_path=Path(b['canonical_manifest']);bundle=Path(b['bundle_dir']);manifest=json.loads(manifest_path.read_text());validate_canonical_instance(manifest,bundle)
    import mujoco
    xml=bundle/'scene.xml';state_path=bundle/'canonical_state.json';state=json.loads(state_path.read_text())
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model);mujoco.mj_setState(model,data,np.asarray(state['integration_state'],dtype=np.float64),mujoco.mjtState.mjSTATE_INTEGRATION)
    physical=(data.qpos.copy(),data.qvel.copy(),data.ctrl.copy(),float(data.time));mujoco.mj_forward(model,data)
    if any(not np.array_equal(a,b) for a,b in zip(physical[:3],(data.qpos,data.qvel,data.ctrl))) or physical[3]!=float(data.time):raise ValueError('static evaluator changed physical state')
    rows=collision_inventory(model,data,inv['workspace_bounds_world_m']);out.mkdir(parents=True,exist_ok=False)
    result=dict(schema_version=1,kind='evaluator_only_native_workspace_inventory',canonical_instance_id=c['canonical_instance_id'],
        source_hashes={str(p):file_hash(p) for p in [components_path,inventory_path,binding_path,manifest_path,xml,state_path]},
        workspace_bounds_world_m=inv['workspace_bounds_world_m'],native_collision_geometries=len(rows),
        workspace_intersection_candidates=sum(r['intersects_declared_workspace'] for r in rows),rows=rows,
        intersection_semantics='conservative compiled collision AABB/plane intersection, not geometric contact or semantic obstacle recovery',
        constructed_component_count=c['built_components'],constructor_unknown_roles=c['inherited_unresolved_roles'],
        canonical_native_task_objects=list(state['object_states']),role_correspondence='UNRESOLVED; no GT-based selection or pose correction performed',
        native_asset_access=True,constructor_access_to_this_audit=False,constructor_mutated=False,simulation_steps=0,
        native_policy_outcomes_accessed=False,L2_READY=False,obstacle_absence='NOT_ESTABLISHED',scope_import='NOT_RUN')
    _write(out/'native_inventory.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['components','inventory','binding','out']:p.add_argument('--'+k,required=True)
    a=p.parse_args();r=audit(a.components,a.inventory,a.binding,a.out);print(json.dumps({k:r[k] for k in ['native_collision_geometries','workspace_intersection_candidates','L2_READY']}))
if __name__=='__main__':main()
