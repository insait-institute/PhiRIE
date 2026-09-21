"""Evaluator-side replacement of a static fixture's direct physical component.

Articulated child bodies, native goal regions and fixture pose stay explicitly
privileged context. Only sealed observed meshes enter the generated component.
No reference geometry or frame is used to estimate or recenter those meshes.
"""
from __future__ import annotations

import copy
import hashlib
import re
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from robo.roundtrip.importers.robocasa import _body, _numbers, import_reconstructed_object


def component_roster(xml, body_name, component_kind='sink_basin'):
    """Privileged evaluator mapping, never a constructor input or object mask."""
    import mujoco
    root = ET.fromstring(xml)
    body, parent = _body(root, body_name)
    parents = {c: p for p in root.iter() for c in p}
    current = body
    while current.tag != 'worldbody':
        if current.tag != 'body' or current.find('joint') is not None or current.find('freejoint') is not None or current.get('mocap') == 'true':
            raise ValueError('fixture component requires a static ancestor chain')
        current = parents[current]
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names, retained = [], []
    for geom in body.findall('geom'):
        name = geom.get('name')
        if not name:
            raise ValueError('fixture component requires named direct geoms')
        gid = model.geom(name).id
        physical = bool(model.geom_contype[gid] or model.geom_conaffinity[gid])
        visible = bool(model.geom_rgba[gid, 3] > 0)
        (names if physical or visible else retained).append(name)
    if not names:
        raise ValueError('no direct physical fixture component')
    retained_physical = []
    if component_kind == 'cabinet_bottom_shelf':
        prefix = body_name.removesuffix('_main')
        selected = [prefix+'_bottom', prefix+'_bottom_visual']
        if not all(n in names for n in selected):
            raise ValueError('cabinet bottom shelf requires separate named collision and visual geoms')
        retained_physical = [n for n in names if n not in selected]
        names = selected
    elif component_kind == 'counter_top':
        prefix=body_name.removesuffix('_main')
        pattern=re.compile(re.escape(prefix)+r'_top(?:_(?:left|right|front|back))?(?:_visual|_\d+)?$')
        selected=[n for n in names if pattern.fullmatch(n)]
        if not selected or not any(n.endswith('_visual') for n in selected):
            raise ValueError('counter top requires official named top visual and collision chunks')
        if not any(bool(model.geom_contype[model.geom(n).id] or model.geom_conaffinity[model.geom(n).id]) for n in selected):
            raise ValueError('counter top lacks physical chunks')
        retained_physical=[n for n in names if n not in selected]
        names=selected
    elif component_kind != 'sink_basin':
        raise ValueError('unsupported static fixture component')
    bid = model.body(body_name).id
    pose = np.eye(4)
    pose[:3, :3] = data.xmat[bid].reshape(3, 3)
    pose[:3, 3] = data.xpos[bid]
    return {'body_name': body_name, 'direct_physical_geoms': names,
            'component_kind':component_kind, 'retained_direct_physical_geoms':retained_physical,
            'retained_direct_metadata_geoms': retained,
            'retained_child_bodies': [b.get('name') for b in body.iter('body') if b is not body],
            'fixture_frame_world': pose.tolist(), 'privileged_evaluator_mapping': True}


def import_fixture_component(xml, *, body_name, object_dir, object_id,
                             role='receptacle', component_kind='sink_basin', **kwargs):
    """Replace direct basin geoms while preserving hierarchy and world estimate.

    This bounded route covers a basin shell, not faucet recovery or a whole sink.
    Cabinet panels need a separate explicit physical-component roster.
    """
    if (role,component_kind) not in (('receptacle','sink_basin'),('support','sink_basin'),('support','cabinet_bottom_shelf'),('support','counter_top')):
        raise ValueError('only static sink_basin receptacle or cabinet_bottom_shelf support components admitted')
    roster = component_roster(xml, body_name, component_kind)
    root = ET.fromstring(xml)
    body, _ = _body(root, body_name)
    removed = set(roster['direct_physical_geoms'])
    for section in ('contact', 'equality', 'tendon', 'sensor'):
        for elem in root.findall(f'./{section}//*'):
            if removed.intersection(elem.attrib.values()):
                raise ValueError('external reference to removed fixture component geom')
    # Reuse the canonical asset importer without compiling its temporary XML.
    # Child joints/actuators remain in the final original tree, never detached.
    scratch = copy.deepcopy(root)
    scratch_body, scratch_parent = _body(scratch, body_name)
    for child in list(scratch_body):
        if child.tag != 'geom' or child.get('name') not in removed:
            scratch_body.remove(child)
    # Canonical importer only needs a static world parent to author world poses.
    scratch_parent.remove(scratch_body)
    scratch.find('worldbody').append(scratch_body)
    generated_xml, receipt = import_reconstructed_object(ET.tostring(scratch, encoding='unicode'),
        body_name=body_name, object_dir=object_dir, object_id=object_id, role=role, **kwargs)
    generated = ET.fromstring(generated_xml)
    generated_body, _ = _body(generated, body_name)
    original_assets = root.find('asset')
    if original_assets is None:
        original_assets = ET.SubElement(root, 'asset')
    existing_names = {e.get('name') for e in original_assets}
    for asset in generated.find('asset'):
        if asset.get('name') not in existing_names:
            original_assets.append(copy.deepcopy(asset))
    estimated = np.eye(4)
    estimated[:3, :3] = Rotation.from_quat(np.asarray(receipt['quaternion_wxyz'])[[1, 2, 3, 0]]).as_matrix()
    estimated[:3, 3] = receipt['position_m']
    local = np.linalg.inv(np.asarray(roster['fixture_frame_world'])) @ estimated
    local_q = Rotation.from_matrix(local[:3, :3]).as_quat()[[3, 0, 1, 2]]
    insertion = min(list(body).index(g) for g in body.findall('geom') if g.get('name') in removed)
    for geom in list(body.findall('geom')):
        if geom.get('name') in removed:
            body.remove(geom)
    for offset, geom in enumerate(generated_body.findall('geom')):
        geom = copy.deepcopy(geom)
        geom.set('pos', _numbers(local[:3, 3]))
        geom.set('quat', _numbers(local_q))
        body.insert(insertion + offset, geom)
    # MuJoCo requires a positive body mass for subtree COM bookkeeping when a
    # static fixture retains articulated children. Use the same generated s6
    # prior as the canonical importer, expressed in this unchanged static frame.
    # No native mass/inertia is copied, and no dynamic ancestor is admitted.
    for inertial in list(body.findall('inertial')):
        body.remove(inertial)
    inertial = copy.deepcopy(generated_body.find('inertial'))
    component_com = np.fromstring(inertial.get('pos'), sep=' ')
    inertial.set('pos', _numbers(local[:3,:3] @ component_com + local[:3,3]))
    inertial.set('quat', _numbers(local_q))
    body.insert(0, inertial)
    output = ET.tostring(root, encoding='unicode')
    receipt.update(import_route='static_fixture_component_v1', component_kind=component_kind,
        scope={'sink_basin':'fixture_direct_basin_shell_only','cabinet_bottom_shelf':'fixture_bottom_support_surface_only','counter_top':'fixture_counter_top_surface_only'}[component_kind], body_semantics='static_component',
        component_roster=roster, component_local_transform=local.tolist(),
        retained_fixture_body_pose=True, retained_fixture_children=True,
        body_local_position_m=None, body_local_quaternion_wxyz=None,
        dynamic_mass_prior_applied=False, static_compilation_inertial_prior='generated s6 box; transformed estimated COM',
        removed_original_geoms=sorted(removed),
        removed_original_sites=[], native_goal_region_source='unchanged privileged evaluator task regions',
        source_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        imported_xml_sha256=hashlib.sha256(output.encode()).hexdigest(),
        native_scorer_binding='PENDING_FIXTURE_BINDING',
        retained_native_context='fixture metadata and all child bodies explicitly retained; see component_roster')
    return output, receipt


def import_fixture_identity(xml, *, body_name, component_kind='sink_basin'):
    """Detach/reinsert the exact component geoms, retaining child hierarchy."""
    roster = component_roster(xml, body_name, component_kind)
    root = ET.fromstring(xml)
    body, _ = _body(root, body_name)
    before = ET.tostring(body)
    for index, child in enumerate(list(body)):
        if child.tag == 'geom' and child.get('name') in roster['direct_physical_geoms']:
            body.remove(child)
            body.insert(index, copy.deepcopy(child))
    output = ET.tostring(root, encoding='unicode')
    return output, {'control': 'U1_static_fixture_component', 'body_name': body_name,
        'component_roster': roster, 'body_xml_equal': ET.tostring(body) == before,
        'privileged_native_asset_access': True, 'runtime_identity': 'NOT_RUN',
        'source_xml_sha256': hashlib.sha256(xml.encode()).hexdigest(),
        'imported_xml_sha256': hashlib.sha256(output.encode()).hexdigest()}


def bind_fixture_component(env, *, fixture_name, receipt):
    """Refresh contact IDs; preserve native goal regions/rubric byte for byte."""
    fixture = env.get_fixture(fixture_name)
    if receipt.get('import_route') != 'static_fixture_component_v1' or fixture.root_body != receipt['body_name']:
        raise ValueError('static fixture role/body binding mismatch')
    removed = set(receipt['removed_original_geoms'])
    prefix = fixture.naming_prefix
    contacts = list(dict.fromkeys([n for n in fixture.contact_geoms if n not in removed] + receipt['contact_geoms']))
    visuals = list(dict.fromkeys([n for n in fixture.visual_geoms if n not in removed] + receipt['visual_geoms']))
    for name in contacts + visuals:
        if not name.startswith(prefix):
            raise ValueError('fixture geom naming prefix mismatch')
        env.sim.model.geom_name2id(name)
    fixture._contact_geoms = [n[len(prefix):] for n in contacts]
    fixture._visual_geoms = [n[len(prefix):] for n in visuals]
    return {'fixture_name': fixture_name, 'contact_geoms': contacts, 'visual_geoms': visuals,
            'native_goal_region_source': 'unchanged privileged evaluator task regions',
            'native_predicate': 'unchanged native function and goal regions',
            'opening_preservation': 'NOT_RUN', 'reconstructed_interior_correspondence': 'NOT_ESTABLISHED'}


from contextlib import contextmanager


@contextmanager
def staged_fixture_components(env, xml, receipts):
    """Stage Python handles before upstream reset generates geom-ID mappings.

    Native metadata reset must precede this context. All plans validate against
    the final XML before mutation. On reset/compile failure, restore old caches;
    after success, the usual post-import binder validates compiled IDs.
    """
    root=ET.fromstring(xml);plans=[];seen=set()
    models=list(env.fixtures.values())+list(env.model.mujoco_objects)
    for receipt in receipts:
        if receipt.get('import_route')!='static_fixture_component_v1':
            raise ValueError('fixture prestaging requires a static-component receipt')
        body,_=_body(root,receipt['body_name'])
        direct={g.get('name') for g in body.findall('geom')}
        required=set(receipt['contact_geoms']+receipt['visual_geoms'])
        if not required.issubset(direct) or set(receipt['removed_original_geoms']) & direct:
            raise ValueError('fixture prestage final XML differs from sealed component receipt')
        matches=[m for m in models if getattr(m,'root_body',None)==receipt['body_name']]
        if not matches:raise ValueError('native fixture metadata handle is missing')
        for fixture in matches:
            if id(fixture) in seen:continue
            seen.add(id(fixture));prefix=fixture.naming_prefix;removed=set(receipt['removed_original_geoms'])
            names={g.get('name') for g in body.iter('geom')}
            fields={}
            for private,public,new in [('_contact_geoms','contact_geoms',receipt['contact_geoms']),('_visual_geoms','visual_geoms',receipt['visual_geoms'])]:
                values=list(dict.fromkeys([n for n in getattr(fixture,public) if n not in removed]+new))
                if any(not n.startswith(prefix) or n not in names for n in values):
                    raise ValueError('fixture prestage contains an absent geom or foreign prefix')
                fields[private]=[n[len(prefix):] for n in values]
            plans.append((fixture,fields,{key:getattr(fixture,key) for key in fields}))
    try:
        for fixture,fields,_ in plans:
            for key,value in fields.items():setattr(fixture,key,value)
        yield
    except BaseException:
        for fixture,_,previous in plans:
            for key,value in previous.items():setattr(fixture,key,value)
        raise


def check_native_sink_scorer_identity(original_xml, imported_xml, *, body_name, task_id='PickPlaceCounterToSink'):
    """Pinned native predicate on an actual scene's unchanged goal regions.

    These are explicit scorer probes (including outside/near-gripper negatives),
    not native policy episodes or generated-interior correspondence evidence.
    """
    import mujoco
    from types import SimpleNamespace as NS
    from robocasa.models.objects.objects import MJCFObject
    from robocasa.models.fixtures import Fixture
    from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToSink, PickPlaceCounterToCabinet
    if task_id not in ('PickPlaceCounterToSink','PickPlaceCounterToCabinet'):
        raise ValueError('unsupported fixture scorer family')
    regions = []
    for xml in (original_xml, imported_xml):
        root = ET.fromstring(xml); body, _ = _body(root, body_name)
        model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        points = {}
        for geom in body.findall('geom'):
            marker = '_reg_basin' if task_id=='PickPlaceCounterToSink' else '_reg_level'
            if marker not in geom.get('name', ''):
                continue
            gid = model.geom(geom.get('name')).id
            size = model.geom_size[gid]; R = data.geom_xmat[gid].reshape(3, 3); p = data.geom_xpos[gid]
            corners = np.asarray([-size, size * [1,-1,-1], size * [-1,1,-1], size * [-1,-1,1]])
            points[geom.get('name')] = corners @ R.T + p
        if not points:
            raise ValueError('actual sink basin goal regions not available')
        regions.append(points)
    if regions[0].keys() != regions[1].keys():
        raise ValueError('native goal region inventory changed')
    equal = all(np.array_equal(regions[0][k], regions[1][k]) for k in regions[0])
    obj = object.__new__(MJCFObject); obj._name = 'obj'
    from itertools import product
    offsets=np.asarray(list(product((-.01,.01),repeat=3)))
    obj.get_bbox_points=lambda trans,rot:offsets+trans
    queries = []
    for name, corners in regions[0].items():
        p0, px, py, pz = corners; basis = np.stack([px-p0, py-p0, pz-p0])
        for label, coeff in [('inside',[.5,.5,.5]), ('outside',[2,.5,.5]),
                             ('below',[.5,.5,-.1]), ('above',[.5,.5,1.1])]:
            pos = p0 + np.asarray(coeff) @ basis
            for retreat in (.1, .4):
                outcomes = []
                for group in regions:
                    fixture = object.__new__(Fixture)
                    fixture.get_int_sites = lambda relative=False, group=group: group
                    env = NS(sim=NS(data=NS(body_xpos=np.asarray([pos]),
                                           body_xquat=np.asarray([[1.,0.,0.,0.]]),
                                           site_xpos=np.asarray([pos+[retreat,0,0]]))),
                             objects={'obj':obj}, obj_body_id={'obj':0}, sink=fixture,
                             robots=[NS(eef_site_id={'right':0})], get_fixture=lambda f:f)
                    env.cab=fixture
                    predicate=PickPlaceCounterToSink if task_id=='PickPlaceCounterToSink' else PickPlaceCounterToCabinet
                    outcomes.append(bool(predicate._check_success(env)))
                queries.append({'region':name, 'probe':label, 'retreat_m':retreat,
                                'original_success':outcomes[0], 'imported_success':outcomes[1]})
    return {'status':'PASS' if equal and all(q['original_success']==q['imported_success'] for q in queries) else 'FAIL',
            'native_goal_region_world_points_byte_equal':equal, 'probes':queries,
            'contains_positive_and_negative':{q['original_success'] for q in queries} == {True,False},
            'task':task_id, 'real_native_episodes':0,
            'protocol':'actual native goal geometry; synthetic scorer queries; unchanged pinned native predicate',
            'generated_interior_correspondence':'NOT_ESTABLISHED'}


def main(argv=None):
    import argparse
    import json
    from pathlib import Path
    import time
    from robo.roundtrip.scope import check_scope_identity_physics
    parser = argparse.ArgumentParser(description='Actual native CPU static-component identity and scorer control')
    parser.add_argument('--xml', required=True); parser.add_argument('--state', required=True)
    parser.add_argument('--body-name', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--component-kind', choices=['sink_basin','cabinet_bottom_shelf'],default='sink_basin')
    args = parser.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); xml = Path(args.xml); state = Path(args.state)
    imported, physics = check_scope_identity_physics(xml.read_text(),
        entities=[{'body_name':args.body_name,'role':'receptacle' if args.component_kind=='sink_basin' else 'support','component_kind':args.component_kind}],
        integration_state=json.loads(state.read_text())['integration_state'], steps=200)
    (out/'physics_identity.json').write_text(json.dumps(physics,indent=2)+'\n')
    scorer = check_native_sink_scorer_identity(xml.read_text(), imported, body_name=args.body_name,
        task_id='PickPlaceCounterToSink' if args.component_kind=='sink_basin' else 'PickPlaceCounterToCabinet')
    result = {'status':'PASS' if physics['runtime_identity']=='PASS' and scorer['status']=='PASS' and scorer['contains_positive_and_negative'] else 'FAIL',
              'physics':physics, 'scorer':scorer, 'renderer_observation_identity':'NOT_RUN',
              'source_hashes':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (xml,state)},
              'elapsed_s':time.monotonic()-started}
    (out/'actual_imported.xml').write_text(imported)
    (out/'import_identity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2)); return 0 if result['status']=='PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
