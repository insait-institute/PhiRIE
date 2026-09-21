"""Transactional L0/L1 rigid-body scope import and explicit retained inventory.

Only sealed existing factory outputs enter reconstruction. Native XML is read on
this evaluator side to retain task/robot identity and convert parent frames.
"""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET

from robo.roundtrip.importers.robocasa import import_identity, import_reconstructed_object, rebind_native_object


def _subtrees(xml, entities):
    root = ET.fromstring(xml)
    names = [e['body_name'] for e in entities]
    if len(names) != len(set(names)):
        raise ValueError('duplicate replacement body')
    found = {}
    for name in names:
        matches = [b for b in root.iter('body') if b.get('name') == name]
        if len(matches) != 1:
            raise ValueError('replacement body missing or ambiguous')
        descendants = {b.get('name') for b in matches[0].iter('body')}
        if (descendants - {name}) & set(names):
            raise ValueError('overlapping replacement subtrees')
        found[name] = descendants
    return root, found


def import_scope(xml, *, entities, scope, workspace_bounds_world_m=None):
    """Pure transaction: return XML, manifest; failure cannot mutate native env.

    entity keys: body_name, object_id, object_dir, role and optional frozen
    rgba/contact_prior/texture_path. L1 requires one target and one destination.
    Arbitrary fixture articulation and L2/L3 are not silently approximated.
    """
    if scope not in ('L0_target_only', 'L1_target_destination'):
        raise ValueError('only bounded L0/L1 import is implemented')
    roles = [e['role'] for e in entities]
    if roles.count('target') != 1:
        raise ValueError('exactly one manipulated target required')
    if scope == 'L0_target_only' and len(entities) != 1:
        raise ValueError('L0 cannot contain undeclared context replacement')
    if scope == 'L1_target_destination' and (len(entities) != 2 or
            sum(r in ('receptacle', 'support') for r in roles) != 1):
        raise ValueError('L1 needs exactly one reconstructed destination receptacle/support')
    root, subtrees = _subtrees(xml, entities)
    output, receipts = xml, []
    for entity in entities:
        if entity.get('component_kind') is not None:
            from robo.roundtrip.fixture_scope import import_fixture_component
            output, receipt = import_fixture_component(output, **entity)
        else:
            output, receipt = import_reconstructed_object(output, **entity)
        receipts.append(receipt)
    covered = {name: receipt['object_id'] for receipt in receipts
               if receipt.get('import_route') != 'static_fixture_component_v1'
               for name in subtrees[receipt['body_name']]}
    inventory = [{'native_body_name': b.get('name'),
                  'classification': 'reconstructed' if b.get('name') in covered else 'retained_reference',
                  'reconstruction_object_id': covered.get(b.get('name'))}
                 for b in root.iter('body')]
    for receipt in receipts:
        if receipt.get('import_route') == 'static_fixture_component_v1':
            inventory.append({'native_component_body': receipt['body_name'],
                'classification': 'reconstructed_direct_physical_geoms',
                'reconstruction_object_id': receipt['object_id'],
                'removed_native_geoms': receipt['removed_original_geoms'],
                'retained_reference_components': receipt['component_roster']})
    # Body inventory alone omits room planes and direct world geoms.
    inventory += [{'native_world_geom_name': g.get('name'), 'classification': 'retained_reference',
                   'reconstruction_object_id': None} for g in root.findall('./worldbody/geom')]
    joint_poses = {r['free_joint_name']: r['position_m'] + r['quaternion_wxyz']
                   for r in receipts if r['body_semantics'] == 'free'}
    if None in joint_poses:
        raise ValueError('multi-body state remap requires named free joints')
    manifest = {'schema_version': 1, 'scope': scope, 'import_status': 'IMPORTED_NOT_EXECUTED',
                'entities': receipts, 'inventory': inventory,
                'intentionally_excluded_entities': [], 'failed_entities': [],
                'workspace_bounds_world_m': workspace_bounds_world_m,
                'oracle_context_retained': True,
                'source_xml_sha256': hashlib.sha256(xml.encode()).hexdigest(),
                'imported_xml_sha256': hashlib.sha256(output.encode()).hexdigest(),
                'replaced_joints': joint_poses,
                'state_remap_policy': 'by named joint; adapter must validate unchanged integration topology',
                'opening_preservation': 'NOT_RUN', 'native_identity': 'NOT_RUN',
                'retained_context_contact_interactions': None,
                'absolute_scorer_correspondence': 'NOT_ESTABLISHED',
                'strong_task_retention_claim': 'NOT_RUN'}
    return output, manifest


def bind_scope_objects(env, *, manifest, native_role_names):
    """Refresh movable native object handles AFTER adapter import, never build.

    Native fixture-specific inside sites/articulations need their own binding;
    this bounded path refuses them instead of retaining stale collision handles.
    """
    receipts = manifest['entities']
    if set(native_role_names) != {r['object_id'] for r in receipts}:
        raise ValueError('every reconstructed object requires exactly one native role')
    if len(set(native_role_names.values())) != len(native_role_names):
        raise ValueError('duplicate native role binding')
    for receipt in receipts:
        name = native_role_names[receipt['object_id']]
        if receipt.get('import_route') == 'static_fixture_component_v1':
            fixture = env.get_fixture(name)
            if fixture.root_body != receipt['body_name']:
                raise ValueError('static fixture body identity differs')
            continue
        if name not in env.objects or env.objects[name].root_body != receipt['body_name']:
            raise ValueError('native fixture binding not implemented or body identity differs')
        for geom in receipt['contact_geoms'] + receipt['visual_geoms']:
            env.sim.model.geom_name2id(geom)
    from robo.roundtrip.fixture_scope import bind_fixture_component
    bindings = [(bind_fixture_component(env, fixture_name=native_role_names[r['object_id']], receipt=r)
                 if r.get('import_route') == 'static_fixture_component_v1' else
                 rebind_native_object(env, object_name=native_role_names[r['object_id']], receipt=r))
                for r in receipts]
    return {'bindings': bindings, 'native_predicate': 'unchanged function; generated origins/radii are binding-specific',
            'absolute_scorer_correspondence': 'NOT_ESTABLISHED', 'whole_mesh_containment_certified': False}


def import_scope_identity(xml, *, entities):
    """Privileged unchanged-import control for declared target/context roles."""
    _subtrees(xml, entities)
    output, receipts = xml, []
    for entity in entities:
        if entity['role'] not in ('target', 'receptacle', 'support', 'obstacle'):
            raise ValueError('unknown identity-control role')
        if entity.get('component_kind') in ('sink_basin','cabinet_bottom_shelf','counter_top'):
            from robo.roundtrip.fixture_scope import import_fixture_identity
            output, receipt = import_fixture_identity(output, body_name=entity['body_name'],component_kind=entity['component_kind'])
        elif entity.get('component_kind') is not None:
            raise ValueError('unsupported identity component kind')
        else:
            output, receipt = import_identity(output, body_name=entity['body_name'])
        receipts.append(dict(receipt, role=entity['role']))
    return output, {'control': 'U1_scope', 'privileged_native_asset_access': True,
                    'entities': receipts, 'runtime_identity': 'NOT_RUN'}


def check_scope_identity_physics(xml, *, entities, integration_state, steps=200):
    """Real-MuJoCo unchanged-role import from an actual native reset state.

    Holds the saved controls constant for this bounded engineering control.
    Policy/renderer/native-predicate controls are explicitly separate, NOT_RUN.
    """
    import mujoco
    import numpy as np
    if not isinstance(steps, int) or not 1 <= steps <= 10000:
        raise ValueError('identity physics steps must be bounded')
    output, report = import_scope_identity(xml, entities=entities)
    models = [mujoco.MjModel.from_xml_string(value) for value in (xml, output)]
    datas = [mujoco.MjData(model) for model in models]
    fields = ('body_pos', 'body_quat', 'body_mass', 'body_inertia', 'body_ipos', 'body_iquat',
              'geom_pos', 'geom_quat', 'geom_size', 'geom_friction', 'geom_contype',
              'geom_conaffinity', 'geom_solimp', 'geom_solref', 'geom_matid', 'mat_rgba',
              'jnt_type', 'jnt_qposadr', 'jnt_dofadr', 'actuator_trnid')
    equality = {key: bool(np.array_equal(getattr(models[0], key), getattr(models[1], key))) for key in fields}
    state = np.asarray(integration_state, float)
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    for model, data in zip(models, datas):
        if state.shape != (mujoco.mj_stateSize(model, kind),) or not np.isfinite(state).all():
            raise ValueError('canonical integration state shape/nonfinite mismatch')
        mujoco.mj_setState(model, data, state, kind)
    state_equal, qpos_error, qvel_error = True, 0.0, 0.0
    for _ in range(steps):
        snapshots = []
        for model, data in zip(models, datas):
            mujoco.mj_step(model, data)
            snapshot = np.empty_like(state)
            mujoco.mj_getState(model, data, snapshot, kind)
            snapshots.append(snapshot)
        state_equal &= bool(np.array_equal(*snapshots))
        qpos_error = max(qpos_error, float(np.max(np.abs(datas[0].qpos - datas[1].qpos))))
        qvel_error = max(qvel_error, float(np.max(np.abs(datas[0].qvel - datas[1].qvel))))
    report.update(mujoco_version=mujoco.__version__, compiled_fields_equal=equality,
                  integration_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                  steps=steps, timestep_s=float(models[0].opt.timestep),
                  integration_states_byte_equal=state_equal, qpos_max_abs=qpos_error, qvel_max_abs=qvel_error,
                  runtime_identity='PASS' if state_equal and all(equality.values()) else 'FAIL',
                  native_predicate_identity='NOT_RUN', renderer_observation_identity='NOT_RUN',
                  control_protocol='saved canonical actuator controls held constant; no policy inference')
    return output, report


def main(argv=None):
    """Bounded CPU role-identity control, not a policy experiment launcher."""
    import argparse
    import json
    import time
    from pathlib import Path
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--xml', required=True)
    parser.add_argument('--canonical-state', required=True)
    parser.add_argument('--entities', required=True, help='JSON list of native body_name/role bindings; privileged U1 only')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('immutable identity output exists')
    started = time.monotonic()
    xml, state = Path(args.xml), Path(args.canonical_state)
    entities = Path(args.entities)
    output, report = check_scope_identity_physics(xml.read_text(),
        entities=json.loads(entities.read_text()),
        integration_state=json.loads(state.read_text())['integration_state'], steps=args.steps)
    report['sources'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (xml, state, entities)}
    report['elapsed_s'] = time.monotonic() - started
    out.mkdir(parents=True, exist_ok=False)
    (out / 'actual_imported.xml').write_text(output)
    (out / 'import_identity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['runtime_identity'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
