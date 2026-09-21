"""Authenticate public grounding selections before canonical E6 graph/features.

This is an artifact adapter. It does not discover objects, evaluate GT, infer
new role assignments, or measure mask/physics/robot quality.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from run.icra2027 import e6_public_reconstruction as shared

PRODUCER_SHA = 'f13248fa7c47db3e7657b638175e349d185ba30d'
KEYS = {'producer', 'config', 'gate'}


def _reference_context(reference):
    if set(reference) != KEYS or set(reference['producer']) != {'path', 'commit'}:
        raise ValueError('public grounding reference schema differs')
    code = Path(reference['producer']['path']).resolve(strict=True)
    shared.sealed.sealed_cpu._inside(code, root=shared.ROOT.parent, label='grounding publisher source')
    snapshot = shared.git_snapshot(code)
    if (reference['producer']['commit'] != PRODUCER_SHA
        or snapshot['commit'] != PRODUCER_SHA or snapshot['dirty']):
        raise ValueError('public grounding publisher source differs')
    for name in ('config', 'gate'):
        ref = reference[name]
        shared.sealed.sealed_cpu._inside(Path(ref['path']), root=shared.ROOT, label='public grounding '+name)
        if shared.identity(ref['path']) != ref:
            raise ValueError('public grounding '+name+' bytes changed')
    config_path = Path(reference['config']['path']);stage = config_path.parent
    gate_path = Path(reference['gate']['path'])
    gate = shared.read(gate_path)
    expected = stage/'audit/public_grounding_cohort'/(gate['scene_id']+'_'+gate['condition_id'])/'terminal/gate.json'
    if gate_path != expected or gate['source_commit'] != PRODUCER_SHA:
        raise ValueError('public grounding gate path/source differs')
    return code, config_path, stage, gate, shared.canonical_hash(reference)


def validate_reference(reference, cache=None):
    code, config_path, stage, gate, key = _reference_context(reference)
    if cache is not None and key in cache:
        if cache[key] != gate:raise ValueError('cached public grounding changed')
        return deepcopy(gate)
    cfg = shared.read(config_path)
    measurement = shared.read(cfg['fresh_config']['path'])
    rgb = shared.read(measurement['rgb_runtime']['path'])
    env = shared.environment(rgb, stage, 'validation')
    env.update(PYTHONPATH=str(code), SIMANY_ROOT=str(code))
    command = [rgb['python'], '-m', 'run.icra2027.e6_grounding_cohort', '--config', str(config_path),
               '--stage-root', str(stage), '--scene', gate['scene_id'], '--condition', gate['condition_id'], '--validate-output']
    result = subprocess.run(command, cwd=code, env=env, text=True, capture_output=True)
    if result.returncode or json.loads(result.stdout) != gate:
        raise ValueError('original public grounding validation failed: '+result.stderr[-1200:])
    if cache is not None:cache[key] = deepcopy(gate)
    return gate



def validate_references(references, *, tier):
    """One original context replay, every fixed unit replay, invocation-local cache."""
    contexts = [_reference_context(ref) for ref in references]
    scenes = list(shared.SCENES) if tier == 'full' else ['behavior_task0020'] if tier == 'pilot' else None
    if scenes is None:raise ValueError('public batch requires fixed pilot or full tier')
    expected = [(scene,c) for scene in scenes for c in shared.CONDITIONS]
    cells = [(ctx[3]['scene_id'],ctx[3]['condition_id']) for ctx in contexts]
    if cells != expected:raise ValueError('public batch cannot omit, add, reorder or duplicate units')
    if any(ref['producer'] != references[0]['producer'] or ref['config'] != references[0]['config'] for ref in references):
        raise ValueError('public batch cannot mix source/config identities')
    code, config_path, stage, _, _ = contexts[0]
    cfg = shared.read(config_path);measurement = shared.read(cfg['fresh_config']['path'])
    rgb = shared.read(measurement['rgb_runtime']['path'])
    env = shared.environment(rgb,stage,'validation');env.update(PYTHONPATH=str(code),SIMANY_ROOT=str(code))
    program = ('import json;from run.icra2027 import e6_grounding_cohort as d;'
        'cfg,roster=d.validate('+repr(str(config_path))+','+repr(str(stage))+');'
        'print(json.dumps([d._validate_output(cfg,roster,'+repr(str(stage))+',scene,condition) '
        'for scene,condition in '+repr(cells)+']))')
    result = subprocess.run([rgb['python'],'-c',program],cwd=code,env=env,text=True,capture_output=True)
    if result.returncode or json.loads(result.stdout) != [ctx[3] for ctx in contexts]:
        raise ValueError('original public batch validation differs: '+result.stderr[-1200:])
    # No module-global or persisted cache: every stage invocation runs this proof.
    return {ctx[4]:deepcopy(ctx[3]) for ctx in contexts}


def bundle_from_gate(gate, reference, *, freeze_id, task_id):
    """Deterministic schema adaptation; no semantic/metric decisions are added."""
    original = gate['original_gate']
    queries = [q for q in original['queries'] if q['task_id'] == task_id]
    if len(queries) != 1:raise ValueError('query absent or duplicated in public grounding')
    query = queries[0]
    if (original['robot_frame'] is not None or original['physics_verified'] is not False
        or original['feature_rows_written'] != 0 or original['paper_ready'] is not False):
        raise ValueError('unsupported upstream robot/physics/feature promotion')
    objects = deepcopy(original['discovered_objects'] or [])
    object_ids = {o['id'] for o in objects}
    if len(object_ids) != len(objects):raise ValueError('duplicate public discovered object')
    # Region observations remain explicit metadata. No solid receptacle or
    # physical support/contact is inferred from a visible surface patch.
    destination = 'receptacle' if 'receptacle' in query['roles'] else 'target'
    roles = ['manipulated_object', destination, 'support', 'obstacle']
    missing = {'support': 'public_support_candidate_is_not_contact_evidence',
               'obstacle': 'robot_workspace_and_collision_evidence_missing'}
    bindings = {}
    for role in roles[:2]:
        observed = query['roles'].get(role, {})
        if observed.get('status') == 'resolved':
            oid = observed.get('selected_object_id')
            selected = [r for r in observed.get('candidates', []) if r['object_id'] == oid]
            if oid not in object_ids or len(selected) != 1:
                raise ValueError('public selected object not in authenticated population')
            bindings[role] = dict(object_id=oid, query_sha256=query['query_sha256'],
                                  source_gate=reference['gate'], overlap_iou=selected[0]['overlap_iou'])
        else:
            missing[role] = observed.get('reason') or ('observed_surface_patch_has_no_physical_volume' if
                observed.get('status') == 'observed_region' else 'public_role_unresolved')
    complete = original['stage_status'] == 'PASS'
    return dict(schema_version=1, freeze_id=freeze_id, scene_id=query['scene_id'], task_id=task_id,
        task_family=query['task_family'], condition_id=gate['condition_id'],
        evidence_source='construction_observation', source_kind='real',
        construction_scope='public_rgb_geometry_and_roles_only',
        build_status='complete' if complete else 'failed', public_grounding=reference,
        scene=dict(scene_id=query['scene_id'], objects=objects if complete else None, robot=None,
                   cameras=[dict(id='required_policy_camera', pos=None, look_at=None, fovy_deg=None)]),
        task=dict(task_id=task_id, query_sha256=query['query_sha256'], roles=roles,
                  construction_role_bindings=bindings, unresolved_roles=missing,
                  public_role_evidence=deepcopy(query['roles']), language={}, rubric={}),
        audit=dict(objects=[dict(object_id=o['id'], checks={}) for o in objects]),
        physics_verified=False, robot_frame=None, paper_ready=False)


def authenticate_bundle(bundle, cache=None):
    reference = bundle.get('public_grounding')
    if reference is None:
        if (bundle.get('task') or {}).get('construction_role_bindings'):
            raise ValueError('public role bindings require authenticated grounding source')
        return
    gate = validate_reference(reference, cache)
    expected = bundle_from_gate(gate, reference, freeze_id=bundle['freeze_id'], task_id=bundle['task_id'])
    if bundle != expected:
        raise ValueError('public bundle differs from exact authenticated role/scene evidence')


def prepare_inputs(config_path, out):
    """Write canonical public bundles/config; feature extraction stays separate."""
    from robo.eval import build_task_support_dataset as producer
    cfg = shared.read(config_path)
    if (set(cfg) != {"schema_version", "source_commit", "freeze_id", "tier", "label_protocol_sha256", "references"}
        or cfg["schema_version"] != 1 or cfg["tier"] not in {"pilot", "full"}
        or not shared.CANONICAL_FREEZE_ID.fullmatch(cfg["freeze_id"])
        or len(cfg["label_protocol_sha256"]) != 64):
        raise ValueError("public feature bridge configuration differs")
    source = shared.git_snapshot(Path(__file__).resolve().parents[2])
    if source['dirty'] or source['commit'] != cfg['source_commit']:
        raise ValueError('public feature bridge requires exact clean source')
    cache = validate_references(cfg['references'],tier=cfg['tier'])
    gates = [validate_reference(ref, cache) for ref in cfg['references']]
    cells = [(g['scene_id'], g['condition_id']) for g in gates]
    scenes = list(shared.SCENES) if cfg['tier'] == 'full' else ['behavior_task0020']
    expected = [(scene, c) for scene in scenes for c in shared.CONDITIONS]
    if cells != expected:raise ValueError('complete fixed public feature condition roster required')
    keys = [];rows = []
    with producer._atomic_output(out) as (staging, output):
        for gate, reference in zip(gates, cfg['references']):
            queries = gate['original_gate']['queries']
            if len(queries) != 4:raise ValueError('exact four public queries required')
            for query in queries:
                bundle = bundle_from_gate(gate, reference, freeze_id=cfg['freeze_id'], task_id=query['task_id'])
                producer._public_payload(bundle)
                key = tuple(bundle[k] for k in producer.JOIN_KEY)
                if key in keys:raise ValueError('duplicate public query')
                keys.append(key)
                name = f'bundles/{len(rows):04d}.json'
                producer._write_json_new(staging/name, bundle)
                rows.append({**{k:bundle[k] for k in producer.JOIN_KEY}, 'task_family':bundle['task_family'],
                    'build_manifest': {'path':str(output/name), 'sha256':producer._sha256(staging/name)}})
        manifest = {'freeze_id':cfg['freeze_id'], 'rows':rows}
        producer._write_json_new(staging/'public_manifest.json', manifest)
        feature_config = dict(schema_version=2, freeze_id=cfg['freeze_id'], paper_ready=False,
            label_definition='construction_validity', tier=cfg['tier'], source_kind='real',
            scene_ids=scenes, conditions=list(shared.CONDITIONS),
            task_ids_by_scene={scene:[r['task_id'] for r in rows if r['scene_id']==scene and r['condition_id']=='clean'] for scene in scenes},
            public_roots=[str(output)], label_protocol_sha256=cfg['label_protocol_sha256'],
            public_manifest={'path':str(output/'public_manifest.json'), 'sha256':producer._sha256(staging/'public_manifest.json')})
        producer._write_json_new(staging/'features_config.json', feature_config)
        producer._write_json_new(staging/'bridge_receipt.json', dict(source=source, config=shared.identity(config_path),
            references=cfg['references'], planned_conditions=len(cells), planned_queries=len(rows),
            feature_rows_written=0, labels_read=False, paper_ready=False,
            output_members={str(p.relative_to(staging)):producer._sha256(p) for p in staging.rglob('*') if p.is_file()}))
    return dict(output=str(output), planned_conditions=len(cells), planned_queries=len(rows), paper_ready=False)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();print(json.dumps(prepare_inputs(args.config,args.out),indent=2))


if __name__=='__main__':main()
