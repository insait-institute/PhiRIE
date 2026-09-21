"""Native CoACD room-context measurements and observation-bounded DEV repair.

Evaluation task outcomes/reference target geometry are never optimization inputs.
L0 native room contact evidence is explicitly privileged oracle context. Accepting
this tool's own probe is not evidence of fidelity or native task preservation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import yaml

from agents.orchestrator.artifact import ProposalRecord, REQUIRED_EVIDENCE, sha256_file
from agents.orchestrator.job_graph import descendants, validate_ledger
from robo.roundtrip.importers.robocasa import import_reconstructed_object
from robo.roundtrip.build import read_train
from robo.roundtrip.capture import backproject_camera_z
from robo.manifest.hash import git_snapshot

DEFAULT_CONFIG = dict(schema_version=1, tier='DEV', max_calls=2, settle_seconds=.5,
                      max_drift_m=.02, max_rotation_deg=15., max_penetration_m=.005,
                      max_support_correction_m=.04, minimum_support_points=30,
                      support_search_height_m=.08, support_annulus_m=.05,
                      support_bin_m=.003, support_clearance_m=.001,
                      bm_order=['reselect_candidate', 'support_correction'])
CONTEXT_CONFIG = dict(DEFAULT_CONFIG, schema_version=3, max_observation_residual_m=.03,
                      max_observation_residual_increase_m=.005, approach_clearance_m=.15,
                      transport_clearance_m=.20, corridor_radius_m=.02, corridor_samples=24,
                      access_settle_seconds=1.5)
V4_CONFIG = dict(CONTEXT_CONFIG, schema_version=4, cabinet_entry_clearance_m=.05,
                 cabinet_release_clearance_m=.02, cabinet_entry_samples=32)
B4_REASON_ACTIONS = (
    ('public_observation_residual','reselect_candidate'),
    ('initial_penetration','support_correction'),
    ('native_destination_access','reselect_candidate'),
    ('approach_corridor_proxy','reselect_candidate'),
    ('transport_corridor_proxy','reselect_candidate'),
    ('settle_translation','support_correction'),
    ('settle_rotation','reselect_candidate'),
)


def next_context_action(method, measured, used):
    """Frozen failure-class scheduler; BM deliberately ignores measured reasons."""
    if method=='BM':
        return next(((a,'fixed_bank_order') for a in DEFAULT_CONFIG['bm_order'] if a not in used),(None,None))
    if measured['passed']:return None,None
    reasons=set(measured['reason_codes'])
    return next(((action,reason) for reason,action in B4_REASON_ACTIONS if reason in reasons and action not in used),(None,None))



def lineage_alternative(pool):
    """Other initial generator, tracing registration retry to its one parent."""
    initial={r['proposal_id']:r for r in pool['initial_candidates']}
    if len(initial)!=len(pool['initial_candidates']):raise ValueError('duplicate initial proposal identity')
    selected=[r for r in pool['outcomes'] if r['native_method']=='B3_AGENT_NATIVE']
    if len(selected)!=1:raise ValueError('one B3 initial selection required')
    pid=selected[0]['selected_proposal_id']
    if pid not in initial:
        retry=pool.get('retry_candidate')
        if not retry or retry['proposal_id']!=pid or len(retry['parent_proposal_ids'])!=1:
            raise ValueError('unestablished B3 generator lineage')
        pid=retry['parent_proposal_ids'][0]
    if pid not in initial or initial[pid]['tool'] not in ('trellis','reconviagen'):
        raise ValueError('unsupported initial generator lineage')
    other='reconviagen' if initial[pid]['tool']=='trellis' else 'trellis'
    alternatives=[r for r in initial.values() if r['tool']==other]
    if not alternatives:return None
    if len(alternatives)!=1:raise ValueError('ambiguous other-generator initial proposal')
    candidate_id=alternatives[0]['proposal_id']
    paths={r['object_dir'] for r in pool['outcomes'] if r['selected_proposal_id']==candidate_id}
    if len(paths)!=1:raise ValueError('other-generator initial artifact path unavailable or ambiguous')
    return {'object_dir':paths.pop(),'proposal_id':candidate_id,'tool':other,
            'selected_generator_lineage':initial[pid]['tool'],'selected_parent_initial_proposal_id':pid}


def validate_config(config):
    version=config.get('schema_version')
    expected={1:DEFAULT_CONFIG,3:CONTEXT_CONFIG,4:V4_CONFIG}.get(version,{})
    if set(config) != set(expected) or version not in (1,3,4) or config['tier'] not in (('DEV','TEST') if version==4 else ('DEV',)):
        raise ValueError('explicit frozen version/tier verification configuration required')
    if config['max_calls'] != 2 or config['bm_order'] != DEFAULT_CONFIG['bm_order']:
        raise ValueError('shared action budget/order changed')
    for key, value in config.items():
        if key not in ('tier', 'bm_order') and (not np.isfinite(value) or value <= 0):
            raise ValueError('positive finite verification configuration required')
    if config['max_support_correction_m'] > .05 or config['settle_seconds'] > 2:
        raise ValueError('DEV diagnostic bounds exceeded')
    if version==4 and any(config[k]!=v for k,v in V4_CONFIG.items() if k!='tier'):
        raise ValueError('v4 parameters are frozen across DEV and TEST')
    if version==4 and (not isinstance(config['cabinet_entry_samples'],int) or config['cabinet_entry_samples']<2):
        raise ValueError('cabinet entry sample count invalid')
    if config['schema_version']>=3 and (not isinstance(config['corridor_samples'],int) or
            config['corridor_samples']<2 or config['access_settle_seconds']>2):
        raise ValueError('context probe sample/time bounds invalid')


def measure_native_context(model, data, receipt, config):
    """Existing native adapter model/data hook; restores all integration bytes."""
    import mujoco
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    saved = np.empty(mujoco.mj_stateSize(model, kind))
    mujoco.mj_getState(model, data, saved, kind)
    gids = {model.geom(n).id for n in receipt['contact_geoms']}
    bid = model.body(receipt['body_name']).id
    try:
        mujoco.mj_forward(model, data)
        start, quat = data.xpos[bid].copy(), data.xquat[bid].copy()
        world_vertices=[]
        for gid in gids:
            mid=model.geom_dataid[gid];adr=model.mesh_vertadr[mid];count=model.mesh_vertnum[mid]
            local=model.mesh_vert[adr:adr+count]
            world_vertices.append(local@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid])
        world_vertices=np.concatenate(world_vertices)
        if not np.isfinite(saved).all():
            raise ValueError('nonfinite native integration state')
        def contact_rows():
            rows = []
            for c in data.contact[:data.ncon]:
                if (c.geom1 in gids) != (c.geom2 in gids):
                    rows.append({'geom1': model.geom(c.geom1).name, 'geom2': model.geom(c.geom2).name,
                                 'distance_m': float(c.dist), 'position_world_m': c.pos.tolist()})
            return rows
        initial = contact_rows()
        nearest = []
        for g in gids:
            for other in range(model.ngeom):
                if other in gids or model.geom_bodyid[other] == bid:
                    continue
                if not (model.geom_contype[other] or model.geom_conaffinity[other]):
                    continue
                points = np.zeros(6)
                distance = mujoco.mj_geomDistance(model, data, g, other, .1, points)
                if distance < .1:
                    nearest.append({'target_geom': model.geom(g).name, 'context_geom': model.geom(other).name,
                                    'distance_m': float(distance), 'closest_points_world_m': points.reshape(2,3).tolist()})
        nearest.sort(key=lambda r:r['distance_m'])
        steps = max(1, round(config['settle_seconds'] / model.opt.timestep))
        positions = []
        for _ in range(steps):
            mujoco.mj_step(model, data)
            positions.append(data.xpos[bid].tolist())
        final = data.xpos[bid].copy()
        drift = float(np.linalg.norm(final - start))
        angle = float(np.degrees(2*np.arccos(np.clip(abs(np.dot(quat, data.xquat[bid])), 0, 1))))
        penetration = max([max(0.,-r['distance_m']) for r in initial], default=0.)
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        reasons = []
        if not finite: reasons.append('nonfinite_after_settle')
        if drift > config['max_drift_m']: reasons.append('settle_translation')
        if angle > config['max_rotation_deg']: reasons.append('settle_rotation')
        if penetration > config['max_penetration_m']: reasons.append('initial_penetration')
        report = {'engine':'mujoco', 'engine_version':mujoco.__version__,
            'oracle_context':True, 'context_scope':'L0_target_only_native_room',
            'initial_position_m':start.tolist(), 'initial_quaternion_wxyz':quat.tolist(),
            'actual_collision_world_bounds_m':[world_vertices.min(0).tolist(),world_vertices.max(0).tolist()],
            'initial_contacts':initial, 'initial_penetration_m':penetration,
            'closest_context_pairs':nearest[:20], 'support_overlap_fraction':None,
            'support_gap_m':None, 'support_gap_note':'closest-context pairs are not a certified support assignment',
            'settle_steps':steps, 'timestep_s':float(model.opt.timestep), 'settle_drift_m':drift,
            'settle_rotation_deg':angle, 'settle_final_position_m':final.tolist(),
            'settle_contacts':contact_rows(), 'settle_trajectory_world_m':positions,
            'finite':finite, 'passed':not reasons, 'reason_codes':reasons,
            'physical_settle_passed':not reasons,
            'actual_collision_parts':{p:h for p,h in receipt['source_hashes'].items() if '/collision/' in p},
            'contact_parameters':receipt['contact_prior'], 'physics_prior':receipt['physics_prior'],
            'native_task_success':None, 'robot_IK':'NOT_RUN', 'camera_visibility':'NOT_RUN',
            'destination_opening':'NOT_RUN', 'independent_fidelity':'NOT_RUN'}
    finally:
        mujoco.mj_setState(model, data, saved, kind)
        mujoco.mj_forward(model, data)
        # forward touches warm-start state; restore again after refreshing caches.
        mujoco.mj_setState(model, data, saved, kind)
    restored = np.empty_like(saved)
    mujoco.mj_getState(model, data, restored, kind)
    report['probe_state_restored_byte_exact'] = bool(np.array_equal(saved, restored))
    if not report['probe_state_restored_byte_exact']:
        raise RuntimeError('native probe failed to restore declared reset')
    return report


def observe_support(capture, asset, config):
    """Horizontal support proxy from TRAIN depth in estimated target annulus."""
    import trimesh
    manifest, rows = read_train(capture)
    asset, capture = Path(asset), Path(capture)
    aligned = json.loads((asset/'aligned.json').read_text())
    T = np.asarray(aligned['T'])
    vertices = np.asarray(trimesh.load(asset/'mesh_sim.obj', process=False, force='mesh').vertices)
    world = vertices @ T[:3,:3].T + T[:3,3]
    lower, upper = world.min(0), world.max(0)
    points = []
    sources = {}
    for row in rows:
        depth_path = capture/row['depth_m']
        xyz, valid = backproject_camera_z(np.load(depth_path, allow_pickle=False), row['K'], row['T_world_from_camera'])
        p = xyz[valid][::8]
        margin = config['support_annulus_m']
        outer = ((p[:,:2] >= lower[:2]-margin)&(p[:,:2]<=upper[:2]+margin)).all(1)
        inner = ((p[:,:2] >= lower[:2])&(p[:,:2]<=upper[:2])).all(1)
        height = np.abs(p[:,2]-lower[2]) <= config['support_search_height_m']
        points.append(p[outer&~inner&height])
        sources[row['depth_m']] = sha256_file(depth_path)
    p = np.concatenate(points) if points else np.empty((0,3))
    report = {'source':'public_TRAIN_depth_horizontal_support_proxy', 'capture_id':manifest['capture_id'],
              'source_hashes':sources, 'points_considered':len(p), 'target_bottom_world_m':float(lower[2]),
              'support_z_world_m':None, 'correction_z_m':None, 'available':False}
    if len(p) < config['minimum_support_points']:
        return report
    bins = np.floor(p[:,2]/config['support_bin_m']).astype(int)
    counts = np.unique(bins, return_counts=True)
    best = counts[0][np.argmax(counts[1])]
    support = p[bins==best]
    if len(support) < config['minimum_support_points']:
        return report
    z = float(np.median(support[:,2]))
    correction = z + config['support_clearance_m'] - float(lower[2])
    report.update(support_z_world_m=z, support_inliers=len(support), correction_z_m=correction,
                  available=abs(correction)<=config['max_support_correction_m'] and abs(correction)>1e-6)
    report['inapplicable_reason'] = None if report['available'] else 'outside_bound_or_no_change'
    return report


def corrected_asset(asset, out, support, *, maximum_displacement_m=.04):
    """New immutable pose proposal; shared mesh/collision bytes never edited."""
    asset, out = Path(asset), Path(out)
    if (not support['available'] or not np.isfinite(support.get('correction_z_m', np.nan))
            or not 1e-6 < abs(support['correction_z_m']) <= maximum_displacement_m <= .05):
        raise ValueError('support action precondition not met')
    shutil.copytree(asset, out, symlinks=False)
    path = out/'aligned.json'
    aligned = json.loads(path.read_text())
    before = np.asarray(aligned['T']).copy()
    aligned['T'][2][3] += support['correction_z_m']
    path.write_text(json.dumps(aligned, indent=2)+'\n')
    invalidated = sorted(descendants([('pose','native_import'),('pose','gs_transform'),
        ('pose','background_removal_completion'),('native_import','task_handles'),
        ('native_import','contact_lists'),('pose','collision_world_transform')], 'pose'))
    return {'before_T':before.tolist(), 'after_T':aligned['T'], 'invalidated_dependencies':invalidated,
            'mesh_and_collision_local_hashes_preserved':True,
            'refresh_required_before_rollout':invalidated}


def asset_identity(asset):
    asset = Path(asset)
    paths = [asset/'aligned.json', asset/'physics.json', asset/'mesh_sim.obj', *sorted((asset/'collision').glob('part_*.obj'))]
    return hashlib.sha256(json.dumps({str(p.relative_to(asset)):sha256_file(p) for p in paths},sort_keys=True).encode()).hexdigest()


def diagnostic_frames(before, after, support, out):
    """Matched CPU orthographic projections of actual imported CoACD parts."""
    import trimesh
    from PIL import Image, ImageDraw, ImageFont
    scenes=[]
    for asset in (Path(before),Path(after)):
        T=np.asarray(json.loads((asset/'aligned.json').read_text())['T'])
        parts=[]
        for path in sorted((asset/'collision').glob('part_*.obj')):
            mesh=trimesh.load(path,process=False,force='mesh')
            world=np.asarray(mesh.vertices)@T[:3,:3].T+T[:3,3]
            parts.append((world,np.asarray(mesh.faces)))
        scenes.append(parts)
    all_points=np.concatenate([p for scene in scenes for p,_ in scene])
    low,high=all_points[:,[0,2]].min(0)-.015,all_points[:,[0,2]].max(0)+.015
    if support.get('support_z_world_m') is not None:low[1]=min(low[1],support['support_z_world_m']-.01)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',28)
    scale=min(1100/(high[0]-low[0]),500/(high[1]-low[1]))
    def project(p):return (90+(p[0]-low[0])*scale,620-(p[2]-low[1])*scale)
    for label,scene in zip(('before','after'),scenes):
        image=Image.new('RGB',(1280,720),(17,24,39));draw=ImageDraw.Draw(image)
        draw.text((40,25),f'{label.upper()}: actual CoACD collision, world X/Z projection',fill='white',font=font)
        draw.text((40,665),'Same projection and scale; public-depth horizontal support proxy in cyan',fill=(190,200,210),font=font)
        for vertices,faces in scene:
            for face in faces:
                draw.line([project(vertices[i]) for i in [*face,face[0]]],fill=(231,167,80),width=1)
        if support.get('support_z_world_m') is not None:
            z=support['support_z_world_m'];draw.line([project([low[0],0,z]),project([high[0],0,z])],fill=(79,209,197),width=3)
        image.save(Path(out)/f'{label}_collision.png')
    return {'kind':'matched_orthographic_CoACD_projection','axes':['world_x','world_z'],
            'native_camera_render':False,'files':{name:sha256_file(Path(out)/name) for name in ('before_collision.png','after_collision.png')}}


def load_and_measure(xml, state, asset, body_name, config, *, checks=None, source_model=None):
    import mujoco
    imported, receipt = import_reconstructed_object(xml, body_name=body_name, object_dir=asset, object_id='target')
    model = mujoco.MjModel.from_xml_string(imported)
    data = mujoco.MjData(model)
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    from robo.roundtrip.adapters.integration_state import remap_integration_state
    if source_model is None:source_model=mujoco.MjModel.from_xml_string(xml)
    vector = remap_integration_state(source_model,model,state['integration_state'])
    mujoco.mj_setState(model,data,vector,kind)
    joint = model.joint(receipt['free_joint_name']).id
    adr = model.jnt_qposadr[joint]
    data.qpos[adr:adr+7] = receipt['position_m']+receipt['quaternion_wxyz']
    report=measure_native_context(model,data,receipt,config)
    report['integration_mapping']='robo.roundtrip.adapters.integration_state.remap_integration_state'
    if config['schema_version']>=3:
        if checks is None:raise ValueError('v3 requires public observations and task context contract')
        from robo.roundtrip.native_context_checks import surface_residual,context_checks
        residual=surface_residual(asset,checks['observed_points'])
        increase=residual-checks['residual_reference']
        fit=residual<=config['max_observation_residual_m'] and increase<=config['max_observation_residual_increase_m']
        report['public_object_surface_guard']={'residual_m':residual,'increase_vs_initial_m':increase,
            'passed':fit,'producer':'agents.assets.s5_align.sym_score','source':'public_TRAIN_auto_mask_points'}
        context=context_checks(imported,model,data,receipt,checks['task_id'],checks['fixture_refs'],config)
        report['task_context_checks']=context
        if not fit:report['reason_codes'].append('public_observation_residual')
        for key in ('approach','transport'):
            if not context['corridors'][key]['passed']:report['reason_codes'].append(key+'_corridor_proxy')
        if not context['destination_access']['passed']:report['reason_codes'].append('native_destination_access')
        report['passed']=not report['reason_codes']
    return report, imported, receipt


def run(*, xml_path, state_path, asset, capture, config, out, method='B4', alternatives=(), body_name='obj_main',
        candidate_pool=None, input_method='B0_DEV_INTEGRATION', freeze_id='DEV_CONTEXT_SMOKE',
        observation_build=None, task_id=None):
    total_started=time.monotonic()
    validate_config(config)
    if method not in ('B3','B4','BM','V1'): raise ValueError('unknown context method')
    out, asset = Path(out).resolve(), Path(asset).resolve()
    if candidate_pool:
        pool=json.loads(Path(candidate_pool).read_text())
        if pool['capture_manifest_sha256']!=sha256_file(Path(capture)/'capture_manifest.json'):
            raise ValueError('candidate bank is bound to another capture')
        for path in [asset,*map(Path,alternatives)]:
            matches=[r for r in pool['outcomes'] if Path(r['object_dir']).name==path.name]
            if not matches:
                raise ValueError('candidate not in frozen action bank')
            row=matches[0]
            if any(sha256_file(path/name)!=digest for name,digest in row['artifact_hashes'].items()):
                raise ValueError('frozen candidate bytes changed')
        if not any(r['native_method']==input_method and Path(r['object_dir']).name==asset.name for r in pool['outcomes']):
            raise ValueError('input does not match declared initial method')
        if config['schema_version']>=3 and input_method=='B3_AGENT_NATIVE':
            alternate=lineage_alternative(pool)
            expected=[] if alternate is None else [Path(alternate['object_dir']).name]
            if [Path(p).name for p in alternatives]!=expected:
                raise ValueError('context action bank must use the other initial generator lineage')
    out.mkdir(parents=True,exist_ok=False)
    xml, state = Path(xml_path).read_text(), json.loads(Path(state_path).read_text())
    checks=None;observation_provenance=None
    if config['schema_version']>=3:
        from robo.roundtrip.native_context_checks import observation_points,surface_residual
        allowed_tasks=('PickPlaceCounterToSink','PickPlaceSinkToCounter')+ (('PickPlaceCounterToCabinet',) if config['schema_version']==4 else ())
        if observation_build is None or task_id not in allowed_tasks:
            raise ValueError('v3 requires bound observation build and resolved native task ID')
        if config['schema_version']==4:
            observed_manifest=json.loads((Path(observation_build)/'build_manifest.json').read_text())
            if observed_manifest.get('tier')!=config['tier']:raise ValueError('context and construction tiers differ')
        points,observation_provenance=observation_points(observation_build,capture)
        checks={'observed_points':points,'residual_reference':surface_residual(asset,points),
                'task_id':task_id,'fixture_refs':state['native_metadata']['fixture_refs']}
    started=time.monotonic()
    import mujoco
    source_model=mujoco.MjModel.from_xml_string(xml)
    original, imported, receipt = load_and_measure(xml,state,asset,body_name,config,checks=checks,source_model=source_model)
    initial_probe_wall=time.monotonic()-started
    started=time.monotonic()
    support = observe_support(capture,asset,config)
    initial_observation_wall=time.monotonic()-started
    if support.get('support_z_world_m') is not None:
        original['observed_support_gap_m']=original['actual_collision_world_bounds_m'][0][2]-support['support_z_world_m']
    candidates = [(asset,original,imported,receipt)]
    actions, skipped = [], []
    # Applicability of support depends on the current candidate after reselection.
    bank = {'support_correction':True, 'reselect_candidate':bool(alternatives)}
    used=set()
    if method in ('B4','BM'):
        while len(actions)<config['max_calls']:
            action,trigger_reason=next_context_action(method,candidates[-1][1],used)
            if action is None:break
            used.add(action)
            if not bank[action]:
                skipped.append({'action':action,'reason':'formal_precondition_unavailable'});continue
            started=time.monotonic();current=candidates[-1][0]
            if action=='support_correction':
                # Re-estimate from the CURRENT candidate, not a native target pose.
                current_support=observe_support(capture,current,config)
                if not current_support['available']:
                    skipped.append({'action':action,'reason':'current_support_precondition_unavailable'});continue
                candidate=out/f'proposal_{len(actions)+1}'
                change=corrected_asset(current,candidate,current_support,maximum_displacement_m=config['max_support_correction_m'])
            else:
                candidate=Path(alternatives[0]).resolve()
                if asset_identity(candidate)==asset_identity(current):
                    skipped.append({'action':action,'reason':'same_artifact_not_new_action'});continue
                change={'selected_existing_candidate':str(candidate),'invalidated_dependencies':['native_import','task_handles','contact_lists','gs_transform','background_removal_completion']}
            measured,new_xml,new_receipt=load_and_measure(xml,state,candidate,body_name,config,checks=checks,source_model=source_model)
            if support.get('support_z_world_m') is not None:
                measured['observed_support_gap_m']=measured['actual_collision_world_bounds_m'][0][2]-support['support_z_world_m']
            candidates.append((candidate,measured,new_xml,new_receipt))
            actions.append({'action':action,'proposal_id':'context-'+asset_identity(candidate)[:20],
                            'trigger_reason':trigger_reason,
                            'parent_proposal_id':'context-'+asset_identity(current)[:20],
                            'artifact':str(candidate),'elapsed_s':time.monotonic()-started,
                            'before_reasons':candidates[-2][1]['reason_codes'],'after_reasons':measured['reason_codes'],
                            'verification_after':measured,**change})
    # Identical final selection rule for both adaptive and fixed-order methods.
    selected=min(candidates,key=lambda c:(not c[1]['passed'],len(c[1]['reason_codes']),c[1]['settle_drift_m'])) if method in ('B4','BM') else candidates[0]
    accepted = selected[1]['passed'] if method in ('B4','BM','V1') else selected[1]['finite']
    def save(name,value): (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    save('verification_before.json',original);save('verification_after.json',selected[1]);save('observed_support.json',support)
    save('verification_candidates.json',[{'asset':str(c[0]),'asset_identity':asset_identity(c[0]),
          'selected':c[0]==selected[0],'verification':c[1]} for c in candidates])
    save('context_privileges.json',{'oracle_context':True,'native_target_pose_optimization':False,
        'native_task_outcomes_accessed':False,'support_source':'public_TRAIN_depth','scope':'L0_target_only'})
    instruction_path=Path(capture)/'task_instruction.txt'
    save('task_dependency_graph.json',{'instruction':instruction_path.read_text() if instruction_path.exists() else None,
         'grounding_status':'caller_bound_reconstructed_target; destination/support retained oracle context',
         'nodes':['known_robot','known_cameras','reconstructed_target','observed_support_proxy','destination'],
         'edges':[['known_robot','reconstructed_target','approach'],['reconstructed_target','destination','transport'],
                  ['observed_support_proxy','reconstructed_target','support'],['known_cameras','reconstructed_target','visibility']],
         'approach_corridor':selected[1].get('task_context_checks',{}).get('corridors',{}).get('approach','NOT_RUN'),
         'transport_corridor':selected[1].get('task_context_checks',{}).get('corridors',{}).get('transport','NOT_RUN'),
         'motion_planning_guarantee':False})
    evidence={key:None for key in REQUIRED_EVIDENCE}
    evidence.update(schema_frame_unit_valid=True,collision_valid=True,usable_convex_parts=len(receipt['contact_geoms']),
        initial_penetration_m=selected[1]['initial_penetration_m'],settle_drift_m=selected[1]['settle_drift_m'],
        settle_stable=selected[1]['physical_settle_passed'],missing_evidence=['independent_geometry','robot_IK']+
        ([] if 'task_context_checks' in selected[1] else ['destination_opening']))
    if checks is not None:
        evidence['symmetric_clipped_registration_residual_m']=selected[1]['public_object_surface_guard']['residual_m']
        evidence['observation_point_count']=len(checks['observed_points'])
    job_id='context-'+sha256_file(state_path)[:20]
    proposal=ProposalRecord(freeze_id=freeze_id,scene_id=Path(state_path).parent.name,object_id='target',
        job_id=job_id,proposal_id='context-'+asset_identity(selected[0])[:20],tool='native_context_verification',
        tool_commit=git_snapshot()['commit'],input_hashes={'public_capture':sha256_file(Path(capture)/'capture_manifest.json')},
        artifact_paths={'object_dir':str(selected[0])},evidence=evidence,decision='accept' if accepted else 'abstain',reason_codes=tuple(selected[1]['reason_codes']))
    ledger={'policy_id':method,'job_id':job_id,'proposal_id':proposal.proposal_id,
            'terminal_action':'accept' if accepted else 'abstain','selected_proposal_id':proposal.proposal_id if accepted else None,
            'proposals':[proposal.as_dict()],'context_actions':actions,'skipped_actions':skipped,'support_label':'not_evaluated'}
    validate_ledger([ledger]);(out/'job_ledger.jsonl').write_text(json.dumps(ledger)+'\n')
    (out/'repair_ledger.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in actions))
    with (out/'action_costs.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['action','proposal_id','elapsed_s']);writer.writeheader();writer.writerows({k:a[k] for k in writer.fieldnames} for a in actions)
    (out/'before.xml').write_text(imported);(out/'after.xml').write_text(selected[2])
    manifest={'method':method,'accepted':accepted,'selected_asset':str(selected[0]),'actual_calls':len(actions),
        'input_method':input_method, 'candidate_pool_sha256':sha256_file(candidate_pool) if candidate_pool else None,
        'capture_manifest_sha256':sha256_file(Path(capture)/'capture_manifest.json'),
        'public_observation_provenance':observation_provenance,'task_id':task_id,
        'alternative_asset_identities':[asset_identity(p) for p in alternatives],
        'maximum_calls':2,'input_asset':str(asset),'config':config,'outcome_claim':'NOT_RUN',
        'dependency_refresh':{'native_import':'REBUILT','contact_lists':'REBUILT','task_handles':'REQUIRED_AT_ADAPTER_IMPORT',
          'gs_transform':'NOT_IMPLEMENTED','background_removal_completion':'NOT_IMPLEMENTED'},
        'source_hashes':{str(p):sha256_file(p) for p in (Path(xml_path),Path(state_path),Path(capture)/'capture_manifest.json')},
        'tier':config['tier'],'context':'L0 retained oracle room; construction context checks only; no native policy outcome'}
    manifest['source_code'] = git_snapshot()
    manifest['implementation_sha256'] = sha256_file(__file__)
    manifest['config_sha256'] = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    manifest['selected_asset_identity'] = asset_identity(selected[0])
    manifest['synchronized_diagnostic_frames'] = diagnostic_frames(asset,selected[0],support,out)
    manifest['timing']={'total_wall_s':time.monotonic()-total_started,'initial_context_probe_wall_s':initial_probe_wall,
                        'initial_observation_support_wall_s':initial_observation_wall,
                        'extra_actions_wall_s':sum(a['elapsed_s'] for a in actions),'gpu_model_calls':0}
    manifest['freeze_id']=freeze_id
    manifest['action_bank_contract']={'actions':['support_correction','reselect_candidate'],
        'maximum_calls_per_target_dependency':2,'support_arguments':'recomputed from current candidate and same public TRAIN support',
        'reselection_arguments':'other initial generator relative to B3 lineage; retry inherits initial parent tool; same for B4/BM',
        'b4_reason_priority':[list(pair) for pair in B4_REASON_ACTIONS],'bm_order':config['bm_order'],
        'exact_compute_match':False}
    manifest['required_gates']={'native_finite':selected[1]['finite'],
        'context_settle_and_penetration':not any(r in selected[1]['reason_codes'] for r in ('settle_translation','settle_rotation','initial_penetration')),
        'public_object_surface_guard':selected[1].get('public_object_surface_guard',{}).get('passed'),
        'task_context_checks':selected[1].get('task_context_checks',{}).get('passed')}
    save('build_manifest.json',manifest)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('xml','state','asset','capture','config','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--method',choices=['B3','B4','BM','V1'],default='B4')
    parser.add_argument('--alternative',action='append',default=[])
    parser.add_argument('--candidate-pool')
    parser.add_argument('--input-method',default='B0_DEV_INTEGRATION')
    parser.add_argument('--freeze-id',default='DEV_CONTEXT_SMOKE')
    parser.add_argument('--observation-build')
    parser.add_argument('--task-id')
    args=parser.parse_args(argv)
    report=run(xml_path=args.xml,state_path=args.state,asset=args.asset,capture=args.capture,
               config=yaml.safe_load(Path(args.config).read_text()),out=args.out,method=args.method,alternatives=args.alternative,
               candidate_pool=args.candidate_pool,input_method=args.input_method,freeze_id=args.freeze_id,
               observation_build=args.observation_build,task_id=args.task_id)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
