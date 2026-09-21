"""Scoped consumers of sealed construction and original TEST cameras.

This module composes existing producers; it defines no render or metric math.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from agents.edit.inpaint_masks import _bound_contract, _checked, _identity, _sealed, _preparation
from agents.edit.inpaint_fill import validate_public_fill
from robo.eval import agentic_ablation as e3
from robo.manifest.hash import git_snapshot
from run.icra2027 import e2_raw_room as raw

SCENES = ('27dd4da69e', '40aec5fffa')
POPULATION = {'27dd4da69e': 7, '40aec5fffa': 10}
SCOPE = 'compact_public_factorized_official_test_pilot'
AUTO = 'Factorized composite, automatic discovery'
FULL_SCOPE = 'full_public_factorized_official_test_evaluation'
PREPARATION_SOURCE = '59c823d4f117c5203753f504a2dc7efb36c93456'
PREPARATION_SEAL = '8442489b9a7c078344b2d55069d415a4b1c3f672c0e54571772347b17d4d99e0'
RAW_SOURCE = 'cbba8ff63bfec9002a740826b55259106373a7af'
RAW_CONFIG = 'c881d9ea03e77bee62022bedd099d1f21db4298dadd30bc2286cb112b9eac57f'
RAW_FREEZE = '20260905-3ff95f4-v2'


def scenes(c):
    return c['scene_ids'] if c.get('scope') == FULL_SCOPE else list(SCENES)


def population(c):
    return c['planned_objects'] if c.get('scope') == FULL_SCOPE else POPULATION


def scope(c):
    return FULL_SCOPE if c.get('scope') == FULL_SCOPE else SCOPE


def full_inventory(c):
    """Authenticate the completed TRAIN inventory without reading TEST pixels.

    The published inventory has an explicit immutable anchor, in addition to
    its original source/config/E0 bindings. Re-sealing a changed decision is
    therefore not a way to admit or remove scenes in this evaluation.
    """
    from run.icra2027 import e2_full_factorized_protocol as protocol
    from run.icra2027.e2_full_preparation import ADMISSION_SHA, SCOPE as PREP_SCOPE
    p = protocol.validate_protocol(_checked(c['full_protocol']))
    spec = c['preparation_inventory']
    source_check(spec)
    if spec['code_commit'] != PREPARATION_SOURCE or spec['seal']['sha256'] != PREPARATION_SEAL:
        raise ValueError('full TRAIN inventory source or immutable seal differs')
    directory, seal = _sealed(_checked(spec['seal']).parent)
    if seal != spec['seal']:
        raise ValueError('full TRAIN inventory seal identity differs')
    pc, contract = _bound_contract(_checked(spec['config']), _checked(spec['contract']),
                                  producer_commit=spec['code_commit'])
    r = json.loads((directory/'inventory.json').read_text())
    if (directory != e3.REPOSITORY_ROOT/'outputs/icra2027'/pc['freeze_id']/'fidelity/preparation_inventory'
            or r['schema_version'] != 1 or r['scope'] != PREP_SCOPE or r['status'] != 'PASS'
            or r['config'] != spec['config'] or r['contract'] != spec['contract']
            or r['freeze_id'] != pc['freeze_id'] or r['full_protocol'] != c['full_protocol']
            or pc['full_protocol'] != c['full_protocol'] or r['background_admission'] != pc['background_admission']
            or r['background_admission']['sha256'] != ADMISSION_SHA
            or r['planned_scenes'] != 50 or r['planned_objects'] != 1871 or r['planned_test_views'] != 400
            or r['mask_fill_inference_invoked'] is not False or r['TEST_quality_used_for_admission'] is not False
            or r['paper_ready'] is not False or [x['scene_id'] for x in r['rows']] != p['scene_ids']
            or c['scene_ids'] != p['scene_ids'] or c['planned_objects'] != p['population']):
        raise ValueError('full TRAIN inventory config/E0/roster binding differs')
    _checked(r['background_admission'])
    for row in r['rows']:
        sid = row['scene_id']; prep = row['preparation']
        if (row['planned_objects'] != p['population'][sid] or row['planned_test_views'] != 8
                or row['paper_ready'] is not False
                or row['status'] not in {'STRUCTURALLY_FILLABLE', 'BLOCKED_UNFILLABLE_ACCEPTED', 'NO_ACCEPTED_OBJECTS'}):
            raise ValueError('full TRAIN inventory scene denominator/status differs')
        if sid == p['pilot_scene']:
            expected = pc['pilot_preparation']
        elif sid in pc.get('prior_preparations', {}):
            expected = pc['prior_preparations'][sid]
        else:
            expected = dict(directory=str(e3.REPOSITORY_ROOT/'outputs/icra2027'/pc['freeze_id']/'fidelity/removal'/sid/'inpaint'),
                seal=prep['seal'], context=pc['contexts'][sid], contract=spec['contract'], producer_commit=spec['code_commit'])
        if prep != expected:
            raise ValueError('full TRAIN inventory preparation source differs')
    required = {x['scene_id'] for x in r['rows'] if x['status'] != 'BLOCKED_UNFILLABLE_ACCEPTED'}
    if set(c['fill_seals']) != required or c.get('paired_protocol') is not None:
        raise ValueError('full background roster or legacy paired protocol differs')
    return p, r


def construction(c, scene):
    """Return an authenticated fill, or the predeclared TRAIN-only blocker.

    Zero accepted scenes require the existing NO_REMOVAL fill seal. A missing
    or invalid fill for any other scene is an error, never a missing-view row.
    """
    if c.get('scope') != FULL_SCOPE:
        fill = validate_public_fill(_checked(c['fill_seals'][scene]).parent)
        if fill['seal_identity'] != c['fill_seals'][scene]:
            raise ValueError('fill seal identity differs')
        return fill
    from run.icra2027.e2_full_preparation import structural_state
    from run.icra2027.e2_full_factorized_protocol import CONSTRUCTION_COMMIT, CONSTRUCTION_FREEZE, read
    p, inventory = full_inventory(c)
    if scene not in p['scene_ids']:
        raise ValueError('scene outside full frozen roster')
    row = next(x for x in inventory['rows'] if x['scene_id'] == scene)
    prep = _preparation(row['preparation'], scene); report = prep['report']
    state, blocked = structural_state(report)
    expected = dict(status=state, planned_objects=report['planned_objects'], accepted_objects=report['accepted_objects'],
        planned_training_views=len(prep['jobs']), blocked_objects=blocked,
        missing_test_views=8 if blocked else None)
    if any(row[k] != value for k, value in expected.items()):
        raise ValueError('TRAIN structural evidence or denominator was relabeled')
    pc = yaml.safe_load(_checked(row['preparation']['context']).read_text())
    factory = read(pc['materialization_manifest'])
    jobs = read(p['resolved_jobs'])
    slots = next([j['object_slot'] for j in x['jobs']] for x in jobs['scenes'] if x['scene_id'] == scene)
    if (factory['e3_code_commit'] != CONSTRUCTION_COMMIT or factory['e3_freeze_id'] != CONSTRUCTION_FREEZE
            or factory['scene_id'] != scene or factory['policy_id'] != 'A4'
            or factory['roster']['object_slots'] != slots or factory['roster']['job_count'] != p['population'][scene]):
        raise ValueError('full canonical construction source or object slots differ')
    if blocked:
        generation = read(pc['generation_config'])
        boundary_path = Path(generation['source_pilot'])/'input_manifest.json'
        if _identity(boundary_path)['sha256'] != generation['source_discovery_hashes']['input_manifest.json']:
            raise ValueError('blocked scene original TRAIN boundary differs')
        boundary = json.loads(boundary_path.read_text())
        _checked(boundary['gaussian'])
        return dict(directory=Path(prep['directory']), seal_identity=prep['seal_identity'], result=dict(
            planned_objects=row['planned_objects'], accepted_objects=row['accepted_objects'], status=state),
            preparation=row['preparation'], source=dict(context=pc, source_boundary=boundary,
                train_images=prep['train_images'], blocked=blocked, factory=Path(prep['source_factory'])))
    fill = validate_public_fill(_checked(c['fill_seals'][scene]).parent)
    chain = fill['source']['erasure']['mask_bundle']['preparation']
    if (fill['seal_identity'] != c['fill_seals'][scene]
            or any(chain[k] != prep[k] for k in ('directory', 'seal_identity', 'context_identity', 'contract_identity'))
            or fill['source']['factory'] != Path(prep['source_factory'])
            or fill['result']['planned_objects'] != row['planned_objects']
            or fill['result']['accepted_objects'] != row['accepted_objects']
            or fill['source']['blocked']
            or fill['result']['status'] != ('NO_REMOVAL' if state == 'NO_ACCEPTED_OBJECTS' else 'COMPLETE')):
        raise ValueError('full fill preparation chain or zero-accepted semantics differ')
    return dict(fill, preparation=row['preparation'])


def original_raw(c, scene):
    if c.get('scope') == FULL_SCOPE:
        return audit_raw(c['raw_source'], scene, full_protocol=c['full_protocol'])
    return audit_raw(c['raw_source'], scene)



def source_check(spec):
    root = Path(spec['code_root'])
    snapshot = git_snapshot(root)
    if snapshot['dirty'] or snapshot['commit'] != spec['code_commit']:
        raise ValueError('archived producer source changed')
    return root


RAW_AUDIT_PROGRAM = '''import json,sys
from pathlib import Path
from run.icra2027 import e2_raw_room as r
c,code,out,contract=r.context(sys.argv[1],sys.argv[2],sys.argv[3])
plan=r.checked_plan(c,code,out,contract,sys.argv[2])
action='RENDER'
if c.get('reuse_bank'):
    bank=json.loads(r.checked_identity(c['reuse_bank']).read_text())
    entries=[x for x in bank['rows'] if x['scene_id']==sys.argv[3]]
    if len(entries)!=1 or entries[0]['action'] not in {'RENDER','VERIFIED_REUSE'}:
        raise ValueError('raw reuse bank has no unique declared scene action')
    action=entries[0]['action']
if action=='VERIFIED_REUSE':
    bank,row,original=r.checked_reuse_source(c,code,plan)
    manifest=r.reused_evaluation_manifest(c,code,out,plan,Path(sys.argv[1]))
    render_ref=row['render_receipt']
    execution_ref=row['execution_receipt']
else:
    receipt_path=r.receipt_root(c,sys.argv[2])/'execution_receipt.json'
    receipt=json.loads(receipt_path.read_text())
    if (receipt['status']!='PASS' or receipt['code_commit']!=code['commit'] or receipt['freeze_id']!=plan['freeze_id']
        or receipt['scene_id']!=sys.argv[3] or receipt['artifact']!=r.identity(out/'bundle/manifest.json')
        or not receipt['stages'] or any(x['exit_code']!=0 for x in receipt['stages'])):
        raise ValueError('original render execution identity did not pass')
    r.check_render_receipt(out,plan,r.sha(Path(sys.argv[1])),r.sha(out/'render_receipt.json'))
    manifest=r.canonical_manifest(plan,out/'bundle')
    render_ref=r.identity(out/'render_receipt.json')
    execution_ref=r.identity(receipt_path)
print(json.dumps(dict(plan=plan,manifest=manifest,bundle=str(out/'bundle'),
    bundle_identity=r.identity(out/'bundle/manifest.json'),render_receipt=render_ref,
    execution_receipt=execution_ref,source_fingerprint=r.reuse_source_fingerprint(code['commit']),
    runtime=c['runtime'],python=c['python'],renderer_dependency_root=c['renderer_dependency_root'],
    lpips_backbone=c['lpips_backbone'],torch_home=c['torch_home'])))
'''


def audit_raw(spec, scene, *, full_protocol=None):
    """Run the untouched original validator in its own clean producer checkout."""
    p = None
    if full_protocol is not None:
        from run.icra2027.e2_full_factorized_protocol import validate_protocol
        p = validate_protocol(_checked(full_protocol))
        if (spec['code_commit'] != RAW_SOURCE or spec['config']['sha256'] != RAW_CONFIG
                or Path(spec['freeze_root']).name != RAW_FREEZE):
            raise ValueError('full raw reuse must use the original frozen producer/config')
    if scene not in (p['scene_ids'] if p else SCENES):
        raise ValueError('outside frozen scene roster')
    code = source_check(spec)
    config = _checked(spec['config']); contract = _checked(spec['contract'])
    if contract != Path(spec['freeze_root'])/'contract/freeze_manifest.json':
        raise ValueError('raw source contract path differs')
    program = RAW_AUDIT_PROGRAM
    env = dict(os.environ, PYTHONPATH=str(code), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
        SIMANY_AUTO='1',SIMANY_NO_GT='1',SIMANY_MESH_SRC='derived',SIMANY_SCENE=scene)
    result = json.loads(subprocess.check_output([spec['python'], '-c', program, str(config),
        spec['freeze_root'], scene], cwd=code, env=env, text=True))
    source_check(spec); _checked(spec['config']); _checked(spec['contract'])
    if p is not None:
        from run.icra2027.e2_full_factorized_protocol import read
        bank = read(p['camera_bank'])
        camera = next(x for x in bank['scenes'] if x['scene_id'] == scene)
        plan = result['plan']; original = read(camera['plan'])
        if (plan['evaluation_images'] != original['evaluation_images']
                or [x['frame'] for x in plan['evaluation_images']] != p['evaluation_frames'][scene]
                or plan['gaussian'] != original['gaussian']
                or plan['optimization_input_frames'] != original['optimization_input_frames']
                or plan['original_camera_plan'] != camera['plan']):
            raise ValueError('raw reuse differs from original full camera bank')
    return result


def validate_scene(fill, original, scene, *, planned_objects=None):
    p = original['plan']; source = fill['source']
    if (source['context']['scene_id'] != scene or p['scene_id'] != scene
            or fill['result']['planned_objects'] != (POPULATION[scene] if planned_objects is None else planned_objects)
            or p['gaussian']['sha256'] != source['source_boundary']['gaussian']['sha256']
            or p['gaussian']['bytes'] != source['source_boundary']['gaussian']['bytes']
            or set(p['optimization_input_frames']) != set(source['train_images'])
            or p['source_gaussian_training_provenance'] != 'FRESH_OFFICIAL_TRAIN_ONLY'):
        raise ValueError('compact source scene/population/TRAIN Gaussian differs')
    raw.validate_camera_rows(p['evaluation_images'])
    if set(r['frame'] for r in p['evaluation_images']) & set(source['train_images']):
        raise ValueError('TEST camera overlaps construction TRAIN inputs')
    # Exact serialized original camera bank, never a new selection or inversion.
    if not p.get('camera_bank') or not p.get('original_camera_plan'):
        raise ValueError('raw render lacks frozen camera-bank provenance')


def context(config_path, contract_path):
    config_path, contract_path = Path(config_path).absolute(), Path(contract_path).absolute()
    c, contract = _bound_contract(config_path, contract_path)
    full = c.get('scope') == FULL_SCOPE
    if (c['schema_version'] != 1 or c['scope'] not in {SCOPE, FULL_SCOPE}
            or c['planned_views_per_scene'] != 8 or c['seed'] != 42
            or c['bootstrap_samples'] != 2000 or c['bootstrap_seed'] != 42
            or c['paper_ready'] is not False or c['full_e3_gt_access'] is not False):
        raise ValueError('fixed evaluation protocol differs')
    if full:
        full_inventory(c)
    elif (c['scene_ids'] != list(SCENES) or c['planned_objects'] != POPULATION
            or set(c['fill_seals']) != set(SCENES)):
        raise ValueError('fixed compact evaluation protocol differs')
    output = e3.REPOSITORY_ROOT/'outputs/icra2027'/c['freeze_id']/'fidelity/public_factorized'
    e3._validate_cli_execution(contract_path,c['freeze_id'],output,config_paths=[config_path])
    source_check(c['raw_source']); source_check(c['metric_source'])
    if _checked(c['metric_source']['module'])!=Path(c['metric_source']['code_root'])/'robo/eval/fidelity_metrics.py':
        raise ValueError('metric module source path differs')
    _checked(c['execution']['lpips_backbone'])
    for ref in c['fill_seals'].values(): _checked(ref)
    return c, contract, output


def seal_output(directory):
    members = {str(p.relative_to(directory)):e3.sha256_file(p) for p in directory.rglob('*') if p.is_file()}
    raw.write_new(directory/'seal.json',{'schema_version':1,'members':members})


def copy_raw(original, destination):
    source = Path(original['bundle']); m = json.loads(_checked(original['bundle_identity']).read_text())
    destination.mkdir(parents=True,exist_ok=False)
    views=[]
    for view in m['views']:
        copied={}
        for key in ('render','gt'):
            p=Path(view[key+'_path']); relative=p.relative_to(source)
            target=destination/relative; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target)
            if e3.sha256_file(target)!=view[key+'_sha256']:
                raise ValueError('reused raw image copy differs')
            copied[key+'_path']=str(target);copied[key+'_sha256']=view[key+'_sha256']
        views.append(dict(view_id=view['view_id'],**copied))
    return views


def render(config_path,contract_path,scene):
    c,contract,out=context(config_path,contract_path)
    if scene not in scenes(c): raise ValueError('outside predeclared roster')
    if c.get('render_source') is not None:raise ValueError('metrics-only recovery cannot rerender')
    target=out/'scenes'/scene;claim=target.with_name(scene+'.claim.json')
    if target.exists() or target.is_symlink() or claim.exists():raise FileExistsError('scene attempt exists')
    fill=construction(c,scene)
    original=original_raw(c,scene);validate_scene(fill,original,scene,planned_objects=population(c)[scene])
    if any(original[k]!=c['execution'][k] for k in ('runtime','python','renderer_dependency_root','lpips_backbone','torch_home')):
        raise ValueError('frozen original render/runtime recipe differs')
    raw.write_new(claim,dict(config=_identity(Path(config_path)),contract=_identity(Path(contract_path))))
    start=time.monotonic();target.mkdir(parents=True,exist_ok=False)
    result=dict(schema_version=1,scope=scope(c),freeze_id=c['freeze_id'],scene_id=scene,
        code_commit=contract['code']['commit'],config_identity=_identity(Path(config_path)),
        contract_identity=_identity(Path(contract_path)),source_fill=c['fill_seals'].get(scene),
        planned_objects=population(c)[scene],accepted_objects=fill['result']['accepted_objects'],
        planned_views=8,raw_source=original,raw_views=[],composite_views=[],
        paper_ready=False,full_e3_gt_access=False,construction_modified=False,
        metric_reuse='NOT_AVAILABLE: original producer stored only full400-view aggregates; no pair LPIPS sufficient statistics')
    if scope(c)==FULL_SCOPE:
        result.update(source_preparation=fill['preparation'],background_status=fill['result']['status'])
    try:
        result['raw_views']=copy_raw(original,target/'raw_reuse')
        if fill['source']['blocked']:
            result.update(status='BLOCKED_UNFILLABLE_ACCEPTED',blocker=fill['source']['blocked'])
        else:
            if str(Path(sys.executable).absolute())!=c['execution']['python']['render']:
                raise ValueError('frozen rendering interpreter differs')
            if raw.runtime('render')!=c['execution']['runtime']['render']:
                raise ValueError('renderer runtime drift')
            from agents.core import common
            from agents.eval.fidelity_room_export import _load_composite
            gaussian,provenance=_load_composite(common,None,fill['source']['factory'],
                use_trellis_snapshot=False,public_fill=fill['directory'])
            result['composite_provenance']=provenance
            result['composite_views']=raw.export_raw(original['plan'],target/'composite',common,
                gaussian=gaussian,method_slug='factorized_auto_discovery')
            if raw.runtime('render')!=c['execution']['runtime']['render']:
                raise ValueError('renderer runtime drift after render')
            result['status']='COMPLETE'
        # Verify construction and source receipts again before publishing any seal.
        if construction(c,scene)['seal_identity']!=fill['seal_identity']:
            raise ValueError('construction changed during rendering')
        if original_raw(c,scene)!=original:raise ValueError('original rendering inputs changed')
        context(config_path,contract_path)
    except Exception as exc:
        result.update(status='FAILED',failure_type=type(exc).__name__,failure=str(exc))
        # Failed partial render pixels remain for diagnosis, never metric inputs.
        result['composite_views']=[]
    result['wall_s']=time.monotonic()-start
    raw.write_new(target/'result.json',result);seal_output(target)
    if result['status']=='FAILED':raise ValueError(result['failure'])
    return result


def scene_result(directory,c,contract,scene):
    directory,seal=_sealed(directory);r=json.loads((directory/'result.json').read_text())
    if (r['scope']!=scope(c) or r['freeze_id']!=c['freeze_id'] or r['scene_id']!=scene
            or r['code_commit']!=contract['code']['commit'] or r['source_fill']!=c['fill_seals'].get(scene)
            or r['planned_objects']!=population(c)[scene] or r['planned_views']!=8
            or r['paper_ready'] is not False or r['full_e3_gt_access'] is not False
            or r['construction_modified'] is not False):raise ValueError('scene result provenance differs')
    if (yaml.safe_load(_checked(r['config_identity']).read_text())!=c
            or json.loads(_checked(r['contract_identity']).read_text())!=contract
            or r['status'] not in {'COMPLETE','BLOCKED_UNFILLABLE_ACCEPTED','FAILED'}):
        raise ValueError('scene result config/E0/status differs')
    fill=construction(c,scene)
    if scope(c)==FULL_SCOPE and (r['source_preparation']!=fill['preparation']
            or r['background_status']!=fill['result']['status']
            or r['accepted_objects']!=fill['result']['accepted_objects']):
        raise ValueError('full scene preparation or background status differs')
    validate_scene(fill,r['raw_source'],scene,planned_objects=population(c)[scene])
    if original_raw(c,scene)!=r['raw_source']:
        raise ValueError('scene original render provenance changed or was resealed')
    if _bound_contract(_checked(r['config_identity']),_checked(r['contract_identity']))!=(c,contract):
        raise ValueError('scene config/E0 source alias differs')
    if (r['status']=='COMPLETE' and fill['source']['blocked']
            or r['status']=='BLOCKED_UNFILLABLE_ACCEPTED' and not fill['source']['blocked']):
        raise ValueError('scene background readiness was relabeled')
    for name in ('raw_views','composite_views'):
        views=r[name]
        expected=8 if name=='raw_views' or r['status']=='COMPLETE' else 0
        if len(views)!=expected:raise ValueError('scene conditional view denominator differs')
        for v in views:
            for key in ('render','gt'):
                path=e3.checked_repo_path(v[key+'_path'],'paired RGB',must_exist=True)
                if not path.is_relative_to(directory) or e3.sha256_file(path)!=v[key+'_sha256']:
                    raise ValueError('paired image escaped or changed')
        if views and [v['view_id'] for v in views]!=[v['frame'] for v in r['raw_source']['plan']['evaluation_images']]:
            raise ValueError('scene views differ from fixed camera roster')
    if scope(c)==FULL_SCOPE:
        if fill['source']['blocked'] and r['status']!='BLOCKED_UNFILLABLE_ACCEPTED':
            raise ValueError('predeclared TRAIN blocker was relabeled as a render failure')
        original_views = r['raw_source']['manifest']['room_methods'][raw.METHOD]['records'][0]['views']
        frozen = {v['view_id']:v for v in original_views}
        if (len(frozen)!=8 or list(frozen)!=[v['frame'] for v in r['raw_source']['plan']['evaluation_images']]
                or any(v[key]!=frozen[v['view_id']][key] for v in r['raw_views']
                       for key in ('render_sha256','gt_sha256'))
                or any(v['gt_sha256']!=frozen[v['view_id']]['gt_sha256'] for v in r['composite_views'])):
            raise ValueError('full copied raw or common-view reference pixels differ from original bundle')
    return r,seal



def paired_protocol(c,results):
    spec=c.get('paired_protocol')
    if spec is None:return None
    protocol=yaml.safe_load(_checked(spec).read_text())
    if (protocol['schema_version']!=1 or protocol['scope']!='predeclared_single_scene_common_view_diagnostic'
            or protocol['freeze_id']!=c['freeze_id'] or protocol['metric_outcomes_inspected'] is not False
            or protocol['original_planned_scenes']!=list(SCENES) or protocol['original_planned_objects']!=17
            or protocol['original_planned_views_per_method']!=16 or protocol['paper_ready'] is not False
            or protocol['metric_module']!=c['metric_source']['module']
            or protocol['metric_source_commit']!=c['metric_source']['code_commit']
            or protocol['methods']!=[raw.METHOD,AUTO] or protocol['metrics']!=['PSNR','SSIM','LPIPS']
            or protocol['bootstrap_samples']!=c['bootstrap_samples'] or protocol['bootstrap_seed']!=c['bootstrap_seed']
            or protocol['source_scene_seals']!=c['render_source']['scene_seals']):
        raise ValueError('predeclared paired analysis scope/source differs')
    common={scene:[v['view_id'] for v in r['composite_views'] if v['view_id'] in {x['view_id'] for x in r['raw_views']}]
            for scene,r in results.items()}
    if (protocol['common_views']!=common or len([v for values in common.values() for v in values])!=8
            or sum(bool(v) for v in common.values())!=1):
        raise ValueError('paired support differs from all originally available common views')
    for scene in SCENES:
        path=_checked(protocol['source_scene_results'][scene])
        if json.loads(path.read_text())!=results[scene]:raise ValueError('paired original scene result differs')
    scene=next(s for s in SCENES if common[s]);r=results[scene];q=protocol['qualitative_view']
    if (q['scene_id']!=scene or q['view_id']!=common[scene][0]
            or q['raw_path']!=r['raw_views'][0]['render_path']
            or q['composite_path']!=r['composite_views'][0]['render_path']
            or q['reference_path']!=r['composite_views'][0]['gt_path']):
        raise ValueError('fixed first-camera qualitative selection differs')
    return protocol

def make_manifest(c,contract,results):
    from robo.eval.fidelity_metrics import ROOM_METHODS,OBJECT_METHODS
    if list(results)!=scenes(c):raise ValueError('all predeclared scenes must remain in metric denominator')
    protocol=paired_protocol(c,results)
    full=scope(c)==FULL_SCOPE
    if full:
        full_inventory(c)
        common={scene:[v['view_id'] for v in r['composite_views'] if v['view_id'] in
                       {x['view_id'] for x in r['raw_views']}] for scene,r in results.items()}
        protocol={'common_views':common}
    room={name:dict(source_id=name,records=[]) for name in ROOM_METHODS};coverage=[]
    for scene,r in results.items():
        p=r['raw_source']['plan']
        for method,key in ((raw.METHOD,'raw_views'),(AUTO,'composite_views')):
            views=r[key]
            coverage.append(dict(scene_id=scene,method=method,planned_views=8,available_views=len(views),
                status='COMPLETE' if len(views)==8 else r['status'],planned_objects=population(c)[scene],
                accepted_objects=r['accepted_objects'],missing_views=[x['frame'] for x in p['evaluation_images'] if x['frame'] not in {v['view_id'] for v in views}]))
            if full:
                coverage[-1].update(background_status=r['background_status'],
                    source_preparation=r['source_preparation'],
                    unchanged_source_background=r['background_status']=='NO_REMOVAL')
            if protocol is not None:
                views=[v for v in views if v['view_id'] in protocol['common_views'][scene]]
                coverage[-1]['analysis_views']=len(views)
            if views:room[method]['records'].append(dict(freeze_id=c['freeze_id'],scene_id=scene,
                source_build=r['source_fill']['sha256'] if key=='composite_views' else r['raw_source']['bundle_identity']['sha256'],
                render_dir=str(Path(views[0]['render_path']).parent),gt_dir=str(Path(views[0]['gt_path']).parent),
                evaluation_frames=[Path(v['render_path']).name for v in views],optimization_input_frames=p['optimization_input_frames'],
                n_views=len(views),coverage={'render_only':[],'gt_only':[]},views=views))
    return dict(schema_version=2,freeze_id=c['freeze_id'],paper_ready=False,
        provenance=dict(code_commit=c['metric_source']['code_commit'],consumer_commit=contract['code']['commit'],scope=scope(c),
            **(dict(full_protocol=c['full_protocol'],preparation_inventory=c['preparation_inventory'],
                    inference_scope='full predeclared roster; conditional quality on all common available views') if full else
               {'paired_protocol':c['paired_protocol'],'inference_scope':'one-scene paired descriptive diagnostic'} if protocol else {})),
        room_split={'unit':'held_out_view'},room_methods=room,
        object_methods={name:dict(source_id=name,records=[]) for name in OBJECT_METHODS},
        coverage={'minimum_room_views_by_method':{},'minimum_room_scenes_by_method':{}}),coverage



def collect_scenes(c,contract,out):
    """Reuse only through the original source's config/E0-bound scene validator."""
    spec=c.get('render_source')
    if spec is None:
        results={};seals={}
        for scene in scenes(c):results[scene],seals[scene]=scene_result(out/'scenes'/scene,c,contract,scene)
        return results,seals
    code=source_check(spec);config_path=_checked(spec['config']);contract_path=_checked(spec['contract'])
    original,original_contract=_bound_contract(config_path,contract_path,producer_commit=spec['code_commit'])
    for key in ('scope','scene_ids','planned_objects','planned_views_per_scene','seed','bootstrap_samples',
                'bootstrap_seed','paper_ready','full_e3_gt_access','fill_seals','raw_source','metric_source','execution'):
        if c[key]!=original[key]:raise ValueError('metrics-only reuse changed a frozen treatment or source')
    if original.get('render_source') is not None or set(spec['scene_seals'])!=set(scenes(c)):
        raise ValueError('reuse requires original complete scene population')
    for scene in scenes(c):_checked(spec['scene_seals'][scene])
    if scope(c)==FULL_SCOPE:
        for key in ('full_protocol','preparation_inventory'):
            if c[key]!=original[key]:raise ValueError('full reuse changed original construction scope')
    program = """import json,sys
from run.icra2027 import e2_public_factorized as p
c,contract,out=p.context(sys.argv[1],sys.argv[2]);results={};seals={}
for scene in (c['scene_ids'] if c.get('scope')=='full_public_factorized_official_test_evaluation' else p.SCENES):results[scene],seals[scene]=p.scene_result(out/'scenes'/scene,c,contract,scene)
print(json.dumps(dict(results=results,seals=seals)))
"""
    env=raw.environment(c['execution']);env['PYTHONPATH']=str(code)
    payload=json.loads(subprocess.check_output([spec['python'],'-c',program,str(config_path),str(contract_path)],
        cwd=code,env=env,text=True))
    if list(payload['results'])!=scenes(c) or payload['seals']!=spec['scene_seals']:
        raise ValueError('reused scene seals/roster differ from new freeze')
    source_check(spec);_checked(spec['config']);_checked(spec['contract'])
    for scene in scenes(c):_checked(spec['scene_seals'][scene])
    return payload['results'],payload['seals']


def metric_runtime(c):
    # The archived metric module can predate this newer runtime-probe helper.
    # Probe from the exact consumer source; execute metrics from their pinned
    # source below. The metric implementation bytes must agree in both sources.
    if e3.sha256_file(raw.CODE/'robo/eval/fidelity_metrics.py')!=c['metric_source']['module']['sha256']:
        raise ValueError('runtime probe and actual metric implementation differ')
    return raw.probe(c['execution'],'metrics')

def metrics(config_path,contract_path):
    c,contract,out=context(config_path,contract_path);destination=out/'metrics'
    if destination.exists() or destination.is_symlink():raise FileExistsError('metric attempt exists')
    results,seals=collect_scenes(c,contract,out)
    manifest,coverage=make_manifest(c,contract,results)
    destination.mkdir(parents=True,exist_ok=False)
    raw.write_new(destination/'fidelity_manifest.json',manifest)
    raw.write_new(destination/'coverage.json',dict(scope=scope(c),planned_scenes=len(scenes(c)),planned_objects=sum(population(c).values()),
        planned_views_per_method=8*len(scenes(c)),rows=coverage,scene_seals=seals,paper_ready=False,
        conditional_quality=True,quality_comparison_paired=scope(c)==FULL_SCOPE or c.get('paired_protocol') is not None,
        paired_protocol=c.get('paired_protocol'),render_reuse_source=c.get('render_source'),
        common_eligible_views=[dict(scene_id=scene,view_id=v['view_id']) for scene,r in results.items()
            for v in r['composite_views'] if v['view_id'] in {x['view_id'] for x in r['raw_views']}],
        quality_scope=('full predeclared roster; all common available views; conditional quality, no headline gain' if scope(c)==FULL_SCOPE else 'one-scene paired descriptive diagnostic; no population inference or headline gain' if c.get('paired_protocol')
            else 'descriptive unequal-support row means; no improvement claim'),full_e3_gt_access=False))
    code=source_check(c['metric_source']);_checked(c['metric_source']['module'])
    env=raw.environment(c['execution'],'metrics');env['PYTHONPATH']=str(code)
    if metric_runtime(c)!=c['execution']['runtime']['metrics']:
        raise ValueError('canonical metric runtime differs')
    command=[c['execution']['python']['metrics'],'-m','robo.eval.fidelity_metrics',
        '--manifest',str(destination/'fidelity_manifest.json'),'--out',str(destination/'table'),
        '--lpips-device','cuda','--bootstrap-samples','2000','--bootstrap-seed','42']
    with (destination/'metrics.log').open('x') as log:
        process=subprocess.run(command,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT)
    if process.returncode:raise ValueError(f'canonical metrics failed: {process.returncode}')
    if metric_runtime(c)!=c['execution']['runtime']['metrics']:
        raise ValueError('canonical metric runtime changed during evaluation')
    source_check(c['metric_source']);_checked(c['metric_source']['module']);context(config_path,contract_path)
    if collect_scenes(c,contract,out)!=(results,seals):raise ValueError('render source changed during metrics')
    table=json.loads((destination/'table/fidelity_table.json').read_text())
    if table['lpips_backend_error']:raise ValueError('LPIPS failed; missing metrics cannot be promoted')
    raw.write_new(destination/'receipt.json',dict(status='PASS',command=command,metric_source=c['metric_source'],
        table=_identity(destination/'table/fidelity_table.json'),coverage=_identity(destination/'coverage.json'),paper_ready=False))
    seal_output(destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--contract',required=True);parser.add_argument('--phase',choices=['context','render','metrics'],required=True)
    parser.add_argument('--scene');a=parser.parse_args()
    if a.phase=='context':context(a.config,a.contract)
    elif a.phase=='render':render(a.config,a.contract,a.scene)
    else:metrics(a.config,a.contract)


if __name__=='__main__':main()
