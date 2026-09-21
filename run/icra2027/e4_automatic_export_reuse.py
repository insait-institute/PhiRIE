"""Explicit historical static/A0 reuse; original XML and measured bytes stay intact."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess

from robo.eval import e4_candidate_screen as screen
from run.icra2027 import e4_compact_materialization as api

KIND = 'e4_automatic_export_reuse'
CORE_OMIT = {'created_utc', 'destination', 'provenance'}


def core(manifest):
    return {k:v for k,v in manifest.items() if k not in CORE_OMIT}


def inventory(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("reuse source root is not a regular directory")
    result = {}
    for member in sorted(directory.rglob('*')):
        if member.is_symlink():
            raise ValueError('reuse source contains a symlink')
        if member.is_file():
            result[str(member.relative_to(directory))] = api.identity(member)
    if not result:
        raise ValueError('reuse source tree is empty')
    return result


def _snapshot(source):
    cwd = Path(source['code_root'])
    top = subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=cwd,text=True).strip()
    if Path(top) != cwd or cwd.resolve() != cwd:
        raise ValueError('original reuse code root differs')
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=cwd,text=True).strip()
    dirty = subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'],cwd=cwd,text=True)
    if head != source['code_commit'] or dirty.strip():
        raise ValueError('original reuse source is not exact and clean')


def validate_source(source, *, require_report=True):
    """Recheck complete original closures; never assign them the current producer."""
    if source.get('schema_version') != 1 or source.get('scope') != KIND:
        raise ValueError('reuse source schema differs')
    expected_keys = {'schema_version','scope','scene_id','freeze_id','code_root','code_commit','runtime','identities','factories','trees'}
    if require_report:
        expected_keys.add('original_validation')
    if set(source) != expected_keys:
        raise ValueError('reuse source field roster differs')
    root = screen.evidence_root()
    stage = root/'outputs/icra2027'/source['freeze_id']
    parent = stage/'automatic_candidates'/source['scene_id']/'materialized'
    expected_paths = {'e0':stage/'contract/freeze_manifest.json',
        'static_manifest':parent/'shared_room_static/manifest.json',
        'static_seal':parent/'shared_room_static/seal.json',
        'static_spec':parent/'shared_room_static_spec.json'}
    if source['factories'] != {p:str(parent/p) for p in screen.POLICIES}:
        raise ValueError('reuse factory paths differ from original stage/scene')
    for p in screen.POLICIES:
        expected_paths[p+'_manifest'] = parent/p/'materialization_manifest.json'
        expected_paths[p+'_seal'] = parent/p/'seal.json'
        m=api.read(expected_paths[p+'_manifest'])
        for name in m['output_members']:
            expected_paths[p+':'+name] = parent/p/name
    if set(source['identities']) != set(expected_paths)|{'config'}:
        raise ValueError('reuse identity roster differs')
    if any(source['identities'][k]['path'] != str(v) for k,v in expected_paths.items()):
        raise ValueError('reuse identity path differs')
    expected_trees = {'static':parent/'shared_room_static','A0_sim':parent/'A0/sim',
                      'A0_sim_export':parent/'A0/sim_export'}
    if set(source['trees']) != set(expected_trees) or any(
        source['trees'][k]['path'] != str(v) for k,v in expected_trees.items()):
        raise ValueError('reuse tree paths differ')
    for name, identity in source['identities'].items():
        allowed_root = Path(source['code_root']) if name == 'config' else root
        screen.sealed_cpu._inside(Path(identity['path']), root=allowed_root, label='reuse identity')
        if api.identity(identity['path']) != identity:
            raise ValueError('original reuse identity changed')
    contract = api.read(source['identities']['e0']['path'])
    api.contract_digest(contract)
    if (contract['code']['commit'] != source['code_commit'] or contract['code']['dirty'] is not False
            or contract['code']['repository'] != source['code_root']
            or contract['freeze_id'] != source['freeze_id']):
        raise ValueError('original reuse E0 source differs')
    if api.resource(contract, 'e4_compact_qualification_config') != source['identities']['config'] \
            or api.resource(contract, 'qualification_python') != source['runtime']:
        raise ValueError('original reuse config/runtime differs from E0')
    if api.identity(source['runtime']['path']) != source['runtime']:
        raise ValueError('original reuse Python changed')
    _snapshot(source)
    for key, item in source['trees'].items():
        if inventory(item['path']) != item['members']:
            raise ValueError('original reuse tree changed: '+key)
    for policy in screen.POLICIES:
        manifest = api.read(source['identities'][policy+'_manifest']['path'])
        if (manifest['scene_id'] != source['scene_id'] or manifest['policy_id'] != policy
                or manifest['destination'] != str(Path(source['factories'][policy]).relative_to(root))
                or manifest['provenance']['code_root'] != source['code_root']
                or manifest['provenance']['materializer_commit'] != source['code_commit']
                or manifest['provenance']['validator_commit'] != source['code_commit']):
            raise ValueError('original reuse materialization binding differs')
    package = api.read(source['identities']['static_manifest']['path'])
    if package.get('source_manifest_sha256') != {p:source['identities'][p+'_manifest']['sha256'] for p in screen.POLICIES}:
        raise ValueError('reused static paired factory binding differs')
    if require_report:
        report=source['original_validation']
        if set(report) != {'materializations','export'} or set(report['materializations']) != set(screen.POLICIES):
            raise ValueError('original validation report schema differs')
        for p in screen.POLICIES:
            m=report['materializations'][p]
            if (m['manifest_sha256'] != source['identities'][p+'_manifest']['sha256']
                or m['materializer_commit'] != source['code_commit']
                or m['validator_commit'] != source['code_commit']
                or m['code_root'] != source['code_root'] or m['scene_id'] != source['scene_id']
                or m['policy_id'] != p):
                raise ValueError('original validator report source differs')
        for role,filename in {'scene_xml':'scene.xml','mujoco_settle':'mujoco_settle.json',
                'room_collision_report':'room_collision_report.json','isaac_manifest':'isaac_manifest.json'}.items():
            if report['export']['artifacts'][role] != screen._identity(parent/'A0/sim_export'/filename,root=root):
                raise ValueError('original validator export identity differs')
    return source


def inspect_source(stage_root, scene_id):
    """Run the actual original validator once, before freezing this declaration."""
    stage = Path(stage_root).resolve(strict=True)
    contract_path = stage/'contract/freeze_manifest.json'
    contract = api.read(contract_path); api.contract_digest(contract)
    parent = stage/'automatic_candidates'/scene_id/'materialized'
    factories = {p:str(parent/p) for p in screen.POLICIES}
    identities = {'e0':api.identity(contract_path),
        'config':api.resource(contract,'e4_compact_qualification_config'),
        'static_manifest':api.identity(parent/'shared_room_static/manifest.json'),
        'static_seal':api.identity(parent/'shared_room_static/seal.json'),
        'static_spec':api.identity(parent/'shared_room_static_spec.json')}
    for p in screen.POLICIES:
        identities[p+'_manifest'] = api.identity(parent/p/'materialization_manifest.json')
        identities[p+'_seal'] = api.identity(parent/p/'seal.json')
    source = dict(schema_version=1, scope=KIND, scene_id=scene_id, freeze_id=stage.name,
        code_root=contract['code']['repository'], code_commit=contract['code']['commit'],
        runtime=api.resource(contract,'qualification_python'), identities=identities,
        factories=factories, trees={})
    # Snapshot all immutable materialized members, including masks/cameras/source mesh.
    for p in screen.POLICIES:
        m = api.read(identities[p+'_manifest']['path'])
        for relative in m['output_members']:
            identities[p+':'+relative] = api.identity(Path(factories[p])/relative)
    for name,path in [('static',parent/'shared_room_static'),('A0_sim',parent/'A0/sim'),
                      ('A0_sim_export',parent/'A0/sim_export')]:
        source['trees'][name] = dict(path=str(path),members=inventory(path))
    validate_source(source, require_report=False)
    script = '''import json,sys
from pathlib import Path
from robo.eval import e4_candidate_screen as s
r=json.load(sys.stdin); fs={p:Path(x) for p,x in r['factories'].items()}
c=s._automatic_export_context(fs,scene_id=r['scene_id'],root=s.evidence_root())
e=s.validate_full_room_export(fs['A0'],scene_id=r['scene_id'],policy='A0',root=s.evidence_root(),expected_object_slots=c['rosters']['A0']['accepted_slots'],expected_discovered_slots=c['object_slots'],automatic_factories=fs)
print(json.dumps({'materializations':c['reports'],'export':e}))
'''
    env = dict(os.environ, SIMANY_EVIDENCE_ROOT=str(screen.evidence_root()),SIMANY_SCENE=scene_id)
    env.pop('PYTHONPATH',None)
    result = subprocess.run([source['runtime']['path'],'-c',script],cwd=source['code_root'],
        env=env,input=json.dumps(source),capture_output=True,text=True,check=True)
    source['original_validation'] = json.loads(result.stdout)
    validate_source(source)
    return source


def publish(parent, source, *, code):
    """Seal the declared origin before copying; reruns only fill missing units."""
    parent=Path(parent); validate_source(source)
    receipt=parent/'export_reuse'
    value=dict(source=source, code=code, destination_parent=str(parent))
    if receipt.exists():
        screen._validate_bundle(receipt,root=screen.evidence_root(),expected_kind=KIND)
        if api.read(receipt/'source.json') != value:
            raise ValueError('existing reuse declaration differs')
    else:
        screen._publish_bundle(receipt,manifest_kind=KIND,
            payloads={'source.json':screen._json_bytes(value)},manifest_fields={'code':code})
    return value


def resolve(factory, factories=None, *, scene_id, root):
    """Return only authenticated original paths; no implicit historical fallback."""
    parent=Path(factory).parent; receipt=parent/'export_reuse'
    if not receipt.exists(): return None
    bundle=screen._validate_bundle(receipt,root=root,expected_kind=KIND)
    value=api.read(receipt/'source.json')
    if value['destination_parent'] != str(parent) or value['code'] != screen._code_snapshot(value['code']['commit']):
        raise ValueError('reuse declaration destination/source differs')
    if bundle['manifest']['code'] != value['code']:
        raise ValueError('reuse declaration producer differs')
    source=validate_source(value['source'])
    if source['scene_id'] != scene_id:
        raise ValueError('reuse declaration scene differs')
    current={p:parent/p for p in screen.POLICIES}
    if factories is not None and {p:str(Path(factories[p])) for p in screen.POLICIES} != {p:str(current[p]) for p in screen.POLICIES}:
        raise ValueError('reuse paired destination differs')
    for p in screen.POLICIES:
        old=api.read(source['identities'][p+'_manifest']['path'])
        new=api.read(current[p]/'materialization_manifest.json')
        if core(old) != core(new):
            raise ValueError('reuse construction/config/object population differs')
    return source


def copy_a0(parent, source):
    """Keep XML, meshdir and settle bytes identical; retain original source paths."""
    parent=Path(parent)
    validate_source(source)
    for name in ('sim','sim_export'):
        original=source['trees']['A0_'+name]
        destination=parent/'A0'/name
        members=original['members']
        if destination.exists():
            actual=inventory(destination)
            if set(actual) != set(members) or any(
                {k:v for k,v in actual[n].items() if k!='path'} !=
                {k:v for k,v in members[n].items() if k!='path'} for n in actual):
                raise ValueError('existing copied A0 export differs')
        else:
            # Atomic directory publication; partial copies cannot masquerade as complete.
            from tempfile import TemporaryDirectory
            with TemporaryDirectory(prefix='.reuse-',dir=parent) as tmp:
                staged=Path(tmp)/name
                shutil.copytree(original['path'],staged)
                actual=inventory(staged)
                if set(actual)!=set(members) or any(
                    {k:v for k,v in actual[n].items() if k!='path'} !=
                    {k:v for k,v in members[n].items() if k!='path'} for n in actual):
                    raise ValueError('A0 export changed during copy')
                os.rename(staged,destination)
    validate_source(source)


def verify_a0(parent, source):
    for name in ('sim','sim_export'):
        expected=source['trees']['A0_'+name]['members']
        actual=inventory(Path(parent)/'A0'/name)
        if set(actual)!=set(expected) or any(
            {k:v for k,v in actual[n].items() if k!='path'} !=
            {k:v for k,v in expected[n].items() if k!='path'} for n in actual):
            raise ValueError('reused A0 exported or measured bytes changed')
