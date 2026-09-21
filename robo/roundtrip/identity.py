"""Evaluator-only canonical instance and perturbation contracts for schema v2."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from robo.manifest.hash import canonical_hash


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def xml_asset_closure(xml_path):
    """Hash only actual XML references, including recursively included XML."""
    files={};visited=set()
    def visit(path):
        path=Path(path).resolve(strict=True)
        if path in visited:return
        visited.add(path)
        root=ET.fromstring(path.read_text());compiler=root.find('compiler')
        options={} if compiler is None else compiler.attrib
        for element in root.iter():
            if 'file' not in element.attrib:continue
            directory=options.get('meshdir' if element.tag=='mesh' else 'texturedir','') if element.tag in {'mesh','texture'} else ''
            member=Path(element.attrib['file'])
            if not member.is_absolute():member=path.parent/directory/member
            member=member.resolve(strict=True)
            if not member.is_file():raise ValueError('XML asset reference is not a file')
            files[str(member)]=file_hash(member)
            if element.tag=='include':visit(member)
    visit(xml_path)
    return files


def create_canonical_instance(bundle_dir, config, asset_closure=None):
    bundle=Path(bundle_dir).resolve(strict=True)
    state=json.loads((bundle/'canonical_state.json').read_text())
    required={'integration_state','controller_state','observable_timing','native_metadata','qpos','qvel','object_states'}
    if not required.issubset(state):raise ValueError('canonical state lacks native identity fields')
    observed_closure=xml_asset_closure(bundle/'scene.xml')
    closure=observed_closure if asset_closure is None else copy.deepcopy(asset_closure)
    # A supplied closure is a reuse hint, never unchecked evidence.
    if closure != observed_closure:raise ValueError('XML asset closure differs')
    physical={'scene_xml_sha256':file_hash(bundle/'scene.xml'),
        'state_sha256':canonical_hash(state), 'asset_closure_sha256':canonical_hash(closure),
        'task':config['instance']['task_id'],'layout_id':config['instance']['layout_id'],
        'style_id':config['instance']['style_id'],'platform':config['platform'],
        'robot':config['robot'],'camera_size':config['camera_size'],
        'policy_sha256':canonical_hash(config['policy'])}
    result={'schema_version':2,'canonical_instance_id':'native-'+canonical_hash(physical),
        'dataset_split':config['instance']['split'],'identity':physical,'asset_closure':closure,
        'native_metadata_sha256':canonical_hash(state['native_metadata']),
        'controller_state_sha256':canonical_hash(state['controller_state']),
        'observable_timing_sha256':canonical_hash(state['observable_timing']),
        'integration_state_sha256':canonical_hash(state['integration_state']),
        'canonical_state_file_sha256':file_hash(bundle/'canonical_state.json'),
        'source_native_seed_is_instance_identity':False,'constructor_access':'FORBIDDEN'}
    result['manifest_sha256']=canonical_hash(result)
    return result


def validate_canonical_instance(manifest,bundle_dir,verify_assets=True):
    body={k:v for k,v in manifest.items() if k!='manifest_sha256'}
    if manifest.get('schema_version')!=2 or manifest.get('manifest_sha256')!=canonical_hash(body):
        raise ValueError('canonical manifest changed')
    if manifest['canonical_instance_id']!='native-'+canonical_hash(manifest['identity']):
        raise ValueError('canonical instance identity changed')
    bundle=Path(bundle_dir);state=json.loads((bundle/'canonical_state.json').read_text())
    if file_hash(bundle/'scene.xml')!=manifest['identity']['scene_xml_sha256'] or canonical_hash(state)!=manifest['identity']['state_sha256']:
        raise ValueError('canonical XML/state differs, seed equality cannot repair identity')
    if file_hash(bundle/'canonical_state.json')!=manifest['canonical_state_file_sha256']:
        raise ValueError('canonical state file bytes differ')
    if verify_assets and xml_asset_closure(bundle/'scene.xml')!=manifest['asset_closure']:
        raise ValueError('canonical XML asset closure differs')
    return manifest['canonical_instance_id']


def validate_delta(delta):
    value=np.asarray(delta,dtype=float)
    if value.shape!=(4,4) or not np.isfinite(value).all() or not np.allclose(value[3],[0,0,0,1],atol=1e-12,rtol=0):
        raise ValueError('reset delta must be a finite homogeneous transform')
    rot=value[:3,:3]
    if not np.allclose(rot.T@rot,np.eye(3),atol=1e-10,rtol=0) or not np.isclose(np.linalg.det(rot),1,atol=1e-10,rtol=0):
        raise ValueError('reset delta must be rigid, no rescaling or reflection')
    return value


def perturb_pose(pose_wxyz,delta_world):
    """Same Delta @ pose for native and estimated pose; no hidden GT snapping."""
    pose=np.asarray(pose_wxyz,dtype=float);delta=validate_delta(delta_world)
    if pose.shape!=(7,) or not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]),1,atol=1e-6,rtol=0):
        raise ValueError('pose must be finite xyz + unit wxyz')
    if np.array_equal(delta,np.eye(4)):
        return pose.tolist()
    if np.array_equal(delta[:3,:3],np.eye(3)):
        return np.r_[pose[:3]+delta[:3,3],pose[3:]].tolist()
    T=np.eye(4);T[:3,:3]=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix();T[:3,3]=pose[:3]
    result=delta@T;q=Rotation.from_matrix(result[:3,:3]).as_quat()
    return np.r_[result[:3,3],q[[3,0,1,2]]].tolist()


def create_reset_bank(instance_manifest,perturbations):
    records=[]
    for entry in perturbations:
        if set(entry)-{'reset_id','delta_world','policy_rng_seed'}:
            raise ValueError('reset contains undeclared fields or an absolute GT pose')
        validate_delta(entry['delta_world'])
        seed=entry['policy_rng_seed']
        if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:raise ValueError('invalid policy RNG seed')
        if not isinstance(entry['reset_id'],str) or not entry['reset_id']:raise ValueError('missing reset_id')
        records.append(copy.deepcopy(entry))
    if not records or len({r['reset_id'] for r in records})!=len(records):raise ValueError('duplicate or empty reset roster')
    bank={'schema_version':2,'canonical_instance_id':instance_manifest['canonical_instance_id'],
        'canonical_manifest_sha256':instance_manifest['manifest_sha256'],
        'unchanged_robot_fixture_state_sha256':instance_manifest['identity']['state_sha256'],
        'perturbation_convention':'Delta_world @ each_arm_own_initial_pose',
        'resets':records}
    bank['reset_contract_sha256']=canonical_hash(bank)
    return bank


def validate_reset_bank(bank,instance_manifest):
    expected=create_reset_bank(instance_manifest,bank['resets'])
    if canonical_hash(bank)!=canonical_hash(expected):raise ValueError('reset/canonical contract differs')
    return bank['reset_contract_sha256']


def assert_unique_instances(manifests):
    ids=[m['canonical_instance_id'] for m in manifests]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate canonical instance, resets are not builds')
