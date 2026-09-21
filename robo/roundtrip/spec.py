"""Small native-track config validation using existing canonical hashes."""
from pathlib import Path
import re
import yaml
from robo.manifest.hash import canonical_hash

PIN = re.compile(r'^[0-9a-f]{40}$')
SHA256 = re.compile(r'^[0-9a-f]{64}$')
NATIVE_HORIZONS = {'PickPlaceCounterToSink': 600, 'PickPlaceSinkToCounter': 900, 'PickPlaceCounterToCabinet': 750}
SCOPES = {'L0_target_only': 'target_only', 'L1_target_destination': 'target_destination',
          'L2_task_workspace': 'task_workspace', 'L3_room': 'room',
          'DEV_partial_observed_workspace':'partial_observed_workspace'}
METHODS = {'REF_NATIVE', 'B0_FIXED_NATIVE', 'B1_FIXED_PRIORITY', 'B2_EVIDENCE', 'B3_AGENT_NATIVE', 'B4_ROOM_REPAIR_NATIVE',
           'BM_BUDGET_MATCHED_NATIVE', 'V1_VERIFY_ONLY_NATIVE', 'B4_GS', 'B4_HC',
           'OBSERVED_SURFACE_NATIVE', 'REF_IMPORT_CONTROL'}

def load_spec(path):
    config = yaml.safe_load(Path(path).read_text())
    validate_spec(config)
    return config

def validate_spec(c):
    if c.get('schema_version') not in (1, 2) or c.get('benchmark') != 'robocasa_native':
        raise ValueError('requires resolved native schema; planning template is not executable')
    if c.get('execution_ready') is not True:
        raise ValueError('execution_ready must be true after pins are resolved')
    for key in ('robocasa_commit', 'robosuite_commit'):
        if not PIN.fullmatch(str(c.get('platform', {}).get(key, ''))):
            raise ValueError(f'unresolved {key}')
    if c['platform'].get('mujoco_version') != '3.3.1':
        raise ValueError('native release requires MuJoCo3.3.1')
    task = c['instance']
    if not task.get('task_id') or not isinstance(task.get('layout_id'), int) or not isinstance(task.get('style_id'), int):
        raise ValueError('resolve native task/layout/style IDs')
    if c['schema_version'] == 1 and task.get('split') != 'development':
        raise ValueError('first vertical slice is DEV only; TEST requires separate admission')
    if c['schema_version'] == 1 and (c.get('replacement_scope') != 'target_only' or c.get('oracle_context') is not True):
        raise ValueError('initial scope must explicitly label target-only oracle context')
    if c.get('horizon', 0) <= 0 or not c.get('reset_seeds') or len(set(c['reset_seeds'])) != len(c['reset_seeds']):
        raise ValueError('positive horizon and unique reset seeds required')
    if c.get('sensor_regime') != 'ideal_rgbd_posed':
        raise ValueError('first slice is ideal posed RGB-D')
    if c['schema_version'] == 2:
        _validate_v2(c)
    return canonical_hash(c)


def _validate_v2(c):
    for field in ('cohort_id', 'canonical_instance_id', 'reset_id'):
        if not isinstance(c.get(field), str) or not c[field].strip():
            raise ValueError(f'v2 requires separate {field}')
    seed=c.get('policy_rng_seed')
    if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:
        raise ValueError('policy_rng_seed must be a separate nonnegative integer')
    for field in ('canonical_manifest_sha256','reset_contract_sha256'):
        if not SHA256.fullmatch(str(c.get(field,''))):
            raise ValueError(f'unresolved {field}')
    if c.get('scope') not in SCOPES or c.get('replacement_scope') != SCOPES[c['scope']]:
        raise ValueError('scope and replacement_scope disagree')
    if c['scope']=='DEV_partial_observed_workspace' and (c['instance'].get('split')!='development' or c.get('controller_method')!='B3_AGENT_NATIVE'):
        raise ValueError('partial observed workspace is a separate B3-target DEV diagnostic only')
    if c['scope'] != 'L3_room' and c.get('oracle_context') is not True:
        raise ValueError('partial replacement must disclose retained oracle context')
    if c.get('controller_method') not in METHODS:
        raise ValueError('unresolved controller_method')
    followup = c.get('mechanism_followup')
    if c.get('controller_method') in {'B1_FIXED_PRIORITY', 'B2_EVIDENCE'} and followup != 'native_mechanism_followup_v1':
        raise ValueError('B1/B2 policy execution requires the declared mechanism follow-up')
    if followup is not None and (followup != 'native_mechanism_followup_v1' or c.get('scope') != 'L0_target_only' or c.get('renderer', 'native') != 'native' or c.get('execution_protocol') != 'primary_native'):
        raise ValueError('invalid mechanism follow-up scope or protocol')
    if c.get('execution_protocol') not in {'primary_native','full_horizon_feedback_diagnostic'}:
        raise ValueError('unresolved execution_protocol')
    if c['instance'].get('split') not in {'development','test'}:
        raise ValueError('unresolved reconstruction split')
    if c['instance']['split']=='test':
        admission=c.get('test_admission',{})
        if not all(SHA256.fullmatch(str(admission.get(k,''))) for k in
                   ('roster_sha256','dev_gate_sha256','thresholds_sha256')):
            raise ValueError('TEST requires sealed roster, DEV gates and thresholds')
    task=c['instance']['task_id']
    if task not in NATIVE_HORIZONS or c['horizon'] != NATIVE_HORIZONS[task]:
        raise ValueError('horizon differs from pinned native task definition or task unresolved')
    if c.get('native_success_threshold_overrides'):
        raise ValueError('native success threshold changes are forbidden')
    policy=c.get('policy',{})
    from robo.roundtrip.native_policy import CHECKPOINT_REVISION, POLICY_CONFIG, UPSTREAM_COMMIT
    if (policy.get('checkpoint_revision') != CHECKPOINT_REVISION or
        policy.get('config') != POLICY_CONFIG or policy.get('code_commit') != UPSTREAM_COMMIT or
        policy.get('replan_steps') != 5 or
        not SHA256.fullmatch(str(policy.get('checkpoint_receipt_sha256','')))):
        raise ValueError('v2 requires unchanged pinned policy and checkpoint content receipt')

def assert_frozen_pair(reference, comparison):
    # Explicit asset-bundle end-to-end contrast. Every other contract is fixed.
    allowed = {'treatment_id', 'build_manifest', 'import_xml'}
    if reference.get('schema_version')==2 and comparison.get('schema_version')==2:
        allowed.add('controller_method')
    a = {k: v for k,v in reference.items() if k not in allowed}
    b = {k: v for k,v in comparison.items() if k not in allowed}
    if canonical_hash(a) != canonical_hash(b):
        raise ValueError('undeclared native paired-field drift')


def paired_config_hash(config):
    allowed={'treatment_id','build_manifest','import_xml'}
    if config.get('schema_version')==2:allowed.add('controller_method')
    return canonical_hash({k:v for k,v in config.items() if k not in allowed})
