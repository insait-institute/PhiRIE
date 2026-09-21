"""Typed automatic adapter boundary tests; camera producer owns geometric replay."""
import copy
import json
from pathlib import Path

import pytest

from robo.eval import e4_reset_eligibility as adapter
from robo.eval import e4_camera_scorer_gate as camera
from robo.eval import e4_candidate_screen as screen
from robo.eval import harness_runner as runner
from robo.manifest import hash as manifest_hash
from robo.eval.harness_spec import HarnessSpec, TreatmentSpec


def adapter_spec(config):
    # Global checkpoint/service contract validation is exercised by harness tests.
    return HarnessSpec(treatments={t["id"]: TreatmentSpec(**t) for t in config["treatments"]},
                       comparisons=(), raw=config)
from robo.eval.harness_validation import task_definition_hash
from run.icra2027 import e4_compact_harness as compact


def fixture(tmp_path, monkeypatch):
    tasks = [{'scene_id': scene, 'task_id': f'{scene}__task{task}'}
             for scene in ('scene0', 'scene1') for task in range(2)]
    protocol = {'fixed_manipulation_tasks': tasks}
    states = compact.definitions(protocol)
    config_path = tmp_path / 'camera.json'
    camera_config = {'study_scope':camera.AUTOMATIC_SCOPE,'freeze_id': 'fixture-v1', 'protocol': {'path': 'protocol', 'sha256': 'p'*64},
                     'menagerie_root': str(tmp_path / 'menagerie')}
    config_path.write_text(json.dumps(camera_config))
    output = tmp_path / 'outputs/icra2027/fixture-v1/harness/automatic_camera_scorer'
    output.mkdir(parents=True)
    summary = {'source': 'complete canonical CPU qualification'}
    gate = {'code': {'commit': 'a'*40, 'dirty': False}, 'upstream': summary}
    (output/'gate.json').write_text(json.dumps(gate))
    scenes, scene_config = {}, []
    for scene in ('scene0', 'scene1'):
        directory = tmp_path/scene
        directory.mkdir()
        manifest = directory/'manifest.json'; manifest.write_text('{}')
        suite = {'scene': scene, 'tasks': [{'task_id': t['task_id'], 'target': 'obj',
                 'instructions': {'default': 'Place the object.'}} for t in tasks if t['scene_id']==scene]}
        paths = {}
        for arm in ('A0', 'A4'):
            (directory/f'{arm}.json').write_text(json.dumps(suite))
            paths[arm] = str(directory/f'{arm}.json')
        planning = directory/'planning.json'; planning.write_text(json.dumps(suite))
        bundle = {'bundle_manifest_sha256': screen._sha256(manifest), 'planning_tasks': str(planning),
                  'variant_tasks': paths, 'factories': {arm: str(directory/arm) for arm in ('A0','A4')},
                  'scene_xml': {arm: str(directory/f'{arm}.xml') for arm in ('A0','A4')}}
        scenes[scene] = {'suites': {'A0': suite, 'A4': copy.deepcopy(suite)}, 'task_bundle': bundle}
        scene_config.append({'id': scene, 'menagerie_root': camera_config['menagerie_root'],
            'tasks_json': str(planning), 'task_freeze_manifest': str(manifest),
            'construction_variants': {variant: {'tasks_json': paths[arm],
                'factory_dir': bundle['factories'][arm], 'scene_xml': bundle['scene_xml'][arm]}
                for arm,variant in [('A0','fixed_single_path'),('A4','agentic')]}})
    rows, cells = [], []
    for state in states:
        for arm in ('A0', 'A4'):
            cpu_pass = arm == 'A4'
            row = {'cell_id': f'{arm.lower()}__{state.reset_state_id}', 'scene_id': state.scene_id,
                'task_id': state.task_id, 'policy_id': arm, 'episode': state.ep,
                'reset_seed': state.reset_seed, 'target': 'obj', 'passed': cpu_pass,
                'checks': {'accepted': cpu_pass}, 'reset_jitter': None}
            rows.append(row)
            cell = {key: row[key] for key in ('cell_id','scene_id','task_id','policy_id','episode','reset_seed','target')}
            cell.update(reset_state_id=state.reset_state_id, executed=cpu_pass, passed=cpu_pass,
                outcome='diagnostic_pass' if cpu_pass else 'build_failure',
                camera_metrics=None, workspace_metrics=None, disambiguation_metrics=None,
                scorer_metrics=None, reset_provenance={'actual': 'sealed'} if cpu_pass else None)
            if cpu_pass:
                cell['checks'] = {'camera': True, 'workspace': True, 'scorer': True,
                                  'disambiguation': True, 'exact_cpu_reset_replay': True}
            cells.append(cell)
    chain = {'rows': rows, 'scenes': scenes, 'summary': summary}
    checked = {'gate': gate, 'cells': cells, 'manifest_sha256': 'b'*64}
    monkeypatch.setattr(camera, 'validate_automatic_chain', lambda *a, **kw: chain, raising=False)
    monkeypatch.setattr(camera, 'validate_automatic_camera_output', lambda **kw: checked, raising=False)
    monkeypatch.setattr(compact, 'checked_protocol', lambda *a: protocol)
    config = {'paper_mode': False, 'policy': 'pi05_droid_jointpos', 'jitter': screen.JITTER_XY_M,
        'jitter_first_episode': True, 'scenes': scene_config,
        'cpu_reset_eligibility': {'kind': 'automatic_compact', 'camera_config': {
            'path': str(config_path), 'sha256': screen._sha256(config_path)},
            'bundle': str(output), 'gate_sha256': screen._sha256(output/'gate.json'), 'source_commit': 'a'*40},
        'contract': {'policy': {'id': 'pi05_droid_jointpos', 'checkpoint_hash': 'c'*64, 'kind': 'real'},
                     'horizon_s': 16.},
        'treatments': [{'id': f'{a}_raster', 'scene': variant, 'collision': 'full_room', 'observation': 'raster'}
                       for a,variant in [('a0','fixed_single_path'),('a4','agentic')]],
        'comparisons': [{'id': 'construction', 'axis': 'scene', 'baseline': 'a0_raster',
                         'treatments': ['a0_raster','a4_raster']}]}
    return config, adapter_spec(config), states, chain, checked


def test_all_40_cells_retained_and_camera_failures_typed(tmp_path, monkeypatch):
    config, spec, states, chain, checked = fixture(tmp_path, monkeypatch)
    checked['cells'][1].update(passed=False, outcome='diagnostic_fail')
    checked['cells'][1]['checks']['camera'] = False
    checked['cells'][3].update(passed=False, executed=False, outcome='diagnostic_error',
                               error='Renderer failed', checks=None, reset_provenance=None)
    result = adapter.load_reset_eligibility(config, spec, states, root=tmp_path)
    assert len(result) == 40
    assert sum(e['passed'] for e in result.values()) == 18
    assert result[('a0_raster',states[0].reset_state_id)]['failure_type'] == 'cpu_reset_validity_failure'
    assert result[('a4_raster',states[0].reset_state_id)]['failure_type'] == 'camera_scorer_failure'
    assert result[('a4_raster',states[1].reset_state_id)]['failed_checks'] == ['diagnostic_error']


@pytest.mark.parametrize('damage', ['config_missing','config_changed','gate_hash','producer','arm',
    'task_manifest','planning_tasks','paired_geometry','menagerie','reset','missing_cell','camera_identity'])
def test_automatic_authentication_drift_fails_closed(tmp_path, monkeypatch, damage):
    config,spec,states,chain,checked = fixture(tmp_path,monkeypatch)
    decl=config['cpu_reset_eligibility'];scene=config['scenes'][0]
    if damage=='config_missing': Path(decl['camera_config']['path']).unlink()
    elif damage=='config_changed': Path(decl['camera_config']['path']).write_text('{}')
    elif damage=='gate_hash': decl['gate_sha256']='0'*64
    elif damage=='producer': checked['gate']['code']['commit']='f'*40
    elif damage=='arm': config['treatments'][1]['observation']='composite_raw';spec=adapter_spec({**config,'comparisons':[]})
    elif damage=='task_manifest': Path(scene['task_freeze_manifest']).write_text('{"changed":true}')
    elif damage=='planning_tasks': scene['tasks_json']='changed'
    elif damage=='paired_geometry': scene['construction_variants']['agentic']['scene_xml']='changed'
    elif damage=='menagerie': scene['menagerie_root']='changed'
    elif damage=='reset': states.pop()
    elif damage=='missing_cell': checked['cells'].pop()
    elif damage=='camera_identity': checked['cells'][1]['reset_seed']+=1
    with pytest.raises((ValueError, OSError, camera.CameraScorerGateError)):
        adapter.load_reset_eligibility(config,spec,states,root=tmp_path)


@pytest.mark.parametrize('outcome', ['build_failure','diagnostic_error'])
def test_unexecuted_camera_cells_cannot_acquire_reset_telemetry(tmp_path, monkeypatch, outcome):
    config,spec,states,chain,checked=fixture(tmp_path,monkeypatch)
    cell=checked['cells'][0 if outcome=='build_failure' else 1]
    cell.update(outcome=outcome,executed=False,passed=False,checks=None,
                reset_provenance={'fabricated':'state'})
    with pytest.raises(ValueError,match='fabricated telemetry'):
        adapter.load_reset_eligibility(config,spec,states,root=tmp_path)


def failure_record(config,spec,states,chain,root):
    state=states[0]
    evidence=adapter.load_reset_eligibility(config,spec,states,root=root)[('a0_raster',state.reset_state_id)]
    task=chain['scenes'][state.scene_id]['suites']['A0']['tasks'][0]
    definition=runner._reset_definition(state,task=task,jitter_xy=screen.JITTER_XY_M)
    controller,camera_hash,action,dim=runner.legacy._frozen_config_hashes()
    contract={**copy.deepcopy(config['contract']), 'policy_execution':'not_invoked_prebuild',
        'policy_id':config['policy'], 'policy_checkpoint_hash':'c'*64,
        'runtime_policy_checkpoint_fingerprint':'c'*64,
        'reset_definition':definition,'reset_definition_hash':manifest_hash.canonical_hash(definition),
        'task_definition_hash':task_definition_hash(task),'controller_config_hash':controller,
        'camera_config_hash':camera_hash,'action_convention':action,'action_dim':dim,
        'task_instruction':'Place the object.','task_id':state.task_id,'reset_state_id':state.reset_state_id,
        'rollout_seed':state.base_seed,'reset_seed':state.reset_seed,
        'construction_artifacts':{'task_freeze_manifest_sha256':chain['scenes'][state.scene_id]['task_bundle']['bundle_manifest_sha256']}}
    return {'outcome':'build_failure','treatment_id':'a0_raster',
        'treatment':spec.treatments['a0_raster'].to_dict(),'scene_id':state.scene_id,
        'task_id':state.task_id,'reset_state_id':state.reset_state_id,'reset_seed':state.reset_seed,
        'base_seed':state.base_seed,'ticks':0,'success':False,'score':0.,'stages':{'place':False},
        'contract':contract,'construction_validity_evidence':evidence}


def test_uninvoked_cpu_failure_certifies_with_planned_jitter_only(tmp_path,monkeypatch):
    config,spec,states,chain,checked=fixture(tmp_path,monkeypatch)
    record=failure_record(config,spec,states,chain,tmp_path)
    assert chain['rows'][0]['reset_jitter'] is None
    assert record['contract']['reset_definition']['jitter'] is not None
    assert 'reset_provenance' not in record
    assert adapter.certify_prebuild_failure(record,spec,root=tmp_path)


@pytest.mark.parametrize('damage',['passed_cell','wrong_arm','jitter','task_digest','camera',
    'checkpoint','server','reset_telemetry','camera_telemetry','policy_execution','success','bundle'])
def test_automatic_prebuild_certificate_rejects_forgery(tmp_path,monkeypatch,damage):
    config,spec,states,chain,checked=fixture(tmp_path,monkeypatch)
    record=failure_record(config,spec,states,chain,tmp_path)
    if damage=='passed_cell':
        record['treatment_id']='a4_raster';record['treatment']=spec.treatments['a4_raster'].to_dict()
        record['construction_validity_evidence']=adapter.load_reset_eligibility(config,spec,states,root=tmp_path)[('a4_raster',states[0].reset_state_id)]
    elif damage=='wrong_arm': record['treatment_id']='a4_raster'
    elif damage=='jitter': record['contract']['reset_definition']['jitter']['offset_xy_m'][0]+=.01
    elif damage=='task_digest': record['contract']['task_definition_hash']='x'*64
    elif damage=='camera': record['contract']['camera_config_hash']='x'*64
    elif damage=='checkpoint': record['contract']['policy']['checkpoint_hash']='x'*64
    elif damage=='server': record['contract']['policy']['server_identity']={'verified':True}
    elif damage=='reset_telemetry': record['reset_provenance']={'actual':False}
    elif damage=='camera_telemetry': record['camera_metrics']={'fabricated':True}
    elif damage=='policy_execution': record['contract']['policy_execution']='invoked'
    elif damage=='success': record['success']=True
    elif damage=='bundle': record['contract']['construction_artifacts']['task_freeze_manifest_sha256']='x'*64
    with pytest.raises(ValueError):
        adapter.certify_prebuild_failure(record,spec,root=tmp_path)


@pytest.mark.parametrize('diagnostic_error', [False, True])
def test_camera_failure_certificate_retains_original_diagnostic(tmp_path,monkeypatch,diagnostic_error):
    config,spec,states,chain,checked=fixture(tmp_path,monkeypatch)
    chain['rows'][0].update(passed=True,checks={'accepted':True})
    cell=checked['cells'][0]
    cell.update(executed=not diagnostic_error,passed=False,
        outcome='diagnostic_error' if diagnostic_error else 'diagnostic_fail',
        error='Renderer failed' if diagnostic_error else None,
        checks=None if diagnostic_error else {'camera':False,'workspace':True,'scorer':True,
                                             'disambiguation':True,'exact_cpu_reset_replay':True},
        reset_provenance=None if diagnostic_error else {'actual':'sealed historical reset'})
    record=failure_record(config,spec,states,chain,tmp_path)
    assert record['construction_validity_evidence']['failure_type']=='camera_scorer_failure'
    assert adapter.certify_prebuild_failure(record,spec,root=tmp_path)
    # Historical diagnostic telemetry stays in its sealed source; it is never
    # relabelled as measurements of an unexecuted canonical policy episode.
    assert record.get('reset_provenance') is None


@pytest.mark.parametrize('error_class', [camera.CameraScorerGateError, screen.CandidateScreenError])
def test_producer_validation_errors_normalized_with_typed_cause(tmp_path,monkeypatch,error_class):
    config,spec,states,chain,checked=fixture(tmp_path,monkeypatch)
    error=error_class('sealed upstream artifact changed')
    def reject(*args,**kwargs):
        raise error
    monkeypatch.setattr(camera,'validate_automatic_chain',reject)
    with pytest.raises(ValueError,match=error_class.__name__) as caught:
        adapter.load_reset_eligibility(config,spec,states,root=tmp_path)
    assert caught.value.__cause__ is error


def test_unexpected_programming_error_is_not_relabelled(tmp_path,monkeypatch):
    config,spec,states,chain,checked=fixture(tmp_path,monkeypatch)
    error=RuntimeError('unexpected implementation failure')
    def reject(*args,**kwargs):
        raise error
    monkeypatch.setattr(camera,'validate_automatic_chain',reject)
    with pytest.raises(RuntimeError) as caught:
        adapter.load_reset_eligibility(config,spec,states,root=tmp_path)
    assert caught.value is error
