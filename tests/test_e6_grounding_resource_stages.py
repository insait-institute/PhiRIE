"""Resource-stage ordering, claims, immutable receipts and failed-query closure."""
from types import SimpleNamespace
from pathlib import Path
import pytest
from run.icra2027 import e6_public_grounding as d


def setup(monkeypatch,tmp_path,fail_phase=None,upstream='PASS'):
    s=d.shared;stage=tmp_path/'stage';stage.mkdir();dest=stage/'audit/public_grounding/scene_clean'
    cfg=dict(source_commit='source',freeze_id='stage',scope='e6_public_grounding_v1')
    runtime=dict(render_python='gs',sam3_python='sam',sam3_source=dict(path='sam'),sam3_checkpoint=dict(path='weights'))
    rgb=dict(python='cpu');reference=dict(terminal_manifest='original');origin=tmp_path/'origin'
    queries=[dict(scene_id='scene',task_id=f'q{i}',task_family='place_in_region',query_sha256='a'*64,
                  source_state_anchor='rgb',roles={}) for i in range(4)]
    plan=dict(source_commit='source',config={'sha256':'cfg'},freeze_id='stage',scene_id='scene',condition_id='clean',
              upstream_status=upstream,planned_queries=4)
    context=(cfg,runtime,rgb,dict(stage_status=upstream),origin,reference,queries,dest,plan)
    monkeypatch.setattr(s,'ROOT',tmp_path)
    monkeypatch.setattr(d,'plan_context',lambda *a:context)
    monkeypatch.setattr(d,'validate',lambda *a:(cfg,runtime,rgb,{}))
    monkeypatch.setattr(s,'environment',lambda *a:{})
    monkeypatch.setattr(d,'associate',lambda *a:dict(discovered_objects=[],queries=queries,robot_frame=None,physics_verified=False))
    for key in ('SLURM_JOB_GPUS','SLURM_STEP_GPUS','SLURM_GPUS_ON_NODE'):
        monkeypatch.delenv(key,raising=False)
    calls=[]
    def invoke(cmd,**kwargs):
        phase=cmd[-1];calls.append(phase)
        if phase!=fail_phase:
            if phase=='render':
                (dest/'mesh_derive').mkdir();(dest/'mesh_derive/view_0000.npz').write_bytes(b'render fixture')
            elif phase=='fuse':(dest/'derived_mesh.ply').write_bytes(b'mesh fixture')
            else:(dest/'auto_instances.npz').write_bytes(b'instances fixture')
        return SimpleNamespace(returncode=9 if phase==fail_phase else 0)
    monkeypatch.setattr(d.subprocess,'run',invoke)
    return stage,dest,calls


def test_resource_stage_dependencies_and_cpu_gpu_separation(monkeypatch,tmp_path):
    stage,dest,calls=setup(monkeypatch,tmp_path)
    d.prepare('config',stage,'scene','clean')
    with pytest.raises(FileExistsError):d.prepare('config',stage,'scene','clean')
    with pytest.raises((ValueError,FileNotFoundError,d.shared.sealed.CandidateScreenError)):
        d.run_phase('config',stage,'scene','clean','fuse')
    assert calls==[]
    render=d.run_phase('config',stage,'scene','clean','render')
    assert render['resource_class']=='gpu' and render['predecessors']=={}
    monkeypatch.setenv('SLURM_GPUS_ON_NODE','1')
    with pytest.raises(ValueError,match='CPU stage'):
        d.run_phase('config',stage,'scene','clean','fuse')
    monkeypatch.delenv('SLURM_GPUS_ON_NODE')
    fuse=d.run_phase('config',stage,'scene','clean','fuse')
    assert fuse['resource_class']=='cpu' and set(fuse['predecessors'])=={'render'}
    discovery=d.run_phase('config',stage,'scene','clean','discover')
    assert set(discovery['predecessors'])=={'render','fuse'}
    result=d.finalize('config',stage,'scene','clean')
    assert result['stage_status']=='PASS' and len(result['queries'])==4
    assert calls==['render','fuse','discover']
    with pytest.raises(FileExistsError):d.run_phase('config',stage,'scene','clean','render')
    with pytest.raises(d.shared.sealed.CandidateScreenError,match='overwrite'):d.finalize('config',stage,'scene','clean')


def test_failed_cpu_stage_closes_all_four_queries_without_discovery(monkeypatch,tmp_path):
    stage,dest,calls=setup(monkeypatch,tmp_path,fail_phase='fuse')
    d.prepare('config',stage,'scene','clean')
    d.run_phase('config',stage,'scene','clean','render')
    failed=d.run_phase('config',stage,'scene','clean','fuse')
    assert failed['returncode']==9
    with pytest.raises(ValueError,match='failed predecessor'):
        d.run_phase('config',stage,'scene','clean','discover')
    result=d.finalize('config',stage,'scene','clean')
    assert result['stage_status']=='FAIL' and result['failed_stage']=='fuse' and len(result['queries'])==4
    assert result['discovered_objects'] is None and result['robot_frame'] is None
    assert calls==['render','fuse']


def test_upstream_failure_finalizes_without_any_resource_stage(monkeypatch,tmp_path):
    stage,dest,calls=setup(monkeypatch,tmp_path,upstream='FAIL')
    d.prepare('config',stage,'scene','clean')
    with pytest.raises(ValueError,match='rejected construction'):
        d.run_phase('config',stage,'scene','clean','render')
    result=d.finalize('config',stage,'scene','clean')
    assert result['stage_status']=='FAIL' and len(result['queries'])==4 and calls==[]


def test_phase_output_and_claim_tampering_rejected(monkeypatch,tmp_path):
    stage,dest,calls=setup(monkeypatch,tmp_path)
    d.prepare('config',stage,'scene','clean')
    d.run_phase('config',stage,'scene','clean','render')
    (dest/'render_claim.json').write_text('{}')
    with pytest.raises(ValueError,match='binding/output'):
        d.run_phase('config',stage,'scene','clean','fuse')
    assert calls==['render']


def test_association_cannot_reserve_gpu(monkeypatch,tmp_path):
    stage,dest,calls=setup(monkeypatch,tmp_path)
    monkeypatch.setenv('SLURM_JOB_GPUS','GPU-example')
    with pytest.raises(ValueError,match='CPU stage'):d.finalize('config',stage,'scene','clean')
