import json
from pathlib import Path
import pytest
from robo.campaign import generator_comparison as gc
from robo.campaign.core import save, receipt


def test_recovery_preserves_frozen_inputs(tmp_path):
    path=tmp_path/'freeze.json'
    gc.frozen(path, {'seed':0}); gc.frozen(path, {'seed':0})
    with pytest.raises(ValueError,match='immutable'):gc.frozen(path,{'seed':42})
    assert json.loads(path.read_text())=={'seed':0}


def make_bundle(root, name, gpu=True):
    b=root/name; flags=['--time=00:30:00']+(['--gres=gpu:h200:1'] if gpu else [])
    save(b/'plan.json',{'runtime':{'environments':{'component':{'sbatch_args':flags}}}})
    save(b/'jobs/task/intent.json',{})
    save(b/'jobs/task/submission.json',{'returncode':0,'job_id':name})
    return str(b)


def test_gpu_budget_counts_failures_and_reserves_full_active_limit(tmp_path,monkeypatch):
    bundles=[make_bundle(tmp_path,'1'),make_bundle(tmp_path,'2'),make_bundle(tmp_path,'3',False)]
    save(tmp_path/'registry.json',{'bundles':bundles})
    monkeypatch.setattr(gc.subprocess,'check_output',lambda *a,**k:'1|FAILED|900|1:0|sof1-h200-0|\n2|RUNNING|100|0:0|sof1-h200-1|\n3|COMPLETED|7200|0:0|sof1-h200-2|\n')
    value=gc.accounting(tmp_path)
    assert value['spent_gpu_hours']==.25
    assert value['reserved_gpu_hours']==.5
    assert value['remaining_unreserved_gpu_hours']==7.25
    assert value['active_gpu_jobs']==1


def test_unknown_scheduler_state_keeps_reservation(tmp_path,monkeypatch):
    save(tmp_path/'registry.json',{'bundles':[make_bundle(tmp_path,'1')]})
    monkeypatch.setattr(gc.subprocess,'check_output',lambda *a,**k:'')
    assert gc.accounting(tmp_path)['reserved_gpu_hours']==.5


def test_ambiguous_submission_blocks_resubmit(tmp_path):
    b=Path(make_bundle(tmp_path,'1'));(b/'jobs/task/submission.json').unlink()
    save(tmp_path/'registry.json',{'bundles':[str(b)]})
    with pytest.raises(RuntimeError,match='uncertain'):gc.accounting(tmp_path)


def test_generation_requires_original_input_seed_and_checkpoint(tmp_path):
    image=tmp_path/'rgba.png';image.write_bytes(b'original alpha')
    mesh=tmp_path/'mesh.ply';mesh.write_bytes(b'original mesh')
    lock=tmp_path/'model.json';save(lock,{'files':{'checkpoint':'abc'}})
    task={'params':{'backend':'trellis2','seed':0,'source_commit':'123','checkpoint_manifest':str(lock),'images':['anchor'],'inference':{}},'inputs':{'anchor':receipt(image)}}
    save(tmp_path/'task.json',task)
    gen={'status':'GENERATION_COMPLETE','backend':'trellis2','seed':0,'source':{'commit':'123'},'checkpoint_manifest':receipt(lock),'inference':{},'input_images':[receipt(image)],'artifacts':{'mesh':receipt(mesh)}}
    save(tmp_path/'generation.json',gen);ref=receipt(tmp_path/'generation.json')
    obj={'rgba_receipt':receipt(image)}
    assert gc.validate_generation(ref,tmp_path/'task.json',obj,'trellis2',0)==gen
    with pytest.raises(ValueError):gc.validate_generation(ref,tmp_path/'task.json',obj,'trellis2',42)
    image.write_bytes(b'changed alpha')
    with pytest.raises(ValueError):gc.validate_generation(ref,tmp_path/'task.json',obj,'trellis2',0)


def test_full_roster_missing_is_not_zero(tmp_path,monkeypatch):
    save(tmp_path/'cohort.json',{'planned_slots':128})
    rows=[dict(slot_id=f'{i}-{b}',object_key=str(i),backend=b,status='MISSING_INPUT') for i in range(32) for b in gc.BACKENDS]
    save(tmp_path/'recovery.json',{'slots':rows})
    monkeypatch.setattr(gc,'accounting',lambda _: {})
    result=gc.collect(tmp_path)
    assert result['planned_slots']==128 and result['common_support']==[]
    assert all(b['planned']==32 and b['measured']==0 and b['available_support_means'] is None for b in result['backends'])
    assert '--' in (tmp_path/'results/generator_comparison.tex').read_text()


def test_evaluation_child_uses_importable_module_when_parent_is_main(tmp_path,monkeypatch):
    from robo.campaign import component_worker as cw
    from types import SimpleNamespace
    save(tmp_path/'batch.json',{'slots':[{'slot_id':'one'}]})
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        assert argv[1:3]==['-m','robo.campaign.component_worker']
        save(Path(argv[-1])/'evaluation.json',{'status':'EVALUATED'})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cw,'__name__','__main__')
    monkeypatch.setattr(cw.subprocess,'run',run)
    cw.evaluate_batch(tmp_path/'batch.json',tmp_path/'result')
    assert len(calls)==1


def test_wall_time_guard_applies_to_cpu_too(tmp_path):
    with pytest.raises(ValueError,match='production limit2h'):
        gc.add_bundle(tmp_path,'cpu','evaluate-batch',tmp_path/'unused.json',minutes=121)


def test_sam_auxiliary_loader_accepts_real_hydra_keyword_and_restores_target(tmp_path,monkeypatch):
    import inspect
    import sys
    import types
    from models.s4_sam3d import load_component_pipeline
    calls=[]
    class MoGe:
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            calls.append(pretrained_model_name_or_path)
            return 'loaded'
    descriptor=inspect.getattr_static(MoGe,'from_pretrained')
    def inference(config,compile):
        assert not compile
        assert MoGe.from_pretrained(pretrained_model_name_or_path='Ruicheng/moge-vitl')=='loaded'
        return 'pipeline'
    for name in ['moge','moge.model','moge.model.v1','inference']:
        monkeypatch.setitem(sys.modules,name,types.ModuleType(name))
    sys.modules['moge.model.v1'].MoGeModel=MoGe
    sys.modules['inference'].Inference=inference
    checkpoint=tmp_path/'model.pt'
    assert load_component_pipeline(tmp_path,tmp_path/'pipeline.yaml',checkpoint)=='pipeline'
    assert calls==[str(checkpoint)]
    assert inspect.getattr_static(MoGe,'from_pretrained') is descriptor


def test_collector_retains_backend_blockers_without_marking_unrun_proposals_failed(tmp_path,monkeypatch):
    save(tmp_path/'cohort.json',{'planned_slots':128})
    save(tmp_path/'recovery.json',{'slots':[dict(slot_id=f'{i}-{b}',object_key=str(i),backend=b,status='MISSING_GENERATION') for i in range(32) for b in gc.BACKENDS]})
    save(tmp_path/'blockers/sam3d.json',{'reason':'infrastructure retry exhausted','attempts':['1','2']})
    monkeypatch.setattr(gc,'accounting',lambda _: {})
    result=gc.collect(tmp_path)
    sam=next(b for b in result['backends'] if b['backend']=='sam3d')
    assert sam['engineering_blocker']['attempts']==['1','2']
    assert sam['generated']==0 and sam['status_counts']=={'MISSING_GENERATION':32}


def test_rvg_export_uses_pinned_path_only_api(tmp_path):
    from robo.campaign.component_worker import save_rvg_gaussian
    class NativeGaussian:
        def save_ply(self,path):Path(path).write_bytes(b'native canonical data')
    save_rvg_gaussian(NativeGaussian(),tmp_path/'native.ply')
    assert (tmp_path/'native.ply').read_bytes()==b'native canonical data'


def test_rvg_export_failure_recovers_geometry_without_regeneration(tmp_path,monkeypatch):
    from robo.campaign import component_worker as cw
    import trimesh
    import numpy as np
    mesh=trimesh.creation.box();mesh.export(tmp_path/'mesh.ply')
    np.savez(tmp_path/'native.npz',vertices=mesh.vertices,faces=mesh.faces,vertex_attrs=np.ones((len(mesh.vertices),3)))
    save(tmp_path/'failed.json',{'status':'GENERATION_FAILED','backend':'reconviagen','slot_id':'smoke','seed':0,'error':"unexpected keyword argument 'transform'"})
    save(tmp_path/'source.json',{'objects':[{'slot_id':'smoke'}],'seed':0,'protocol':{},'evaluation_matching':{}})
    save(tmp_path/'prior.json',{'wall_s':70.,'initialization_seconds':59.})
    cfg={'failed_generation':receipt(tmp_path/'failed.json'),'source_config':receipt(tmp_path/'source.json'),'failed_batch':receipt(tmp_path/'prior.json'),'mesh':receipt(tmp_path/'mesh.ply'),'native_arrays':receipt(tmp_path/'native.npz')}
    save(tmp_path/'config.json',cfg)
    def evaluate(config,out):
        save(out/'evaluation.json',{'status':'EVALUATED'});return {'status':'EVALUATED'}
    monkeypatch.setattr(cw,'evaluate',evaluate)
    monkeypatch.setattr(cw,'rvg_generate',lambda *a:pytest.fail('recovery must not generate'))
    cw.recover_rvg_smoke(tmp_path/'config.json',tmp_path/'out')
    result=json.loads((tmp_path/'out/batch.json').read_text())
    assert result['status']=='VALIDATED' and result['rows'][0]['no_regeneration']
    assert 'gaussians' not in result['rows'][0]['artifacts']


def test_additional_sam_smoke_requires_bound_renewed_authorization(tmp_path):
    save(tmp_path/'inputs/sam3d-smoke-retry1.json',{'seed':0})
    assert gc.smoke_name(tmp_path,'sam3d')=='sam3d-smoke-retry1'
    save(tmp_path/'inputs/sam3d-smoke-retry2.json',{'seed':0})
    assert gc.smoke_name(tmp_path,'sam3d')=='sam3d-smoke-retry1'
    save(tmp_path/'blockers/sam3d.json',{'status':'BLOCKED_RETRY_LIMIT'})
    authorization=dict(backend='sam3d',additional_smoke_attempts=1,smoke_name='sam3d-smoke-retry2',user_request='submit failed work',
        historical_blocker=receipt(tmp_path/'blockers/sam3d.json'),smoke_config=receipt(tmp_path/'inputs/sam3d-smoke-retry2.json'))
    save(tmp_path/'authorizations/sam3d-resubmit-20260915.json',authorization)
    assert gc.smoke_name(tmp_path,'sam3d')=='sam3d-smoke-retry2'
    (tmp_path/'inputs/sam3d-smoke-retry2.json').write_text('{"seed":42}')
    with pytest.raises(ValueError,match='identity mismatch'):gc.smoke_name(tmp_path,'sam3d')


def test_hala_profile_uses_one_a6000_and_preserves_sam_attention(tmp_path,monkeypatch):
    from robo.campaign import runner
    save(tmp_path/'registry.json',{'bundles':[]});save(tmp_path/'config.json',{'seed':0})
    monkeypatch.setattr(runner,'prepare',lambda *a:None)
    gc.add_bundle(tmp_path,'sam-tail','generate',tmp_path/'config.json',gpu=True,minutes=15,node_profile='hala')
    runtime=json.loads((tmp_path/'inputs/sam-tail-runtime.json').read_text())['environments']['component']
    flags=runtime['sbatch_args']
    assert '--nodelist=hala' in flags and '--gres=gpu:a6000:1' in flags
    assert '--gres=gpu:h200:1' not in flags and not any('--array' in f for f in flags)
    assert runtime['env']['ATTN_BACKEND']==runtime['env']['SPARSE_ATTN_BACKEND']=='flash_attn'


def test_unknown_node_profile_is_not_submitted(tmp_path):
    with pytest.raises(ValueError,match='unknown node profile'):
        gc.add_bundle(tmp_path,'bad','generate',tmp_path/'config.json',gpu=True,node_profile='typo')
