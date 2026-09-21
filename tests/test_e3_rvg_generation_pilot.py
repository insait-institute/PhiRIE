import importlib.util
import json
from pathlib import Path
import pytest
from models.rvg_pinned import resolve_declared

@pytest.fixture
def pilot():
    path=Path(__file__).resolve().parents[1]/'run/icra2027/e3_rvg_generation_pilot.py';spec=importlib.util.spec_from_file_location('rvgpilot',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_no_unpinned_model_fallback():
    mapping={'expected/model':'/pinned/model'}
    assert resolve_declared('expected/model',mapping)=='/pinned/model'
    assert resolve_declared('/pinned/model',mapping)=='/pinned/model'
    for value in ['unexpected/model','/another/model','https://example.org/model']:
        with pytest.raises(ValueError,match='undeclared'):resolve_declared(value,mapping)

def test_rvg_preserves_all_jobs_when_generation_or_views_unavailable(pilot,tmp_path):
    jobs=[{'job_id':f'scene:auto:{1000+i}','automatic_instance_id':1000+i,'prepared':i<3,'output_index':i,'input':{'sha256':'initial-crop'}} for i in range(6)]
    rdir=tmp_path/'construction/objects/obj_00/rvg';rdir.mkdir(parents=True)
    for name in ['rvg_mesh.ply','rvg_gs.ply']:(rdir/name).write_bytes(b'generated')
    records=tmp_path/'producer_records';records.mkdir()
    (records/'object_00.json').write_text(json.dumps({'status':'generated','seed':42}))
    (records/'object_01.json').write_text(json.dumps({'status':'unavailable','reason':'fewer_than_two_usable_training_views'}))
    rows=pilot.collect_records(tmp_path,{'jobs':jobs},'freeze',1)
    assert len(rows)==6 and sum(r['prepared'] for r in rows)==3
    assert [r['status'] for r in rows]==['available','unavailable','generation_failed','unavailable','unavailable','unavailable']
    assert rows[0]['input_sha256']=='initial-crop'
    assert rows[0]['proposal_id']=='freeze:scene:auto:1000:reconviagen:initial:seed42'
    (rdir/'rvg_gs.ply').unlink()
    assert pilot.collect_records(tmp_path,{'jobs':jobs},'freeze',0)[0]['status']!='available'

def test_source_manifest_hash_closes_training_boundary(pilot,tmp_path):
    for name in ['pilot_summary.json','output_hashes.json','input_manifest.json']:(tmp_path/name).write_text('{}')
    c={'source_pilot':str(tmp_path),'source_summary_sha256':pilot.sha(tmp_path/'pilot_summary.json'),'source_output_hashes_sha256':pilot.sha(tmp_path/'output_hashes.json'),'source_input_manifest_sha256':'wrong'}
    with pytest.raises(pilot.PilotError,match='input_manifest'):pilot.source_boundary(c)

def test_pinned_loader_routes_every_nested_resource_and_restores_bindings(tmp_path,monkeypatch):
    import sys,types,torch
    from models.rvg_pinned import load_pipeline
    root=types.ModuleType('trellis');root.__path__=[]
    pipelines=types.ModuleType('trellis.pipelines');pipelines.__path__=[]
    module=types.ModuleType('trellis.pipelines.trellis_image_to_3d')
    root.pipelines=pipelines;pipelines.trellis_image_to_3d=module
    for name,mod in [('trellis',root),('trellis.pipelines',pipelines),('trellis.pipelines.trellis_image_to_3d',module)]:monkeypatch.setitem(sys.modules,name,mod)
    calls=[]
    class Model:
        @classmethod
        def from_pretrained(cls,path,**kwargs):calls.append((cls.__name__,path,kwargs));return object()
    class VGGT(Model):pass
    class BiRef(Model):pass
    module.VGGT=VGGT;module.AutoModelForImageSegmentation=BiRef
    def original_hub(path,name,**kwargs):calls.append(('hub',path,kwargs))
    monkeypatch.setattr(torch.hub,'load',original_hub)
    def original_dreamsim(**kwargs):calls.append(('dreamsim',kwargs['cache_dir'],kwargs))
    module.dreamsim=original_dreamsim
    class Pipeline:
        @staticmethod
        def from_pretrained(path):
            assert path=='/pinned/rvg_snapshot'
            VGGT.from_pretrained('Stable-X/vggt-object-v0-1')
            BiRef.from_pretrained('ZhengPeng7/BiRefNet')
            torch.hub.load('facebookresearch/dinov2','dinov2_vitl14_reg')
            torch.hub.load('facebookresearch/dino:main','dino_vitb16')
            module.dreamsim(dreamsim_type='dino_vitb16',cache_dir='wrong')
            return 'pipeline'
    module.TrellisVGGTTo3DPipeline=Pipeline
    config=tmp_path/'models.json';config.write_text(json.dumps({'models':{k:{'path':'/pinned/'+k} for k in ['rvg_snapshot','vggt_snapshot','birefnet_snapshot','dinov2_source','dreamsim']}}))
    assert load_pipeline(config)=='pipeline'
    assert calls[0][1]=='/pinned/vggt_snapshot' and calls[0][2]['local_files_only']
    assert calls[1][1]=='/pinned/birefnet_snapshot' and calls[1][2]['local_files_only']
    assert calls[2][1]=='/pinned/dinov2_source' and calls[2][2]['source']=='local'
    assert calls[3][1]=='/pinned/dreamsim/facebookresearch_dino_main'
    assert calls[4][1]=='/pinned/dreamsim'
    assert torch.hub.load is original_hub and module.dreamsim is original_dreamsim
