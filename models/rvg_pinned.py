"""Resolve existing ReconViaGen dependencies to audited local snapshots.

No sampler/model/treatment changes. The original full loader still constructs
VGGT, BiRefNet, DINOv2 and DreamSim; only their source/cache locations differ.
"""
from pathlib import Path
import json


def resolve_declared(value, mapping):
    if str(value) in mapping:
        return mapping[str(value)]
    if str(value) in mapping.values():
        return str(value)
    raise ValueError(f'undeclared pretrained resource requested: {value}')


def load_pipeline(config_path):
    import torch
    import trellis.pipelines.trellis_image_to_3d as module
    cfg=json.loads(Path(config_path).read_text())
    mappings={name:item['path'] for name,item in cfg['models'].items()}
    vggt_map={'Stable-X/vggt-object-v0-1':mappings['vggt_snapshot']}
    biref_map={'ZhengPeng7/BiRefNet':mappings['birefnet_snapshot']}
    hub_map={'facebookresearch/dinov2':mappings['dinov2_source'],
             'facebookresearch/dino:main':str(Path(mappings['dreamsim'])/'facebookresearch_dino_main')}
    original_vggt=module.VGGT.from_pretrained
    original_biref=module.AutoModelForImageSegmentation.from_pretrained
    original_hub=torch.hub.load
    original_dreamsim=module.dreamsim
    original_hub_dir=torch.hub.get_dir()
    def vggt(path,*args,**kwargs):
        kwargs['local_files_only']=True
        return original_vggt(resolve_declared(path,vggt_map),*args,**kwargs)
    def biref(path,*args,**kwargs):
        kwargs['local_files_only']=True
        return original_biref(resolve_declared(path,biref_map),*args,**kwargs)
    def hub(path,name,*args,**kwargs):
        kwargs['source']='local'
        return original_hub(resolve_declared(path,hub_map),name,*args,**kwargs)
    def dreamsim(*args,**kwargs):
        if kwargs.get('dreamsim_type')!='dino_vitb16':
            raise ValueError('undeclared DreamSim model variant')
        kwargs['cache_dir']=mappings['dreamsim']
        return original_dreamsim(*args,**kwargs)
    module.VGGT.from_pretrained=staticmethod(vggt)
    module.AutoModelForImageSegmentation.from_pretrained=staticmethod(biref)
    module.dreamsim=dreamsim
    torch.hub.load=hub
    torch.hub.set_dir(str(Path(mappings['dinov2_source']).parent))
    try:
        return module.TrellisVGGTTo3DPipeline.from_pretrained(mappings['rvg_snapshot'])
    finally:
        module.VGGT.from_pretrained=original_vggt
        module.AutoModelForImageSegmentation.from_pretrained=original_biref
        module.dreamsim=original_dreamsim
        torch.hub.load=original_hub
        torch.hub.set_dir(original_hub_dir)
