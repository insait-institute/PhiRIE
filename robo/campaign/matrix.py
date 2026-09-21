"""Build a finite component campaign from real, frozen scene/object bindings.

No invented scene IDs. Native policy blocks are bound to existing paired-harness
recipes, not inferred from the fact that a dataset has textured rooms.
"""
from __future__ import annotations
import copy
from pathlib import Path
from .core import digest, load, rows, save
from .data import validate_inventory
from .runner import validate_tasks


def build(scenes_file, models_file, out, *, stages=None, max_tasks=20000):
    scenes = validate_inventory(rows(scenes_file)); models = load(models_file)
    stages = set(stages or ('video','generator','inpaint','chorus','harmonizer_sequence','official','native'))
    tasks=[]; missing=[]
    for scene in scenes:
        base={'split':scene['split'],'sensor':scene.get('sensor','posed_rgb'),
              'scale_source':scene.get('scale_source'),'camera_source':scene.get('camera_source')}
        sid=digest([scene['dataset'],scene['release'],scene['scene_id']])[:16]
        bindings=scene.get('bindings',{})
        def append(stage, name, config, inputs, suffix):
            t={**base,'id':f'{sid}-{stage}-{name}-{suffix}','kind':stage,
               'params':copy.deepcopy(config['params']),'environment':config['environment'],'inputs':inputs,
               'dataset':scene['dataset'],'scene_id':scene['scene_id'],'group_id':scene['group_id']}
            tasks.append(t); return t
        if 'video' in stages and bindings.get('video') and models.get('video',{}).get('enabled'):
            append('rgb_video_reconstruction','colmap_gs',models['video'],{'video':bindings['video']},'capture')
        if 'generator' in stages:
            for obj in bindings.get('objects',[]):
                for name,c in models.get('generators',{}).items():
                    if not c.get('enabled'): continue
                    images=obj['images'] if c['params']['backend']=='reconviagen' else [obj['images'][0]]
                    inputs={f'image{i}':im for i,im in enumerate(images)}
                    t=append('generator',name,c,inputs,digest(obj['object_id'])[:10]);t['params']['images']=list(inputs)
        if 'inpaint' in stages:
            for case in bindings.get('inpaint_cases',[]):
                for name,c in models.get('inpainters',{}).items():
                    if not c.get('enabled'): continue
                    t=append('inpaint',name,c,{'rgb':case['rgb'],'mask':case['mask']},digest(case['case_id'])[:10])
                    if t['params']['backend']=='gemini':t['params']['scene_upload_allowed']=scene.get('external_api_permitted') is True
        if 'chorus' in stages and bindings.get('gaussians'):
            c=models['chorus']
            if c.get('enabled'): append('chorus','frozen',c,bindings['gaussians'],'encode')
        if 'harmonizer_sequence' in stages:
            for seq in bindings.get('visual_sequences',[]):
                for name,c in models.get('harmonizers',{}).items():
                    if c.get('enabled'):
                        t=append('harmonizer_sequence',name,c,seq['inputs'],digest(seq['sequence_id'])[:10])
                        t['params']['frames']=seq['frames'];t['params']['stream_id']=seq['sequence_id']
        if 'official' in stages:
            for name,c in models.get('official',{}).items():
                if not c.get('enabled'):continue
                if name=='simfoundry' and bindings.get('video'):
                    t=append('simfoundry',name,c,{'video':bindings['video']},'reconstruct')
                    t['params']['scene_upload_allowed']=scene.get('external_api_permitted') is True
                elif name=='polaris' and bindings.get('polaris_bundle'):
                    t=append('polaris_bundle',name,c,{},'validate');t['params'].update(bindings['polaris_bundle'])
                else:missing.append({'scene_id':scene['scene_id'],'component':name,'reason':'missing actual native input/bundle; not a method failure'})
        if 'native' in stages:
            if bindings.get('native_block'):
                c=bindings['native_block'];append('command','native',c,c.get('inputs',{}),'paired')
            else:missing.append({'scene_id':scene['scene_id'],'component':'native','reason':'paired policy recipe not yet admitted'})
        if not bindings:missing.append({'scene_id':scene['scene_id'],'component':'all','reason':'inventory-only; prepare captures first'})
    if len(tasks)>max_tasks:raise ValueError(f'job budget exceeded: {len(tasks)} > {max_tasks}')
    if tasks:validate_tasks(tasks)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    save(out/'tasks.jsonl',tasks,jsonl=True);save(out/'unbound.json',missing)
    save(out/'matrix.json',{'tasks':len(tasks),'unbound':len(missing),'scenes':len(scenes),
         'stage_counts':{s:sum(t['kind']==s for t in tasks) for s in sorted({t['kind'] for t in tasks})},
         'full_data_ready':not missing,'no_cartesian_policy_product':True})
    return {'tasks':len(tasks),'unbound':len(missing)}
