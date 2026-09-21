"""Actual NVIDIA model sequence inference plus optional 3D-state correction.

Use one task per episode/camera. Official and constrained treatments are separate
jobs with identical raw frames. No identity backend is allowed here.
"""
from pathlib import Path
import time
import numpy as np
from PIL import Image
from .core import load, local_model, receipt, save


def run(config, inputs, out):
    from .models import source
    from .visual import run as state_correct
    from integrations.harmonizer.server import NvidiaPix2PixBackend
    source(config); weights=local_model(config)
    # The existing NVIDIA wrapper imports pix2pix_turbo_harmonizer from src/.
    source_dir=Path(config['source_root'])/config.get('module_subdir','src')
    load_start=time.perf_counter()
    backend=NvidiaPix2PixBackend(str(source_dir),weights,temporal=config.get('temporal',True),
        resolution=int(config.get('resolution',1024)),timestep=int(config.get('timestep',250)))
    load_s=time.perf_counter()-load_start
    stream=config['stream_id'];backend.reset_stream(stream);out=Path(out);records=[];previous=None
    for i,frame in enumerate(config['frames']):
        started=time.perf_counter()
        state=load(inputs[frame['state']])
        if state['stream_id']!=stream or state['frame_index']!=i:
            raise ValueError('sequence must be contiguous, single episode and single camera')
        raw=np.asarray(Image.open(inputs[frame['rgb']]).convert('RGB'))
        target=out/f'{i:06d}';target.mkdir()
        result=backend.enhance(raw,stream)
        Image.fromarray(result).save(target/'enhanced.png')
        final=target/'enhanced.png'
        if config.get('state_conditioned'):
            aligned={'rgb':inputs[frame['rgb']],'enhanced':str(final),
                     'state':inputs[frame['state']],'buffers':inputs[frame['buffers']]}
            if previous:
                aligned.update(previous_state=previous['state'],previous_buffers=previous['buffers'],
                               previous_rgb=previous['rgb'])
            correction=target/'state';correction.mkdir()
            final=state_correct(config.get('correction',{}),aligned,correction)['rgb']
            # History must condition on the actually emitted, protected image.
            backend.history[stream][-1]=np.asarray(Image.open(final).convert('RGB'))
        backend.history[stream]=backend.history[stream][-backend.min_history:]
        previous={'state':inputs[frame['state']],'buffers':inputs[frame['buffers']],'rgb':str(final)}
        records.append({'frame':i,'state':receipt(inputs[frame['state']]),'rgb':receipt(final),
                        'end_to_end_ms':1000*(time.perf_counter()-started)})
    save(out/'sequence.json',{'stream_id':stream,'frames':records,'state_conditioned':bool(config.get('state_conditioned')),
        'checkpoint_manifest':receipt(config['checkpoint_manifest']),'official_model':True,
        'model_load_s':load_s,'latency_p95_ms':float(np.percentile([r['end_to_end_ms'] for r in records],95)) if records else None,
        'closed_loop_policy':'NOT_RUN','history':'causal emitted outputs; finite window'})
    return {'sequence':out/'sequence.json'}
