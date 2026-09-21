"""Replay exact CPU-selected RVG masks under an explicit new config/E0 contract.

No new visibility threshold, camera selection, crop, or generator input is made.
Old source/config/E0 and actual view receipts must all authenticate before use.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path

import numpy as np
import yaml

from run.icra2027.e3_auto_discovery_pilot import PilotError, sha
from run.icra2027.e3_fresh_generation_contract import checked_identity, checked_contract, FRESH

MODE = 'exact_frozen_cpu_views_v1'


def authenticate_source(c, boundary):
    from run.icra2027 import e3_rvg_generation_pilot as rvg
    spec=c['frozen_view_replay']
    if set(spec)!={'mode','producer_commit','source_config','source_contract','view_manifest','views_receipt'} or spec['mode']!=MODE:
        raise PilotError('RVG frozen replay recipe differs')
    source_config=checked_identity(spec['source_config'])
    original=yaml.safe_load(source_config.read_text())
    if 'frozen_view_replay' in original:
        raise PilotError('RVG frozen replay requires original CPU collection, not a replay chain')
    contract=checked_contract(spec['source_contract'],spec['producer_commit'],sha(source_config),original['contract_resource_id'])
    manifest_path=checked_identity(spec['view_manifest']);receipt_path=checked_identity(spec['views_receipt'])
    source_root=Path(spec['source_contract']['path']).parent.parent
    source_out=rvg.output_directory(original,source_root)
    if (manifest_path!=source_out/'view_manifest.json' or receipt_path!=source_out/'views_receipt.json'
            or original['freeze_id']!=contract['freeze_id']):
        raise PilotError('RVG original CPU receipt directory differs')
    for key in ('scene_id','source_pilot','source_discovery_hashes','models','seed','runtime_sha256'):
        if c.get(key)!=original.get(key):
            raise PilotError('RVG frozen replay changes construction/model recipe: '+key)
    if c.get('source_gaussian_training_provenance')!=FRESH:
        raise PilotError('RVG frozen replay requires official TRAIN-only provenance')
    source_manifest,source_boundary=rvg.read_manifest(original,source_out)
    if (source_manifest['code_commit']!=spec['producer_commit']
            or source_manifest['config_sha256']!=sha(source_config)):
        raise PilotError('RVG original CPU source/config identity differs')
    if source_boundary!=boundary:
        raise PilotError('RVG frozen replay changes training images or cameras')
    record=rvg.validate_view_receipt(original,source_out,source_manifest,boundary)
    return source_out,record


def collect(c,out,boundary,*,write):
    """Materialize or read byte-identical source masks, never raycast them again."""
    from run.icra2027 import e3_rvg_generation_pilot as rvg
    from agents.core import common as C
    from agents.discover.training_views import select_training_views
    source_out,source_record=authenticate_source(c,boundary)
    if Path(out)==source_out:
        raise PilotError('RVG frozen replay requires a new output freeze')
    K,W,H,_=C.load_intrinsics()
    frames,training=select_training_views(sorted(C.load_colmap_w2c().items()),
        out/'inputs/data'/rvg.scene_id(c)/'dslr/train_test_lists.json',1,
        boundary['boundary'].get('max_train_frames',48))
    if training['training_frames']!=source_record['training_frames']:
        raise PilotError('RVG frozen replay training population differs')
    objects=json.loads((out/'construction/objects/objects.json').read_text())
    instances={x['object_id']:x for x in C.load_instances()}
    record=copy.deepcopy(source_record);view_data={}
    if [(o['index'],o['gt_object_id']) for o in objects]!=[(r['object_index'],r['automatic_instance_id']) for r in record['rows']]:
        raise PilotError('RVG frozen replay object population differs')
    for obj,row in zip(objects,record['rows']):
        views=[]
        for view in row['views']:
            path=Path(view['mask_path'])
            if sha(path)!=view['mask_sha256']:raise PilotError('RVG source mask changed')
            mask=np.load(path,allow_pickle=False)
            views.append((view['frame'],tuple(view['bbox']),mask))
        actual=rvg.freeze_object_views(out,obj['index'],views,boundary,W,H,C.IMAGES_DIR,write=write)
        expected=copy.deepcopy(row['views'])
        for k,view in enumerate(expected):
            view['mask_path']=str(out/'frozen_views'/f'obj_{obj["index"]:02d}'/f'mask_{k:02d}.npy')
        if actual!=expected:
            raise PilotError('RVG replay changed frozen RGB/mask bytes or crop')
        row['views']=actual
        view_data[obj['index']]=(obj,instances[obj['gt_object_id']],views)
    if write:rvg.write_new(out/'view_manifest.json',record)
    elif record!=json.loads((out/'view_manifest.json').read_text()):
        raise PilotError('RVG replay manifest differs from config-bound CPU source')
    # run_rvg uses only precollected_views; no raycasting scene is needed.
    return (K,W,H,frames,None,None,None),view_data,record
