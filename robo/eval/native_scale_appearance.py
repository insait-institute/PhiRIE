"""Heldout native full-frame appearance receipts; no object-only/GS inference."""
from collections import defaultdict
from pathlib import Path
import math
from robo.eval.fidelity_metrics import __file__ as metric_file


def appearance_table(units,paths,methods,src):
    from robo.eval.native_scale_tables import _sha
    if not methods or len(set(methods))!=len(methods):raise ValueError('declare distinct appearance methods')
    populations=defaultdict(set)
    for u in units:
        if u['canonical_instance_id'] is not None:
            populations[(u['cohort_id'],u['scope'],u['sensor_regime'],u['renderer'],u['split'])].add(u['canonical_instance_id'])
    found={};expected_views={};backends=set();reference_hashes={};render_protocols={}
    for path in paths:
        metric=src.read(path);render=src.read(metric['source']['path'],metric['source']['sha256'])
        geometry=src.read(render['geometry_receipt']['path'],render['geometry_receipt']['sha256'])
        binding=src.read(render['binding']['path'],render['binding']['sha256'])
        canonical=src.read(binding['canonical_manifest'],geometry['canonical_manifest_sha256'])
        bundle=Path(binding['bundle_dir'])
        src.bind(bundle/'scene.xml',canonical['identity']['scene_xml_sha256'])
        src.bind(bundle/'canonical_state.json',canonical['canonical_state_file_sha256'])
        for file,digest in canonical['asset_closure'].items():src.bind(file,digest)
        if geometry['binding_sha256']!=render['binding']['sha256'] or geometry['canonical_instance_id']!=binding['canonical_instance_id']:
            raise ValueError('appearance geometry canonical binding differs')
        for k in ('canonical_instance_id','cohort_id','scope','sensor_regime','renderer','tier','planned_views_per_method','support','intervention'):
            if metric[k]!=render[k]:raise ValueError('appearance metric/render identity differs')
        protocol=render.get('render_protocol','cold_import_v1')
        if protocol not in ('cold_import_v1','capture_train_prelude_v1') or metric.get('render_protocol','cold_import_v1')!=protocol:
            raise ValueError('appearance render protocol differs')
        if protocol=='capture_train_prelude_v1':
            prelude=render['prelude_source'];plan=src.read(prelude['capture_plan_path'],prelude['capture_plan_sha256'])
            prefix=[]
            for frame in plan['frames']:
                if frame['split']=='test':break
                if frame['split']!='train':raise ValueError('appearance prelude is not TRAIN-only')
                prefix.append(frame['frame_id'])
            if not prefix or prefix!=prelude['frame_ids'] or prelude['evaluation_rgb_accessed_for_prelude'] is not False:
                raise ValueError('appearance prelude declaration differs')
            expected={(m,f) for m in ['REF_IMPORT_CONTROL',*[r['method'] for r in geometry['rows']]] for f in prefix}
            checks=render['prelude_state_checks']
            if len(checks)!=len(expected) or {(r['method'],r['frame_id']) for r in checks}!=expected or not all(r['physical_control_state_exact'] for r in checks):
                raise ValueError('appearance prelude state evidence incomplete')
        if render.get('metadata_import_overrides'):
            from robo.roundtrip.fidelity_native import metadata_render_methods
            proofs={}
            for override in render['metadata_import_overrides']:
                r=override['correction_receipt'];proofs[r['path']]=src.bind(r['path'],r['sha256'])
            _,verified=metadata_render_methods(geometry['rows'],geometry['frozen_estimated_artifacts'],list(proofs.values()))
            if verified!=render['metadata_import_overrides']:raise ValueError('appearance metadata import provenance differs')
        if render['source_code']['dirty'] or render['evaluator_alignment']!='NONE' or render['exposure_matching']!='NONE':
            raise ValueError('appearance requires frozen direct rendering')
        if metric['metric_implementation_sha256']!=_sha(metric_file):raise ValueError('appearance metric implementation differs')
        src.bind(metric_file,metric['metric_implementation_sha256'])
        planned=metric['planned_views_per_method'];identities=render['native_import_render_identity']
        if type(planned) is not int or planned<1 or len(identities)!=planned or not all(r['byte_exact'] for r in identities):
            raise ValueError('native heldout render identity incomplete')
        frame_ids={r['frame_id'] for r in identities}
        if len(frame_ids)!=planned:raise ValueError('duplicate native heldout control view')
        provenance=metric['lpips_provenance']
        if provenance['network']!='alex' or not provenance['backend_class']:raise ValueError('unbound LPIPS backend')
        hashes=[]
        for field in ('backbone_checkpoint','linear_checkpoint'):
            receipt=provenance[field]
            if not receipt or not receipt.get('sha256'):raise ValueError('unbound LPIPS checkpoint')
            hashes.append(receipt['sha256'])
        backends.add(tuple([provenance['backend_class'],provenance['package_version'],*hashes]))
        iid=metric['canonical_instance_id'];block=(metric['cohort_id'],metric['scope'],metric['sensor_regime'],metric['renderer'],{'DEV':'development','TEST':'test'}.get(metric['tier'],metric['tier']))
        if block in render_protocols and render_protocols[block]!=protocol:raise ValueError('mixed cold/warm appearance protocols within comparison')
        render_protocols[block]=protocol
        if iid not in populations.get(block,set()):raise ValueError('unplanned appearance instance/block')
        if block in expected_views and expected_views[block]!=planned:raise ValueError('unequal planned appearance view counts')
        expected_views[block]=planned
        renders={(r['method'],r['frame_id']):r for r in render['records']}
        if len(renders)!=len(render['records']):raise ValueError('duplicate rendered view')
        required={(r['method'],frame) for r in geometry['rows'] for frame in frame_ids}
        if set(renders)!=required:raise ValueError('rendered method/view population incomplete')
        matched=set()
        for row in metric['rows']:
            key=(row['method'],row['frame_id'])
            if key in matched:raise ValueError('duplicate appearance metric view')
            matched.add(key)
            if row['method'] not in methods or row['frame_id'] not in frame_ids or key not in renders:
                raise ValueError('unplanned appearance method/view')
            original=renders[key]
            if any(row[k]!=original[k] for k in original):raise ValueError('render/metric image roster differs')
            if row['width']!=1280 or row['height']!=720:raise ValueError('heldout image resolution differs')
            for field,digest in (('pred_rgb','pred_sha256'),('reference_rgb','reference_sha256')):src.bind(row[field],row[digest])
            refkey=(*block,iid,row['frame_id'])
            if refkey in reference_hashes and reference_hashes[refkey]!=row['reference_sha256']:raise ValueError('method reference views differ')
            reference_hashes[refkey]=row['reference_sha256']
            for name in ('ssim','lpips'):
                if isinstance(row[name],bool) or not isinstance(row[name],(int,float)) or not math.isfinite(row[name]):raise ValueError('nonfinite appearance metric')
            psnr_status=row.get('psnr_status','FINITE')
            if psnr_status=='POSITIVE_INFINITY':
                if row['psnr'] is not None or row.get('exact_rgb_match') is not True:
                    raise ValueError('infinite PSNR requires exact RGB receipt')
                import numpy as np
                from PIL import Image
                with Image.open(row['pred_rgb']) as pred, Image.open(row['reference_rgb']) as ref:
                    a,b=np.asarray(pred.convert('RGB')),np.asarray(ref.convert('RGB'))
                if a.shape!=(720,1280,3) or not np.array_equal(a,b):
                    raise ValueError('infinite PSNR requires equal decoded RGB pixels')
            elif psnr_status!='FINITE' or isinstance(row['psnr'],bool) or not isinstance(row['psnr'],(int,float)) or not math.isfinite(row['psnr']):
                raise ValueError('invalid PSNR status/value')
            if not -1<=row['ssim']<=1 or row['lpips']<0:raise ValueError('invalid appearance metric range')
            entry=(*block,iid,*key)
            if entry in found:raise ValueError('duplicate appearance instance/method/view')
            found[entry]=row
        if matched!=set(renders):raise ValueError('missing rendered appearance metric row')
        for row in geometry['rows']:
            for name,digest in geometry['frozen_estimated_artifacts'][row['method']].items():src.bind(Path(row['object_dir'])/name,digest)
    if len(backends)>1:raise ValueError('mixed LPIPS checkpoint/backend identity')
    table=[]
    for block,ids in sorted(populations.items()):
        nviews=expected_views.get(block)
        support={m:{iid for iid in ids if nviews and sum((*block,iid,m,f) in found for f in {k[-1] for k in found if k[:5]==block})==nviews} for m in methods}
        common=set.intersection(*support.values()) if support else set()
        for iid in common:
            sets=[{k[-1] for k in found if k[:7]==(*block,iid,m)} for m in methods]
            if any(frames!=sets[0] for frames in sets[1:]):raise ValueError('common appearance frame sets differ')
        for method in methods:
            values=[]
            for iid in sorted(common):
                views=[r for k,r in found.items() if k[:7]==(*block,iid,method)]
                values.append(dict(psnr=math.inf if any(r.get('psnr_status')=='POSITIVE_INFINITY' for r in views) else sum(r['psnr'] for r in views)/len(views), perfect_views=sum(r.get('psnr_status')=='POSITIVE_INFINITY' for r in views), **{m:sum(r[m] for r in views)/len(views) for m in ('ssim','lpips')}))
            table.append(dict(zip(('cohort_id','scope','sensor_regime','renderer','split'),block),method=method,
                render_protocol=render_protocols.get(block),scenes_available=len(support[method]),scenes_planned=len(ids),common_scenes=len(common),
                views_available=len(support[method])*nviews if nviews else 0,views_planned=len(ids)*nviews if nviews else None,
                common_views=len(common)*nviews if nviews else 0,matched_instance_ids=sorted(common),
                psnr=None if not values or any(math.isinf(v['psnr']) for v in values) else sum(v['psnr'] for v in values)/len(values),
                psnr_status='UNMEASURED' if not values else ('POSITIVE_INFINITY' if any(math.isinf(v['psnr']) for v in values) else 'FINITE'),
                perfect_views=sum(v['perfect_views'] for v in values),
                **{m:sum(v[m] for v in values)/len(values) if values else None for m in ('ssim','lpips')},
                support='full heldout frames; common instance/view support; conditional instance macro mean',
                intervention='uniform reconstructed material; native context retained',object_only_quality=None,gaussian_appearance=None))
    return table
