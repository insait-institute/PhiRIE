import json
import pytest
from pathlib import Path
from robo.eval.native_scale_evidence import Sources
from robo.eval.native_scale_appearance import appearance_table
from robo.eval.native_scale_tables import _sha
from robo.eval.fidelity_metrics import __file__ as metric_file


def put(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data));return path


@pytest.fixture
def appearance(tmp_path):
    root=tmp_path;state=put(root/'bundle/canonical_state.json',{});scene=put(root/'bundle/scene.xml',{})
    canonical=put(root/'canonical.json',dict(identity={'scene_xml_sha256':_sha(scene)},canonical_state_file_sha256=_sha(state),asset_closure={}))
    binding=put(root/'binding.json',dict(canonical_instance_id='i',canonical_manifest=str(canonical),bundle_dir=str(state.parent)))
    geometry=put(root/'geometry.json',dict(binding_sha256=_sha(binding),canonical_manifest_sha256=_sha(canonical),canonical_instance_id='i',rows=[{'method':'B0_FIXED_NATIVE','object_dir':str(root)}],frozen_estimated_artifacts={'B0_FIXED_NATIVE':{}}))
    records=[]
    for f in ('f0','f1'):
        image=put(root/(f+'.png'),[f])
        records.append(dict(method='B0_FIXED_NATIVE',frame_id=f,pred_rgb=str(image),pred_sha256=_sha(image),reference_rgb=str(image),reference_sha256=_sha(image),width=1280,height=720))
    render=dict(schema_version=2,canonical_instance_id='i',cohort_id='c',scope='L0_target_only',sensor_regime='ideal',renderer='native',tier='DEV',planned_views_per_method=2,support='full-frame',intervention='uniform',source_code={'dirty':False},geometry_receipt={'path':str(geometry),'sha256':_sha(geometry)},binding={'path':str(binding),'sha256':_sha(binding)},evaluator_alignment='NONE',exposure_matching='NONE',native_import_render_identity=[dict(frame_id=f,byte_exact=True) for f in ('f0','f1')],records=records)
    rp=put(root/'render.json',render)
    metric={k:render[k] for k in ('schema_version','canonical_instance_id','cohort_id','scope','sensor_regime','renderer','tier','support','intervention','planned_views_per_method')}
    metric.update(source={'path':str(rp),'sha256':_sha(rp)},rows=[dict(r,psnr=20.,ssim=.8,lpips=.2) for r in records],metric_implementation_sha256=_sha(metric_file),lpips_provenance=dict(network='alex',backend_class='lpips.LPIPS',package_version='0.1.4',backbone_checkpoint={'sha256':'a'*64},linear_checkpoint={'sha256':'b'*64}))
    mp=put(root/'metrics.json',metric)
    units=[dict(canonical_instance_id=i,cohort_id='c',scope='L0_target_only',sensor_regime='ideal',renderer='native',split='development') for i in ('i','missing')]
    return mp,units


def test_full_frame_common_support_retains_failed_build_denominator(appearance):
    path,units=appearance
    row=appearance_table(units,[path],['B0_FIXED_NATIVE'],Sources())[0]
    assert row['scenes_planned']==2 and row['common_scenes']==1 and row['views_planned']==4
    assert row['psnr']==20 and row['object_only_quality'] is None and row['gaussian_appearance'] is None
    missing=appearance_table(units,[path],['B0_FIXED_NATIVE','B3_AGENT_NATIVE'],Sources())
    assert all(r['psnr'] is None for r in missing)


@pytest.mark.parametrize('change',['duplicate','missing','image','metric','identity','absolute'])
def test_appearance_rejects_tampering_and_missing_views(appearance,change):
    path,units=appearance;d=json.loads(path.read_text())
    if change=='duplicate':d['rows'].append(d['rows'][0])
    if change=='missing':d['rows'].pop()
    if change=='image':Path(d['rows'][0]['pred_rgb']).write_text('changed')
    if change=='metric':d['metric_implementation_sha256']='0'*64
    if change in ('identity','absolute'):
        p=Path(d['source']['path']);render=json.loads(p.read_text())
        if change=='identity':render['native_import_render_identity'][0]['byte_exact']=False
        else:render['evaluator_alignment']='GT_ICP'
        put(p,render);d['source']['sha256']=_sha(p)
    put(path,d)
    with pytest.raises(ValueError):appearance_table(units,[path],['B0_FIXED_NATIVE'],Sources())


def perfect_receipt(path, equal=True):
    from PIL import Image
    d=json.loads(path.read_text());rp=Path(d['source']['path']);r=json.loads(rp.read_text())
    row=d['rows'][0]
    pred=Path(row['pred_rgb']);ref=pred.with_name('reference.png')
    Image.new('RGB',(1280,720),(10,20,30)).save(pred)
    Image.new('RGB',(1280,720),(10,20,30) if equal else (10,20,31)).save(ref)
    for target in (row,r['records'][0]):
        target.update(reference_rgb=str(ref),reference_sha256=_sha(ref),pred_sha256=_sha(pred))
    row.update(psnr=None,psnr_status='POSITIVE_INFINITY',exact_rgb_match=True)
    put(rp,r);d['source']['sha256']=_sha(rp);put(path,d)
    return d


def test_perfect_view_preserves_extended_mean_and_denominators(appearance):
    p,u=appearance;perfect_receipt(p)
    row=appearance_table(u,[p],['B0_FIXED_NATIVE'],Sources())[0]
    assert row['psnr'] is None and row['psnr_status']=='POSITIVE_INFINITY'
    assert row['perfect_views']==1 and row['common_views']==2 and row['scenes_planned']==2
    assert row['ssim']==.8 and row['lpips']==.2
    json.dumps(row,allow_nan=False)


@pytest.mark.parametrize('bad',['pixels','flag','finite_value','unencoded_infinity'])
def test_infinite_psnr_requires_actual_pixel_equality(appearance,bad):
    p,u=appearance;d=perfect_receipt(p,equal=bad!='pixels')
    if bad=='flag':d['rows'][0]['exact_rgb_match']=False
    if bad=='finite_value':d['rows'][0]['psnr']=100.
    if bad=='unencoded_infinity':d['rows'][0].update(psnr=float('inf'),psnr_status='FINITE')
    put(p,d)
    with pytest.raises(ValueError):appearance_table(u,[p],['B0_FIXED_NATIVE'],Sources())


def warm_receipt(path):
    metric=json.loads(path.read_text());render=json.loads(Path(metric['source']['path']).read_text());root=path.parent
    plan=put(root/'capture_plan.json',dict(frames=[dict(frame_id='train0',split='train'),dict(frame_id='f0',split='test'),dict(frame_id='f1',split='test')]))
    render.update(render_protocol='capture_train_prelude_v1',prelude_source=dict(capture_plan_path=str(plan),capture_plan_sha256=_sha(plan),frame_ids=['train0'],evaluation_rgb_accessed_for_prelude=False),prelude_state_checks=[dict(method=m,frame_id='train0',physical_control_state_exact=True) for m in ['REF_IMPORT_CONTROL','B0_FIXED_NATIVE']])
    rp=put(root/'warm_render.json',render);metric.update(render_protocol='capture_train_prelude_v1',source=dict(path=str(rp),sha256=_sha(rp)))
    return put(root/'warm_metrics.json',metric)


def test_cold_warm_protocols_cannot_mix_within_comparison(appearance):
    p,units=appearance;warm=warm_receipt(p)
    row=appearance_table(units,[warm],['B0_FIXED_NATIVE'],Sources())[0]
    assert row['render_protocol']=='capture_train_prelude_v1'
    with pytest.raises(ValueError,match='mixed cold/warm'):appearance_table(units,[p,warm],['B0_FIXED_NATIVE'],Sources())


@pytest.mark.parametrize('bad',['state','missing','metric_protocol'])
def test_warm_protocol_requires_complete_preserved_state_evidence(appearance,bad):
    p,units=appearance;warm=warm_receipt(p);metric=json.loads(warm.read_text());rp=Path(metric['source']['path']);render=json.loads(rp.read_text())
    if bad=='state':render['prelude_state_checks'][0]['physical_control_state_exact']=False
    elif bad=='missing':render['prelude_state_checks'].pop()
    else:metric['render_protocol']='cold_import_v1'
    put(rp,render);metric['source']['sha256']=_sha(rp);put(warm,metric)
    with pytest.raises(ValueError):appearance_table(units,[warm],['B0_FIXED_NATIVE'],Sources())
