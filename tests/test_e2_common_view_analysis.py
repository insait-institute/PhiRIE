import copy,json
from pathlib import Path
import pytest,yaml
from agents.edit.inpaint_masks import _identity
from run.icra2027 import e2_public_factorized as p,e2_raw_room as raw
from tests.test_e2_public_factorized import original
from tests.test_e2_raw_room import common_mock


@pytest.fixture
def pair(tmp_path):
    results={};seals={};identities={}
    for scene in p.SCENES:
        origin=original(tmp_path/scene,scene);plan=origin['plan'];common,_=common_mock(plan)
        rawviews=raw.export_raw(plan,tmp_path/scene/'raw',common)
        auto=(raw.export_raw(plan,tmp_path/scene/'auto',common,gaussian=object(),method_slug='factorized_auto_discovery')
            if scene==p.SCENES[0] else [])
        results[scene]=dict(raw_source=dict(origin,bundle_identity={'sha256':'a'*64}),source_fill={'sha256':'b'*64},raw_views=rawviews,
            composite_views=auto,status='COMPLETE' if auto else 'BLOCKED_UNFILLABLE_ACCEPTED',accepted_objects=1 if auto else 4)
        result=tmp_path/scene/'result.json';result.write_text(json.dumps(results[scene]));identities[scene]=_identity(result)
        seal=tmp_path/scene/'seal.json';seal.write_text('{}');seals[scene]=_identity(seal)
    module=tmp_path/'metric.py';module.write_text('frozen implementation')
    c=dict(freeze_id='new',bootstrap_samples=2000,bootstrap_seed=42,metric_source={'module':_identity(module),'code_commit':'a'*40},render_source={'scene_seals':seals})
    r=results[p.SCENES[0]]
    protocol=dict(schema_version=1,scope='predeclared_single_scene_common_view_diagnostic',freeze_id='new',metric_outcomes_inspected=False,
        original_planned_scenes=list(p.SCENES),original_planned_objects=17,original_planned_views_per_method=16,paper_ready=False,
        metric_module=c['metric_source']['module'],metric_source_commit='a'*40,methods=[raw.METHOD,p.AUTO],metrics=['PSNR','SSIM','LPIPS'],
        bootstrap_samples=2000,bootstrap_seed=42,source_scene_seals=seals,source_scene_results=identities,
        common_views={s:[v['view_id'] for v in x['composite_views']] for s,x in results.items()},
        qualitative_view=dict(scene_id=p.SCENES[0],view_id=r['composite_views'][0]['view_id'],raw_path=r['raw_views'][0]['render_path'],
            composite_path=r['composite_views'][0]['render_path'],reference_path=r['composite_views'][0]['gt_path']))
    path=tmp_path/'protocol.yaml';path.write_text(yaml.safe_dump(protocol));c['paired_protocol']=_identity(path)
    return c,results,protocol,path


def test_paired_manifest_uses_common8_but_keeps_original16_coverage(pair):
    c,results,_,_=pair
    manifest,coverage=p.make_manifest(c,{'code':{'commit':'b'*40}},results)
    for method in (raw.METHOD,p.AUTO):
        records=manifest['room_methods'][method]['records']
        assert len(records)==1 and records[0]['scene_id']==p.SCENES[0] and records[0]['n_views']==8
        assert sum(r['planned_views'] for r in coverage if r['method']==method)==16
    assert sum(r['available_views'] for r in coverage if r['method']==raw.METHOD)==16
    assert sum(r['analysis_views'] for r in coverage if r['method']==raw.METHOD)==8
    assert manifest['provenance']['inference_scope']=='one-scene paired descriptive diagnostic'


@pytest.mark.parametrize('kind',['subset','qualitative','seal','result','method','post_outcome'])
def test_predeclared_support_and_sources_cannot_change(pair,kind):
    c,results,protocol,path=pair
    if kind=='subset':protocol['common_views'][p.SCENES[0]].pop()
    elif kind=='qualitative':protocol['qualitative_view']['view_id']='favorable_later.jpg'
    elif kind=='seal':protocol['source_scene_seals']=copy.deepcopy(protocol['source_scene_seals']);protocol['source_scene_seals'][p.SCENES[0]]['sha256']='changed'
    elif kind=='result':Path(protocol['source_scene_results'][p.SCENES[0]]['path']).write_text('{}')
    elif kind=='method':protocol['methods']=[p.AUTO]
    else:protocol['metric_outcomes_inspected']=True
    path.write_text(yaml.safe_dump(protocol));c['paired_protocol']=_identity(path)
    with pytest.raises(ValueError):p.make_manifest(c,{'code':{'commit':'b'*40}},results)
