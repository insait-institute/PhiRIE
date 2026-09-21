"""No data/GPU needed: prove omitted failures, mismatched pairs and promotion fail."""
import hashlib
from pathlib import Path
import pytest

from robo.eval import paper_full_appearance as full


def fixture():
    roster = [f's{i:02}' for i in range(50)]
    protocol = dict(scene_ids=roster, planned_scenes=50, planned_objects=1871,
        planned_views=400, views_per_scene=8, population={s: 1 for s in roster},
        evaluation_frames={s: [f'test{i}.JPG' for i in range(8)] for s in roster})
    protocol['population'][roster[0]] = 1822
    coverage = dict(scope=full.SCOPE, planned_scenes=50, planned_objects=1871,
        planned_views_per_method=400, paper_ready=False, conditional_quality=True,
        quality_comparison_paired=True, full_e3_gt_access=False, scene_seals=dict.fromkeys(roster, {}),
        rows=[], common_eligible_views=[])
    manifest = dict(freeze_id='freeze', room_methods={m: dict(records=[]) for m in full.METHODS})
    for i, sid in enumerate(roster):
        status = 'COMPLETE' if i < 36 else 'NO_REMOVAL' if i < 40 else 'BLOCKED_UNFILLABLE_ACCEPTED'
        n = 8 if i < 40 else 0
        for j, method in enumerate(full.METHODS):
            coverage['rows'].append(dict(scene_id=sid, method=method, planned_views=8,
                planned_objects=protocol['population'][sid], accepted_objects=0 if status == 'NO_REMOVAL' else 1,
                background_status=status, unchanged_source_background=status=='NO_REMOVAL',
                source_preparation={}, available_views=8 if j == 0 else n, analysis_views=n,
                missing_views=protocol['evaluation_frames'][sid] if j == 1 and not n else [],
                status='BLOCKED_UNFILLABLE_ACCEPTED' if j == 1 and not n else 'COMPLETE'))
            if n:
                views=[dict(view_id=v,render_path=f'/images/{sid}/{j}/{Path(v).stem}.png',
                            gt_sha256='a'*64) for v in protocol['evaluation_frames'][sid]]
                manifest['room_methods'][method]['records'].append(dict(scene_id=sid,freeze_id='freeze',
                    n_views=n,views=views,optimization_input_frames=['train.JPG'],
                    evaluation_frames=[Path(v['render_path']).name for v in views]))
        if n:
            coverage['common_eligible_views'].extend(dict(scene_id=sid,view_id=v) for v in protocol['evaluation_frames'][sid])
    rows=[]
    for j, method in enumerate(full.METHODS):
        values=dict(psnr=21-j,ssim=.8-j*.01,lpips=.3+j*.02)
        rows.append(dict(method=method,unit='room',n_images=320,n_scenes=40,**values,
            metric_samples=dict.fromkeys(values,320),metric_source_hashes=dict.fromkeys(values,'b'*64),
            metrics={k:dict(value=v,n=320,source_artifact_hash='b'*64,
                bootstrap=dict(unit='scene',n_units=40,samples=2000)) for k,v in values.items()}))
    for i in range(6):rows.insert(1, dict(method='missing'+str(i),n_images=0))
    table=dict(rows=rows,rows_count=8,paper_ready=False,lpips_backend_error=None,
        validation=dict(valid=True,generator_leakage_count=0,registration_evaluation_hash_collision_count=0,
                        lpips_provenance_complete=True))
    return table,coverage,manifest,protocol


def test_negative_means_and_original_null_rows_are_retained():
    table,coverage,manifest,protocol=fixture()
    assert full.validate_rows(table,coverage,manifest,protocol) is table
    assert table['rows'][1]['method']=='missing5'
    assert table['rows'][-1]['psnr'] < table['rows'][0]['psnr']


@pytest.mark.parametrize('damage', ['duplicate_scene','omitted_scene','substituted_scene','denominator',
    'raw_availability','composite_availability','missing_views','common_subset','common_duplicate',
    'unequal_support','reference_mismatch','leakage','noop_relabel','noop_object','different_construction',
    'promote','gt_access','lpips_failure','nonfinite','missing_metric','metric_n','metric_value',
    'metric_bootstrap','metric_source','unmeasured_promotion','missing_method','frame_drift','analysis_drift',
    'registration_leakage','false_validation'])
def test_fail_closed(damage):
    t,c,m,p=fixture()
    if damage=='duplicate_scene':c['rows'][0]=c['rows'][2]
    elif damage=='omitted_scene':c['rows'].pop()
    elif damage=='substituted_scene':c['rows'][0]['scene_id']='unplanned'
    elif damage=='denominator':c['planned_views_per_method']=320
    elif damage=='raw_availability':c['rows'][0]['available_views']=7
    elif damage=='composite_availability':c['rows'][-1]['available_views']=8
    elif damage=='missing_views':c['rows'][-1]['missing_views']=[]
    elif damage=='common_subset':c['common_eligible_views'].pop()
    elif damage=='common_duplicate':c['common_eligible_views'][1]=c['common_eligible_views'][0]
    elif damage=='unequal_support':m['room_methods'][full.METHODS[1]]['records'].pop()
    elif damage=='reference_mismatch':m['room_methods'][full.METHODS[1]]['records'][0]['views'][0]['gt_sha256']='c'*64
    elif damage=='leakage':m['room_methods'][full.METHODS[1]]['records'][0]['optimization_input_frames']=['test0.png']
    elif damage=='noop_relabel':c['rows'][72]['unchanged_source_background']=False
    elif damage=='noop_object':c['rows'][72]['accepted_objects']=1
    elif damage=='different_construction':c['rows'][0]['accepted_objects']=3
    elif damage=='promote':t['paper_ready']=True
    elif damage=='gt_access':c['full_e3_gt_access']=True
    elif damage=='lpips_failure':t['lpips_backend_error']='failed'
    elif damage=='nonfinite':t['rows'][0]['psnr']=float('nan')
    elif damage=='missing_metric':t['rows'][0]['psnr']=None
    elif damage=='metric_n':t['rows'][0]['metric_samples']['lpips']=319
    elif damage=='metric_value':t['rows'][0]['metrics']['psnr']['value']=999
    elif damage=='metric_bootstrap':t['rows'][0]['metrics']['psnr']['bootstrap']['unit']='view'
    elif damage=='metric_source':t['rows'][0]['metrics']['psnr']['source_artifact_hash']='different'
    elif damage=='unmeasured_promotion':t['rows'][1]['psnr']=100
    elif damage=='missing_method':t['rows'].pop()
    elif damage=='frame_drift':m['room_methods'][full.METHODS[0]]['records'][0]['evaluation_frames'][0]='other.png'
    elif damage=='analysis_drift':c['rows'][0]['analysis_views']=7
    elif damage=='registration_leakage':t['validation']['registration_evaluation_hash_collision_count']=1
    elif damage=='false_validation':t['validation']['valid']=False
    with pytest.raises(ValueError):full.validate_rows(t,c,m,p)


def test_dirty_producer_cannot_publish(monkeypatch,tmp_path):
    monkeypatch.setattr(full,'git_snapshot',lambda _:dict(commit=full.PRODUCER,dirty=True))
    with pytest.raises(ValueError,match='producer source'):
        full.validate_source({'producer':dict(path=str(tmp_path),commit=full.PRODUCER)},None,None,None,None)


def test_original_scope_not_a_condition_on_quality():
    t,c,m,p=fixture()
    # A future positive numerical result would remain admissible: this consumer
    # must not gate source validity on attaining a desired sign.
    t['rows'][-1]['psnr']=22
    t['rows'][-1]['metrics']['psnr']['value']=22
    assert full.validate_rows(t,c,m,p) is t


def test_pinned_metadata_hash_and_alias_cannot_change(tmp_path):
    path=tmp_path/'seal.json';path.write_text('{}')
    ref=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert full.checked(ref)==path
    alias=tmp_path/'alias.json';alias.symlink_to(path)
    with pytest.raises(ValueError,match='alias'):full.checked({**ref,'path':str(alias)})
    path.write_text('{"members":{}}')
    with pytest.raises(ValueError,match='hash'):full.checked(ref)
