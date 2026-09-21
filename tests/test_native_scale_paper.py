import pytest
from robo.eval.native_scale_paper import render_section


def rows():
    base=dict(cohort_id='c',scope='L0_target_only',sensor_regime='ideal',renderer='native',execution_protocol='primary_native',split='DEV',policy_id='p',instances=2,layouts=1,planned=10,executed=4,measured=8,successes_observed=3,complete=False,success_per_executed=.75)
    return [dict(base,method='REF_NATIVE',executed=10,measured=10,successes_observed=7,complete=True,success_per_executed=.7),dict(base,method='B0_FIXED_NATIVE')]


def test_generated_paper_preserves_partial_and_scope(tmp_path):
    render_section(tmp_path,{'T2_manipulation':rows()})
    text=(tmp_path/'native_dev_results.tex').read_text()
    assert '7/10' in text and '4/10 & --' in text
    section=(tmp_path/'native_dev_section.tex').read_text()
    assert '18/20' in section and 'DEV' in section and 'privileged native geometry' in section
    assert 'not independent physics validation' in section


def test_paper_rejects_unapproved_scope_or_mixed_block(tmp_path):
    r=rows();r[1]['scope']='L1_target_destination'
    with pytest.raises(ValueError):render_section(tmp_path,{'T2_manipulation':r})


def test_generated_paper_keeps_positive_infinite_psnr(tmp_path):
    a=dict(method='B0_FIXED_NATIVE',common_scenes=1,scenes_planned=2,psnr=None,psnr_status='POSITIVE_INFINITY',ssim=1.,lpips=0.)
    render_section(tmp_path,{'T2_manipulation':rows(),'T1a_appearance':[a]})
    assert r'$\infty$' in (tmp_path/'native_dev_appearance.tex').read_text()
    assert 'includes every declared view' in (tmp_path/'native_dev_section.tex').read_text()


def test_generated_conditional_gain_retains_coverage_penalty(tmp_path):
    ref=rows()[0]
    a=dict(ref,method='B3_AGENT_NATIVE',executed=8,success_per_executed=.25,success_per_planned=.2)
    b=dict(ref,method='B4_ROOM_REPAIR_NATIVE',executed=2,success_per_executed=.5,success_per_planned=.1)
    render_section(tmp_path,{'T2_manipulation':[ref,a,b]})
    assert 'coverage penalty' in (tmp_path/'native_dev_section.tex').read_text()


def test_standard_t1a_renderer_retains_infinity(tmp_path):
    from robo.eval.native_scale_tables import _latex
    path=tmp_path/'T1a_appearance.tex'
    _latex(path,[dict(method='B0',psnr=None,psnr_status='POSITIVE_INFINITY')])
    assert r'$\infty$' in path.read_text()


def test_test_output_separate_partial_rates_disabled_and_no_dev_overwrite(tmp_path):
    import json
    r=rows()
    for row in r:row['split']='TEST'
    (tmp_path/'native_dev_section.tex').write_text('historical DEV source')
    outputs=render_section(tmp_path,{'T2_manipulation':r,'test_conclusion_ledger':[
        dict(claim_id='test_pair_fabricated',gate='PASS',cohort_complete=True,enabled_sentence='MUST NOT APPEAR')]})
    assert outputs['native_test_section.tex']=='paper_sections/06_native_test.tex'
    assert (tmp_path/'native_dev_section.tex').read_text()=='historical DEV source'
    section=(tmp_path/'native_test_section.tex').read_text()
    assert 'Incomplete TEST snapshot' in section and 'MUST NOT APPEAR' not in section
    assert 'noninferiority or preservation' in section and 'not pooled' in section
    assert '4/10 & --' in (tmp_path/'native_test_results.tex').read_text()
    gates=json.loads((tmp_path/'native_paper_claim_gates.json').read_text())
    assert all(r['enabled_sentence']=='' and r['scientific_gate']=='NOT_RUN' for r in gates)
    assert {r['claim_id'] for r in gates}>={'preservation','full_l2','full_room','gs_occlusion','feedback'}


def test_complete_test_consumes_existing_paired_sentence_only(tmp_path):
    r=rows()
    for row in r:row.update(split='TEST',complete=True,measured=row['planned'])
    sentence='The complete paired TEST B3_AGENT_NATIVE minus B0_FIXED_NATIVE difference is -1.23 percentage points; this does not establish preservation.'
    render_section(tmp_path,{'T2_manipulation':r,'test_conclusion_ledger':[
        dict(claim_id='test_pair_B3_B0',gate='PASS',cohort_complete=True,enabled_sentence=sentence),
        dict(claim_id='test_pair_missing',gate='NOT_RUN',cohort_complete=False,enabled_sentence='MUST NOT APPEAR')]})
    text=(tmp_path/'native_test_section.tex').read_text()
    assert '-1.23' in text and 'Incomplete TEST snapshot' not in text and 'MUST NOT APPEAR' not in text
    assert 'B3: agent minus B0: fixed' in text


def test_mixed_dev_test_rejected(tmp_path):
    r=rows();r[1]['split']='TEST'
    with pytest.raises(ValueError,match='one declared block'):render_section(tmp_path,{'T2_manipulation':r})


def scope_sources(complete=True):
    base=dict(scope_run_id='scope_phase',cohort_id='test',policy_id='frozen',sensor_regime='ideal',
        renderer='native',execution_protocol='primary_native',split='test',instances=24,layouts=4,
        planned=240,executed=10,measured=240 if complete else 200,unmeasured=0 if complete else 40,
        complete=complete,successes_observed=2,success_per_executed=.2,
        inherited_target_nonexecution_observed=90,additional_scope_nonexecution_observed=20)
    methods=('B3_AGENT_NATIVE','B4_ROOM_REPAIR_NATIVE')
    scopes=('L0_target_only','L1_target_destination')
    return dict(T4_scope=[dict(base,method=m,scope=s) for m in methods for s in scopes],
        T4_native_controls=[dict(base,method='REF_NATIVE',scope=scopes[0],executed=240)],
        T4_scope_comparisons=[dict(method=m,scope_run_id='scope_phase',baseline_scope=scopes[0],scope=scopes[1],
            planned_pairs=240,paired_population_complete=complete,delta_pp=-1.23 if complete else None,
            ci95_pp=[-2.34, .56] if complete else [None,None]) for m in methods])


def test_compact_scope_uses_existing_estimates_and_separate_controls(tmp_path):
    from robo.eval.native_scale_paper import render_scope
    output,text=render_scope(tmp_path,scope_sources(),require_complete=True)
    joined=' '.join(text)
    assert '24 instances across 4 layouts' in joined
    assert 'separate from the primary cohort' in joined
    assert '-1.23' in joined and '-2.34' in joined and '0.56' in joined
    assert '90 inherited' in joined and '20 additional' in joined
    assert 'articulations' in joined and 'goal handles' in joined
    assert output['native_test_scope.tex']=='tables/native_test_scope.tex'
    assert (tmp_path/'native_test_scope.tex').read_text().count('10/240 & 2/240')==4


def test_partial_scope_preview_has_no_paired_population_conclusion(tmp_path):
    from robo.eval.native_scale_paper import render_scope
    _,text=render_scope(tmp_path,scope_sources(False))
    joined=' '.join(text)
    assert 'Incomplete scope phase' in joined and 'difference' not in joined
    assert '2/240' not in (tmp_path/'native_test_scope.tex').read_text()
    with pytest.raises(ValueError,match='scope phase outcome terminal'):
        render_scope(tmp_path,scope_sources(False),require_complete=True)


@pytest.mark.parametrize('defect',['duplicate','missing','control','phase','interval'])
def test_scope_formatter_rejects_inconsistent_sources(tmp_path,defect):
    from robo.eval.native_scale_paper import render_scope
    source=scope_sources()
    if defect=='duplicate':source['T4_scope'].append(source['T4_scope'][0])
    elif defect=='missing':source['T4_scope_comparisons'].pop()
    elif defect=='control':source['T4_native_controls']=[]
    elif defect=='phase':source['T4_scope'][0]['scope_run_id']='different'
    else:source['T4_scope_comparisons'][0]['ci95_pp']=[None,None]
    with pytest.raises(ValueError):render_scope(tmp_path,source,require_complete=True)


def test_complete_test_escapes_actual_generated_interval_percent(tmp_path):
    r=rows()
    for row in r:row.update(split='TEST',complete=True,measured=row['planned'])
    sentence='The complete paired TEST B3_AGENT_NATIVE minus B0_FIXED_NATIVE difference is -1.23 percentage points (descriptive hierarchical 95% interval [-2.0, 0.0]); this does not establish noninferiority or preservation.'
    render_section(tmp_path,{'T2_manipulation':r,'test_conclusion_ledger':[
        dict(claim_id='test_pair_B3_B0',gate='PASS',cohort_complete=True,enabled_sentence=sentence)]})
    text=(tmp_path/'native_test_section.tex').read_text()
    assert r'95\% interval' in text and '95% interval' not in text
    assert '[-2.0, 0.0]' in text


def test_adjudication_prose_comes_from_machine_counts_and_preserves_failure_nuance(tmp_path):
    r=rows()
    for row in r:row.update(split='TEST',complete=True,measured=row['planned'])
    audit=dict(method='B0_FIXED_NATIVE',split='test',classified_units=7,affected_instances=2,
        all_watertight_near_threshold_nonconvex=True,producer_accepted_objects=9,native_executed_instances=6)
    render_section(tmp_path,{'T2_manipulation':r,'native_import_adjudication_summary':[audit]})
    text=(tmp_path/'native_test_section.tex').read_text()
    assert '7 prepolicy' in text and '2 instance(s)' in text
    assert 'acceptance covers 9 assets' in text and '6 instances have native execution' in text
    assert r'CODE\_FAILED' in text and r'BUILD\_FAILED' in text
    assert 'watertight' in text and 'near-threshold' in text
    assert 'native-success values remain unchanged' in text


def test_compact_test_reuses_complete_pair_and_keeps_full_detail_extended(tmp_path):
    r=rows()
    for row in r:row.update(split='TEST',complete=True,measured=row['planned'])
    pair=dict(method='B3_AGENT_NATIVE',reference='B0_FIXED_NATIVE',paired_population_complete=True,
        delta_pp=8.75,ci95_pp=[2.083333333,16.25])
    render_section(tmp_path,{'T2_manipulation':r},comparisons=[pair])
    text=(tmp_path/'native_test_section.tex').read_text()
    assert r'\conf{' in text and r'\ext{' in text
    assert '8.75 percentage points' in text and r'95\% interval [2.08, 16.25]' in text
    assert 'does not isolate its components or control extra computation' in text
    assert 'horizons remain fixed' in text
    # The original full protocol was moved, not erased.
    assert 'task-asset intervention' not in text  # No newly invented intervention axis.
    assert 'complete target-asset intervention' in text
    pair['paired_population_complete']=False
    with pytest.raises(ValueError,match='complete B3/B0 pairs'):
        render_section(tmp_path,{'T2_manipulation':r},comparisons=[pair])


def test_compact_scope_does_not_promote_partial_phase():
    from robo.eval.native_scale_paper import compact_test_text
    data=scope_sources(False);data['T2_manipulation']=rows()
    text=compact_test_text(data,[])
    assert 'Adding destination' not in text
