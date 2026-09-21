"""Synthetic formatter tests; no result here is experimental evidence."""
import copy
import csv
import json
from pathlib import Path

import pytest
import yaml

from robo.eval import paper_pipeline as p
from robo.eval.agentic_ablation import AGENTIC_UNCERTAINTY_PROTOCOL
from tests.test_paper_pipeline_audit import full_fixture, _refresh_full


def fixture(tmp_path):
    path, config, aggregate, audit = full_fixture(tmp_path)
    rows = []
    for a, b in AGENTIC_UNCERTAINTY_PROTOCOL['contrasts']:
        for metric in AGENTIC_UNCERTAINTY_PROTOCOL['metrics']:
            coverage = metric == 'build_coverage'
            geometry = metric in {'f1_20', 'cd_cm'}
            n = 1871 if coverage else 80 if geometry else 100
            ns = 50 if coverage else 2
            mean = 100 / 1871 if coverage else .5
            rows.append(dict(contrast=f'{b}-{a}',baseline=a,treatment=b,metric=metric,
                support='all_planned_jobs' if coverage else 'common_accepted_matched_geometry'
                        if geometry else 'common_accepted_construction_probes',
                planned_scenes=50,planned_jobs=1871,baseline_accepted_jobs=100,treatment_accepted_jobs=100,
                baseline_metric_eligible_jobs=n,treatment_metric_eligible_jobs=n,common_accepted_jobs=100,
                paired_jobs=n,paired_scenes=ns,excluded_pair_jobs=1871-n,unsupported_scenes=50-ns,
                paired_scene_ids=[f'scene{i:02}' for i in range(ns)],paired_job_ids_sha256='a'*64,
                baseline_mean=mean,treatment_mean=mean,delta=0.,ci95=[-.05,.05],status='ESTIMATED',reason=None,
                selection_conditioned=not coverage,independent_physical_validation=False))
    data=dict(schema_version=1,protocol=copy.deepcopy(AGENTIC_UNCERTAINTY_PROTOCOL),rows=rows,
              freeze_id=aggregate['freeze_id'],construction_freeze_id=aggregate['construction_freeze_id'],
              planned_scenes=50,planned_jobs=1871,paper_ready=False,headline_eligible=False,claim_gate='NOT_RUN')
    spec=dict(path=str(tmp_path/'paired.json'))
    config['agentic_paired_uncertainty']=spec
    refresh(path,config,aggregate,audit,data)
    return path,config,aggregate,audit,data


def refresh(path,config,aggregate,audit,data):
    spec=config['agentic_paired_uncertainty'];source=Path(spec['path']);source.write_text(json.dumps(data))
    spec['sha256']=p._sha256(source)
    audit['evidence_hashes'][str(source)]=dict(sha256=spec['sha256'],size_bytes=source.stat().st_size)
    _refresh_full(path,config,aggregate,audit)


def test_all_contrasts_are_generated_and_every_number_has_source_field(tmp_path):
    path,config,aggregate,audit,data=fixture(tmp_path)
    report=p.generate(path,tmp_path/'out')
    tex=(tmp_path/'out/generated_tables/agentic_paired_uncertainty.tex').read_text()
    for a,b in AGENTIC_UNCERTAINTY_PROTOCOL['contrasts']:
        assert tex.count(f'{b}-{a} &')==4
    assert '80/1871' in tex and '1871/1871' in tex
    assert 'common accepted, matched' in tex and 'pointwise descriptive' in tex
    assert 'not multiplicity-adjusted' in tex and 'independent physics gains' in tex
    assert not report['paper_ready']
    copied=json.loads((tmp_path/'out/sources/agentic_paired_uncertainty.json').read_text())
    assert copied==data
    ledger=list(csv.DictReader((tmp_path/'out/claim_ledger.csv').open()))
    ci=[r for r in ledger if r['source_field'].endswith('.ci95[0]')]
    assert len(ci)==16 and all(r['value_in_text']=='-0.050' for r in ci)
    assert all(r['source_sha256']==config['agentic_paired_uncertainty']['sha256'] for r in ci)


@pytest.mark.parametrize('empty',[False,True])
def test_unestimable_intervals_stay_null_without_dropping_contrast(tmp_path,empty):
    path,config,aggregate,audit,data=fixture(tmp_path)
    row=data['rows'][1];n=0 if empty else 1
    row.update(paired_jobs=n,paired_scenes=n,paired_scene_ids=[] if empty else ['scene00'],
               excluded_pair_jobs=1871-n,unsupported_scenes=50-n,ci95=[None,None],
               status='NOT_ESTIMABLE',reason='fewer_than_two_supported_scenes')
    if empty:row.update(baseline_mean=None,treatment_mean=None,delta=None)
    refresh(path,config,aggregate,audit,data);p.generate(path,tmp_path/'out')
    tex=(tmp_path/'out/generated_tables/agentic_paired_uncertainty.tex').read_text()
    assert '[--, --]' in tex and tex.count('A1-A0 &')==4


@pytest.mark.parametrize('mutation',[
    'drop_contrast','wrong_order','wrong_protocol','wrong_freeze','promote_claim','missing_audit_binding',
    'wrong_accepted','wrong_eligible','conditional_all_jobs','coverage_excluded','bool_count',
    'wrong_delta','nonfinite_ci','reversed_ci','one_scene_ci','fake_null_ci','wrong_support','physical_claim'])
def test_source_or_denominator_mismatch_cannot_publish(tmp_path,mutation):
    path,config,aggregate,audit,data=fixture(tmp_path);row=data['rows'][1]
    if mutation=='drop_contrast':data['rows'].pop()
    elif mutation=='wrong_order':data['rows'].reverse()
    elif mutation=='wrong_protocol':data['protocol']['seed']=42
    elif mutation=='wrong_freeze':data['freeze_id']='other'
    elif mutation=='promote_claim':data['headline_eligible']=True
    elif mutation=='wrong_accepted':row['baseline_accepted_jobs']=101
    elif mutation=='wrong_eligible':row['baseline_metric_eligible_jobs']=79
    elif mutation=='conditional_all_jobs':row.update(paired_jobs=1871,excluded_pair_jobs=0)
    elif mutation=='coverage_excluded':data['rows'][0].update(paired_jobs=1870,excluded_pair_jobs=1)
    elif mutation=='bool_count':row['paired_scenes']=True
    elif mutation=='wrong_delta':row['delta']=.4
    elif mutation=='nonfinite_ci':row['ci95'][0]=float('nan')
    elif mutation=='reversed_ci':row['ci95']=[.1,-.1]
    elif mutation=='one_scene_ci':row.update(paired_scenes=1,unsupported_scenes=49,paired_scene_ids=['scene00'])
    elif mutation=='fake_null_ci':row['ci95']=[None,None]
    elif mutation=='wrong_support':row['support']='all_planned_jobs'
    elif mutation=='physical_claim':row['independent_physical_validation']=True
    refresh(path,config,aggregate,audit,data)
    if mutation=='missing_audit_binding':
        audit['evidence_hashes'].pop(config['agentic_paired_uncertainty']['path'])
        _refresh_full(path,config,aggregate,audit)
    with pytest.raises(ValueError):p.generate(path,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_optional_table_does_not_change_original_tables(tmp_path):
    path,config,aggregate,audit,data=fixture(tmp_path)
    p.generate(path,tmp_path/'with')
    del config['agentic_paired_uncertainty'];path.write_text(yaml.safe_dump(config))
    p.generate(path,tmp_path/'without')
    for table in (tmp_path/'without/generated_tables').glob('*.tex'):
        assert table.read_bytes()==(tmp_path/'with/generated_tables'/table.name).read_bytes()
