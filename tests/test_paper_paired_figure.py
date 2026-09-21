import json

import pytest
import yaml

from robo.eval import paper_pipeline as p
from tests.test_paper_paired_uncertainty import fixture, refresh


def test_paired_vector_figure_preserves_every_statistic_and_support(tmp_path):
    path,config,aggregate,audit,data=fixture(tmp_path)
    config['agentic_paired_uncertainty_figure']=True
    path.write_text(yaml.safe_dump(config))
    out=tmp_path/'out';paper=tmp_path/'paper';paper.mkdir()
    report=p.generate(path,out,paper)
    figures=out/'generated_figures'
    manifest=json.loads((figures/'figure_manifest.json').read_text())
    assert len(manifest['plotted_rows'])==16
    for row in manifest['plotted_rows']:
        source=data['rows'][row['source_row']]
        for key in ('contrast','metric','delta','ci95','paired_jobs','planned_jobs','paired_scenes','status'):
            assert row[key]==source[key]
    assert manifest['metric_recomputation'] is False and manifest['bootstrap_repeated'] is False
    assert manifest['source']['sha256']==config['agentic_paired_uncertainty']['sha256']
    for name,identity in report['figures'].items():
        assert p._sha256(figures/name)==identity['sha256']
        assert (paper/'figures'/name).read_bytes()==(figures/name).read_bytes()
    assert (figures/'agentic_paired_uncertainty.pdf').read_bytes().startswith(b'%PDF')
    svg=(figures/'agentic_paired_uncertainty.svg').read_text()
    assert '<text' in svg and 'Common accepted' in svg


def test_null_intervals_are_visible_and_not_silently_zero(tmp_path):
    path,config,aggregate,audit,data=fixture(tmp_path)
    r=data['rows'][1];r.update(paired_jobs=0,paired_scenes=0,paired_scene_ids=[],excluded_pair_jobs=1871,
        unsupported_scenes=50,baseline_mean=None,treatment_mean=None,delta=None,ci95=[None,None],
        status='NOT_ESTIMABLE',reason='fewer_than_two_supported_scenes')
    config['agentic_paired_uncertainty_figure']=True
    refresh(path,config,aggregate,audit,data);p.generate(path,tmp_path/'out')
    root=tmp_path/'out/generated_figures'
    manifest=json.loads((root/'figure_manifest.json').read_text())
    row=next(r for r in manifest['plotted_rows'] if r['source_row']==1)
    assert row['delta'] is None and row['ci95']==[None,None]
    assert 'unmeasured' in (root/'agentic_paired_uncertainty.svg').read_text()


def test_percentile_interval_can_exclude_point_estimate(tmp_path):
    path,config,aggregate,audit,data=fixture(tmp_path)
    data['rows'][0]['ci95']=[.01,.02]
    config['agentic_paired_uncertainty_figure']=True
    refresh(path,config,aggregate,audit,data);p.generate(path,tmp_path/'out')
    manifest=json.loads((tmp_path/'out/generated_figures/figure_manifest.json').read_text())
    row=next(r for r in manifest['plotted_rows'] if r['source_row']==0)
    assert row['ci95']==[.01,.02] and row['delta']==0.


@pytest.mark.parametrize('setting',[True,'true',1])
def test_figure_cannot_bypass_full_source_authentication(tmp_path,setting):
    path,config,_,_,_=fixture(tmp_path)
    config.pop('agentic_paired_uncertainty')
    config['agentic_paired_uncertainty_figure']=setting
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):p.generate(path,tmp_path/'out')
    assert not (tmp_path/'out').exists()
