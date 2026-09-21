import copy
import hashlib
import json
from pathlib import Path
import pytest
import yaml

from robo.eval import paper_room_diagnostic as diagnostic
from robo.eval import paper_pipeline as pipeline
from tests.test_paper_pipeline_audit import fixture_config


def fixture():
    table = dict(validation={'valid': True}, lpips_backend_error=None, rows=[
        dict(method=diagnostic.METHODS[0], unit='room', n_scenes=1, n_images=8,
             psnr=21.25, ssim=.8, lpips=.2, metric_samples=dict(psnr=8,ssim=8,lpips=8)),
        dict(method='unmeasured intervening row', psnr=None),
        dict(method=diagnostic.METHODS[1], unit='room', n_scenes=1, n_images=8,
             psnr=20.125, ssim=.7, lpips=.3, metric_samples=dict(psnr=8,ssim=8,lpips=8))])
    coverage = dict(planned_scenes=2, planned_objects=17, planned_views_per_method=16,
        paper_ready=False, full_e3_gt_access=False, quality_comparison_paired=True,
        rows=[dict(method=m, planned_views=8, available_views=8 if s==0 or i==0 else 0,
                   analysis_views=8 if s==0 else 0) for i,m in enumerate(diagnostic.METHODS) for s in range(2)])
    return table, coverage


def identity(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def test_original_row_indices_and_unmeasured_rows_are_preserved():
    table, coverage = fixture()
    assert diagnostic.validate_rows(table, coverage) is table
    assert table['rows'][1]['psnr'] is None


@pytest.mark.parametrize('damage', ['unpaired','planned','available','subset','scenes','missing','nan','invalid','gt','metric_count'])
def test_scope_denominators_and_quality_fail_closed(damage):
    table,coverage=fixture()
    if damage=='unpaired':coverage['quality_comparison_paired']=False
    elif damage=='planned':coverage['planned_views_per_method']=8
    elif damage=='available':coverage['rows'][3]['available_views']=8
    elif damage=='subset':coverage['rows'][0]['analysis_views']=7
    elif damage=='scenes':table['rows'][0]['n_scenes']=2
    elif damage=='missing':table['rows'][2]['lpips']=None
    elif damage=='nan':table['rows'][2]['ssim']=float('nan')
    elif damage=='invalid':table['validation']['valid']=False
    elif damage=='metric_count':table['rows'][0]['metric_samples']['lpips']=7
    else:coverage['full_e3_gt_access']=True
    with pytest.raises(ValueError):diagnostic.validate_rows(table,coverage)


@pytest.mark.parametrize('damage', ['manifest_sha256','manifest_path','freeze_id','source','provenance_manifest','promotion'])
def test_metric_identity_rejects_rebound_table(damage):
    manifest=dict(path='/immutable/manifest.json',sha256='a'*64)
    table=dict(freeze_id='freeze',manifest_path=manifest['path'],manifest_sha256=manifest['sha256'],paper_ready=False,
        provenance=dict(code_commit='b'*40,freeze_id='freeze',manifest_path=manifest['path'],manifest_sha256=manifest['sha256']))
    diagnostic.validate_metric_identity(table,manifest,'freeze','b'*40)
    if damage=='source':table['provenance']['code_commit']='c'*40
    elif damage=='provenance_manifest':table['provenance']['manifest_sha256']='d'*64
    elif damage=='promotion':table['paper_ready']=True
    else:table[damage]='different'
    with pytest.raises(ValueError,match='table-to-manifest'):diagnostic.validate_metric_identity(table,manifest,'freeze','b'*40)


def test_alias_and_content_drift_rejected(tmp_path):
    source=tmp_path/'source';source.write_text('{}');ref=identity(source)
    assert diagnostic.checked(ref)==source
    alias=tmp_path/'alias';alias.symlink_to(source)
    with pytest.raises(ValueError,match='alias'):diagnostic.checked({**ref,'path':str(alias)})
    source.write_text('changed')
    with pytest.raises(ValueError,match='hash'):diagnostic.checked(ref)


def test_dirty_or_wrong_original_producer_rejected(monkeypatch,tmp_path):
    code=tmp_path/'code';code.mkdir()
    monkeypatch.setattr(diagnostic,'git_snapshot',lambda _:dict(commit=diagnostic.PRODUCER,dirty=True))
    spec={'producer':{'path':str(code),'commit':diagnostic.PRODUCER}}
    with pytest.raises(ValueError,match='producer source'):diagnostic.validate_source(spec,None,None,None,None)


def test_generated_diagnostic_uses_exact_source_fields_and_planned_coverage(monkeypatch,tmp_path):
    config_path,config=fixture_config(tmp_path)
    table,coverage=fixture()
    metrics=tmp_path/'metrics';(metrics/'table').mkdir(parents=True)
    source=metrics/'table/fidelity_table.json';source.write_text(json.dumps(table))
    coverage_path=metrics/'coverage.json';coverage_path.write_text(json.dumps(coverage))
    seal=metrics/'seal.json';seal.write_text('{}')
    config['engineering_appendix']={'room_common_view':{**identity(source),'completion_audit':identity(seal)}}
    config_path.write_text(yaml.safe_dump(config))
    def validate(spec,path,payload,seal_path,sealed):
        assert path==source and seal_path==seal
        return diagnostic.validate_rows(payload,coverage),coverage,{'canonical_metric_recomputed':False}
    monkeypatch.setattr(diagnostic,'validate_source',validate)
    out=tmp_path/'publication';pipeline.generate(config_path,out,tmp_path/'paper')
    tex=(out/'generated_tables/room_common_view_engineering.tex').read_text()
    assert tex.count('8/16')==2 and 'descriptive means' in tex and '20.12' in tex
    assert 'confidence interval is interpreted' in tex
    import csv
    claims=list(csv.DictReader((out/'claim_ledger.csv').open()))
    metric=next(r for r in claims if r['claim_id']=='room_common_view.2.psnr')
    assert metric['source_artifact']==str(source) and metric['source_field']=='rows[2].psnr'
    denominator=next(r for r in claims if r['claim_id']=='room_common_view_coverage.None.planned_views_per_method')
    assert denominator['source_artifact']==str(coverage_path) and denominator['value_in_source']=='16'
    assert json.loads((out/'sources/room_common_view.json').read_text())==table
    assert json.loads((out/'sources/room_common_view_coverage.json').read_text())==coverage
    with pytest.raises(FileExistsError):pipeline.generate(config_path,out,tmp_path/'paper')
