"""Full-source publication preserves conditional support and missing methods."""
import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from robo.eval.paper_pipeline import generate, _engineering_sources
from tests.test_paper_pipeline_audit import fixture_config


def ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_full_appearance_uses_common_support_and_preserves_unmeasured_rows(tmp_path, monkeypatch):
    from robo.eval import paper_full_appearance
    path, config = fixture_config(tmp_path)
    table = json.loads(Path(config['audited_artifacts']['fidelity']['path']).read_text())
    for i, row in enumerate(table['rows']):
        row.update(n_images=320 if i in (0, 2) else 0, n_scenes=40 if i in (0, 2) else 0,
                   psnr=21.5 if i == 0 else 20.4 if i == 2 else None,
                   ssim=.84 if i == 0 else .83 if i == 2 else None,
                   lpips=.30 if i == 0 else .32 if i == 2 else None)
    source=tmp_path/'full.json';source.write_text(json.dumps(table))
    seal=tmp_path/'seal.json';seal.write_text('{}')
    coverage=dict(planned_scenes=50,planned_views_per_method=400,paper_ready=False)
    (tmp_path/'coverage.json').write_text(json.dumps(coverage))
    calls=[]
    def verified(*args):
        calls.append(args)
        return table, coverage, {'verification_scope':'synthetic metadata fixture'}
    monkeypatch.setattr(paper_full_appearance,'validate_source',verified)
    config['engineering_appendix']={'full_appearance':{
        **ref(source),'completion_audit':ref(seal)}}
    path.write_text(yaml.safe_dump(config));out=tmp_path/'out'
    report=generate(path,out)
    text=(out/'generated_tables/object_factorization_main.tex').read_text()
    assert len(calls)==1 and text.count('40/50 & 320/400')==2
    assert '0/50 & 0/400 & -- & -- & --' in text
    assert 'GT discovery' in text and 'Harmonizer' in text
    assert 'no paired-difference significance' in text and 'Zero-object' in text
    assert report['paper_ready'] is False
    rows=list(csv.DictReader((out/'claim_ledger.csv').open()))
    psnr=next(r for r in rows if r['claim_id']=='full_appearance.2.psnr')
    assert psnr['source_field']=='rows[2].psnr' and psnr['source_sha256']==ref(source)['sha256']
    planned=next(r for r in rows if r['claim_id']=='full_appearance_coverage.None.planned_views_per_method')
    assert planned['source_field']=='planned_views_per_method'
    assert planned['source_sha256']==ref(tmp_path/'coverage.json')['sha256']


def test_full_construction_authentication_is_required_before_scoped_display(tmp_path, monkeypatch):
    from robo.eval import paper_full_construction
    path, config=fixture_config(tmp_path)
    source=Path(config['audited_artifacts']['construction']['path'])
    table=json.loads(source.read_text())
    for row in table['rows']:
        row.update(stable_instances=2,tested_instances=3,valid_for_paper=False)
    table.update(paper_ready=False,preliminary_override=True)
    source.write_text(json.dumps(table));seal=tmp_path/'seal.json';seal.write_text('{}')
    config['audited_artifacts']['construction']={**ref(source),'completion_audit':ref(seal)}
    path.write_text(yaml.safe_dump(config))
    def reject(*args):raise ValueError('source seal rejected')
    monkeypatch.setattr(paper_full_construction,'validate_source',reject)
    with pytest.raises(ValueError,match='source seal rejected'):generate(path,tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()
    monkeypatch.setattr(paper_full_construction,'validate_source',lambda *args:dict(paper_ready=False))
    out=tmp_path/'out';generate(path,out)
    text=(out/'generated_tables/automatic_construction_main.tex').read_text()
    assert 'stable/tested' in text and '2/3' in text
    assert 'independent isolated-body drop' in text and 'exports' in text
    assert json.loads((out/'sources/construction.json').read_text())['paper_ready'] is False


def test_pilot_and_full_common_view_cannot_mix():
    with pytest.raises(ValueError,match='do not mix'):
        _engineering_sources({'room_common_view':{},'full_appearance':{}})
