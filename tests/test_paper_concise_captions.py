import json
import pytest
import yaml
from robo.eval import paper_pipeline as p
from tests.test_paper_pipeline_audit import full_fixture, fixture_config


def test_caption_style_preserves_all_generated_values_and_sources(tmp_path):
    path, config, _, _ = full_fixture(tmp_path)
    p.generate(path, tmp_path/'verbose')
    config['concise_captions'] = True
    path.write_text(yaml.safe_dump(config))
    p.generate(path, tmp_path/'concise')
    for source in (tmp_path/'verbose/sources').iterdir():
        assert source.read_bytes() == (tmp_path/'concise/sources'/source.name).read_bytes()
    assert (tmp_path/'verbose/claim_ledger.csv').read_bytes() == (tmp_path/'concise/claim_ledger.csv').read_bytes()
    tex=(tmp_path/'concise/generated_tables/agentic_ablation_main.tex').read_text()
    assert 'A4 changes that subset' in tex and 'acceptance-time construction probe' in tex
    assert 'Preliminary audit' not in tex and 'Complete automatic-discovery controller cohort' in tex


def test_concise_style_cannot_label_incomplete_cohort_as_complete(tmp_path):
    path, config = fixture_config(tmp_path)
    config['concise_captions'] = True
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='authenticated full controller cohort'):
        p.generate(path, tmp_path/'out')
