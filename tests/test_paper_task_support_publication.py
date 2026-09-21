"""Publication lineage checks; synthetic counts are never experiment evidence."""
import csv
import json
import pytest
import yaml
from robo.eval import paper_pipeline as pipeline
from robo.eval import paper_task_support_closure as closure
from tests.test_paper_pipeline_audit import fixture_config


def test_original_qa_fields_and_null_metrics_are_preserved(monkeypatch, tmp_path):
    path, config = fixture_config(tmp_path)
    qa = dict(planned_conditions=18, planned_queries=72,
              failed_constructor_query_rows=16, invalid_queries=72,
              metrics=dict.fromkeys(closure.METRICS), loso_status='NOT_RUN')
    source = tmp_path / 'qa.json'
    source.write_text(json.dumps(qa))
    ref = dict(path=str(source), sha256=pipeline._sha256(source))
    config['engineering_appendix'] = {'task_support_closure': {**ref, 'completion_audit': ref}}
    path.write_text(yaml.safe_dump(config))
    def validate(p, *, expected_sha256):
        assert p == source and expected_sha256 == ref['sha256']
        return dict(producer_commit='a'*40, contract_sha256='b'*64,
                    feature_contract_sha256='c'*64, feature_seal={}, label_join_manifest={})
    monkeypatch.setattr(closure, 'validate_closure', validate)
    out = tmp_path / 'out'
    pipeline.generate(path, out)
    assert json.loads((out/'sources/task_support_closure.json').read_text()) == qa
    tex = (out/'generated_tables/task_support_closure_engineering.tex').read_text()
    assert '18 & 72 & 16 & 72 & not run' in tex
    assert 'unmeasured, not zero' in tex
    claims = list(csv.DictReader((out/'claim_ledger.csv').open()))
    for key in ('planned_conditions', 'planned_queries', 'failed_constructor_query_rows', 'invalid_queries'):
        claim = next(r for r in claims if r['claim_id'] == f'task_support_closure.None.{key}')
        assert claim['source_artifact'] == str(source) and claim['source_field'] == key
        assert float(claim['value_in_source']) == qa[key]
    source.write_text('{}')
    with pytest.raises(ValueError, match='SHA256'):
        pipeline.generate(path, tmp_path/'tampered')
