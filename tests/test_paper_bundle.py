import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from robo.eval import paper_bundle as bundle


def dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture
def publication(tmp_path):
    paper = tmp_path / 'paper'
    paper.mkdir()
    (paper / 'root.tex').write_text('Missing trials remain unmeasured.')
    (paper / 'root.pdf').write_bytes(b'fixture pdf')
    (paper / 'root.log').write_text('fixture build')
    source = tmp_path / 'source.json'
    dump(source, {'rows': [{'value': .12555, 'missing': None, 'interval': [1.23456, 2.4]}]})
    tables = tmp_path / 'freeze/paper_tables'
    tables.mkdir(parents=True)
    config = tmp_path / 'config.yaml'
    config.write_text('mode: preliminary_audit\n')
    dump(tables / 'paper_table_provenance.json', dict(freeze_id='freeze', formatter_commit='producer',
         config=str(config), config_sha256=bundle.sha(config), tables={}, figures={}))
    dump(tables.parent / 'contract/freeze_manifest.json', dict(freeze_id='freeze', code=dict(commit='producer', dirty=False)))
    rows = []
    for expression, value, text in [('rows[0].value', .12555, '12.6\\%'),
                                    ('rows[0].missing', None, '--'), ('rows[0].interval[0]', 1.23456, '1.235')]:
        rows.append(dict(claim_id=expression, source_artifact=str(source), source_sha256=bundle.sha(source),
                         source_field=expression, value_in_source=value, value_in_text=text, enabled_sentence=''))
    rows.append(dict(claim_id='limitation', source_artifact='', source_sha256='', source_field='',
                     value_in_source='', value_in_text='', enabled_sentence='Missing trials remain unmeasured.'))
    with (tables / 'claim_ledger.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    qa = paper / 'audit/qa.json'
    dump(qa, dict(status='PASS', documents={'root': dict(sha256=bundle.sha(paper / 'root.pdf'),
         pages_including_references=1, page_limit_pass=True, fonts=[dict(embedded=True)])}, generated_artifacts={}))
    subprocess.run(['git', 'init', '-q', str(paper)], check=True)
    subprocess.run(['git', 'add', '.'], cwd=paper, check=True)
    subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                    'commit', '-qm', 'fixture'], cwd=paper, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=paper, text=True).strip()
    receipt = tmp_path / 'receipt.json'
    dump(receipt, dict(freeze_id='freeze', producer_commit='producer', paper_commit=commit,
         paper_qa=dict(path=str(qa), sha256=bundle.sha(qa)),
         table_provenance_sha256=bundle.sha(tables / 'paper_table_provenance.json'), scientific_submission_gate='FAIL'))
    return tables, paper, receipt, tmp_path / 'bundle'


def test_bundle_relocates_without_original_cluster_paths(publication, tmp_path):
    result = bundle.package(*publication)
    moved = tmp_path / 'relocated'
    shutil.move(publication[-1], moved)
    for path in publication[:3]:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    (tmp_path / 'source.json').unlink()
    run = subprocess.run([sys.executable, '-I', str(moved / 'verify.py'), '--verify', str(moved),
                          '--expected-sha256', result['manifest_sha256']], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    assert report['numeric_claims'] == 3 and report['claim_decisions'] == 1
    assert report['scientific_submission_gate'] == 'FAIL'


def test_no_overwrite(publication):
    bundle.package(*publication)
    with pytest.raises(FileExistsError): bundle.package(*publication)


@pytest.mark.parametrize('change', ['source', 'extra', 'symlink', 'missing', 'manifest'])
def test_tampered_bundle_rejected(publication, change):
    result = bundle.package(*publication)
    root = publication[-1]
    source = next((root / 'numeric_sources').glob('*.json'))
    if change == 'source': source.write_text('{}')
    elif change == 'extra': (root / 'extra').write_text('unexpected')
    elif change == 'missing': source.unlink()
    elif change == 'manifest': (root / 'bundle_manifest.json').write_text('{}')
    else:
        data = source.read_bytes(); source.unlink()
        target = root.parent / 'alias.json'; target.write_bytes(data); source.symlink_to(target)
    with pytest.raises((ValueError, FileNotFoundError)): bundle.verify(root, result['manifest_sha256'])


@pytest.mark.parametrize('column,value', [('value_in_source','0.9'), ('value_in_text','99.9\\%'),
                                        ('source_field','rows[1].value'), ('source_sha256','0'*64)])
def test_stale_numeric_claim_rejected(publication, column, value):
    path = publication[0] / 'claim_ledger.csv'
    rows = list(csv.DictReader(path.open()))
    rows[0][column] = value
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    with pytest.raises((ValueError, IndexError)): bundle.package(*publication)


def test_wrong_paper_revision_rejected(publication):
    receipt = bundle.read(publication[2]); receipt['paper_commit'] = 'wrong'; dump(publication[2], receipt)
    with pytest.raises(ValueError, match='different paper'): bundle.package(*publication)


def test_pipeline_existing_bundle_does_not_generate(publication, monkeypatch, capsys):
    from robo.eval import paper_pipeline
    monkeypatch.setattr(paper_pipeline, 'generate', lambda *a: pytest.fail('must not rerun producers'))
    tables, paper, receipt, out = publication
    assert paper_pipeline.main(['--bundle-existing', str(tables), '--paper-root', str(paper),
                               '--publication-receipt', str(receipt), '--out', str(out)]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'PASS'


def test_source_field_accepts_digit_prefixed_scene_ids():
    assert bundle.field({'source_qualification_summaries': {'40aec5fffa': {'fixed_selected_cells': 40}}},
                        'source_qualification_summaries.40aec5fffa.fixed_selected_cells') == 40
    assert bundle.field({'rows': [{'interval': [1.0, 2.0]}]}, 'rows[0].interval[1]') == 2.0
    with pytest.raises(ValueError): bundle.field({}, '../outside')
