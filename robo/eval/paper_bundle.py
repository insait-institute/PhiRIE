"""Portable audit of existing canonical paper products; standard library only.

This packages measurements without recomputing them. Offline verification proves
byte identity, numeric lineage and prose decisions, not scientific validity or
reproduction of restricted raw data/model execution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def field(payload, expression):
    if not re.fullmatch(r'[A-Za-z_0-9]+(?:\[\d+\]|\.[A-Za-z_0-9]+)*', expression):
        raise ValueError('invalid numeric source field')
    for index, key in re.findall(r'\[(\d+)\]|([A-Za-z_0-9]+)', expression):
        payload = payload[key] if key else payload[int(index)]
    return payload


def local(root, name):
    path = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError('unsafe bundle member path')
    return path


def verify(root, expected_manifest_sha256):
    root = Path(root).resolve(strict=True)
    manifest_path = root / 'bundle_manifest.json'
    if sha(manifest_path) != expected_manifest_sha256:
        raise ValueError('bundle manifest differs from external expected SHA256')
    manifest = read(manifest_path)
    if manifest['scope'] != 'canonical_paper_numeric_lineage_and_publication_bytes':
        raise ValueError('unsupported bundle audit scope')
    members = manifest['members']
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != set(members) | {'bundle_manifest.json'}:
        raise ValueError('bundle member roster differs')
    for name, identity in members.items():
        path = local(root, name)
        if path.stat().st_size != identity['size_bytes'] or sha(path) != identity['sha256']:
            raise ValueError('bundle member bytes differ: ' + name)
    provenance = read(local(root, 'tables/paper_table_provenance.json'))
    receipt = read(local(root, 'publication_receipt.json'))
    if (receipt['freeze_id'] != provenance['freeze_id'] or
            receipt['producer_commit'] != provenance['formatter_commit'] or
            receipt['paper_commit'] != manifest['paper_commit'] or
            receipt['table_provenance_sha256'] != sha(root / 'tables/paper_table_provenance.json')):
        raise ValueError('publication identity differs')
    if sha(local(root, 'producer_config.yaml')) != provenance['config_sha256']:
        raise ValueError('original producer config differs')
    if sha(local(root, manifest['paper_qa'])) != receipt['paper_qa']['sha256']:
        raise ValueError('paper QA differs')
    e0 = read(local(root, 'producer_e0.json'))
    if (e0['freeze_id'] != provenance['freeze_id'] or e0['code']['commit'] != provenance['formatter_commit']
            or e0['code']['dirty'] is not False):
        raise ValueError('producer E0 identity differs')
    qa = read(local(root, manifest['paper_qa']))
    if qa['status'] != 'PASS':
        raise ValueError('paper layout QA is not PASS')
    for name, doc in qa['documents'].items():
        if (sha(local(root, f'paper/{name}.pdf')) != doc['sha256'] or
                not doc['page_limit_pass'] or doc['pages_including_references'] > 8 or
                not doc['fonts'] or not all(font['embedded'] for font in doc['fonts'])):
            raise ValueError('PDF differs from passed page/font audit')
    for name, digest in qa['generated_artifacts'].items():
        if sha(local(root, 'paper/' + name)) != digest:
            raise ValueError('paper copy differs from generated artifact QA')
    for family, directory in [('tables', 'generated_tables'), ('figures', 'generated_figures')]:
        for name, identity in provenance.get(family, {}).items():
            if sha(local(root, f'tables/{directory}/{name}')) != identity['sha256']:
                raise ValueError('generated provenance bytes differ')
    prose = '\n'.join(p.read_text() for p in (root / 'paper').rglob('*.tex'))
    checked = decisions = 0
    with (root / 'tables/claim_ledger.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if not row['source_artifact']:
                sentence = row.get('enabled_sentence', '')
                if sentence and sentence not in prose:
                    raise ValueError('enabled claim sentence missing: ' + row['claim_id'])
                decisions += 1
                continue
            identity = manifest['numeric_sources'][row['source_artifact']]
            path = local(root, identity['member'])
            if sha(path) != row['source_sha256'] or identity['sha256'] != row['source_sha256']:
                raise ValueError('numeric source hash differs')
            value = field(read(path), row['source_field'])
            rendered = row['value_in_text']
            if value is None:
                valid = row['value_in_source'] == '' and rendered == '--'
            else:
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError('numeric claim is nonfinite or nonnumeric')
                percent = rendered.endswith('\\%')
                number = rendered[:-2] if percent else rendered
                if not re.fullmatch(r'-?\d+(?:\.\d+)?', number):
                    raise ValueError('numeric display is malformed')
                digits = len(number.split('.')[1]) if '.' in number else 0
                valid = (str(value) == row['value_in_source'] and
                         number == f'{value * (100 if percent else 1):.{digits}f}')
            if not valid:
                raise ValueError('numeric source or rounding differs: ' + row['claim_id'])
            checked += 1
    if checked != manifest['numeric_claims'] or decisions != manifest['claim_decisions']:
        raise ValueError('claim roster differs')
    return dict(status='PASS', scope=manifest['scope'], freeze_id=provenance['freeze_id'],
                paper_commit=manifest['paper_commit'], numeric_claims=checked,
                claim_decisions=decisions, members=len(members),
                scientific_submission_gate=receipt['scientific_submission_gate'],
                raw_data_or_model_execution_reproduced=False)


def package(tables, paper_root, publication_receipt, out):
    tables, paper = Path(tables).resolve(strict=True), Path(paper_root).resolve(strict=True)
    out = Path(out).absolute()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=paper):
        raise ValueError('paper must be clean before packaging')
    paper_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=paper, text=True).strip()
    provenance = read(tables / 'paper_table_provenance.json')
    receipt = read(publication_receipt)
    if receipt['paper_commit'] != paper_commit:
        raise ValueError('receipt is for a different paper commit')
    out.mkdir(parents=True, exist_ok=False)
    members = {}
    def copy(source, name):
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise ValueError('source must be a regular nonsymlink file')
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as dest, source.open('rb') as src:
            shutil.copyfileobj(src, dest)
        members[name] = dict(sha256=sha(destination), size_bytes=destination.stat().st_size)
    for path in sorted(tables.rglob('*')):
        if path.is_file():
            if path.suffix not in {'.json', '.csv', '.yaml', '.tex', '.pdf', '.svg', '.png'}:
                raise ValueError('unexpected canonical table asset type')
            copy(path, 'tables/' + str(path.relative_to(tables)))
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=paper, text=True).split('\0')
    for name in filter(None, tracked):
        copy(paper / name, 'paper/' + name)
    for name in read(receipt['paper_qa']['path'])['documents']:
        for suffix in ('.pdf', '.log'):
            member = 'paper/' + name + suffix
            if member not in members:
                copy(paper / (name + suffix), member)
    copy(provenance['config'], 'producer_config.yaml')
    copy(tables.parent / 'contract/freeze_manifest.json', 'producer_e0.json')
    copy(publication_receipt, 'publication_receipt.json')
    copy(Path(__file__), 'verify.py')
    qa_name = 'paper/' + str(Path(receipt['paper_qa']['path']).relative_to(paper))
    sources = {}
    rows = list(csv.DictReader((tables / 'claim_ledger.csv').open(newline='')))
    for row in rows:
        origin = row['source_artifact']
        if origin and origin not in sources:
            if Path(origin).suffix != '.json':
                raise ValueError('numeric source must be JSON')
            name = 'numeric_sources/' + row['source_sha256'] + '.json'
            if name not in members:
                copy(origin, name)
            sources[origin] = dict(member=name, sha256=row['source_sha256'])
    manifest = dict(schema_version=1, scope='canonical_paper_numeric_lineage_and_publication_bytes',
                    paper_commit=paper_commit, paper_qa=qa_name, members=members,
                    numeric_sources=sources, numeric_claims=sum(bool(r['source_artifact']) for r in rows),
                    claim_decisions=sum(not r['source_artifact'] for r in rows),
                    restrictions='Raw datasets and checkpoints are excluded; original paths are provenance labels only.')
    with (out / 'bundle_manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write('\n')
    digest = sha(out / 'bundle_manifest.json')
    result = verify(out, digest)
    return dict(**result, bundle=str(out), manifest_sha256=digest,
                verify_command=['python3', str(out / 'verify.py'), '--verify', str(out), '--expected-sha256', digest])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', required=True)
    parser.add_argument('--expected-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.verify, args.expected_sha256), indent=2))
