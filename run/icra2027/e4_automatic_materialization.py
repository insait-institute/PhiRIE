"""Source-bound CPU launcher for the existing automatic E3 materializer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
import time
from importlib.metadata import version

import yaml

from robo.eval import e3_factory_materializer as materializer
from robo.eval import agentic_ablation as e3
from robo.manifest.hash import canonical_hash

CODE = Path(__file__).resolve().parents[2]


def validate_inputs(config, root, contract, snapshot, config_path):
    if (materializer.CODE_ROOT != CODE
            or Path(materializer.__file__).resolve().parents[2] != CODE
            or Path(e3.__file__).resolve().parents[2] != CODE):
        raise ValueError('imported materialization implementation differs from launcher source')
    if (config.get('schema_version') != 1 or config.get('paper_ready') is not False
            or config.get('scope') != 'automatic_materialization_engineering'
            or config.get('policies') != ['A0', 'A4']
            or type(config.get('planned_jobs_per_policy')) is not int
            or config['planned_jobs_per_policy'] < 0
            or config.get('freeze_id') != root.name
            or contract.get('freeze_id') != root.name
            or contract['code']['commit'] != snapshot['commit']
            or contract['code'].get('dirty') is not False):
        raise ValueError('materialization scope, population or exact source contract differs')
    expected = canonical_hash({key:value for key,value in contract.items()
                               if key not in {'created_utc','environment','contract_sha256'}})
    if contract.get('contract_sha256') != expected:
        raise ValueError('E0 contract digest changed')
    resources = [row for row in contract['resource_inventory']
                 if row['id'] == 'e4_automatic_materialization_config']
    if len(resources) != 1 or resources[0]['sha256'] != e3.sha256_file(config_path):
        raise ValueError('materialization config differs from exact E0 resource')
    interpreter = Path(config['python']).resolve(strict=True)
    python_resources = [row for row in contract['resource_inventory'] if row['id'] == 'materialization_python']
    if (interpreter != Path(sys.executable).resolve() or len(python_resources) != 1
            or python_resources[0]['sha256'] != e3.sha256_file(interpreter)):
        raise ValueError('materialization Python differs from exact E0 runtime')
    if materializer.REPOSITORY_ROOT != Path(config['evidence_root']).resolve():
        raise ValueError('explicit evidence root differs')
    for name, identity in config['source_files'].items():
        path = e3.checked_repo_path(identity['path'], name, kind='file')
        if path.stat().st_size != identity['size_bytes'] or e3.sha256_file(path) != identity['sha256']:
            raise ValueError(f'frozen source file changed: {name}')
    source_root = e3.checked_repo_path(config['e3_root'], 'E3 root', kind='dir')
    jobs, _, audit, _ = materializer._verify_inventory(source_root)
    if (jobs['freeze_id'] != config['e3_freeze_id']
            or jobs['source_contract']['code_commit'] != config['e3_source_commit']
            or [scene['scene_id'] for scene in jobs['scenes']] != [config['scene_id']]
            or jobs['counts']['jobs'] != config['planned_jobs_per_policy']):
        raise ValueError('sealed E3 source identity or full scene population differs')
    return jobs, audit


def run(config_path, root, phase):
    started = time.monotonic()
    config_path = Path(config_path).resolve(strict=True)
    root = e3.checked_repo_path(root, 'materialization freeze', kind='dir')
    config = yaml.safe_load(config_path.read_text())
    contract = json.loads((root/'contract/freeze_manifest.json').read_text())
    snapshot = materializer._require_clean_code_snapshot()
    jobs, audit = validate_inputs(config, root, contract, snapshot, config_path)
    directory = e3.checked_repo_path(root/'harness', 'harness output', must_exist=False, kind='dir')
    directory.mkdir(exist_ok=True)
    descriptor = directory/'automatic_scene_descriptor.json'
    payload = materializer._json_bytes(config['automatic_scene_descriptor'])
    if descriptor.exists():
        if descriptor.is_symlink() or descriptor.read_bytes() != payload:
            raise ValueError('published automatic descriptor changed')
    else:
        materializer._write_inside(descriptor, payload)
    source = materializer._automatic_source_context(jobs, audit, config['scene_id'], descriptor)
    report = dict(schema_version=1, phase=phase, paper_ready=False,
                  scope=config['scope'], freeze_id=root.name, code_commit=snapshot['commit'],
                  e3_source_commit=config['e3_source_commit'], e3_freeze_id=config['e3_freeze_id'],
                  planned_jobs_per_policy=len(source['jobs']), planned_policy_object_rows=2*len(source['jobs']),
                  source_gaussian_training_provenance=source['source_gaussian_training_provenance'],
                  source_files=config['source_files'], descriptor_sha256=e3.sha256_file(descriptor),
                  hostname=socket.gethostname(), python=sys.version,
                  package_versions={name:version(name) for name in ('numpy','plyfile','trimesh')},
                  gpu_requested=False)
    if phase == 'materialize':
        host = socket.gethostname().split('.')[0].lower()
        if host != 'hala' and not host.startswith(('gcp','sof1')):
            raise ValueError('CPU job is outside the declared allowed hosts')
        report['policies'] = {}
        for policy in config['policies']:
            destination = e3.checked_repo_path(directory/'materialized'/policy,
                'materialized policy output', must_exist=False, kind='dir')
            if not destination.exists():
                materializer.materialize_factory_variant(e3_root=config['e3_root'],
                    scene_id=config['scene_id'], policy_id=policy, out=destination,
                    automatic_scene_contract=descriptor)
            validated = materializer.validate_materialized_factory(destination,
                expected_scene_id=config['scene_id'], expected_policy_id=policy)
            manifest = json.loads((destination/'materialization_manifest.json').read_text())
            if (e3.checked_repo_path(manifest['e3_root'], 'adopted E3 source', kind='dir')
                    != e3.checked_repo_path(config['e3_root'], 'configured E3 source', kind='dir')
                    or manifest['e3_freeze_id'] != config['e3_freeze_id']
                    or manifest['e3_code_commit'] != config['e3_source_commit']
                    or manifest['source_scene']['automatic_scene_descriptor'] != materializer._identity(descriptor)):
                raise ValueError('existing materialization belongs to another frozen source')
            report['policies'][policy] = validated
    report['wall_s'] = time.monotonic()-started
    report_path = directory/(phase+'_report.json')
    materializer._write_inside(report_path, materializer._json_bytes(report))
    print(json.dumps({'report':str(report_path), 'sha256':e3.sha256_file(report_path),
                      'planned_jobs_per_policy':report['planned_jobs_per_policy'], 'paper_ready':False}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--freeze-root', required=True)
    parser.add_argument('--phase', choices=['preflight','materialize'], required=True)
    args = parser.parse_args()
    run(args.config, args.freeze_root, args.phase)


if __name__ == '__main__':
    main()
