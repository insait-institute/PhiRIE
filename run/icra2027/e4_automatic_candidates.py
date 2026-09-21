"""Exact-source CPU launcher for the canonical automatic candidate producer."""
import argparse
import json
from pathlib import Path
import socket
import yaml

from run.icra2027 import e4_automatic_materialization as shared
from robo.eval import e4_candidate_screen as screen

CODE = Path(__file__).resolve().parents[2]


def run(config_path, phase):
    config_path = Path(config_path).resolve(strict=True)
    config = yaml.safe_load(config_path.read_text())
    root = shared.e3.checked_repo_path(Path(config['evidence_root'])/'outputs/icra2027'/config['freeze_id'],
                                     'candidate freeze', kind='dir')
    contract = json.loads((root/'contract/freeze_manifest.json').read_text())
    snapshot = shared.materializer._require_clean_code_snapshot()
    if screen.CODE_ROOT != CODE or Path(screen.__file__).resolve().parents[2] != CODE:
        raise ValueError('candidate implementation differs from launcher source')
    shared.validate_inputs(config, root, contract, snapshot, config_path)
    directory = shared.e3.checked_repo_path(root/'harness', 'candidate descriptor output', must_exist=False, kind='dir')
    directory.mkdir(exist_ok=True)
    descriptor = directory/'automatic_scene_descriptor.json'
    payload = shared.materializer._json_bytes(config['automatic_scene_descriptor'])
    if descriptor.exists():
        if descriptor.is_symlink() or descriptor.read_bytes() != payload:
            raise ValueError('published candidate descriptor changed')
    else:
        shared.materializer._write_inside(descriptor, payload)
    population = screen.automatic_candidate_population(e3_root=config['e3_root'], scene_id=config['scene_id'],
                                                      automatic_scene_descriptor=descriptor)
    if type(config['planned_semantic_pairs']) is not int or population['counts']['semantic_pairs'] != config['planned_semantic_pairs']:
        raise ValueError('declared semantic pair denominator changed')
    if (type(config.get('planned_reset_cells')) is not int
            or config['planned_reset_cells'] != population['counts']['semantic_pairs']*len(screen.POLICIES)*screen.EPISODES):
        raise ValueError('declared reset-cell denominator changed')
    from robo.eval import e4_camera_scorer_gate as camera
    camera._menagerie_snapshot(Path(config['menagerie']['root']), config['menagerie']['commit'])
    if phase != 'preflight':
        host = socket.gethostname().split('.')[0].lower()
        if host != 'hala' and not host.startswith(('gcp', 'sof1')):
            raise ValueError('candidate CPU job is outside allowed hosts')
    if phase == 'prepare':
        result = screen.prepare_automatic_candidates(screen_id=config['freeze_id'], scene_id=config['scene_id'],
            e3_root=config['e3_root'], automatic_scene_descriptor=descriptor,
            expected_commit=snapshot['commit'], export=True)
    elif phase == 'tasks':
        result = screen.prepare_automatic_task_suites(screen_id=config['freeze_id'], scene_id=config['scene_id'],
            expected_commit=snapshot['commit'])
    elif phase == 'qualify':
        result = screen.qualify_scene(screen_id=config['freeze_id'], scene_id=config['scene_id'],
            expected_commit=snapshot['commit'], menagerie_root=config['menagerie']['root'],
            expected_menagerie_commit=config['menagerie']['commit'], automatic_population=True)
    else:
        result = dict(population, code_commit=snapshot['commit'], phase=phase)
    path = directory/(phase+'_report.json')
    shared.materializer._write_inside(path, shared.materializer._json_bytes(result))
    print(json.dumps({'report':str(path), 'sha256':shared.e3.sha256_file(path),
        'planned_semantic_pairs':config['planned_semantic_pairs'], 'paper_ready':False}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--phase', choices=['preflight','prepare','tasks','qualify'], required=True)
    args = parser.parse_args();run(args.config, args.phase)


if __name__ == '__main__':
    main()
