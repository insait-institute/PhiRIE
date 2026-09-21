import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from robo.eval import native_publication as pub
from robo.eval import native_final_release as release
from robo.eval.native_scale_paper import DISABLED_CLAIMS


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def complete():
    return (dict(planned_units=2400, measured_units=2400, state='COMPLETE_BLOCKS'),
            dict(planned_render_units=86, state='COMPLETE_AVAILABLE_RENDER_ATTEMPTS',
                 ready={str(i): {} for i in range(82)}, metrics={str(i): {} for i in range(82)},
                 failed={str(i): {} for i in range(82, 86)}, metric_failures={}, unattempted=[]),
            dict(planned=1200, terminal=1200, groups=[dict(unmeasured=0)]))


def test_complete_attempts_admit_render_failures_without_exclusion():
    assert pub.closed_matrix_gate(*complete())


@pytest.mark.parametrize('part', ['primary', 'scope', 'warm'])
def test_incomplete_matrix_waits(part):
    primary, warm, scope = complete()
    if part == 'primary': primary.update(measured_units=2399, state='INCOMPLETE')
    elif part == 'scope': scope.update(terminal=1199, groups=[dict(unmeasured=1)])
    else: warm.update(unattempted=['untried'])
    assert not pub.closed_matrix_gate(primary, warm, scope)


@pytest.mark.parametrize('defect', ['population', 'missing_metric', 'overlap', 'missing_failure'])
def test_changed_or_incomplete_fidelity_denominator_rejected(defect):
    primary, warm, scope = complete()
    if defect == 'population': scope['planned'] = 1190
    elif defect == 'missing_metric': warm['metrics'].pop('0')
    elif defect == 'overlap': warm['failed']['0'] = {}
    else: warm['failed'].pop('85')
    with pytest.raises(ValueError): pub.closed_matrix_gate(primary, warm, scope)


def test_wait_for_primary_only_publisher_without_touching_paper(tmp_path):
    assert pub.readiness(dict(primary_release_receipt=str(tmp_path/'absent'))) is None
    put(tmp_path/'blocked', {'state': 'PIPELINE_FAILED'})
    with pytest.raises(ValueError, match='failed'):
        pub.readiness(dict(primary_release_receipt=str(tmp_path/'blocked')))


def command(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True)


def paper_repository(tmp_path):
    paper = tmp_path/'paper'; paper.mkdir()
    command('git', 'init', '-b', 'main', str(paper))
    command('git', '-C', str(paper), 'config', 'user.name', 'Guard test')
    command('git', '-C', str(paper), 'config', 'user.email', 'test@example.invalid')
    (paper/'title.tex').write_text('Approved title\n')
    (paper/'legacy_negative.tex').write_text('Geometry F1 improves while stability drops.\n')
    (paper/'build.sh').write_text('#!/bin/sh\nexit 0\n')
    command('git', '-C', str(paper), 'add', '.')
    command('git', '-C', str(paper), 'commit', '-m', 'Original')
    remote = tmp_path/'remote.git'; command('git', 'init', '--bare', str(remote))
    command('git', '-C', str(paper), 'remote', 'add', 'origin', str(remote))
    command('git', '-C', str(paper), 'push', 'origin', 'main')
    plan = dict(paper_repo=str(paper), paper_remote='origin', paper_base_commit=pub.git(paper, 'rev-parse', 'HEAD'),
                paper_worktree=str(tmp_path/'isolated'), paper_branch='publish', code_commit='frozen-source', qa_python='qa-python')
    return plan, paper


def test_pin_and_old_negative_bytes_are_strict(tmp_path):
    plan, paper = paper_repository(tmp_path)
    pub.check_paper_base(plan)
    preserved = pub.preserve_snapshot(paper)
    (paper/'legacy_negative.tex').write_text('Unsupported positive claim')
    with pytest.raises(ValueError, match='dirty'): pub.check_paper_base(plan)
    with pytest.raises(ValueError, match='changed'): pub.verify_preserved(paper, preserved)


def fake_generation(tmp_path, paper_root):
    tables = tmp_path/'tables'; tables.mkdir()
    source = tables/'native_test_section.tex'; source.write_text('Complete target-only TEST accounting. No preservation claim.\n')
    gates = [dict(claim_id=k, scientific_gate='NOT_RUN', enabled_sentence='', disabled_sentence=s) for k, s, _ in DISABLED_CLAIMS]
    put(tables/'native_paper_claim_gates.json', gates)
    destination = 'paper_sections/06_native_test.tex'
    target = Path(paper_root)/destination; target.parent.mkdir(parents=True); target.write_bytes(source.read_bytes())
    put(Path(paper_root)/'audit/native_test_transfer.json', dict(primary_complete=True,
        files={source.name: dict(destination=destination, sha256=release._sha(source))}))
    manifest = put(tables/'release_manifest.json', dict(planned_units=2400, measured_units=2400,
        artifacts={p.name: release._sha(p) for p in tables.iterdir()}))
    return dict(state='GENERATED', release_manifest=release.bind(manifest))


def test_duplicate_disabled_claim_gate_rejected(tmp_path):
    paper = tmp_path/'paper'; paper.mkdir(); command('git', 'init', str(paper))
    result = fake_generation(tmp_path, paper)
    tables = Path(result['release_manifest']['path']).parent
    gates = pub.read(tables/'native_paper_claim_gates.json'); gates.append(gates[0]);put(tables/'native_paper_claim_gates.json', gates)
    with pytest.raises(ValueError, match='claim'): pub.verify_generated_claims(tables, paper, {})


@pytest.mark.parametrize('qa_pass', [False, True])
def test_publication_pushes_only_after_qa_and_preserves_original_bytes(tmp_path, monkeypatch, qa_pass):
    plan, paper = paper_repository(tmp_path);control=tmp_path/'control';control.mkdir()
    monkeypatch.setattr(pub, 'check_code', lambda plan: tmp_path)
    monkeypatch.setattr(release, 'run_release', lambda *args, paper_root=None: fake_generation(tmp_path, paper_root))
    def demo(plan,tables,control):
        put(control/'native_demo_receipt.json', {'state':'COMPLETE_NATIVE_TRACK'})
        return {'state':'COMPLETE_NATIVE_TRACK'}
    monkeypatch.setattr(pub, 'render_native_demo', demo)
    real_run = subprocess.run
    def run(argv, **kwargs):
        if argv[0] == 'qa-python':
            if not qa_pass: raise subprocess.CalledProcessError(1, argv)
            put(control/'pdf_qa/qa.json', {'status':'PASS'})
            return SimpleNamespace(returncode=0)
        return real_run(argv, **kwargs)
    monkeypatch.setattr(subprocess, 'run', run)
    original = pub.preserve_snapshot(paper)
    if not qa_pass:
        with pytest.raises(subprocess.CalledProcessError) as error:
            pub.publish(plan, ({}, ({}, {})), control)
        audit = pub.preserve_failure(plan, control, error.value)
        assert not audit['push_attempted'] and not audit['remote_published']
        assert 'paper_sections/06_native_test.tex' in audit['preserved_generated_files']
        assert pub.git(paper, 'ls-remote', 'origin', 'refs/heads/main').split()[0] == plan['paper_base_commit']
    else:
        result = pub.publish(plan, ({}, ({}, {})), control)
        assert result['state'] == 'PUBLISHED'
        assert result['paper_commit'] != plan['paper_base_commit']
        assert pub.git(paper, 'rev-parse', 'HEAD') == result['remote_commit']
        pub.verify_preserved(paper, original)


@pytest.mark.parametrize('defect', ['pages', 'fonts', 'undefined', 'overfull', 'placeholder'])
def test_pdf_qa_rejects_actual_publication_gate_failures(tmp_path, monkeypatch, defect):
    class Page:
        def get_text(self): return 'TBD' if defect == 'placeholder' else 'Measured results'
        def get_pixmap(self, **kwargs): return SimpleNamespace(save=lambda path: Path(path).write_bytes(b'png'))
        def get_fonts(self, **kwargs): return [(1, None, None, 'CMR')]
    class Doc:
        def __len__(self): return 9 if defect == 'pages' else 1
        def __iter__(self): return iter([Page()])
        def extract_font(self, xref): return ('font', 'type', 'ext', b'' if defect == 'fonts' else b'font')
    monkeypatch.setitem(sys.modules, 'fitz', SimpleNamespace(open=lambda path: Doc(), Matrix=lambda *args: args))
    for name in ('conference', 'root'):
        (tmp_path/(name+'.pdf')).write_bytes(b'pdf')
        (tmp_path/(name+'.log')).write_text({'undefined':'Reference X undefined', 'overfull':'Overfull \\hbox'}.get(defect, 'Clean'))
    with pytest.raises(ValueError): pub.pdf_qa(tmp_path, tmp_path/'qa')
    assert not (tmp_path/'qa/qa.json').exists()


def test_pinned_code_accepts_main_advancement_but_not_source_edits(tmp_path, monkeypatch):
    plan, repo = paper_repository(tmp_path)
    source = pub.git(repo, 'rev-parse', 'HEAD')
    command('git', '-C', str(repo), 'branch', 'frozen', source)
    (repo/'later-status.txt').write_text('Later main-only status')
    command('git', '-C', str(repo), 'add', '.')
    command('git', '-C', str(repo), 'commit', '-m', 'Later status')
    command('git', '-C', str(repo), 'checkout', 'frozen')
    monkeypatch.setattr(pub, '__file__', str(repo/'robo/eval/native_publication.py'))
    assert pub.check_code({'code_commit': source}) == repo
    (repo/'title.tex').write_text('Unfrozen edit')
    with pytest.raises(ValueError, match='dirty'): pub.check_code({'code_commit': source})


def test_remote_paper_change_blocks_even_clean_original_checkout(tmp_path):
    plan, paper = paper_repository(tmp_path)
    other = tmp_path/'other'
    command('git', '-C', str(paper), 'worktree', 'add', '-b', 'other', str(other))
    (other/'other.txt').write_text('Concurrent editorial work')
    command('git', '-C', str(other), 'add', '.')
    command('git', '-C', str(other), 'commit', '-m', 'Other author')
    command('git', '-C', str(other), 'push', 'origin', 'HEAD:main')
    with pytest.raises(ValueError, match='remote paper'): pub.check_paper_base(plan)


@pytest.mark.parametrize('defect', [None, 'release', 'seam', 'source'])
def test_final_native_demo_requires_same_release_and_existing_qa(tmp_path, monkeypatch, defect):
    tables=tmp_path/'paper_tables';tables.mkdir();put(tables/'release_manifest.json', {'state':'COMPLETE_BLOCKS'})
    control=tmp_path/'control';control.mkdir();source=tmp_path/'original.mp4';source.write_bytes(b'continuous')
    def render(argv, **kwargs):
        assert argv[3]=='native-hero' and kwargs['timeout']==900
        out=tmp_path/'native_demo';out.mkdir()
        for name in ('native_test_90s_1080p.mp4','teaser_30s_1080p.mp4','static_evidence_loop_8s.mp4','poster.png','native_test.srt','B4_repair_continuous.mp4'):
            (out/name).write_bytes(b'generated')
        put(out/'demo_source_manifest.json',dict(state='COMPLETE_TEST_PRESENTATION',release=str(tables.resolve()),
            release_sha256='wrong' if defect=='release' else release._sha(tables/'release_manifest.json'),
            native_track_demo_ready=True,full_demo_ready=False,source_files={str(source):'wrong' if defect=='source' else release._sha(source)}))
        put(out/'qa.json',dict(status='PASS',decoded_all_hero_frames=True,frames=2700,duration_s=90,
            resolution=[1920,1080],loop_seam_exact=defect!='seam',native_track_demo_ready=True))
    monkeypatch.setattr(subprocess,'run',render)
    if defect:
        with pytest.raises(ValueError):pub.render_native_demo({'demo_python':'native-python'},tables,control)
        assert not (control/'native_demo_receipt.json').exists()
    else:
        assert pub.render_native_demo({'demo_python':'native-python'},tables,control)['state']=='COMPLETE_NATIVE_TRACK'
