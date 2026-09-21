"""Selection cancellation and repeatable scene presets."""
import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from physicalview.phiview import Demo
from physicalview.phiview_demo import DemoPreparation, complete_fragments
from physicalview.phiview_selection import ClickSelection, object_directory, restore_objects


def test_deselect_clears_selection_and_pin_without_disabling_objects():
    d = Demo.__new__(Demo)
    d.selected = 'obj_01'
    d.mode = 'clean_selected'
    d.pinned_frame_id = 7
    d.keys, d.look = ['w'], np.ones(2)
    d.physics = SimpleNamespace(enabled={'obj_01'}, data=SimpleNamespace(time=0))
    d.click_selection = ClickSelection(d)
    d.click_selection.pending = object()
    d.preparation = SimpleNamespace(cancel=lambda: None)
    d.audit = io.StringIO()
    d.execute({'op': 'deselect'})
    assert d.selected is None and d.pinned_frame_id is None
    assert d.click_selection.pending is None and d.click_selection.status['state'] == 'idle'
    assert d.mode == 'original' and not d.keys and not np.any(d.look)
    assert d.physics.enabled == {'obj_01'}


@pytest.mark.parametrize('fail', [False, True])
def test_deselect_rejects_late_segmentation_result_and_error(tmp_path, monkeypatch, fail):
    started, release = threading.Event(), threading.Event()
    def run(*args, **kwargs):
        started.set()
        assert release.wait(5)
        if fail:
            raise RuntimeError('late GPU failure')
    monkeypatch.setattr('physicalview.phiview_selection.subprocess.run', run)
    d = SimpleNamespace(out=tmp_path, pipeline_thread=None, physics=SimpleNamespace(running=False),
        frames={7: (None, np.ones((4, 4)), np.eye(4), np.eye(3))},
        frame_images={7: np.zeros((4, 4, 3), np.uint8)},
        config=SimpleNamespace(interpreter=lambda _: '/unused/python'))
    selection = ClickSelection(d)
    selection.start(7, 2, 2)
    assert started.wait(5)
    selection.cancel()
    release.set(); selection.thread.join(5)
    assert not selection.busy and selection.pending is None
    assert selection.status == {'state': 'idle', 'message': 'Selection cleared.'}


def test_demo_fragment_completion_protects_other_objects():
    torch = pytest.importorskip('torch')
    points = torch.tensor([[2., 2., 1.], [3., 3., 1.], [4., 4., 1.], [5., 5., 1.],
                           [6., 6., 1.], [0., 0., 1.], [7., 7., 1.]])
    names = ['partial', 'physical', 'proposal', 'spanning']
    indices = dict(zip(names, [torch.tensor([0, 1]), torch.tensor([2]), torch.tensor([3]), torch.tensor([4, 5])]))
    objects = {n: SimpleNamespace(meta={'interactive': n != 'proposal', 'selection_work': '/'+n}) for n in names}
    d = SimpleNamespace(scene=SimpleNamespace(raw={'means': points}, labels=torch.tensor([1, 1, 2, 3, 4, 4, 0]), indices=indices),
                        state=SimpleNamespace(objects=objects), physics=SimpleNamespace(available={'physical'}))
    mask = np.zeros((10, 10), bool); mask[2:8, 2:8] = True
    result, old = complete_fragments(d, mask, np.ones((10, 10)), np.eye(4), np.eye(3), torch.tensor([6]))
    assert result.tolist() == [0, 1, 6]
    assert old == [{'name': 'partial', 'selection_work': '/partial'}]


def test_superseded_clicks_restore_by_identity_and_keep_files(tmp_path):
    rs = SimpleNamespace(splat_ply=Path('/scene.ply'), out_dir=tmp_path/'build')
    state = SimpleNamespace(result_set=rs, objects={})
    old_receipts = [('obj_00', '/old'), ('obj_01', '/unrelated')]
    for i, (name, receipt) in enumerate(old_receipts + [('obj_02', '/complete')]):
        directory = object_directory(state, tmp_path, name); directory.mkdir(parents=True)
        meta = {'name': name, 'index': i, 'label': name, 'aabb': [[0, 0, 0], [1, 1, 1]],
                'scene_splat': '/scene.ply', 'scene_build': str(rs.out_dir), 'selection_work': receipt}
        if i == 2:
            meta['supersedes'] = [{'name': 'obj_00', 'selection_work': '/old'},
                                  {'name': 'obj_01', 'selection_work': '/wrong'}]
        (directory/'selection.json').write_text(json.dumps(meta))
    restore_objects(state, tmp_path)
    assert set(state.objects) == {'obj_01', 'obj_02'}
    assert (object_directory(state, tmp_path, 'obj_00')/'selection.json').exists()


def test_preset_is_scene_specific_and_deselect_does_not_restart_it():
    d = SimpleNamespace(state=SimpleNamespace(result_set=SimpleNamespace(splat_ply=Path('/other.ply'))))
    assert DemoPreparation(d, enabled=True).status == {'available': False, 'state': 'idle'}
    d.state.result_set.splat_ply = Path('/fb5a96b1a2.ply')
    preset = DemoPreparation(d, enabled=True)
    assert preset.status['state'] == 'pending'
    preset.cancel(); preset.advance()
    assert preset.status['state'] == 'cancelled'
