"""Curated scene demo setup; segmentation and physics remain server-side."""
from __future__ import annotations

GREEN_BOTTLE = {
    'id': 'fb5a96b1a2-green-bottle',
    'scene': 'fb5a96b1a2',
    'label': 'Green spray bottle',
    'camera': 'DSC03413.JPG',
    'box': [878/1752, 824/1168, 954/1752, 1025/1168],
}


def complete_fragments(demo, mask, depth, w2c, K, unassigned):
    """Join unprepared partial clicks fully covered by the fresh bottle mask.

    Existing physical bodies and discovered proposals are never reassigned.
    The old selection files are retained and referenced by the new receipt.
    """
    import torch
    from physicalview.phiview_selection import lift_mask

    scene = demo.scene
    candidates = lift_mask(scene.raw['means'], torch.zeros_like(scene.labels), mask, depth, w2c, K)
    visible = torch.zeros_like(scene.labels, dtype=torch.bool)
    visible[candidates] = True
    parts, supersedes = [unassigned], []
    for name, rec in demo.state.objects.items():
        indices = scene.indices[name]
        if (not rec.meta.get('interactive') or name in demo.physics.available
                or not rec.meta.get('selection_work') or not len(indices)):
            continue
        if float(visible[indices].float().mean()) >= .9:
            parts.append(indices)
            supersedes.append({'name': name, 'selection_work': rec.meta['selection_work']})
    return torch.unique(torch.cat(parts)), supersedes


class DemoPreparation:
    def __init__(self, demo, enabled=False):
        self.demo = demo
        self.preset = GREEN_BOTTLE if demo.state.result_set.splat_ply.stem == GREEN_BOTTLE['scene'] else None
        self.status = {'available': self.preset is not None, 'state': 'pending' if enabled and self.preset else 'idle'}

    def cancel(self):
        if self.status['state'] in ('pending', 'segmenting'):
            self.status.update(state='cancelled', message='Demo preparation cancelled.')

    def start(self):
        d, p = self.demo, self.preset
        if p is None:
            raise ValueError('This scene has no automatic object preset')
        if d.click_selection.busy or (d.pipeline_thread and d.pipeline_thread.is_alive()):
            raise ValueError('Wait for the current selection or scene build')
        if d.physics.robot:
            raise ValueError('Prepare the demo object before placing a robot')
        d.policy.stop()
        d.physics.running = False
        d.execute({'op': 'camera', 'name': p['camera']})
        d.wh = d.native_wh
        d.mode = 'original'
        d.selected = None
        d.highlight = False
        d.keys = []; d.look[:] = 0
        existing = next((n for n, r in d.state.objects.items() if r.meta.get('demo_preset') == p['id']), None)
        if existing:
            self.ready(existing)
            return
        d.render()
        from physicalview.phiview_box import pixel_box
        box = pixel_box(p['box'], d.frames[d.frame_id][0].shape)
        d.click_selection.start(d.frame_id, (box[0]+box[2])//2, (box[1]+box[3])//2, box=box, preset=p)
        self.status.update(state='segmenting', message='Preparing the green bottle…')

    def ready(self, name):
        d = self.demo
        d.select_known(name)
        d.execute({'op': 'enable'})
        d.physics.running = False
        self.status.update(state='ready', object=name, message='Green bottle ready. Choose Fall, Throw, or Place robot.')

    def advance(self):
        try:
            if self.status['state'] == 'pending':
                self.start()
            elif self.status['state'] == 'segmenting':
                selection = self.demo.click_selection.status
                if selection['state'] == 'failed':
                    self.status.update(state='failed', message=selection['message'])
                elif not self.demo.click_selection.busy and selection['state'] == 'selected':
                    name = selection['object']
                    if self.demo.state.objects[name].meta.get('demo_preset') == self.preset['id']:
                        self.ready(name)
        except Exception as exc:
            self.status.update(state='failed', message=str(exc))
