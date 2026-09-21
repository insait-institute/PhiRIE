"""Actual-GPU demo checks and continuous footage; failures are retained and raised."""
import json
import time
import traceback
import numpy as np


def check_demo(demo):
    from physicalview.phiview import save_json
    import imageio.v2 as imageio
    out = demo.out/'validation'; out.mkdir(exist_ok=True)
    results = {}
    def check(name, fn):
        t0 = time.monotonic()
        try:
            info = fn()
            results[name] = {'status': 'PASS', 'seconds': time.monotonic()-t0, 'evidence': info}
            print(f'[PhiView] PASS {name}: {info}', flush=True)
        except Exception as exc:
            results[name] = {'status': 'FAIL', 'error': str(exc), 'traceback': traceback.format_exc()}
            print(f'[PhiView] FAIL {name}: {exc}', flush=True)
        save_json(out/'checks.json', results)

    def original():
        demo.highlight = False; demo.mode = 'original'
        img = demo.render()
        assert img.shape == (demo.wh[1], demo.wh[0], 3) and img.std() > 5
        assert len(demo.scene.raw['means']) == demo.scene.count
        (out/'original.jpg').write_bytes(demo.jpeg)
        from agents.core.common import render_view
        w2c, K = demo.camera.matrices(demo.wh)
        ref, _, _ = render_view(demo.scene.raw, w2c, K, *demo.wh)
        ref = (np.clip(ref, 0, 1)*255+.5).astype(np.uint8)
        assert np.array_equal(img, ref), 'Original view changes the reference Gaussian render'
        return {'gaussians': demo.scene.count, 'wh': demo.wh, 'reference_pixel_exact': True}
    check('original_full_gaussians', original)

    def native():
        old = demo.wh; demo.wh = demo.native_wh
        try:
            image = demo.render()
            assert image.shape[:2] == (demo.native_wh[1], demo.native_wh[0])
            (out/'native-resolution.jpg').write_bytes(demo.jpeg)
            return {'resolution': demo.native_wh}
        finally:
            demo.wh = old
    check('native_resolution', native)

    def selection():
        demo.render()
        mask = demo.frames[demo.frame_id][0]
        visible = [n for n in sorted(demo.physics.available) if np.sum(mask == demo.scene.ids[n]) > 40]
        assert visible, 'No visible simulatable object in starting view'
        # Prefer a small bottle for physical interactions, without filtering task outcomes.
        name = next((n for n in visible if 'bottle' in demo.state.objects[n].label), visible[0])
        ys, xs = np.where(mask == demo.scene.ids[name]); mid = len(xs)//2
        demo.execute({'op': 'pick', 'frame': demo.frame_id,
                      'x': (float(xs[mid])+.5)/mask.shape[1], 'y': (float(ys[mid])+.5)/mask.shape[0]})
        assert demo.selected == name
        demo.highlight = True; image = demo.render()
        (out/'selected-highlight.jpg').write_bytes(demo.jpeg)
        return {'selected': name, 'visible_objects': visible, 'server_pixel_pick': True}
    check('automatic_highlights_and_click', selection)

    def clean():
        name = demo.selected_required()
        demo.execute({'op': 'view', 'mode': 'clean_selected'}); demo.render()
        mask = demo.frames[demo.frame_id][0]
        assert not np.any(mask == demo.scene.ids[name])
        (out/'clean-selected.jpg').write_bytes(demo.jpeg)
        demo.execute({'op': 'view', 'mode': 'clean_all'}); demo.render()
        mask = demo.frames[demo.frame_id][0]
        for n in demo.physics.available:
            assert not np.any(mask == demo.scene.ids[n]), f'{n} survives clean-all'
        (out/'clean-all.jpg').write_bytes(demo.jpeg)
        return {'removed_selected': name, 'removed_all': len(demo.physics.available)}
    check('clean_selected_and_all', clean)

    def physics():
        n = demo.selected_required()
        demo.execute({'op': 'enable', 'all': True})
        demo.execute({'op': 'fall'})
        p0 = demo.physics.data.xpos[demo.physics.model.body(n).id].copy()
        for _ in range(15):
            demo.physics.step(1/30)
        p1 = demo.physics.data.xpos[demo.physics.model.body(n).id].copy()
        assert np.linalg.norm(p1-p0) > .01 and p1[2] < p0[2]
        demo.physics.reset()
        demo.execute({'op': 'friction_value', 'value': .4}); demo.execute({'op': 'friction'})
        qa, da = demo.physics.addresses(n)
        assert abs(demo.physics.data.qvel[da]-.7) < 1e-9
        for _ in range(15):
            demo.physics.step(1/30)
        assert np.isfinite(demo.physics.data.qpos).all()
        return {'fall_displacement_m': (p1-p0).tolist(), 'parameters': demo.physics.parameters(n)}
    check('fall_friction_parameters', physics)

    def throw_video():
        demo.physics.reset(); demo.execute({'op': 'throw', 'strength': 2.})
        n = demo.selected_required(); p0 = demo.physics.data.xpos[demo.physics.model.body(n).id].copy()
        writer = imageio.get_writer(out/'throw.mp4', fps=15, codec='libx264', macro_block_size=1)
        try:
            for _ in range(45):
                demo.physics.step(1/15); writer.append_data(demo.render())
        finally:
            writer.close()
        p1 = demo.physics.data.xpos[demo.physics.model.body(n).id].copy()
        assert np.linalg.norm(p1-p0) > .1
        return {'displacement_m': float(np.linalg.norm(p1-p0)), 'frames': 45}
    check('throw_video', throw_video)

    def shooting():
        demo.physics.reset()
        p = demo.physics.data.xpos[demo.physics.model.body(demo.selected).id].copy()
        origin = p+np.array([.30, 0, .03])
        demo.physics.shoot(origin, p-origin, 3.)
        for _ in range(30):
            demo.physics.step(1/60)
        demo.render(); (out/'shooting.jpg').write_bytes(demo.jpeg)
        assert demo.physics.events, 'No projectile contact recorded'
        return {'projectiles': demo.physics.projectile, 'contacts': demo.physics.events}
    check('physical_projectile_contacts', shooting)

    def alternatives():
        options = [(n, s) for n, rec in demo.state.objects.items() for s, p in rec.proposals.items()
                   if p.gs_ply and p.aligned and p.aligned.get('T') and not p.aligned.get('rejected')]
        assert options, 'No registered alternative is available'
        n, source = options[0]
        demo.execute({'op': 'select', 'object': n})
        demo.execute({'op': 'variant', 'source': source}); demo.render()
        (out/'generated-alternative.jpg').write_bytes(demo.jpeg)
        demo.execute({'op': 'variant', 'source': 'original'})
        return {'object': n, 'source': source, 'registered_options': len(options)}
    check('generated_alternative', alternatives)

    def robot():
        candidates = [n for n in demo.physics.available if 'bottle' in demo.state.objects[n].label]
        n = sorted(candidates)[0] if candidates else demo.selected
        demo.physics.enabled = {n}
        demo.execute({'op': 'select', 'object': n})
        demo.execute({'op': 'robot_command', 'command': 'reach the object'})
        q0 = demo.physics.data.qpos[demo.physics.robot['qadr']].copy()
        overlay_pixels = 0
        writer = imageio.get_writer(out/'robot-reach.mp4', fps=15, codec='libx264', macro_block_size=1)
        try:
            for _ in range(120):
                demo.physics.step(1/30)
                if _ % 2 == 0:
                    writer.append_data(demo.render())
                    overlay_pixels = max(overlay_pixels, int(np.sum(demo.physics.overlay_mask)))
        finally:
            writer.close()
        q1 = demo.physics.data.qpos[demo.physics.robot['qadr']].copy()
        assert np.linalg.norm(q1-q0) > .01
        assert overlay_pixels > 200, 'Robot moves but is not visible in the streamed image'
        return {'joint_motion_norm': float(np.linalg.norm(q1-q0)), 'robot': demo.physics.robot_status,
                'robot_overlay_pixels': overlay_pixels, 'task_success_claimed': False}
    check('robot_placement_command_and_motion', robot)

    demo.physics.reset(); demo.mode = 'original'; demo.highlight = True
    demo.execute({'op': 'camera'}); demo.render()
    save_json(out/'final-status.json', demo.status())
    if any(row['status'] == 'FAIL' for row in results.values()):
        raise RuntimeError(f'PhiView validation failed: {out / "checks.json"}')
