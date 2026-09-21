# Scene media capture

Run these entry points with the compatible GPU environment from `run/env.sh`.
They require existing scene assets and do not install datasets or checkpoints.

| Tool | Output |
| --- | --- |
| `capture_green_bottle.py` | Original/moved GS images, scripted robot push, projectile video |
| `record_shooting_trails.py` | Two-shot MuJoCo trajectory and independently recorded contacts |
| `render_shooting_trails.py` | GS video with colored curling wakes, impact sparks, clean video and PNGs |

For the refined `fb5a96b1a2` spray bottle, use the existing holding-capture asset
folder (geometry JSON, refined-scene XML and meshes, camera, removal indices,
refined bottle Gaussians and table-fill Gaussians):

```bash
python tools/demos/record_shooting_trails.py --assets "$HOLDING_ASSETS" --out "$NEW_ROLLOUT"
python tools/demos/render_shooting_trails.py --assets "$HOLDING_ASSETS" --rollout "$NEW_ROLLOUT" --out "$NEW_MEDIA"
```

The recorder requires two distinct projectile–bottle contacts before it succeeds.
Projectiles are 30 mm collision spheres of 80 g, launched at 5 m/s by default.
Predictive aiming selects their initial velocities; the bottle is not teleported.
The renderer reads the saved model and states without rerunning capture code.
Its first 160 frames play at 0.25× speed and the remainder at real time, at 30 FPS.
`--preview` renders selected PNGs before a full export.

`physicalview/projectile_effects.py` generates world-space GS particles from the
recorded trajectory. The luminous projectile cores follow measured positions;
curling wisps and contact sparks are decorative effects, not collision geometry.
They share the room's Gaussian depth compositing. No frontend geometry is added.

These captures use GT-assisted collision geometry, estimated mass/friction, and
a LaMa-colored GS tabletop fill. The original partial GS bottle can expose missing
surfaces when it rotates. The clips are demonstrations, not reconstruction or
simulation accuracy benchmarks. This pipeline does not alter a live viewer session.
