"""Studio UI panels. Each module exposes ``build(ctx: physicalview.app.Context)``.

Panels must:
  * create all GUI elements inside the tab context they are called from;
  * subscribe to ctx.events topics instead of importing other panels;
  * keep long work off the viser thread (use ctx.jobs for pipeline stages and a daemon
    thread for physics/policy loops);
  * be safe to build without a GPU (degrade, never crash).
"""
