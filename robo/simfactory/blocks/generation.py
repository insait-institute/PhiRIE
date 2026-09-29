"""Generation adapters over the stable backend modules."""
from __future__ import annotations

from robo.simfactory.registry import register
from robo.simfactory.runner import sh

# Single-method runs generate with ONE model; 'hybrid' generates with every
# candidate then keeps the best-registering asset per object (sym_score).

_GEN_ENV = {"trellis": ("agents.models.s4_trellis", "venv"),
            "trellis2": ("agents.models.s4_trellis2", "trellis2"),
            "sam3d": ("agents.models.s4_sam3d", "sam3d"),
            "reconviagen": ("agents.models.s4_reconviagen", "venv")}


def _gen_single(name):
    def run(ctx, entry):
        module, env_kind = _GEN_ENV[name]
        sh(ctx, module, env_kind=env_kind)
        sh(ctx, "agents.assets.factory_align")
        sh(ctx, "agents.assets.s6_physics")
    return run


for _n, _h in [("trellis", "TRELLIS single-view image-to-3D"),
               ("trellis2", "TRELLIS.2-4B O-Voxel image-to-3D"),
               ("sam3d", "SAM 3D Objects (won the c50 pilot 3/3)"),
               ("reconviagen", "ReconViaGen VGGT-conditioned multi-view")]:
    register("generation", _n, help=_h)(_gen_single(_n))


@register("generation", "hybrid",
          help="generate with all candidates, keep best per object by "
               "registration sym_score (candidates: [...] in config)")
def gen_hybrid(ctx, entry):
    cands = entry.get("candidates",
                      ctx.opt("candidates", ["trellis", "reconviagen"]))
    alias = {"reconviagen": "rvg"}
    for c in cands:
        module, env_kind = _GEN_ENV[c]
        sh(ctx, module, env_kind=env_kind)
    sh(ctx, "agents.assets.factory_align")
    sh(ctx, "agents.assets.factory_hybrid", extra_env={
        "SIMANY_HYBRID_CANDIDATES": ",".join(alias.get(c, c)
                                             for c in cands)})
    sh(ctx, "agents.assets.s6_physics")
