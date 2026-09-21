"""Stage 6: sim-ready physics annotation per object.

- CoACD convex decomposition of the canonical sim mesh -> collision/part_*.obj
- Physics parameters (mass/friction/restitution) from a local VLM
  (Qwen2.5-7B-Instruct stands in for the paper's Gemini V_scene slot),
  with a density-table fallback on parse failure.
- Writes object.urdf (visual + N collision meshes, canonical frame, scale
  baked into the URDF <mesh scale>), spawned later at (t, R) from s5.
"""
import json
import os
import re

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch  # noqa: F401  MUST import before coacd runs: importing torch
#              after run_coacd() segfaults (bundled-libgomp clash, verified)

from agents.core import common as C

MAX_PARTS = 16
COACD_THRESHOLD = 0.05


def coacd_parts(odir, world_max_dim=0.5):
    import coacd
    import trimesh
    tm = trimesh.load(odir / "mesh_sim.ply", process=False)
    mesh = coacd.Mesh(np.asarray(tm.vertices), np.asarray(tm.faces))
    budget = MAX_PARTS if world_max_dim <= 1.2 else MAX_PARTS * 2
    parts = coacd.run_coacd(mesh, threshold=COACD_THRESHOLD,
                            max_convex_hull=budget)
    cdir = odir / "collision"
    cdir.mkdir(exist_ok=True)
    files = []
    for k, (pv, pf) in enumerate(parts):
        f = cdir / f"part_{k:02d}.obj"
        trimesh.Trimesh(pv, pf, process=False).export(f)
        files.append(f.name)
    return files


def annotate_physics(items):
    """items: [{label, dims}] -> [{mass_kg, friction, restitution, source}]"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen2.5-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16).to("cuda").eval()

    out = []
    for it in items:
        dims = ", ".join(f"{d:.3f}" for d in it["dims"])
        msgs = [
            {"role": "system", "content":
             "You annotate physical properties of household objects for a "
             "rigid-body robotics simulator. Answer with a single JSON object "
             "and nothing else."},
            {"role": "user", "content":
             f"Object: '{it['label']}' (as found on an office desk). "
             f"Axis-aligned bounding box: [{dims}] meters. Estimate typical "
             "values: {\"mass_kg\": float, \"friction\": float (0.2-1.0), "
             "\"restitution\": float (0.0-0.5)}"},
        ]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=120, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        rec = None
        m = re.search(r"\{.*?\}", text, re.S)
        if m:
            try:
                j = json.loads(m.group(0))
                rec = {"mass_kg": float(j["mass_kg"]),
                       "friction": float(j["friction"]),
                       "restitution": float(j.get("restitution", 0.1)),
                       "source": "qwen2.5-7b"}
            except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                rec = None
        in_range = rec is not None and (0.005 <= rec["mass_kg"] <= 20.0
                                        and 0.2 <= rec["friction"] <= 1.0
                                        and 0.0 <= rec["restitution"] <= 0.5)
        if not in_range:
            mass, fric = C.FALLBACK_PHYSICS.get(it["label"], (0.3, 0.5))
            rec = {"mass_kg": mass, "friction": fric, "restitution": 0.1,
                   "source": "fallback" if rec is None else "fallback-range"}
        out.append(rec)
        print(f"[s6] physics {it['label']}: {rec}")
    del model
    torch.cuda.empty_cache()
    return out


def write_urdf(odir, name, scale, dims_world, phys, collision_files, com):
    m = phys["mass_kg"]
    dx, dy, dz = np.maximum(dims_world, 1e-3)
    ixx = m / 12 * (dy ** 2 + dz ** 2)
    iyy = m / 12 * (dx ** 2 + dz ** 2)
    izz = m / 12 * (dx ** 2 + dy ** 2)
    s = f"{scale:.6f} {scale:.6f} {scale:.6f}"
    coll = "\n".join(
        f'    <collision><geometry>'
        f'<mesh filename="collision/{f}" scale="{s}"/></geometry></collision>'
        for f in collision_files)
    urdf = f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="base">
    <inertial>
      <origin xyz="{com[0]:.5f} {com[1]:.5f} {com[2]:.5f}"/>
      <mass value="{m:.4f}"/>
      <inertia ixx="{ixx:.6e}" iyy="{iyy:.6e}" izz="{izz:.6e}"
               ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual><geometry>
      <mesh filename="mesh_sim.obj" scale="{s}"/></geometry></visual>
{coll}
  </link>
</robot>
"""
    (odir / "object.urdf").write_text(urdf)


def main():
    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    import trimesh

    todo = []
    for meta in objects:
        odir = C.OUT / "objects" / f"obj_{meta['index']:02d}"
        aligned = json.loads((odir / "aligned.json").read_text())
        if aligned.get("rejected"):
            print(f"[s6] {odir.name}: skip ({aligned['rejected']})")
            continue
        tm = trimesh.load(odir / "mesh_sim.ply", process=False)
        ext_canon = tm.vertices.max(axis=0) - tm.vertices.min(axis=0)
        s = aligned["scale"]
        files = coacd_parts(odir, world_max_dim=float((ext_canon * s).max()))
        todo.append({"meta": meta, "odir": odir, "scale": s,
                     "label": meta["label"], "dims": (ext_canon * s).tolist(),
                     "com": tm.vertices.mean(axis=0) * s,
                     "collision": files})
        print(f"[s6] {odir.name} {meta['label']}: {len(files)} convex parts")

    phys = annotate_physics(todo)
    for it, ph in zip(todo, phys):
        write_urdf(it["odir"], f"obj_{it['meta']['index']:02d}", it["scale"],
                   np.array(it["dims"]), ph, it["collision"], it["com"])
        C.save_json(it["odir"] / "physics.json", ph)
    print(f"[s6] wrote {len(todo)} URDFs")


if __name__ == "__main__":
    main()
