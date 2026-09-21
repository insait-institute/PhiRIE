"""One-time offline conversion: URDF -> OmniGibson USD dataset, for every
object across every task in the Tier-1 manifest (both the "ours" CoACD-collision
URDF and the "baseline" convex-hull URDF).

import_og_asset_from_urdf() asserts zero scenes exist in the simulator (it's
designed as an offline dataset-prep step, not something to interleave with an
active scene) - so this MUST run standalone, once, before any of
import_and_run.py's Scene()/DatasetObject() calls. Run inside `behavior1k`:

  OMNIGIBSON_HEADLESS=1 python convert_assets.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "outputs" / "omnigibson_export" / "manifest.json"
DATASET_NAME = "phiroom_custom"


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())

    import omnigibson as og
    from omnigibson.utils.asset_conversion_utils import import_og_asset_from_urdf
    from omnigibson.utils.asset_utils import get_dataset_path

    import headless_stubs
    headless_stubs.apply()

    og.launch()
    assert len(og.sim.scenes) == 0

    done = set()
    n_converted = 0
    for task_name, task in manifest["tasks"].items():
        for o in task["objects"]:
            for arm, urdf_key, suffix in (("ours", "urdf", ""), ("baseline", "urdf_baseline", "_bl")):
                model_id = (o["model_id"] + suffix)[:42]
                key = (o["category"], model_id)
                if key in done:
                    continue
                usd_path = (Path(get_dataset_path(DATASET_NAME)) / "objects" / o["category"]
                            / model_id / "usd" / f"{model_id}.usd")
                if usd_path.exists():
                    print(f"[convert] {task_name}/{arm} {o['category']}/{model_id}: "
                          f"already converted, skipping")
                    done.add(key)
                    continue
                urdf_rel = o[urdf_key]
                import_og_asset_from_urdf(
                    category=o["category"], model=model_id, dataset_name=DATASET_NAME,
                    urdf_path=str(ROOT / urdf_rel), collision_method=None, overwrite=True,
                )
                done.add(key)
                n_converted += 1
                print(f"[convert] {task_name}/{arm} {o['category']}/{model_id}: converted")

    print(f"[convert] done - {n_converted} newly converted, {len(done)} total unique (category, model)")


if __name__ == "__main__":
    main()
