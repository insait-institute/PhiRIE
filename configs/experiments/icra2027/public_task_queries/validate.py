"""Validate prospective annotations only; never imports an evaluator or reads GT.

Run from any directory: python path/to/public_task_queries/validate.py
"""
import copy
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENES = ["behavior_task0020", "behavior_task0011", "behavior_task0023",
          "behavior_task0027", "behavior_task0045", "behavior_task0002"]
CONDITIONS = ["clean", "mild", "severe"]
FORBIDDEN_KEYS = {"object_id", "object_ids", "role_refs", "benchmark_mapping",
                  "invalid_label", "invalid_reasons", "success", "validity_label",
                  "ground_truth", "gt_pose", "gt_scale", "policy_success"}


def require(test, message):
    if not test:
        raise ValueError(message)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def scan_keys(obj):
    if isinstance(obj, dict):
        require(not FORBIDDEN_KEYS.intersection(obj), "forbidden identity/label key")
        for value in obj.values():
            scan_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            scan_keys(value)


def validate(sources, roster, expansion, verify_source_bytes=True):
    for obj in (sources, roster, expansion):
        scan_keys(obj)
        require(obj["schema_version"] == 1, "unsupported schema version")
    require(roster["paper_ready"] is False, "annotations cannot be paper ready")
    require(roster["scene_ids"] == SCENES, "fixed scene roster changed")
    require(roster["conditions"] == CONDITIONS, "fixed conditions changed")
    require([s["scene_id"] for s in sources["scenes"]] == SCENES, "source scene mismatch")
    # Preserve the frozen declaration while locating its unchanged bytes after
    # the workspace move. Only these two exact roots are approved.
    relocated_root = "${SIMANY_ROOT}/data/recon_scenes/data"
    source_root = sources["public_root"]
    require(source_root == relocated_root, "unapproved source root")
    public_root = Path(os.environ.get("SIMANY_ROOT", HERE.parents[3])) / "data/recon_scenes/data"
    by_id = {}
    for scene in sources["scenes"]:
        require(len(scene["sources"]) == 7, "expected five RGB and two camera files")
        image_dir = public_root / scene["scene_id"] / "dslr" / "resized_undistorted_images"
        images = sorted(image_dir.glob("*.jpg"))
        expected_indices = [round((len(images)-1)*q/4) for q in range(5)]
        require(scene["public_rgb_count"] == len(images), "public image list changed")
        require(scene["selected_indices"] == expected_indices, "selection rule changed")
        rgb_entries = [s for s in scene["sources"] if s["kind"] == "rgb"]
        require([Path(s["relative_path"]).name for s in rgb_entries] ==
                [images[i].name for i in expected_indices], "selected images changed")
        for source in scene["sources"]:
            sid = source["source_id"]
            require(sid not in by_id, "duplicate source ID")
            by_id[sid] = source
            relative = Path(source["relative_path"])
            require(not relative.is_absolute() and ".." not in relative.parts, "unsafe source path")
            prefix = (scene["scene_id"], "dslr")
            require(relative.parts[:2] == prefix, "source scene mismatch")
            require(relative.parts[2] in {"resized_undistorted_images", "colmap", "nerfstudio"}, "source outside public allowlist")
            path = public_root / relative
            require(path.resolve().is_relative_to((public_root / scene["scene_id"] / "dslr").resolve()), "source escaped public tree")
            if verify_source_bytes:
                data = path.read_bytes()
                require(len(data) == source["bytes"], "source byte count changed")
                require(hashlib.sha256(data).hexdigest() == source["sha256"], "source hash changed")
            if source["kind"] == "rgb":
                require(source["visually_inspected"] is True, "uninspected source image")
                require(source["camera_record"] is not None, "missing public camera record")
    require(len(by_id) == 42, "source count changed")
    keys = set()
    counts = Counter()
    frames = defaultdict(set)
    query_map = {}
    for query in roster["queries"]:
        key = (query["scene_id"], query["task_id"])
        require(key not in keys, "duplicate scene/task")
        keys.add(key)
        counts[query["scene_id"]] += 1
        query_map[key] = query
        unhashed = {k:v for k,v in query.items() if k != "query_sha256"}
        require(digest(unhashed) == query["query_sha256"], "query hash changed")
        require(query["conditions"] == CONDITIONS, "query conditions changed")
        require(query["annotation_status"] in {"annotated_public_rgb", "missing_annotation"}, "unknown annotation status")
        if query["annotation_status"] == "missing_annotation":
            require(bool(query["missing_annotation"]), "missing annotation reason absent")
            continue
        require(query["missing_annotation"] is None, "annotated query has missing reason")
        require(query["instruction"].startswith("Move the "), "task instruction absent")
        roles = query["roles"]
        require("manipulated_object" in roles and bool({"receptacle", "target"}.intersection(roles)), "required semantic roles absent")
        require(set(query["required_unresolved_roles"]) == {"robot_frame", "policy_cameras", "support_contact", "swept_workspace_obstacles", "one_hop_visibility"}, "missing unresolved evidence declaration")
        frame = query["source_state_anchor"]
        frames[query["scene_id"]].add(frame)
        require(frame in by_id and by_id[frame]["kind"] == "rgb", "unknown reference frame")
        require(frame.startswith(query["scene_id"] + "."), "cross-scene reference")
        width, height = by_id[frame]["width"], by_id[frame]["height"]
        for role in roles.values():
            require(role["source_image_id"] == frame, "multiple states mixed in one query")
            require(role["grounding_status"] == "image_region_annotated_3d_unresolved", "unsupported 3D grounding assertion")
            x0,y0,x1,y1 = role["bbox_xyxy_pixels"]
            require(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, "box outside image")
            expected = [round(v/(width if j%2 == 0 else height),8) for j,v in enumerate([x0,y0,x1,y1])]
            require(role["bbox_xyxy_normalized"] == expected, "normalized box mismatch")
            if "polygon_xy_pixels" in role:
                points = role["polygon_xy_pixels"]
                require(len(points) >= 3 and all(x0 <= x <= x1 and y0 <= y <= y1 for x,y in points), "bad polygon bounds")
                area2 = sum(points[i][0]*points[(i+1)%len(points)][1] - points[(i+1)%len(points)][0]*points[i][1] for i in range(len(points)))
                require(abs(area2) > 0, "zero-area target polygon")
                require(role["polygon_xy_normalized"] == [[round(x/width,8),round(y/height,8)] for x,y in points], "normalized polygon mismatch")
    require(counts == Counter({scene:4 for scene in SCENES}), "expected 24 query slots")
    require(all(len(frames[s]) == 1 for s in SCENES), "multiple source states per scene")
    expected_rows = {(scene,task,c) for scene,task in keys for c in CONDITIONS}
    actual_rows = set()
    for row in expansion["rows"]:
        require(set(row) == {"scene_id", "task_id", "condition_id", "query_sha256", "query_ref", "annotation_status", "experiment_status"}, "condition row may not override query content")
        key = (row["scene_id"], row["task_id"])
        row_key = (*key, row["condition_id"])
        require(row_key not in actual_rows, "duplicate condition row")
        actual_rows.add(row_key)
        query = query_map[key]
        require(row["query_sha256"] == query["query_sha256"], "condition query hash mismatch")
        require(row["query_ref"] == f"queries.json#{key[0]}/{key[1]}", "query reference mismatch")
        require(row["experiment_status"] == "not_run", "annotations cannot claim experiment completion")
        require(row["annotation_status"] == query["annotation_status"], "condition annotation status mismatch")
    require(actual_rows == expected_rows and len(expansion["rows"]) == 72, "condition expansion incomplete")
    return {"integrity": "PASS", "queries": len(keys), "condition_rows": len(actual_rows),
            "public_sources_hashed": len(by_id), "rgb_images_viewed": 30,
            "reference_frames_per_scene": {s:len(frames[s]) for s in SCENES},
            "missing_annotations": sum(q["annotation_status"] == "missing_annotation" for q in roster["queries"]),
            "experiment_status": "not_run", "paper_ready": False}


def negative_tests(sources, roster, expansion):
    cases = []
    r = copy.deepcopy(roster); r["queries"][0]["instruction"] += " tampered"; cases.append((r, expansion))
    e = copy.deepcopy(expansion); e["rows"].pop(); cases.append((roster, e))
    e = copy.deepcopy(expansion); e["rows"][0]["instruction"] = "condition-specific task"; cases.append((roster, e))
    r = copy.deepcopy(roster); r["queries"][0]["roles"]["manipulated_object"]["object_id"] = "hidden"; cases.append((r, expansion))
    r = copy.deepcopy(roster); q = r["queries"][0]; q["roles"]["manipulated_object"]["bbox_xyxy_pixels"][0] = -1
    q["query_sha256"] = digest({k:v for k,v in q.items() if k != "query_sha256"}); cases.append((r, expansion))
    for index, (r,e) in enumerate(cases):
        try:
            validate(sources, r, e, verify_source_bytes=False)
        except ValueError:
            continue
        raise AssertionError(f"negative test {index} unexpectedly accepted invalid input")
    return len(cases)


if __name__ == "__main__":
    sources, roster, expansion = [json.loads((HERE/name).read_text()) for name in
                                  ("sources.json", "queries.json", "condition_expansion.json")]
    result = validate(sources, roster, expansion)
    result["negative_tests_passed"] = negative_tests(sources, roster, expansion)
    print(json.dumps(result, indent=2, sort_keys=True))
