"""Manifest-backed demo scaffolding; draft artifacts never claim paper readiness.

Reuses demo_movie's typography and easing without changing its legacy movie.
Rendering consumes cached stills/videos and E3 selected-asset records only.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import uuid

import yaml

from agents.orchestrator.artifact import sha256_file
from robo.manifest.hash import git_snapshot


class DemoError(ValueError):
    pass


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _path(value, root):
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _check_file(record, root):
    path = _path(record["path"], root)
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise DemoError(f"missing source or SHA256 mismatch: {path}")
    return path


def _comparison(record, event, root):
    """Authenticate construction evidence; never consume evaluation metrics."""
    import numpy as np
    spec = record["evidence_comparison"]
    if set(spec) != {"kind", "records"} or spec["kind"] not in {"initial_selection", "registration_retry"}:
        raise DemoError("invalid evidence comparison schema")
    if len(spec["records"]) != 2:
        raise DemoError("comparison requires exactly two proposal records")
    candidates, closure = [], []
    for ref in spec["records"]:
        path = _check_file(ref, root)
        other = _json(path)
        if any(other.get(k) != event.get(k) for k in ("freeze_id", "scene_id", "object_slot", "job_id")):
            raise DemoError("comparison must use the same frozen object job")
        selected = other.get("selected_asset") or {}
        paths, hashes = selected.get("artifact_paths", {}), selected.get("artifact_hashes", {})
        if not {"raw_mesh", "transform", "registration", "evidence", "probe"} <= paths.keys():
            raise DemoError("comparison missing registered mesh or construction evidence")
        closure.append({"path": str(path), "sha256": ref["sha256"], "role": "comparison_record"})
        resolved = {}
        for role, value in paths.items():
            artifact = {"path": value, "sha256": hashes.get(role)}
            checked = _check_file(artifact, root)
            resolved[role] = str(checked)
            closure.append({**artifact, "path": str(checked), "role": role})
        evidence = _json(resolved["evidence"])
        values, probe = evidence.get("raw_values", {}), _json(resolved["probe"])
        if evidence.get("input_hashes", {}).get("raw_mesh") != hashes["raw_mesh"]:
            raise DemoError("construction evidence is not bound to the displayed mesh")
        required = ("symmetric_clipped_registration_residual_m", "support_overlap_fraction")
        if any(not isinstance(values.get(k), (int, float)) or not math.isfinite(values[k]) for k in required):
            raise DemoError("comparison missing finite construction measurements")
        if type(values.get("settle_stable")) is not bool or values["settle_stable"] != probe.get("settle_stable"):
            raise DemoError("construction evidence and archived probe verdict disagree")
        transform = np.load(resolved["transform"], allow_pickle=False)
        registration = _json(resolved["registration"])
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.array_equal(transform, np.asarray(registration.get("T")))):
            raise DemoError("display transform and archived registration disagree")
        candidates.append({"event": other, "paths": resolved, "hashes": hashes,
                           "values": values, "transform": transform.tolist()})
    ids = [c["event"]["proposal_id"] for c in candidates]
    if len(set(ids)) != 2 or event["proposal_id"] not in ids:
        raise DemoError("comparison must contain distinct proposals and the recorded selection")
    if next(c["event"] for c in candidates if c["event"]["proposal_id"] == event["proposal_id"]) != event:
        raise DemoError("displayed selection differs from its pinned decision record")
    if spec["kind"] == "initial_selection":
        if (event.get("policy_id") != "A2" or "construction_evidence_selection" not in event["reason_codes"]
                or {c["event"]["selected_asset"]["tool"] for c in candidates} != {"trellis", "reconviagen"}):
            raise DemoError("initial comparison requires the shared TRELLIS/ReconViaGen selection")
    else:
        parent, retry = candidates
        if retry["event"]["proposal_id"] != event["proposal_id"] or "retry" not in retry["paths"]:
            raise DemoError("retry comparison must end with the recorded retry")
        action = _json(retry["paths"]["retry"])
        if (not event.get("retry_invoked") or not event.get("retry_produced")
                or action.get("action") != "registration_signed_source_up_restart"
                or action.get("proposal_id") != ids[1] or action.get("parent_proposal_id") != ids[0]
                or parent["hashes"]["raw_mesh"] != retry["hashes"]["raw_mesh"]
                or parent["hashes"]["transform"] == retry["hashes"]["transform"]
                or np.array_equal(parent["transform"], retry["transform"])):
            raise DemoError("retry must preserve its real parent and execute a new registration")
    return {"kind": spec["kind"], "candidates": candidates,
            "evidence_scope": {"origin_freeze_id": event["freeze_id"],
                "kind": "original construction evidence; not a paper aggregate",
                "final_common_freeze_deliverable": False, "scientific_claim_enabled": False},
            "rendering_protocol": {"renderer": "MuJoCo triangle mesh rasterization",
                "geometry": "complete raw mesh with archived registration; no simplification",
                "camera": {"azimuth_degrees": 135, "elevation_degrees": -18,
                           "fit": "all vertices from both proposals within central 60 percent of frame"},
                "normalization": "one shared bounds center and scale for both panels",
                "simulation_steps": 0, "probe_values": "archived construction evidence"}}, closure


def resolve(config_path):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") == 2:
        return resolve_final(config_path)
    if config.get("schema_version") != 1 or config.get("mode") not in {"draft", "smoke"}:
        raise DemoError("this scaffold supports draft/smoke only; final paper gates are unmet")
    freeze_id = config.get("freeze_id", "")
    if not freeze_id or freeze_id == "UNASSIGNED":
        raise DemoError("assign an immutable freeze before rendering")
    root = _path(config.get("source_root", "."), config_path.parent)
    width, height = config["resolution"]
    fps = config["fps"]
    if [width, height, fps] != [960, 540, 15]:
        raise DemoError("draft scaffold profile is fixed to 960x540 at 15 fps")
    sources = {}
    for record in config.get("sources", []):
        key = record["id"]
        if key in sources or record.get("freeze_id") != freeze_id:
            raise DemoError(f"duplicate source or mixed freeze: {key}")
        path = _check_file(record, root)
        if record["kind"] not in {"image", "video", "selection"}:
            raise DemoError(f"unsupported source kind: {record['kind']}")
        item = {**record, "path": str(path)}
        if record["kind"] == "selection":
            event = _json(path)
            origin = record.get("origin_freeze_id", freeze_id)
            if event.get("freeze_id") != origin:
                raise DemoError("E3 event has an undeclared source freeze")
            if not event.get("selected_asset") or not event.get("reason_codes"):
                raise DemoError("E3 source must be a complete selected-asset record")
            closure = event["selected_asset"]
            item["artifact_closure"] = []
            for role, artifact in closure["artifact_paths"].items():
                entry = {"path": artifact, "sha256": closure["artifact_hashes"][role]}
                checked = _check_file(entry, root)
                item["artifact_closure"].append({**entry, "path": str(checked), "role": role})
            if event.get("retry_produced"):
                if not event.get("retry_invoked") or "retry" not in closure["artifact_paths"]:
                    raise DemoError("retry requires an invoked action and hashed retry artifact")
                retry = _json(_path(closure["artifact_paths"]["retry"], root))
                if (retry.get("proposal_id") != event["proposal_id"]
                        or retry.get("parent_proposal_id") == event["proposal_id"]
                        or not retry.get("parent_proposal_id") or not retry.get("action")):
                    raise DemoError("retry must record a distinct proposal, parent, and action")
            item["event"] = event
            if "evidence_comparison" in record:
                item["comparison"], comparison_closure = _comparison(record, event, root)
                item["artifact_closure"].extend(comparison_closure)
        sources[key] = item
    shots = config.get("shots", [])
    if not sources or not shots:
        raise DemoError("no frozen sources/shots; framework is not a finished demo")
    seen, episode_blocks = set(), {}
    total_frames = 0
    for shot in shots:
        sid = shot["shot_id"]
        if sid in seen:
            raise DemoError("duplicate shot ID")
        seen.add(sid)
        duration = float(shot["duration_s"])
        if not math.isfinite(duration) or duration <= 0 or duration * fps != round(duration * fps):
            raise DemoError("shot duration must be a positive integral number of frames")
        total_frames += round(duration * fps)
        if shot.get("metric_fields"):
            raise DemoError("experimental result cards await the E9 final claim gate")
        overlays = shot.get("overlay_text", [])
        if len(overlays) > 3 or any(len(t.split()) > 7 for t in overlays):
            raise DemoError("at most three overlays, seven words each")
        if any(re.search(r"(?<!\w)\d", t) for t in overlays):
            raise DemoError("numeric overlays require E9 provenance, unavailable in draft mode")
        source = sources[shot["source_id"]]
        episode = shot.get("episode_id")
        if episode != source.get("episode_id"):
            raise DemoError("shot/source episode IDs differ")
        if episode:
            block = shot.get("episode_block", "policy_episode")
            if block in episode_blocks and episode_blocks[block] != episode:
                raise DemoError("multiple episodes cannot form one apparent policy episode")
            episode_blocks[block] = episode
        if source["kind"] == "selection":
            event = source["event"]
            if shot.get("decision") == "retry" and not event.get("retry_produced"):
                raise DemoError("a retry shot needs a genuine produced retry")
            if shot.get("decision") == "selection" and "construction_evidence_selection" not in event["reason_codes"]:
                raise DemoError("selection shot needs an evidence-selected E3 decision")
    if total_frames != 12 * fps:
        raise DemoError("engineering draft timeline must be exactly 12 seconds")
    return {"schema_version": 1, "freeze_id": freeze_id, "mode": config["mode"],
            "paper_ready": False, "full_demo_ready": False,
            "config_path": str(config_path), "config_sha256": sha256_file(config_path),
            "code": git_snapshot(Path(__file__).resolve().parents[1]),
            "resolution": [width, height], "fps": fps, "frame_count": total_frames,
            "sources": list(sources.values()), "shots": shots,
            "limitations": config.get("limitations", [])}


def _card(size, lines):
    from PIL import Image, ImageDraw
    from interface.demo_movie import _font
    image = Image.new("RGB", size, (22, 25, 31))
    draw = ImageDraw.Draw(image)
    for idx, line in enumerate(lines):
        font = _font((58 if idx == 0 else 42) if size[0] >= 1920 else (28 if idx == 0 else 20))
        if draw.textlength(line, font=font) > size[0] * .86:
            raise DemoError("card text exceeds safe area")
        draw.text((size[0] * .07, size[1] * .30 + idx * (96 if size[0] >= 1920 else 48)), line,
                  font=font, fill=(244, 241, 230) if idx == 0 else (178, 188, 202))
    return image


def _decorate(image, overlays, mode):
    from PIL import ImageDraw
    from interface.demo_movie import _font
    draw = ImageDraw.Draw(image)
    label = "FROZEN SOURCE EVIDENCE" if mode == "final" else ("PRELIMINARY ENGINEERING DRAFT" if mode == "draft" else "SYNTHETIC FRAMEWORK TEST")
    draw.rectangle((0, 0, image.width, 125 if mode == "final" else 68), fill=(22, 25, 31))
    draw.text((image.width * .07, image.height * .07 if mode == "final" else 37), label,
              font=_font(28 if mode == "final" else 16), fill=(241, 170, 74))
    for idx, text in enumerate(overlays):
        font = _font(34 if mode == "final" else 20)
        if draw.textlength(text, font=font) > image.width * .86:
            raise DemoError("overlay exceeds safe area")
        y = (image.height * .93 - 46 - 50 * (len(overlays) - 1 - idx) if mode == "final"
             else image.height - 55 - 34 * (len(overlays) - 1 - idx))
        draw.rectangle((0, y - 4, image.width, y + 29), fill=(22, 25, 31))
        draw.text((image.width * .07, y), text, font=font, fill=(244, 241, 230))
    return image


def _srt_time(seconds):
    milliseconds = round(seconds * 1000)
    return f"{milliseconds // 3600000:02d}:{milliseconds // 60000 % 60:02d}:{milliseconds // 1000 % 60:02d},{milliseconds % 1000:03d}"


def evidence_comparison_image(source):
    """Render actual raw mesh plus archived registration with one shared camera.

    MuJoCo draws the complete triangle mesh. No simplification, new alignment,
    simulation, evaluation reference, or estimated probe outcome is introduced.
    """
    import numpy as np
    import trimesh
    import mujoco
    from PIL import Image, ImageDraw
    from interface.demo_movie import _font

    comparison = source["comparison"]
    meshes = []
    for candidate in comparison["candidates"]:
        mesh = trimesh.load(candidate["paths"]["raw_mesh"], force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise DemoError("display mesh is empty or nonfinite")
        mesh.apply_transform(np.asarray(candidate["transform"]))
        meshes.append(mesh)
    bounds = np.asarray([m.bounds for m in meshes])
    lo, hi = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    center, radius = (lo + hi) / 2, float(np.linalg.norm(hi - lo) / 2)
    if not math.isfinite(radius) or radius <= 0:
        raise DemoError("invalid common display bounds")
    shared_vertices = np.concatenate([m.vertices - center for m in meshes])
    comparison["rendering_checks"] = []
    image = Image.new("RGB", (1920, 1080), (22, 25, 31))
    draw = ImageDraw.Draw(image)
    def text(position, value, *, font, fill):
        box = draw.textbbox(position, value, font=font)
        if box[0] < 1920 * .07 or box[1] < 1080 * .07 or box[2] > 1920 * .93 or box[3] > 1080 * .93:
            raise DemoError("evidence text exceeds the seven-percent safe area")
        draw.text(position, value, font=font, fill=fill)
    retry = comparison["kind"] == "registration_retry"
    text((135, 78), "PRELIMINARY CONSTRUCTION EVIDENCE", font=_font(30), fill=(241, 170, 74))
    text((135, 130), "A genuine registration retry" if retry else "Compare the initial proposals",
              font=_font(54), fill=(244, 241, 230))
    labels = ["Parent registration", "New signed-axis registration"] if retry else [
        "TRELLIS" if c["event"]["selected_asset"]["tool"] == "trellis" else "ReconViaGen"
        for c in comparison["candidates"]]
    for index, (candidate, mesh, label) in enumerate(zip(comparison["candidates"], meshes, labels)):
        x = 135 + index * 845
        mesh.vertices -= center
        mesh_bytes = mesh.export(file_type="obj", include_color=False, include_texture=False).encode()
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><compiler inertiafromgeom="false"/><visual><global offwidth="780" offheight="500"/>'
            '<rgba haze="0.086 0.098 0.122 1"/></visual><asset><mesh name="proposal" file="proposal.obj"/></asset>'
            '<worldbody><light pos="0 -2 4" dir="0 0 -1" castshadow="false"/>'
            '<geom type="mesh" mesh="proposal" contype="0" conaffinity="0" rgba="0.35 0.65 0.88 1"/>'
            '</worldbody></mujoco>', {"proposal.obj": mesh_bytes})
        model.stat.extent = radius
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = 0
        camera.distance, camera.azimuth, camera.elevation = radius * 3.2, 135, -18
        renderer = mujoco.Renderer(model, height=500, width=780)
        try:
            renderer.update_scene(data, camera=camera)
            view = renderer.scene.camera[0]
            forward, up = np.asarray(view.forward), np.asarray(view.up)
            right = np.cross(forward, up)
            tan_y = float(view.frustum_top / view.frustum_near)
            tan_x = tan_y * 780 / 500
            depth_offset = shared_vertices @ forward
            required = np.maximum(np.abs(shared_vertices @ right) / (.60 * tan_x),
                                  np.abs(shared_vertices @ up) / (.60 * tan_y)) - depth_offset
            camera.distance = float(required.max()) * 1.0001
            renderer.update_scene(data, camera=camera)
            # Verify the full union, including disconnected/outlier vertices, against
            # the actual camera. No object-specific crops or zooms are permitted.
            eye = np.mean([c.pos for c in renderer.scene.camera], axis=0)
            relative = shared_vertices - eye
            depth = relative @ forward
            projected = np.column_stack((relative @ right / (depth * tan_x),
                                         relative @ up / (depth * tan_y)))
            if (not np.isfinite(projected).all() or (depth <= view.frustum_near).any()
                    or (depth >= view.frustum_far).any() or np.abs(projected).max() > .601):
                raise DemoError("shared camera clips complete proposal geometry")
            comparison["rendering_checks"].append({"panel": index,
                "camera_distance": camera.distance, "all_union_vertices": len(shared_vertices),
                "projected_ndc_min": projected.min(axis=0).tolist(),
                "projected_ndc_max": projected.max(axis=0).tolist(), "clipping": False})
            image.paste(Image.fromarray(renderer.render()), (x, 272))
        finally:
            renderer.close()
        text((x, 225), label, font=_font(32), fill=(205, 221, 242))
        values = candidate["values"]
        lines = [f'Registration residual: {values["symmetric_clipped_registration_residual_m"] * 1000:.1f} mm',
                 f'Support overlap: {values["support_overlap_fraction"] * 100:.1f}%',
                 'Archived settle probe: ' + ('PASS' if values["settle_stable"] else 'FAIL')]
        for row, line in enumerate(lines):
            text((x, 790 + row * 42), line, font=_font(30), fill=(226, 231, 240))
        if candidate["event"]["proposal_id"] == source["event"]["proposal_id"]:
            verdict = "RETRY CREATED" if retry else "SELECTED BY EVIDENCE"
            text((x, 932), verdict + " / " + source["event"]["support_label"].upper(),
                      font=_font(28), fill=(241, 170, 74))
    text((135, 968), "Same object, frame and camera. Archived probes; no manipulation claim.",
              font=_font(28), fill=(175, 186, 202))
    return image


def _export_evidence_clip(image, source_id, staging):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", source_id):
        raise DemoError("evidence clip source ID must be a plain filename stem")
    directory = staging / "evidence_clips"
    directory.mkdir(exist_ok=True)
    poster = directory / f"{source_id}_1080p.png"
    image.save(poster)
    movie = directory / f"{source_id}_1080p.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-loop", "1", "-i", str(poster),
                    "-t", "6", "-r", "30", "-an", "-c:v", "libx264", "-threads", "2",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-n", str(movie)], check=True)
    return {"source_id": source_id, "poster": str(poster.relative_to(staging)),
            "video": str(movie.relative_to(staging)), "duration_s": 6, "fps": 30,
            "resolution": [1920, 1080], "motion": "static evidence comparison; no simulated action"}


def render(config_path, out_dir):
    from PIL import Image, ImageOps
    import imageio.v2 as imageio

    if yaml.safe_load(Path(config_path).read_text()).get("schema_version") == 2:
        return render_final(config_path, out_dir)
    manifest = resolve(config_path)
    output = Path(out_dir).resolve()
    if output.exists():
        raise DemoError(f"refusing to overwrite demo output: {output}")
    if output.parent.name != manifest["freeze_id"] or output.name != "demo":
        raise DemoError("output must be <freeze_id>/demo")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".demo-{uuid.uuid4().hex}")
    staging.mkdir()
    size = tuple(manifest["resolution"])
    video = staging / "engineering_draft_12s.mp4"
    command = ["ffmpeg", "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{size[0]}x{size[1]}", "-r", str(manifest["fps"]), "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-threads", "2", "-crf", "18", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", "-n", str(video)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    sources = {r["id"]: r for r in manifest["sources"]}
    subtitles, elapsed, comparisons = [], 0.0, []
    try:
        for index, shot in enumerate(manifest["shots"]):
            source = sources[shot["source_id"]]
            reader = None
            if source["kind"] == "image":
                still = Image.open(source["path"]).convert("RGB")
            elif source["kind"] == "selection":
                event = source["event"]
                if "comparison" in source:
                    still = evidence_comparison_image(source)
                    comparisons.append(_export_evidence_clip(still, source["id"], staging))
                else:
                    decision = "RETRY CREATED" if event.get("retry_produced") else "EVIDENCE SELECTION"
                    state = "Still unsupported" if event["support_label"] == "unsupported" else event["support_label"].capitalize()
                    still = _card(size, [decision, event["scene_id"] + " / " + event["object_slot"],
                                        state, "Recorded construction evidence"])
            else:
                reader = imageio.get_reader(source["path"])
                source_fps = float(reader.get_meta_data()["fps"])
            n = round(shot["duration_s"] * manifest["fps"])
            for frame_index in range(n):
                if reader is not None:
                    source_time = shot.get("source_start_s", 0) + frame_index / manifest["fps"]
                    still = Image.fromarray(reader.get_data(round(source_time * source_fps)))
                frame = ImageOps.pad(still, size, color=(22, 25, 31))
                frame = _decorate(frame, shot.get("overlay_text", []), manifest["mode"])
                if index == 0 and frame_index == 0:
                    frame.save(staging / "poster.png")
                proc.stdin.write(frame.tobytes())
            if reader is not None:
                reader.close()
            text = " / ".join(shot.get("overlay_text", []))
            subtitles.append(f"{index+1}\n{_srt_time(elapsed)} --> {_srt_time(elapsed+shot['duration_s'])}\n{text}\n")
            elapsed += shot["duration_s"]
        proc.stdin.close()
        stderr = proc.stderr.read().decode()
        if proc.wait() != 0:
            raise DemoError(f"ffmpeg failed: {stderr}")
        (staging / "engineering_draft_12s.srt").write_text("\n".join(subtitles))
        _write(staging / "source_manifest.json", manifest)
        if comparisons:
            _write(staging / "evidence_clips.json", {"clips": comparisons, "full_demo_ready": False,
                                                     "result_metrics_displayed": False})
        shutil.copyfile(config_path, staging / "resolved_demo_config.yaml")
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)],
                               capture_output=True, text=True, check=True)
        metadata = json.loads(probe.stdout)
        stream = next(s for s in metadata["streams"] if s["codec_type"] == "video")
        if int(stream["nb_frames"]) != 180 or abs(float(metadata["format"]["duration"]) - 12) > .01:
            raise DemoError("encoded draft frame count/duration mismatch")
        _write(staging / "qa.json", {"profile_pass": True, "frames": 180, "duration_s": 12,
                                    "width": stream["width"], "height": stream["height"],
                                    "full_demo_ready": False, "unmet_criteria": manifest["limitations"]})
        _write(staging / "artifact_hashes.json", {str(p.relative_to(staging)): sha256_file(p)
                                                 for p in staging.rglob("*") if p.is_file()})
        _write(staging / "presentation.json", {"freeze_id": manifest["freeze_id"],
               "fallback": {"path": str(output / video.name), "sha256": sha256_file(video),
                            "size_bytes": video.stat().st_size, "mtime_ns": video.stat().st_mtime_ns},
               "video_command": ["ffplay", "-loglevel", "error", "-autoexit", "-fs", "{video}"],
               "live_command": [], "health_commands": []})
        staging.rename(output)
    except BaseException:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if staging.exists():
            # Preserve partial outputs and encoder evidence for audit.
            (staging / "FAILED").write_text("draft render failed; inspect preserved artifacts\n")
        raise
    return output


def assemble(demo_dir):
    """Validate the immutable draft package; never synthesize missing hero footage."""
    root = Path(demo_dir)
    hashes = _json(root / "artifact_hashes.json")
    for name, digest in hashes.items():
        _check_file({"path": name, "sha256": digest}, root)
    manifest = _json(root / "source_manifest.json")
    if manifest.get("schema_version") == 2:
        _require(resolve_final(manifest["config_path"]) == manifest, "final source closure changed")
    for source in manifest["sources"]:
        _check_file(source, root)
        for artifact in source.get("artifact_closure", []):
            _check_file(artifact, root)
    presentation = _json(root / "presentation.json")
    if presentation["freeze_id"] != manifest["freeze_id"]:
        raise DemoError("presentation and source manifest have mixed freezes")
    _check_file(presentation["fallback"], root)
    return {"draft_package_valid": manifest.get("schema_version") == 1,
            "full_demo_ready": manifest.get("full_demo_ready", False), "freeze_id": manifest["freeze_id"]}


def _stop(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=.3)


def live_healthy(commands, budget=1.0):
    if not commands:
        return False
    deadline = time.monotonic() + budget
    for command in commands:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        try:
            proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError:
            return False
        try:
            if proc.wait(timeout=remaining) != 0:
                return False
        except subprocess.TimeoutExpired:
            _stop(proc)
            return False
    return True


def present(manifest_path, mode="auto"):
    """Preflight checks SHA256 before the show; live mode uses pinned file stat.

    One-second aggregate probe deadlines apply initially and throughout live
    playback. A service timeout or live-process death starts local video with
    no retry loop. Child commands never execute through a shell.
    """
    config = _json(manifest_path)
    fallback = config["fallback"]
    video = Path(fallback["path"])
    st = video.stat()
    if (st.st_size, st.st_mtime_ns) != (fallback["size_bytes"], fallback["mtime_ns"]):
        raise DemoError("fallback changed since checksum preflight; preflight again")
    command = [part.replace("{video}", str(video)) for part in config["video_command"]]

    def fallback_now():
        return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

    if mode == "video" or not config.get("live_command") or not live_healthy(config.get("health_commands")):
        return fallback_now()
    try:
        live = subprocess.Popen(config["live_command"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError:
        return fallback_now()
    try:
        while live.poll() is None:
            time.sleep(.15)
            if not live_healthy(config["health_commands"]):
                _stop(live)
                return fallback_now()
        return fallback_now()
    except KeyboardInterrupt:
        _stop(live)
        raise


# Final presentation uses the same source resolver, encoder and live watchdog.
# A plan may be inspected before evidence arrives; encoding cannot bypass gates.
FINAL_DURATIONS = {'hero': 90, 'teaser': 30, 'loop': 8}
FINAL_VIDEO_PATHS = {'hero':'master/simanyroom_hero_90s_1080p.mp4',
                     'teaser':'teaser/simanyroom_teaser_30s_1080p.mp4',
                     'loop':'loop/simanyroom_loop_8s.mp4'}
FINAL_ROLES = {'capture', 'reconstruction', 'selection', 'retry', 'identity',
               'synchronized_episode', 'harmonizer', 'scale', 'real_capture', 'results'}
FULL_E3_FREEZE = '20260906-357caca-v1'
HERO_RULE = 'scene_id_ascending_no_quality_or_policy_outcome_v1'


def _require(value, message):
    if not value:
        raise DemoError(message)


def _identity(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256_file(path)}


def full_e3_selection(spec, root):
    """Predeclared construction ordering; evaluation fields are not used for ordering."""
    audit_path = _check_file(spec['completion_audit'], root)
    audit = _json(audit_path)
    base = audit_path.parent/'agentic'
    _require(audit['freeze_id'] == FULL_E3_FREEZE and audit['status'] == 'PASS'
             and audit['planned_scenes'] == 50 and audit['planned_jobs'] == 1871
             and audit['source_commit'] == '1b3c5165a42d0b0bef0db39f591d5af44db89f78'
             and audit['construction_freeze_id'] == '20260905-859f51d-v1',
             'full E3 completion/denominator differs')
    _require(spec['rule'] == HERO_RULE, 'hero selection rule differs')
    refs = audit['evidence_hashes']
    seal_path = base/'aggregate_seal.json'
    _check_file({'path': str(seal_path), 'sha256': refs[str(seal_path)]['sha256']}, root)
    seal = _json(seal_path)
    ledger_path = base/'job_ledger.jsonl'
    _check_file({'path': str(ledger_path), 'sha256': seal['members']['job_ledger.jsonl']}, root)
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line]
    _require(len(rows) == 9355 and len({(r['job_id'],r['policy_id']) for r in rows}) == 9355,
             'full E3 ledger denominator differs')
    scenes = sorted({r['scene_id'] for r in rows})
    _require(len(scenes) == 50 and all(sum(r['policy_id']==p for r in rows)==1871
             for p in ['A0','A1','A2','A3','A4']), 'full E3 policy roster differs')
    candidates = [{'scene_id': scene, 'accepted_objects': sum(r['scene_id']==scene and
                   r['policy_id']=='A4' and r['terminal_action']=='accept' for r in rows)} for scene in scenes]
    candidates.sort(key=lambda r:r['scene_id'])

    def event(row):
        rel=f"{row['policy_id']}/{row['scene_id']}/{row['object_slot']}.json"
        path=base/'selected_assets'/rel
        _check_file({'path':str(path),'sha256':seal['selected_asset_members'][rel]['sha256']},root)
        value=_json(path)
        _require({k:value[k] for k in row}==row, 'selected asset and ledger differ')
        return value,_identity(path)

    by_key={(r['job_id'],r['policy_id']):r for r in rows}
    selection=retry=None
    # Lexicographic first real cross-tool choice and changed retry, independent
    # of GT fidelity, final probe success, or learned-policy outcome.
    for row in sorted(rows,key=lambda r:(r['scene_id'],r['object_slot'],r['policy_id'])):
        if selection is None and row['policy_id']=='A2' and 'construction_evidence_selection' in row['reason_codes']:
            parent=by_key[(row['job_id'],'A0')]
            if row['proposal_id'] != parent['proposal_id'] and row['proposal_id']:
                selected,sref=event(row);prior,pref=event(parent)
                record={**sref,'evidence_comparison':{'kind':'initial_selection','records':[pref,sref]}}
                comparison,_=_comparison(record,selected,root)
                selection={'record':record,'job_id':row['job_id'],'comparison':comparison}
        if retry is None and row['policy_id']=='A3' and row.get('retry_produced'):
            parent=by_key[(row['job_id'],'A2')]
            if row['proposal_id'] != parent['proposal_id'] and row['proposal_id']:
                selected,sref=event(row);prior,pref=event(parent)
                record={**sref,'evidence_comparison':{'kind':'registration_retry','records':[pref,sref]}}
                comparison,_=_comparison(record,selected,root)
                retry={'record':record,'job_id':row['job_id'],'comparison':comparison}
        if selection and retry:break
    _require(selection and retry, 'full source has no genuine selection/retry pair')
    return {'rule':HERO_RULE,'source':spec['completion_audit'],'candidates':candidates,
            'selection':selection,'retry':retry,'hero_scene_id':None,
            'hero_requires':['at_least_four_accepted_objects','clean_background',
                             'full_room_collision','usable_synchronized_episode'],
            'uses_policy_success_for_selection':False,'uses_gt_quality_for_selection':False}


def _paper_source(spec, root):
    """Exact E9 publication bytes; numeric text comes from machine source fields."""
    path=_check_file(spec['provenance'],root);report=_json(path);directory=path.parent
    _require(path.name=='paper_table_provenance.json' and report['freeze_id']==spec['freeze_id'],
             'E9 freeze/provenance differs')
    config_path=_check_file({'path':report['config'],'sha256':report['config_sha256']},root)
    snapshot=git_snapshot(config_path.parent)
    _require(snapshot['commit']==report['formatter_commit'] and snapshot['dirty'] is False,
             'E9 formatter source differs')
    artifacts={}
    for group,folder in [('tables','generated_tables'),('figures','generated_figures')]:
        for name,ref in report.get(group,{}).items():
            artifact=directory/folder/name
            _check_file({'path':str(artifact),'sha256':ref['sha256']},root)
            artifacts[group+'/'+name]=_identity(artifact)
    sources={}
    for name,ref in report['sources'].items():
        if isinstance(ref,dict) and 'path' in ref and 'sha256' in ref:
            sources[name]={'reference':ref,'payload':_json(_check_file(ref,root))}
    audit=_json(_check_file(spec['submission_audit'],root))
    _require(_path(spec['submission_audit']['path'],root)==directory/'submission_audit.json',
             'E9 submission audit path differs')
    return {'reference':spec,'provenance':report,'artifacts':artifacts,'sources':sources,
            'submission_pass':audit.get('submission_freeze')=='PASS' and report.get('paper_ready') is True}


def _result_text(field, paper):
    _require(set(field)=={'source','pointer','label','format'},'result field schema differs')
    value=paper['sources'][field['source']]['payload']
    for part in field['pointer']:
        value=value[part]
    _require(type(value) in {int,float} and math.isfinite(value),'result must be a measured finite source value')
    _require(field['format'] in {'d','.1f','.2f','.3f','.1%'},'unapproved numeric rounding')
    _require(not re.search(r'\d',field['label']) and len(field['label'].split())<=5,'result label contains numbers')
    return field['label']+' '+format(value,field['format'])


def _episode_source(record, root):
    """Consume canonical ledger/trace; paired renders are still missing today.

    ``synchronized_frames`` is a D0 render manifest, not another rollout ledger.
    Each panel must identify the same canonical tick and body-state digest.
    Its producer/config/receipt are pinned by the caller before rendering.
    """
    from robo.eval import episode_log
    from robo.eval.harness_spec import load_harness_spec
    from robo.eval.harness_validation import validate_records, validate_saved_treatment_records
    from robo.manifest.hash import canonical_hash
    ledger=_check_file(record['ledger'],root)
    rows=[json.loads(l) for l in ledger.read_text().splitlines() if l]
    matches=[r for r in rows if r.get('episode_id')==record['episode_id']]
    _require(len(matches)==1,'episode absent or duplicated in canonical ledger')
    row=matches[0]
    config_path=_check_file(record['harness_config'],root)
    bank=_check_file(record['reset_bank'],root)
    spec=load_harness_spec(config_path)
    harness_cfg=yaml.safe_load(config_path.read_text())
    treatment=spec.treatments[row['treatment_id']]
    _require(harness_cfg.get('contract',{}).get('policy',{}).get('kind')=='real'
             and treatment.scene in {'simany','agentic'} and treatment.collision=='full_room',
             'hero requires real policy, agentic scene and full-room collision')
    reset_ids={state.reset_state_id for state in episode_log.load_reset_states(bank)}
    for check in [validate_records(rows,spec,reset_ids),
                  validate_saved_treatment_records(rows,spec,reset_ids,manifest_root=ledger.parent)]:
        _require(check['ok'],'canonical harness ledger/manifest validation failed')
    _require(row['outcome'] in {'success','task_failure','policy_timeout','safety_termination','environment_crash'}
             and row.get('ticks',0)>0 and row.get('contract',{}).get('policy_execution')!='not_invoked_prebuild',
             'episode was not actually invoked')
    manifest_path=_check_file(record['episode_manifest'],root)
    _require(str(manifest_path)==row['manifest_path'],'canonical episode manifest differs')
    manifest=_json(manifest_path)
    trace=_check_file(record['trace'],root)
    _require(str(trace)==row['trace_path']==manifest['artifacts']['trace_path'],'canonical trace differs')
    ticks=episode_log.read_timeseries(trace)
    _require(len(ticks)==row['ticks'] and [t['t'] for t in ticks]==list(range(len(ticks))),
             'canonical episode ticks are missing or reordered')
    sync_path=_check_file(record['synchronized_frames'],root);sync=_json(sync_path)
    _require(sync['episode_id']==record['episode_id'] and sync['trace']==record['trace']
             and sync['ledger']==record['ledger'] and sync['episode_manifest']==record['episode_manifest']
             and sync['collision_mode']=='full_room' and sync['simulation_reexecuted'] is False,
             'paired recording is not a full-room same-episode replay')
    for name in ['producer','config','execution_receipt']:_check_file(sync[name],root)
    producer=_path(sync['producer']['path'],root)
    producer_source=git_snapshot(producer.parent)
    execution=_json(_path(sync['execution_receipt']['path'],root))
    _require(producer.suffix=='.py' and producer_source['dirty'] is False
             and sync['schema_version']==1 and sync['scope']=='canonical_harness_render_replay'
             and execution['status']=='PASS' and execution['source']==producer_source
             and execution['producer']==sync['producer'] and execution['config']==sync['config']
             and execution['ledger']==record['ledger'] and execution['trace']==record['trace']
             and execution['frames_sha256']==canonical_hash(sync['frames']),
             'paired rendering lacks source-bound execution evidence')
    clean=_check_file(sync['clean_background'],root)
    _require(clean==Path(manifest['factory_dir']).resolve()/'inpaint/clean_background.ply',
             'hero clean background differs from canonical factory')
    from robo.envs import pi05_env
    _require(sync['frame_hz']==pi05_env.rig.CONTROL_HZ, 'paired recording control frequency differs')
    _require(len(sync['frames'])==len(ticks),'paired recording omitted episode ticks')
    for tick,frame in zip(ticks,sync['frames']):
        _require(frame['tick']==tick['t'] and frame['state_sha256']==canonical_hash(tick)
                 and set(frame['views'])=={'mujoco','photoreal'},'paired frame state/camera mismatch')
        for ref in frame['views'].values():_check_file(ref,root)
    _require(sync['video']=={'path':str(_check_file(record,root)),'sha256':record['sha256']},
             'paired video differs from render receipt')
    return {'episode_id':row['episode_id'],'scene_id':row['scene_id'],'outcome':row['outcome'],
            'success':row['success'],'ticks':row['ticks'],'frames':sync['frames'],
            'frame_hz':sync['frame_hz'],'source':record,
            'recorded_candidates': sorted([{'episode_id':r['episode_id'],'scene_id':r['scene_id']}
                for r in rows if r.get('ticks',0)>0 and r.get('video_path')
                and spec.treatments[r['treatment_id']].scene in {'simany','agentic'}
                and spec.treatments[r['treatment_id']].collision=='full_room'],key=lambda r:r['episode_id'])}


def _final_timeline(shots, sources, paper, name):
    _require(shots and len({s['shot_id'] for s in shots})==len(shots),'empty/duplicate final shots')
    total=0;episode_spans=[];resolved=[]
    for shot in shots:
        source=sources[shot['source_id']];duration=shot['duration_s']
        if shot['role']=='synchronized_episode':
            _require(source['kind']=='synchronized_episode','synchronized role requires canonical paired frames')
        if shot['role'] in {'selection','retry'}:
            _require(source['kind']=='selection' and source.get('decision')==shot['role'],
                     'decision shot requires the predeclared genuine E3 record')
        if shot['role']=='results':
            _require(source['kind'] in {'figure','result_card'},'results require exact E9 figure/card')
        _require(type(duration) in {int,float} and math.isfinite(duration) and duration>0
                 and duration*30==round(duration*30),'final duration must have exact frames')
        overlays=shot.get('overlay_text',[])
        _require(len(overlays)<=3 and all(len(t.split())<=7 and not re.search(r'(?<!\w)\d',t)
                 for t in overlays),'numeric text must use E9 field binding')
        fields=shot.get('metric_fields',[])
        _require(not fields or shot['role']=='results','metrics outside frozen results card')
        texts=overlays+[_result_text(f,paper) for f in fields]
        _require(len(texts)<=3,'too many final overlays')
        if source['kind']=='synchronized_episode' and 'episode' in source:
            ep=source['episode'];start=shot.get('source_start_s',0)
            _require(start>=0 and (start+duration)*ep['frame_hz']<=ep['ticks'],
                     'episode excerpt extends past recorded frames')
            _require(not any(word in ' '.join(overlays).lower() for word in ['success','successful','placed'])
                     or ep['success'] is True,'unsuccessful episode caption claims success')
            episode_spans.append((ep['episode_id'],start,start+duration))
        if name=='loop':
            _require(source['kind'] in {'image','figure','selection','result_card'},
                     'loop may hold authentic imagery; it cannot loop/reverse policy motion')
        resolved.append({**shot,'overlay_text':texts});total+=round(duration*30)
    _require(total==FINAL_DURATIONS[name]*30,'final '+name+' duration differs')
    _require(len({s[0] for s in episode_spans})<=1,'multiple episodes appear as one rollout')
    _require(all(a[2]==b[1] for a,b in zip(episode_spans,episode_spans[1:])),
             'episode shots skip, overlap or reverse recorded time')
    if name=='loop':_require(len(shots)==1,'seamless hold loop requires one authentic source')
    return resolved


def resolve_final(config_path, *, require_ready=True):
    config_path=Path(config_path).resolve();cfg=yaml.safe_load(config_path.read_text())
    _require(cfg.get('schema_version')==2 and cfg.get('mode')=='final','invalid final demo schema')
    _require(cfg.get('freeze_id') not in {None,'','UNASSIGNED'} and cfg.get('resolution')==[1920,1080]
             and cfg.get('fps')==30,'final demo requires assigned freeze and1080p30')
    root=_path(cfg['source_root'],config_path.parent)
    selection=full_e3_selection(cfg['e3'],root);paper=_paper_source(cfg['e9'],root)
    agentic=paper['provenance']['sources'].get('agentic',{})
    _require(agentic.get('completion_audit',{}).get('sha256')==cfg['e3']['completion_audit']['sha256'],
             'E9 result freeze does not bind the same full E3 source')
    sources={};missing=[]
    for source in cfg['sources']:
        _require(source['id'] not in sources,'duplicate final source ID')
        item=dict(source)
        if source.get('path') is None:
            missing.append('missing_source:'+source['id']);sources[source['id']]=item;continue
        item['path']=str(_check_file(source,root))
        _require(source['kind'] in {'image','video','selection','figure','result_card','synchronized_episode'},
                 'unsupported final source kind')
        if source['kind']=='selection':
            chosen=selection[source['decision']]
            _require(source['path']==chosen['record']['path'] and source['sha256']==chosen['record']['sha256'],
                     'manual E3 selection differs from predeclared rule')
            item['comparison']=chosen['comparison'];item['event']=_json(item['path'])
        elif source['kind']=='figure':
            _require(source['e9_artifact'] in paper['artifacts'] and
                     {'path':item['path'],'sha256':source['sha256']}==paper['artifacts'][source['e9_artifact']],
                     'displayed figure differs from E9 exact artifact')
        elif source['kind']=='result_card':
            _require({'path':item['path'],'sha256':source['sha256']}==cfg['e9']['provenance'],
                     'result card must use the exact E9 provenance')
        elif source['kind']=='synchronized_episode':item['episode']=_episode_source(source,root)
        sources[source['id']]=item
    _require(set(cfg['timelines'])==set(FINAL_DURATIONS),'final package requires all three timelines')
    timelines={name:_final_timeline(shots,sources,paper,name) for name,shots in cfg['timelines'].items()}
    _require({s['role'] for s in timelines['hero']}==FINAL_ROLES,'hero storyboard is incomplete')
    episode_sources=[s for s in sources.values() if s['kind']=='synchronized_episode' and 'episode' in s]
    if not episode_sources:missing.append('genuine_synchronized_policy_episode')
    hero=None
    if episode_sources:
        cohort=episode_sources[0]['episode']['recorded_candidates']
        _require(all(s['ledger']==episode_sources[0]['ledger'] for s in episode_sources),
                 'final film mixes canonical rollout ledgers')
        for candidate in selection['candidates']:
            episodes=[r for r in cohort if r['scene_id']==candidate['scene_id']]
            if candidate['accepted_objects']>=4 and episodes:
                expected=episodes[0]
                _require(all(s['episode']['episode_id']==expected['episode_id'] for s in episode_sources),
                         'episode differs from outcome-independent full-roster selection')
                hero=candidate['scene_id'];break
    if hero is None:missing.append('eligible_full_room_hero')
    if not paper['submission_pass']:missing.append('E9_common_submission_freeze')
    e0=cfg.get('execution_contract')
    if e0 is None:missing.append('D0_exact_source_config_E0')
    else:
        from robo.manifest.hash import canonical_hash
        contract=_json(_path(e0,root));code=git_snapshot(Path(__file__).resolve().parents[1])
        _require(contract['contract_sha256']==canonical_hash({k:v for k,v in contract.items()
                 if k not in {'created_utc','environment','contract_sha256'}})
                 and contract['freeze_id']==cfg['freeze_id'] and contract['code']['commit']==code['commit']
                 and contract['code']['dirty'] is False and code['dirty'] is False,'D0 E0/source differs')
        _require(any(r.get('sha256')==sha256_file(config_path) and r.get('resolved_path')==str(config_path)
                 for r in contract['resource_inventory']),'D0 config absent from E0')
    # Preservation must come from the existing visual-metric producer; no
    # placeholder or raw-RGB fallback can satisfy Option C.
    harmony=cfg.get('harmonizer_gate')
    if harmony is None:missing.append('E5_preservation_certified_frames')
    else:
        from robo.eval.harmony_visual_metrics import deterministic_core_check
        table_ref=harmony['table'];table=_json(_check_file(table_ref,root))
        declared=paper['provenance']['sources'].get('harmony_visual',{})
        _require(declared.get('path')==table_ref['path'] and declared.get('sha256')==table_ref['sha256'],
                 'Harmonizer table is not in the exact E9 publication')
        source_manifest=_check_file(harmony['manifest'],root);hm=_json(source_manifest)
        _require(table['manifest']==str(source_manifest) and len(hm['conditions'])==5
                 and {r['condition'] for r in table['rows']}==set(hm['conditions']),
                 'canonical five-condition Harmonizer coverage missing')
        rows=[r for r in table['rows'] if r['condition']==harmony['condition']]
        _require(len(rows)==1 and rows[0]['robot_core_exact'] is True
                 and all(rows[0].get(k) is not None for k in ['target_iou','robot_iou','tlpips','p95_ms_per_camera'])
                 and all(rows[0].get(k,0)>0 for k in ['n_images','target_pairs','robot_pairs','temporal_pairs','latency_samples']),
                 'Option C preservation measurements incomplete or failed')
        condition=hm['conditions'][harmony['condition']]
        latency=_check_file(harmony['latency_records'],root)
        _require(str(latency)==condition['latency_records'] and
                 deterministic_core_check(latency,harmony['condition']) is True,
                 'canonical robot-core evidence is absent or changed')
        frame_refs=harmony['frames']
        for ref in frame_refs:
            fp=_check_file(ref,root)
            _require(fp.is_relative_to(Path(condition['render_dir']).resolve()),
                     'Harmonizer source is not an evaluated restored frame')
        for shot in timelines['hero']:
            if shot['role']=='harmonizer':
                src=sources[shot['source_id']]
                _require(src['kind']=='image' and {'path':src.get('path'),'sha256':src.get('sha256')} in frame_refs,
                         'Harmonizer shot is not an exact evaluated restored frame')
    missing=sorted(set(missing))
    if require_ready and missing:raise DemoError('final demo blocked: '+', '.join(missing))
    return {'schema_version':2,'mode':'final','freeze_id':cfg['freeze_id'],
            'config_path':str(config_path),'config_sha256':sha256_file(config_path),
            'code':git_snapshot(Path(__file__).resolve().parents[1]),'resolution':[1920,1080],'fps':30,
            'sources':list(sources.values()),'timelines':timelines,'e3_selection':selection,
            'e9':paper,'hero_scene_id':hero,'missing_prerequisites':missing,
            'full_demo_ready':not missing,'paper_ready':paper['submission_pass'] and not missing,
            'limitations':['Loop is an explicit seamless hold of authentic evidence, not repeated policy motion.']}


def _final_frame(source, frame_index, shot):
    from PIL import Image
    import imageio.v2 as imageio
    if source['kind']=='selection':return evidence_comparison_image(source)
    if source['kind']=='result_card':return _card((1920,1080),['Frozen results','Source-bound paper evidence'])
    if source['kind']=='synchronized_episode':
        from PIL import ImageOps
        tick=int((shot.get('source_start_s',0)+frame_index/30)*source['episode']['frame_hz'])
        frame=source['episode']['frames'][tick]
        canvas=Image.new('RGB',(1920,1080),(22,25,31))
        for offset,name in [(0,'mujoco'),(960,'photoreal')]:
            view=Image.open(frame['views'][name]['path']).convert('RGB')
            canvas.paste(ImageOps.pad(view,(960,1080),color=(22,25,31)),(offset,0))
        return canvas
    if source['kind'] in {'image','figure'}:return Image.open(source['path']).convert('RGB')
    reader=imageio.get_reader(source['path'])
    try:
        fps=float(reader.get_meta_data()['fps'])
        return Image.fromarray(reader.get_data(round((shot.get('source_start_s',0)+frame_index/30)*fps)))
    finally:reader.close()


def render_final(config_path, out_dir):
    """CPU assembly only. Missing scientific footage aborts before any output."""
    from PIL import Image, ImageOps
    manifest=resolve_final(config_path);output=Path(out_dir).resolve()
    _require(not output.exists(),'refusing to overwrite final demo')
    _require(output.name=='demo' and output.parent.name==manifest['freeze_id'],'final output freeze differs')
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=output.with_name('.demo-final-'+uuid.uuid4().hex);staging.mkdir()
    sources={s['id']:copy.deepcopy(s) for s in manifest['sources']};profiles={}
    try:
        for name,shots in manifest['timelines'].items():
            video=staging/FINAL_VIDEO_PATHS[name];video.parent.mkdir(parents=True,exist_ok=True)
            cmd=['ffmpeg','-v','error','-nostdin','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080',
                 '-r','30','-i','pipe:0','-an','-c:v','libx264','-threads','4','-crf','18',
                 '-pix_fmt','yuv420p','-movflags','+faststart','-n',str(video)]
            proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
            elapsed=0;subtitles=[]
            try:
                for idx,shot in enumerate(shots):
                    source=sources[shot['source_id']];still=None;reader=None
                    if source['kind']=='video':
                        import imageio.v2 as imageio
                        reader=imageio.get_reader(source['path']);source_fps=float(reader.get_meta_data()['fps'])
                    for frame in range(round(shot['duration_s']*30)):
                        if still is None or source['kind'] in {'video','synchronized_episode'}:
                            still=(Image.fromarray(reader.get_data(int((shot.get('source_start_s',0)+frame/30)*source_fps))) if reader is not None else _final_frame(source,frame,shot))
                        image=ImageOps.pad(still,(1920,1080),color=(22,25,31))
                        image=_decorate(image,shot['overlay_text'],'final')
                        if name=='hero' and idx==0 and frame==0:
                            (staging/'poster').mkdir(exist_ok=True);image.save(staging/'poster/simanyroom_poster.png')
                        proc.stdin.write(image.tobytes())
                    if reader is not None:reader.close()
                    subtitles.append(f"{idx+1}\n{_srt_time(elapsed)} --> {_srt_time(elapsed+shot['duration_s'])}\n"+
                                     ' / '.join(shot['overlay_text'])+'\n')
                    elapsed+=shot['duration_s']
                proc.stdin.close();error=proc.stderr.read().decode()
                _require(proc.wait()==0,'final encoder failed: '+error)
            finally:
                if proc.poll() is None:proc.kill();proc.wait()
            video.with_suffix('.srt').write_text('\n'.join(subtitles))
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],text=True))
            stream=next(s for s in probe['streams'] if s['codec_type']=='video')
            _require(int(stream['nb_frames'])==FINAL_DURATIONS[name]*30 and stream['width']==1920
                     and stream['height']==1080 and abs(float(probe['format']['duration'])-FINAL_DURATIONS[name])<.01,
                     'encoded final video profile differs')
            profiles[name]={'frames':int(stream['nb_frames']),'duration_s':FINAL_DURATIONS[name],
                            'width':1920,'height':1080,'fps':30}
        subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(staging/FINAL_VIDEO_PATHS['hero']),
                        '-c','copy','-n',str(staging/'master/simanyroom_hero_90s_master.mov')],check=True)
        subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(staging/FINAL_VIDEO_PATHS['loop']),
                        '-c:v','libwebp_anim','-loop','0','-n',str(staging/'loop/simanyroom_loop_8s.webp')],check=True)
        # Exact source revalidation before atomic publication protects a long encode.
        _require(resolve_final(config_path)==manifest,'sources changed during final rendering')
        _write(staging/'source_manifest.json',manifest);shutil.copyfile(config_path,staging/'resolved_demo_config.yaml')
        _write(staging/'qa.json',{'profiles':profiles,'full_demo_ready':True,'loop':'static evidence hold; exact cyclic visual content',
                                   'rendering_checks':{k:v['comparison'].get('rendering_checks',[]) for k,v in sources.items() if 'comparison' in v}})
        _write(staging/'artifact_hashes.json',{str(p.relative_to(staging)):sha256_file(p) for p in staging.rglob('*') if p.is_file()})
        video=staging/FINAL_VIDEO_PATHS['hero'];_write(staging/'presentation.json',{'freeze_id':manifest['freeze_id'],
            'fallback':{'path':str(output/FINAL_VIDEO_PATHS['hero']),'sha256':sha256_file(video),
                        'size_bytes':video.stat().st_size,'mtime_ns':video.stat().st_mtime_ns},
            'video_command':['ffplay','-loglevel','error','-autoexit','-fs','{video}'],
            'live_command':[],'health_commands':[]})
        staging.rename(output)
    except BaseException:
        if staging.exists():(staging/'FAILED').write_text('Final assembly failed; original partial output retained.\n')
        raise
    return output


def resolve_evidence_clips(config_path):
    """Admissible engineering clips; never bypass the final-film gates."""
    from robo.manifest.hash import canonical_hash
    config_path=Path(config_path).resolve();cfg=yaml.safe_load(config_path.read_text())
    _require(cfg['schema_version']==2 and cfg['mode']=='engineering_evidence_clips', 'engineering clip scope differs')
    root=Path(cfg['source_root']).resolve()
    selection=full_e3_selection(cfg['e3'],root);paper=_paper_source(cfg['e9'],root)
    _require(paper['provenance']['sources']['agentic']['completion_audit']['sha256']==
             cfg['e3']['completion_audit']['sha256'], 'E3/E9 completion audit differs')
    receipt_path=_check_file(cfg['publication_receipt'],root);receipt=_json(receipt_path)
    _require(receipt['freeze_id']==cfg['e9']['freeze_id'] and
             receipt['producer_commit']==paper['provenance']['formatter_commit'] and
             receipt['table_provenance_sha256']==cfg['e9']['provenance']['sha256'] and
             receipt['full_e3_evidence_freeze']==FULL_E3_FREEZE, 'paper transfer receipt differs')
    code=git_snapshot(Path(__file__).resolve().parents[1])
    contract_path=_path(cfg['execution_contract'],root);contract=_json(contract_path)
    _require(contract['contract_sha256']==canonical_hash({k:v for k,v in contract.items()
             if k not in {'created_utc','environment','contract_sha256'}}) and
             contract['freeze_id']==cfg['freeze_id'] and contract['code']['commit']==code['commit'] and
             contract['code']['dirty'] is False and code['dirty'] is False, 'engineering E0/source differs')
    _require(any(r.get('sha256')==sha256_file(config_path) and r.get('resolved_path')==str(config_path)
                 for r in contract['resource_inventory']), 'engineering config absent from E0')
    figure=cfg['result_figure']
    _require(figure in paper['artifacts'] and figure.startswith('figures/') and figure.endswith('.png'),
             'result poster must be an exact E9 PNG')
    sources=[]
    for decision in ['selection','retry']:
        selected=selection[decision]
        sources.append({'id':decision,'kind':'selection','event':_json(selected['record']['path']),
                        'comparison':selected['comparison'],'reference':selected['record']})
    return {'schema_version':2,'mode':cfg['mode'],'freeze_id':cfg['freeze_id'],
            'config':_identity(config_path),'code':code,'execution_contract':_identity(contract_path),
            'sources':sources,'e3_selection':selection,'e9':paper,
            'publication_receipt':cfg['publication_receipt'],'result_figure':paper['artifacts'][figure],
            'full_demo_ready':False,'paper_ready':False,'result_metrics_displayed':True,
            'scope':'Engineering evidence clips; original full E3 comparisons and exact E9 figure. Not the final same-freeze D0 deliverable.',
            'unmet_criteria':['genuine synchronized E4 episode','E5 preservation footage','common scientific submission gate']}


def render_evidence_clips(config_path,out_dir):
    from PIL import Image
    manifest=resolve_evidence_clips(config_path);output=Path(out_dir).resolve()
    cfg=yaml.safe_load(Path(config_path).read_text());root=Path(cfg['source_root']).resolve()
    _require(output==root/'outputs/icra2027'/manifest['freeze_id']/'demo', 'engineering output must use its new freeze')
    _require(not output.exists(),'engineering output already exists')
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=output.parent/('.demo-'+uuid.uuid4().hex);staging.mkdir()
    try:
        clips=[]
        rendering_checks={}
        for original in manifest['sources']:
            source=copy.deepcopy(original)
            clips.append(_export_evidence_clip(evidence_comparison_image(source),source['id'],staging))
            rendering_checks[source['id']]=source['comparison'].get('rendering_checks',[])
        figure=Image.open(manifest['result_figure']['path']).convert('RGB')
        from PIL import ImageOps
        figure=ImageOps.pad(figure,(1920,1080),color=(255,255,255))
        clips.append(_export_evidence_clip(figure,'frozen_results',staging))
        for clip in clips:
            video=staging/clip['video']
            metadata=json.loads(subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],
                                               capture_output=True,text=True,check=True).stdout)
            stream=next(s for s in metadata['streams'] if s['codec_type']=='video')
            _require(int(stream['nb_frames'])==180 and stream['width']==1920 and stream['height']==1080 and
                     abs(float(metadata['format']['duration'])-6)<.01,'engineering clip profile differs')
            subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(video),'-f','null','-'],check=True)
            (staging/clip['video']).with_suffix('.srt').write_text('1\n00:00:00,000 --> 00:00:06,000\nArchived evidence; engineering preview.\n')
        _require(resolve_evidence_clips(config_path)==manifest,'sources changed during engineering render')
        _write(staging/'source_manifest.json',manifest)
        _write(staging/'qa.json',{'status':'PASS','clips':clips,'rendering_checks':rendering_checks,'full_demo_ready':False})
        _write(staging/'artifact_hashes.json',{str(p.relative_to(staging)):sha256_file(p) for p in staging.rglob('*') if p.is_file()})
        staging.rename(output)
    except BaseException:
        (staging/'FAILED').write_text('Engineering render failed; partial evidence preserved.\n')
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    child = sub.add_parser("native-scope-diagnostic")
    child.add_argument("--diagnosis", required=True)
    child.add_argument("--out", required=True)
    child = sub.add_parser("native-hero")
    child.add_argument("--release", required=True)
    child.add_argument("--out", required=True)
    child = sub.add_parser("native-progress")
    child.add_argument("--release", required=True)
    child.add_argument("--out", required=True)
    child = sub.add_parser("preflight")
    inputs = child.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--config")
    inputs.add_argument("--demo-dir")
    child = sub.add_parser("plan-final")
    child.add_argument("--config", required=True)
    child = sub.add_parser("select-full-e3")
    child.add_argument("--spec", required=True)
    child.add_argument("--root", required=True)
    child = sub.add_parser("evidence-clips")
    child.add_argument("--config", required=True)
    child.add_argument("--out", required=True)
    child = sub.add_parser("render")
    child.add_argument("--config", required=True)
    child.add_argument("--out", required=True)
    child = sub.add_parser("assemble")
    child.add_argument("--demo-dir", required=True)
    child = sub.add_parser("present")
    child.add_argument("--manifest", required=True)
    child.add_argument("--mode", choices=("video", "live", "auto"), default="auto")
    args = parser.parse_args(argv)
    try:
        if args.action == "native-scope-diagnostic":
            from interface.demo_native_progress import render_scope_diagnostic
            print(render_scope_diagnostic(args.diagnosis, args.out))
        elif args.action == "native-hero":
            from interface.demo_native_hero import render as render_hero
            print(render_hero(args.release, args.out))
        elif args.action == "native-progress":
            from interface.demo_native_progress import render as render_native
            print(render_native(args.release, args.out))
        elif args.action == "plan-final":
            print(json.dumps(resolve_final(args.config, require_ready=False)))
        elif args.action == "select-full-e3":
            print(json.dumps(full_e3_selection(_json(args.spec), Path(args.root))))
        elif args.action == "preflight":
            result = assemble(args.demo_dir) if args.demo_dir else resolve(args.config)
            print(json.dumps({"source_preflight": "PASS", "freeze_id": result["freeze_id"], "full_demo_ready": result.get("full_demo_ready", False)}))
        elif args.action == "evidence-clips":
            print(render_evidence_clips(args.config,args.out))
        elif args.action == "render":
            print(render(args.config, args.out))
        elif args.action == "assemble":
            print(json.dumps(assemble(args.demo_dir)))
        else:
            process = present(args.manifest, args.mode)
            return process.wait()
    except (DemoError, KeyError, OSError, ValueError) as exc:
        print(f"demo failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
