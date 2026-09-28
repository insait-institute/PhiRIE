"""Closed-loop pi0.5 evaluation over SimAny pick-and-place task suites.

Server side (openpi, separate JAX venv, e.g. on the same A6000):
  cd ${OPENPI_ROOT} && \
  OPENPI_DATA_HOME=${OPENPI_DATA_HOME} \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py \
    policy:checkpoint --policy.config=pi05_droid_jointpos \
    --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos

Client side (this venv):
  MUJOCO_GL=egl python pi05_eval.py \
    --tasks ../outputs/c50d2d1d42_factory/sim_export/pi05_tasks.json \
    --out runs/pi05_c50 [--policy scripted]

Actions from the jointpos server are ABSOLUTE joint position targets
(7) + gripper [0,1]; we execute open_loop_horizon actions per query at
15 Hz (RoboLab default for pi05: the full 15-step chunk).
"""
import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import yaml

from robo.envs import pi05_env
from robo.manifest import hash as manifest_hash
from robo.manifest import io as manifest_io
from robo.manifest.schema import ObservationPreprocessing, RolloutManifest, StagedProgress
from robo.tasks import pi05_tasks

# --- Task 01 (robo.manifest) provenance wiring --------------------------
ROOT = Path(__file__).resolve().parents[2]
FROZEN_FIELDS_PATH = ROOT / "configs" / "experiments" / "frozen_fields.yaml"
# Bump this if configs/experiments/frozen_fields.yaml's `rubric` section
# (stages/credit-per-stage) ever changes -- docs/ICRA_RESEARCH_CONTRACT.md
# section 5 requires a new version rather than a silent redefinition.
RUBRIC_VERSION = "staged_0.25_per_stage_v1"


def _frozen_config_hashes():
    """(controller_config_hash, camera_config_hash, action_convention,
    action_dim) for RolloutManifest, sourced from
    configs/experiments/frozen_fields.yaml when present (the actual frozen
    contract every table comparison must match), else from this file's own
    hardcoded conventions (matches robo/rigs/pi05_rig.py) with a loud
    warning -- eval still runs, but the manifest says so.
    """
    if FROZEN_FIELDS_PATH.exists():
        frozen = yaml.safe_load(FROZEN_FIELDS_PATH.read_text())
        control, cameras = frozen["control"], frozen["cameras"]
        return (manifest_hash.hash_config_section(control),
                manifest_hash.hash_config_section(cameras),
                control["action_convention"], control["action_dim"])
    print(f"[eval] WARNING: {FROZEN_FIELDS_PATH} missing; manifest "
          f"controller/camera hashes fall back to hardcoded defaults "
          f"instead of the frozen contract file", flush=True)
    return (manifest_hash.canonical_hash(
                {"action_convention": "absolute_joint_position",
                 "rate_hz": pi05_env.rig.CONTROL_HZ}),
            manifest_hash.canonical_hash({"note": "frozen_fields.yaml missing"}),
            "absolute_joint_position", 8)


def _policy_checkpoint_hash(args):
    """Best-effort policy_checkpoint_hash for the manifest.

    `--policy scripted` has no checkpoint (matches frozen_fields.yaml's
    `scripted_sinusoid: checkpoint_path: null`). `--policy server` only
    knows a host:port, not which checkpoint that server loaded, unless the
    caller also passes --checkpoint-path; without it the manifest still
    carries a value (never silently omits the field) but it is a documented
    placeholder, not a real fingerprint -- flagged loudly so a benchmark run
    launched without --checkpoint-path is easy to catch in review.
    """
    if args.policy == "scripted":
        return None
    if args.checkpoint_path:
        p = Path(args.checkpoint_path)
        if p.exists():
            return manifest_hash.hash_checkpoint_path(p)
        print(f"[eval] WARNING: --checkpoint-path {p} not found locally; "
              f"manifest policy_checkpoint_hash falls back to hashing the "
              f"path string only, not the checkpoint", flush=True)
        return manifest_hash.canonical_hash({"checkpoint_path": str(p)})
    print("[eval] WARNING: --policy server with no --checkpoint-path given; "
          "manifest policy_checkpoint_hash is a host:port PLACEHOLDER, not "
          "a real checkpoint fingerprint -- pass --checkpoint-path for a "
          "real one before treating this run as benchmark evidence",
          flush=True)
    return manifest_hash.canonical_hash(
        {"server_host": args.host, "server_port": args.port,
         "note": "no --checkpoint-path given"})


def _failure_label(res):
    """None on success, else the last staged-progress checkpoint the
    episode reached before stalling (grasp/lift/hover/place order, per
    robo/tasks/pi05_tasks.py TaskScorer) -- e.g. 'stalled_after_lift' if it
    grasped and lifted but never hovered/placed, 'no_grasp' if it never even
    grasped the target."""
    if res["success"]:
        return None
    last_ok = None
    for stage in ("grasp", "lift", "hover", "place"):
        if res["stages"].get(stage):
            last_ok = stage
        else:
            break
    return f"stalled_after_{last_ok}" if last_ok else "no_grasp"


class ServerPolicy:
    def __init__(self, host, port, open_loop_horizon=15):
        self._host, self._port = host, port
        self.client = self._connect()
        self.horizon = open_loop_horizon
        self._chunk, self._i = None, 0

    def _connect(self):
        from openpi_client import websocket_client_policy
        return websocket_client_policy.WebsocketClientPolicy(
            host=self._host, port=self._port)

    def _infer_retry(self, req, tries=5):
        # first inference triggers JAX jit (minutes); the websockets 20 s
        # keepalive kills the connection meanwhile -> reconnect and resend
        # (same reason RoboLab's client has _infer_with_retry)
        for k in range(tries):
            try:
                return self.client.infer(req)
            except Exception as e:
                if k == tries - 1:
                    raise
                print(f"[eval] infer failed ({type(e).__name__}); "
                      f"reconnecting ({k + 1}/{tries})", flush=True)
                time.sleep(10)
                try:
                    self.client = self._connect()
                except Exception:
                    pass

    def warmup(self, obs, prompt):
        self._infer_retry(self._request(obs, prompt))

    def reset(self):
        self._chunk, self._i = None, 0

    @staticmethod
    def _request(obs, prompt):
        from openpi_client import image_tools

        return {
            "observation/exterior_image_1_left":
                image_tools.resize_with_pad(
                    obs["observation/exterior_image_1_left"], 224, 224),
            "observation/wrist_image_left":
                image_tools.resize_with_pad(
                    obs["observation/wrist_image_left"], 224, 224),
            "observation/joint_position":
                obs["observation/joint_position"],
            "observation/gripper_position":
                obs["observation/gripper_position"],
            "prompt": prompt,
        }

    def __call__(self, obs, prompt):
        if self._chunk is None or self._i >= min(self.horizon,
                                                 len(self._chunk)):
            res = self._infer_retry(self._request(obs, prompt))
            self._chunk = np.asarray(res["actions"])
            assert self._chunk.ndim == 2 and self._chunk.shape[1] >= 8, \
                f"unexpected action chunk {self._chunk.shape}"
            self._i = 0
        a = self._chunk[self._i, :8]
        self._i += 1
        return a


class ScriptedPolicy:
    """Env/scorer smoke test: sinusoidal joint sway + gripper open/close."""

    def __init__(self, home):
        self.home, self.t = home, 0

    def reset(self):
        self.t = 0

    def __call__(self, obs, prompt):
        self.t += 1
        a = np.zeros(8)
        a[:7] = self.home + 0.25 * np.sin(self.t / 22.0) * np.array(
            [1.0, 0.5, 0.0, 0.4, 0.0, -0.4, 0.0])
        a[7] = 1.0 if (self.t // 45) % 2 else 0.0
        return a


def run_episode(env, task, policy, prompt, time_limit_s, video_frames=None,
                jitter=0.0, rng=None, debug_dir=None):
    obs = env.reset(jitter_body=task["target"] if jitter > 0 else None,
                    jitter_xy=jitter, rng=rng)
    policy.reset()
    scorer = pi05_tasks.TaskScorer(env, task)
    ticks = int(time_limit_s * pi05_env.rig.CONTROL_HZ)
    t_infer = 0.0
    grip_cmds, dq_means = [], []
    tgt_bid = env.model.body(task["target"]).id
    ee_dists = []
    for k in range(ticks):
        t0 = time.time()
        action = policy(obs, prompt)
        t_infer += time.time() - t0
        if debug_dir and k in (0, 75, 150):
            from openpi_client import image_tools
            import imageio.v2 as iio
            for key, tag in (("observation/exterior_image_1_left", "ext"),
                             ("observation/wrist_image_left", "wr")):
                iio.imwrite(Path(debug_dir) /
                            f"{task['task_id']}_t{k}_{tag}224.png",
                            image_tools.resize_with_pad(obs[key], 224, 224))
        grip_cmds.append(float(action[7]))
        dq_means.append(float(np.abs(
            action[:7] - obs["observation/joint_position"]).mean()))
        pinch = env.data.site("robot/2f85/pinch").xpos
        ee_dists.append(float(np.linalg.norm(
            pinch - env.data.xpos[tgt_bid])))
        env.apply_action(action)
        obs = env.get_obs()
        scorer.update()
        if video_frames is not None:
            ext = obs["observation/exterior_image_1_left"]
            wr = obs["observation/wrist_image_left"]
            h = ext.shape[0]
            scale = h / wr.shape[0]
            import cv2
            wr = cv2.resize(wr, (int(wr.shape[1] * scale), h))
            video_frames.append(np.concatenate([ext, wr], axis=1))
        if scorer.success:
            break
    out = scorer.summary()
    g = np.asarray(grip_cmds)
    out.update(ticks=k + 1, infer_s=round(t_infer, 2),
               grip_close_frac=float((g > 0.5).mean()),
               grip_first_close=int(np.argmax(g > 0.5)) if (g > 0.5).any()
               else -1,
               dq_mean=float(np.mean(dq_means)),
               ee_dist_min=float(np.min(ee_dists)),
               ee_dist_final=float(ee_dists[-1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True,
                    help="pi05_tasks.json suite file(s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="server",
                    choices=["server", "scripted"])
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--checkpoint-path", default="",
                    help="local path to the served policy checkpoint, used "
                         "only for robo.manifest provenance "
                         "(policy_checkpoint_hash); leave unset for "
                         "--policy scripted, or when serving from a "
                         "non-local path (gs://...) in which case the "
                         "manifest records a documented placeholder instead")
    ap.add_argument("--open-loop-horizon", type=int, default=15)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--variant", default="default",
                    choices=["vague", "default", "specific"])
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="uniform xy jitter (m) on the target before settle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--task-filter", default="")
    ap.add_argument("--time-limit", type=float, default=0.0,
                    help="override suite time_limit_s")
    ap.add_argument("--debug-obs", action="store_true",
                    help="dump the exact 224px policy inputs at 3 ticks")
    ap.add_argument("--obs", default="raster",
                    choices=["raster", "composite"],
                    help="composite = gsplat photoreal (GPU; single suite; "
                         "requires SIMANY_SCENE/SIMANY_OUT to match the suite)")
    # --- demo-render overrides. These change what the POLICY SEES, so they
    # must never be used for benchmark numbers: clutter realism is the point
    # of the benchmark. They exist so a presentable clip can be rendered
    # without editing (and thereby invalidating) the task suite.
    ap.add_argument("--ext-cam-frame", default="",
                    help="DEMO ONLY: pin the exterior camera to this scan "
                         "frame (e.g. DSC01616) instead of the suite's pick")
    ap.add_argument("--demo-declutter", action="store_true",
                    help="DEMO ONLY: composite in only the task target and "
                         "receptacle; every other object is dropped from the "
                         "sim, so clean_background.ply shows through where it "
                         "was removed and inpainted")
    ap.add_argument("--render-wh", type=int, nargs=2, default=(640, 360),
                    metavar=("W", "H"),
                    help="composite render size per camera (policy input is "
                         "resized to 224 regardless; 1280 720 for demos)")
    args = ap.parse_args()
    if args.obs == "composite":
        assert len(args.tasks) == 1, \
            "composite mode: one suite per invocation (common.py binds the " \
            "scene at import time)"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # --- Task 01 (robo.manifest): per-run provenance, computed once -------
    git_snap = manifest_hash.git_snapshot()
    controller_config_hash, camera_config_hash, action_convention, action_dim = \
        _frozen_config_hashes()
    policy_checkpoint_hash = _policy_checkpoint_hash(args)

    results = []
    for suite_path in args.tasks:
        suite = json.loads(Path(suite_path).read_text())
        # Stand-in for a full SceneBuildManifest hash (Tasks 09/20's job to
        # emit one) until the factory pipeline writes one: hashes exactly
        # the scene-identifying fields of the suite file itself, so a
        # rollout against a differently-posed/robot/table suite is
        # distinguishable even before that upstream manifest exists.
        scene_manifest_hash = manifest_hash.canonical_hash({
            "scene": suite["scene"], "scene_xml": suite["scene_xml"],
            "robot": suite["robot"], "table": suite["table"],
            "ext_cam": suite["ext_cam"],
            "exclude_objects": sorted(suite.get("exclude_objects", ())),
        })
        time_limit_s = args.time_limit or suite.get("time_limit_s", 16.0)
        fact_dir = Path(suite_path).parent.parent
        ext_cam = suite["ext_cam"]
        if args.ext_cam_frame:
            pinned = pi05_tasks.ext_cam_from_frame(args.ext_cam_frame)
            if pinned is None:
                raise SystemExit(
                    f"--ext-cam-frame {args.ext_cam_frame} is not a scan frame "
                    f"of {suite['scene']}")
            print(f"[eval] DEMO ext cam pinned to {pinned['frame']} "
                  f"(suite default was {ext_cam.get('frame')})", flush=True)
            ext_cam = pinned
        exclude = list(suite.get("exclude_objects", ()))
        if args.demo_declutter:
            # Union over the whole suite so one env serves every task in it.
            keep = {t["target"] for t in suite["tasks"]}
            keep |= {t["receptacle"] for t in suite["tasks"] if t.get("receptacle")}
            rows = pi05_tasks._load_objects(fact_dir)
            dropped = [o["name"] for o in rows if o["name"] not in keep]
            exclude = sorted(set(exclude) | set(dropped))
            print(f"[eval] DEMO declutter: keeping {sorted(keep)}, "
                  f"dropping {len(dropped)} objects", flush=True)
        env = pi05_env.DroidSimEnv(
            suite["scene_xml"], suite["robot"]["base_pos"],
            suite["robot"]["base_yaw"], table_box=suite["table"],
            ext_cam=ext_cam,
            exclude_objects=tuple(exclude),
            xml_dump=out_dir / f"{suite['scene']}_rig.xml")
        env._task_rows = pi05_tasks._load_objects(fact_dir)
        if args.obs == "composite":
            from robo.rendering import pi05_render
            env.composite = pi05_render.CompositeObs(
                env, fact_dir,
                width=args.render_wh[0], height=args.render_wh[1])
        if args.policy == "server":
            policy = ServerPolicy(args.host, args.port,
                                  args.open_loop_horizon)
            print("[eval] warmup inference (jit compile)...", flush=True)
            policy.warmup(env.reset(settle_s=0.1), "warmup")
            print("[eval] warmup done", flush=True)
        else:
            policy = ScriptedPolicy(env.info["home"])

        for task in suite["tasks"]:
            if args.task_filter and args.task_filter not in task["task_id"]:
                continue
            prompt = task["instructions"][args.variant]
            for ep in range(args.episodes):
                frames = [] if args.video else None
                res = run_episode(env, task, policy, prompt,
                                  time_limit_s, frames,
                                  jitter=args.jitter if ep > 0 else 0.0,
                                  rng=rng,
                                  debug_dir=out_dir if args.debug_obs
                                  else None)
                res.update(task_id=task["task_id"], episode=ep,
                           prompt=prompt, scene=suite["scene"])
                results.append(res)
                tag = "OK " if res["success"] else \
                    f"{res['score']:.2f}"
                print(f"[eval] {tag} {task['task_id']} ep{ep} "
                      f"stages={res['stages']} ticks={res['ticks']}")
                video_path = None
                if frames:
                    import imageio.v2 as imageio
                    video_path = out_dir / f"{task['task_id']}_ep{ep}.mp4"
                    imageio.mimwrite(
                        video_path, frames, fps=pi05_env.rig.CONTROL_HZ,
                        quality=8, macro_block_size=None)

                # --- Task 01 (robo.manifest): emit alongside results.json.
                # Never overwrites a completed manifest with different
                # content (manifest_io.write_manifest's guard) -- a rerun
                # of the exact same config into the same --out is an
                # idempotent no-op; a rerun with a changed contract raises.
                rollout_manifest = RolloutManifest(
                    created_utc=datetime.datetime.now(datetime.timezone.utc)
                        .isoformat(),
                    git_dirty=git_snap["dirty"],
                    scene_build_commit=git_snap["commit"],
                    scene_manifest_hash=scene_manifest_hash,
                    policy_checkpoint_hash=policy_checkpoint_hash,
                    controller_config_hash=controller_config_hash,
                    camera_config_hash=camera_config_hash,
                    task_id=task["task_id"],
                    initial_state_id=f"{task['task_id']}__ep{ep}",
                    rollout_seed=args.seed,
                    scene_id=suite["scene"],
                    language=prompt,
                    rubric_version=RUBRIC_VERSION,
                    horizon_s=time_limit_s,
                    observation_preprocessing=ObservationPreprocessing(
                        mode=args.obs, resize_hw=(224, 224)),
                    action_convention=action_convention,
                    action_dim=action_dim,
                    video_path=str(video_path) if video_path else None,
                    success=bool(res["success"]),
                    staged_progress=StagedProgress(**res["stages"]),
                    failure_label=_failure_label(res),
                )
                manifest_io.write_manifest(
                    out_dir / "manifests" / f"{task['task_id']}__ep{ep}",
                    rollout_manifest)

    n = len(results)
    succ = sum(r["success"] for r in results)
    mean_score = float(np.mean([r["score"] for r in results])) if n else 0.0
    summary = {"n_episodes": n, "successes": succ,
               "success_rate": succ / n if n else 0.0,
               "mean_score": mean_score, "variant": args.variant,
               "policy": args.policy, "results": results}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=1))
    print(f"[eval] {succ}/{n} success, mean staged score {mean_score:.3f} "
          f"-> {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
