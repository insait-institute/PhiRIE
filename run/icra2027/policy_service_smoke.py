"""Synthetic bound-policy service smoke; no environment or episode execution.

Authenticates existing HTTP/websocket identity, warms up once, and consumes
15 client actions from the next chunk using immutable zero RGB/fixed joints.
Every exit preserves a receipt and logs; only this process's child group is
terminated. Configuration and source must match an already-published E0 freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

import numpy as np
import yaml

from robo.manifest.hash import canonical_hash
from robo.policy.clients.pi05_client import Pi05PolicyClient
from robo.policy.registry import PolicyRegistry
from robo.policy.runtime_identity import (
    build_server_identity, runtime_checkpoint_path, validate_server_identity,
    verify_clean_git_checkout,
)

ROOT = Path(__file__).resolve().parents[2]
SCOPE = "synthetic_policy_service_infrastructure_only"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def command(argv):
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT,
                                   timeout=20).strip()


def synthetic_observation(config):
    """Construct only the declared synthetic observation; never reads images."""
    value = config["synthetic_observation"]
    if (config.get("study_scope") != SCOPE or config.get("paper_ready") is not False
            or config.get("policy_id") != "pi05_droid_jointpos"
            or config.get("action_count") != 15 or config.get("action_dim") != 8
            or value["image_shape"] != [224, 224, 3]
            or value["image_dtype"] != "uint8" or value["image_fill"] != 0):
        raise ValueError("synthetic-only policy service contract differs")
    joints = np.asarray(value["joint_position"], dtype=np.float32)
    gripper = np.asarray(value["gripper_position"], dtype=np.float32)
    if joints.shape != (7,) or gripper.shape != (1,) or not (
            np.isfinite(joints).all() and np.isfinite(gripper).all()):
        raise ValueError("synthetic fixed joint/gripper observation is invalid")
    observation = {
        "observation/exterior_image_1_left": np.zeros((224,224,3),dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((224,224,3),dtype=np.uint8),
        "observation/joint_position": joints,
        "observation/gripper_position": gripper,
    }
    for array in observation.values():
        array.flags.writeable = False
    return observation, value["prompt"]


def observation_manifest(observation, prompt):
    return {"prompt":prompt,"arrays":{
        name:{"shape":list(value.shape),"dtype":str(value.dtype),
              "sha256":hashlib.sha256(value.tobytes()).hexdigest()}
        for name,value in observation.items()}}


def check_contract(contract_dir, config_path, config, expected_commit):
    manifest_path = contract_dir / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_hash = canonical_hash({k:v for k,v in manifest.items()
        if k not in {"created_utc","environment","contract_sha256"}})
    if manifest.get("contract_sha256") != expected_hash:
        raise ValueError("E0 contract checksum differs")
    if (manifest["freeze_id"] != config["freeze_id"]
            or manifest["code"]["commit"] != expected_commit
            or manifest["code"]["dirty"] is not False
            or manifest.get("paper_ready") is not False):
        raise ValueError("E0 freeze/source scope differs")
    inventory = manifest["resource_inventory"]
    for path in (config_path, Path(__file__), ROOT/"run/icra2027/policy_service_smoke.sbatch"):
        matches = [r for r in inventory if r.get("resolved_path") == str(path.resolve())]
        if len(matches) != 1 or matches[0].get("hash_method") != "content_sha256" \
                or matches[0].get("sha256") != sha256(path):
            raise ValueError(f"E0 inventory does not authenticate {path}")
    preflight = (contract_dir/"preflight.log").read_text()
    if "PREFLIGHT=PASS" not in preflight:
        raise ValueError("E0 contract lacks completed exact-source preflight")
    return {"path":str(manifest_path),"sha256":sha256(manifest_path),
            "contract_sha256":expected_hash}


class LoggedTransport:
    """Observe calls to the existing OpenPI transport; preserve its protocol."""
    def __init__(self, client, events):
        self.client, self.events = client, events

    def get_server_metadata(self):
        return self.client.get_server_metadata()

    def infer(self, request):
        start = time.monotonic()
        event = {"request_index":len(self.events),"status":"started"}
        self.events.append(event)
        try:
            response = self.client.infer(request)
            actions = np.asarray(response["actions"])
            event["raw_chunk_shape"] = list(actions.shape)
            if actions.ndim != 2 or actions.shape[0] < 15 or actions.shape[1] < 8:
                raise ValueError(f"raw policy action chunk shape differs: {actions.shape}")
            if not np.isfinite(actions[:,:8]).all():
                raise ValueError("raw policy action chunk contains nonfinite values")
            event["raw_first_eight_channels_sha256"] = hashlib.sha256(
                np.ascontiguousarray(actions[:,:8]).tobytes()).hexdigest()
            event["status"] = "finite_chunk_received"
            return response
        except Exception as exc:
            event.update(status="failed",error_type=type(exc).__name__,error=str(exc))
            raise
        finally:
            event["elapsed_s"] = time.monotonic()-start


def stop_child(process, grace_s):
    if process is None:
        return {"started":False}
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
    return {"started":True,"pid":process.pid,"returncode":process.returncode,
            "terminated":process.poll() is not None}


def run(args):
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    process = None
    stage = "source_and_contract"
    events = []
    result = {"schema_version":1,"study_scope":SCOPE,"paper_ready":False,
              "freeze_id":config["freeze_id"],"status":"FAIL",
              "actions_completed":0,"manipulation_episodes":0,
              "interpretation":"Service/identity/action-shape check only; no manipulation evidence.",
              "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}

    def deadline(signum, frame):
        raise TimeoutError(f"bounded service smoke interrupted by signal {signum}")

    for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):
        signal.signal(signum, deadline)
    signal.alarm(int(config["total_timeout_s"]))
    try:
        result["source"] = verify_clean_git_checkout(ROOT,args.expected_code_commit,owner="SimAny service smoke")
        result["e0_contract"] = check_contract(args.contract.resolve(),config_path,config,args.expected_code_commit)
        result["openpi"] = verify_clean_git_checkout(config["openpi_root"],config["openpi_commit"],owner="OpenPI")
        observation,prompt = synthetic_observation(config)
        observation_identity = observation_manifest(observation,prompt)
        write_new(output/"synthetic_observation.json",observation_identity)
        write_new(output/"config_resolved.json",config)
        if not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("GPU smoke must execute inside its Slurm allocation")
        hostname = socket.gethostname().split(".")[0]
        if hostname != config["slurm"]["node"]:
            raise ValueError(f"allocation node differs: {hostname}")
        allocation = {key:os.environ.get(key) for key in (
            "SLURM_JOB_ID","SLURM_JOB_NODELIST","SLURM_JOB_PARTITION","SLURM_CPUS_PER_TASK",
            "SLURM_MEM_PER_NODE","SLURM_JOB_GPUS","SLURM_STEP_GPUS","CUDA_VISIBLE_DEVICES")}
        result["runtime"] = {"hostname":hostname,"python":sys.executable,
            "python_version":sys.version,"allocation":allocation,
            "scontrol_job":command(["scontrol","show","job","-o",os.environ["SLURM_JOB_ID"]]),
            "gpu_inventory":command(["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.total","--format=csv,noheader"]),
            "package_versions":{name:importlib.metadata.version(name) for name in ("jax","jaxlib","numpy","websockets")}}
        write_new(output/"runtime_start.json",result["runtime"])
        registry = PolicyRegistry.from_config_dir()
        entry = registry.get(config["policy_id"])
        registry.verify_checkpoint_hash(entry.id)
        if registry.validate_control_contract(entry.id):
            raise ValueError("registry policy control contract differs")
        # Job-scoped scratch prevents accidentally reusing a service/identity receipt.
        cache = Path(os.environ.get("SIMANY_SCRATCH", "/scratch/" + os.environ.get("USER", "phirie")))/"icra2027/policy-service"/config["freeze_id"]/os.environ["SLURM_JOB_ID"]
        cache.mkdir(parents=True,exist_ok=False)
        result["checkpoint_cache_root"] = str(cache)
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1",0))
            port = reservation.getsockname()[1]
        expected = build_server_identity(policy_id=entry.id,checkpoint_path=entry.checkpoint_path,
            checkpoint_fingerprint=entry.checkpoint_hash,training_config=entry.training_config,
            openpi_root=config["openpi_root"],openpi_commit=config["openpi_commit"],port=port,
            runtime_checkpoint_path_value=runtime_checkpoint_path(cache,entry.id,entry.checkpoint_hash),
            verify_runtime_checkpoint_now=False)
        write_new(output/"expected_server_identity.json",expected)
        identity_path = output/"server_identity.json"
        env = dict(os.environ, SIMANY_POLICY_ID=entry.id,SIMANY_OPENPI_ROOT=config["openpi_root"],
            SIMANY_OPENPI_COMMIT=config["openpi_commit"],SIMANY_OPENPI_PYTHON=config["openpi_python"],
            SIMANY_POLICY_IDENTITY_FILE=str(identity_path),SIMANY_CHECKPOINT_CACHE_ROOT=str(cache),
            PYTHONDONTWRITEBYTECODE="1", XLA_PYTHON_CLIENT_MEM_FRACTION="0.85")
        stage = "checkpoint_restore_and_health"
        with (output/"server.log").open("x") as server_log:
            process = subprocess.Popen(["bash",str(ROOT/"run/pi05_serve_bound.sh"),"--port",str(port)],
                cwd=ROOT,env=env,stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True)
            startup = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"bound server exited before readiness: {process.returncode}")
                if time.monotonic()-startup > config["startup_timeout_s"]:
                    raise TimeoutError("bound server readiness timeout")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz",timeout=3) as response:
                        health = json.load(response)
                except (urllib.error.URLError,TimeoutError,ConnectionError):
                    time.sleep(1)
                    continue
                if health.get("status") != "ok":
                    raise ValueError("bound server health is not ok")
                receipt = json.loads(identity_path.read_text())
                validated = validate_server_identity(health.get("identity"),expected=expected)
                if validated != receipt:
                    raise ValueError("health identity differs from restored server receipt")
                result["health_identity"] = validated
                result["startup_elapsed_s"] = time.monotonic()-startup
                write_new(output/"health_verified.json",health)
                break
            stage = "websocket_identity"
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
            def connector():
                return LoggedTransport(WebsocketClientPolicy(host="127.0.0.1",port=port),events)
            client = Pi05PolicyClient.from_entry(entry,host="127.0.0.1",port=port,
                connector=connector,expected_server_identity=expected,strict_warmup=True)
            result["websocket_identity"] = client.verify_server_identity()
            if result["websocket_identity"] != result["health_identity"]:
                raise ValueError("websocket and health identity differ")
            stage = "synthetic_warmup"
            warmup_start = time.monotonic()
            client.warmup(observation,prompt)
            result["warmup_elapsed_s"] = time.monotonic()-warmup_start
            stage = "synthetic_actions"
            actions = []
            with (output/"actions.jsonl").open("x") as action_log:
                for index in range(config["action_count"]):
                    action_start = time.monotonic()
                    action = client(observation,prompt)
                    if action.shape != (8,) or not np.isfinite(action).all():
                        raise ValueError("client action failed finite 8-dimensional check")
                    row = {"index":index,"action":action.tolist(),"client_call_elapsed_s":time.monotonic()-action_start}
                    action_log.write(json.dumps(row,allow_nan=False)+"\n");action_log.flush()
                    actions.append(row);result["actions_completed"] = len(actions)
            if observation_manifest(observation,prompt) != observation_identity:
                raise ValueError("synthetic observation changed during inference")
            result["action_shape"] = [len(actions),8]
            result["action_sampling"] = "15 sequential client actions after warmup; chunk-cache calls are not independent inference latency measurements."
            processes = command(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_gpu_memory","--format=csv,noheader"])
            result["server_gpu_processes"] = [line for line in processes.splitlines() if line.split(",",1)[0].strip()==str(process.pid)]
            if not result["server_gpu_processes"]:
                raise ValueError("cannot verify bound server PID on an actual GPU")
            verify_clean_git_checkout(ROOT,args.expected_code_commit,owner="SimAny after smoke")
            verify_clean_git_checkout(config["openpi_root"],config["openpi_commit"],owner="OpenPI after smoke")
            result["status"] = "PASS"
    except Exception as exc:
        result.update(failed_stage=stage,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        result["server_cleanup"] = stop_child(process,config["server_terminate_grace_s"])
        result["transport_requests"] = events
        result["elapsed_s"] = time.monotonic()-start
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
        write_new(output/("result.json" if result["status"] == "PASS" else "failure.json"),result)
        print(json.dumps({k:result[k] for k in ("status","freeze_id","actions_completed","elapsed_s","paper_ready")}),flush=True)
    return 0 if result["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--expected-code-commit",required=True)
    parser.add_argument("--contract",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
