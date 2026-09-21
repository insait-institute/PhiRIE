"""Serve the pinned official RoboCasa policy with explicit paired RNG keys."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading

from robo.roundtrip.native_policy import CHECKPOINT_REVISION, POLICY_CONFIG, UPSTREAM_COMMIT


def verified_checkpoint(receipt_path: Path, checkpoint: Path) -> dict:
    raw = receipt_path.read_bytes()
    receipt = json.loads(raw)
    if receipt.get("status") != "PASS" or receipt.get("revision") != CHECKPOINT_REVISION:
        raise ValueError("checkpoint receipt is not the pinned verified revision")
    files = receipt.get("files", [])
    if len(files) != 20 or receipt.get("total_bytes") != 12440863932:
        raise ValueError("checkpoint inference file roster/size differs")
    for member in files:
        path = Path(member["path"])
        if not path.is_relative_to(checkpoint) or not path.is_file():
            raise ValueError("checkpoint member is missing or outside checkpoint root")
        if path.stat().st_size != member["size_bytes"]:
            raise ValueError("checkpoint member size differs")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != member["sha256"]:
            raise ValueError(f"checkpoint bytes changed: {path.name}")
    norm = checkpoint / "assets/norm_stats.json"
    return {"checkpoint_receipt_sha256": hashlib.sha256(raw).hexdigest(),
            "checkpoint_revision": CHECKPOINT_REVISION,
            "checkpoint_total_bytes": receipt["total_bytes"],
            "normalization_sha256": hashlib.sha256(norm.read_bytes()).hexdigest()}


class SeededPolicy:
    def __init__(self, policy):
        self.policy = policy
        self._request_lock = threading.Lock()

    def infer(self, observation):
        import jax
        observation = dict(observation)
        seed = observation.pop("_simany_rng_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("paired chunk RNG seed must be uint32")
        # Atomic across threaded callers: one request cannot replace another's
        # key between key assignment and model inference. No connection RNG.
        with self._request_lock:
            self.policy._rng = jax.random.key(seed)
            result = self.policy.infer(observation)
        return {**result, "simany_rng_seed": seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--metadata-out", type=Path)
    parser.add_argument("--engine-id")
    parser.add_argument("--engine-protocol")
    parser.add_argument("--canonical-instance-id")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "-C", str(args.openpi_root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(args.openpi_root), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if commit != UPSTREAM_COMMIT or dirty:
        raise ValueError("official OpenPI checkout must match the clean pinned revision")
    metadata = verified_checkpoint(args.receipt.resolve(), args.checkpoint.resolve())
    tokenizer = Path(os.environ["OPENPI_DATA_HOME"]) / "big_vision/paligemma_tokenizer.model"
    tokenizer_hash = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    if tokenizer_hash != "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6":
        raise ValueError("PaliGemma tokenizer content differs from declared cache")
    metadata.update({"tokenizer_sha256": tokenizer_hash, "policy_config": POLICY_CONFIG, "openpi_commit": commit,
                     "rng_protocol": "simany-sr0-policy-v1/per-chunk-jax-key",
                     "action_dim": 12, "model_action_dim": 32,
                     "model_action_horizon": 50, "replan_steps": 5,
                     "policy_type": "learned_visual_pi05_robocasa",
                     "native_camera_resolution": [256, 256], "network_resolution": [224, 224]})
    from openpi.training.config import DataConfig, get_config
    from openpi.shared.normalize import load as load_norm_stats
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    config = get_config(POLICY_CONFIG)
    norm_stats = load_norm_stats(args.checkpoint.resolve() / "assets")
    # Official data.create otherwise reads all training dataset metadata before
    # create_trained_policy reaches its checkpoint-local normalization loader.
    # Supply those same checkpoint stats through the existing public config API.
    config = dataclasses.replace(config, data=dataclasses.replace(
        config.data, base_config=dataclasses.replace(
            config.data.base_config or DataConfig(), norm_stats=norm_stats)))
    policy = create_trained_policy(config, args.checkpoint.resolve(), norm_stats=norm_stats)
    if any((args.engine_id,args.engine_protocol,args.canonical_instance_id)):
        from robo.roundtrip.policy_engine import engine_metadata
        metadata['policy_engine']=engine_metadata(args.engine_id,args.engine_protocol,args.canonical_instance_id)
    if args.metadata_out:
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metadata_out.open("x") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
    print(json.dumps({"event": "policy_loaded", **metadata}), flush=True)
    WebsocketPolicyServer(SeededPolicy(policy), host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
