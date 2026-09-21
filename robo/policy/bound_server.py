"""Serve one registry policy with fail-closed runtime identity.

Run this module with an OpenPI environment but with ``PYTHONPATH`` pointing
at the explicitly selected clean OpenPI worktree.  The launcher validates
the imported module paths again, so an editable environment cannot silently
redirect execution to a different (for example, dirty) checkout.
"""
from __future__ import annotations

import argparse
import http
import inspect
import json
from pathlib import Path

from robo.policy.control_contract import ControlContractMismatchError
from robo.policy.registry import PolicyRegistry, PolicyUnavailableError
from robo.policy.runtime_identity import (
    IDENTITY_METADATA_KEY,
    PolicyRuntimeIdentityError,
    build_server_identity,
    health_document,
    stage_checkpoint,
)


def _atomic_json(path: Path, value: dict) -> None:
    path = path.expanduser()
    if not path.is_absolute():
        raise PolicyRuntimeIdentityError("identity file path must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _require_import_from(module, root: Path, relative_root: str) -> None:
    source = Path(inspect.getfile(module)).resolve()
    expected = (root / relative_root).resolve()
    if not source.is_relative_to(expected):
        raise PolicyRuntimeIdentityError(
            f"imported {module.__name__} from {source}, outside pinned {expected}")


def _install_identity_health_check(server_module, identity: dict) -> None:
    """Make OpenPI's existing websocket port return the identity at healthz."""
    body = health_document(identity)

    def identity_health_check(connection, request):
        if request.path == "/healthz":
            response = connection.respond(http.HTTPStatus.OK, body)
            response.headers["Content-Type"] = "application/json"
            return response
        return None

    # WebsocketPolicyServer.run resolves this module global when it starts.
    # Keeping the protocol handler itself upstream avoids a second inference
    # implementation while giving health and websocket metadata one source.
    server_module._health_check = identity_health_check


def serve(
    *, policy_id: str, openpi_root: str | Path, openpi_commit: str,
    port: int, identity_file: str | Path,
    checkpoint_cache_root: str | Path | None = None,
    deterministic_sampling: bool = False,
) -> None:
    registry = PolicyRegistry.from_config_dir()
    entry = registry.get(policy_id)
    if entry.client_kind != "pi05_server" or entry.status == "unavailable":
        raise PolicyUnavailableError(
            f"policy {policy_id!r} is not an available pi05_server entry")
    if not entry.checkpoint_path or not entry.checkpoint_hash:
        raise PolicyRuntimeIdentityError(
            f"policy {policy_id!r} lacks checkpoint path/hash")
    if not entry.training_config:
        raise PolicyRuntimeIdentityError(
            f"policy {policy_id!r} lacks an OpenPI training config")
    registry.verify_checkpoint_hash(policy_id)
    control_issues = registry.validate_control_contract(policy_id)
    if control_issues:
        raise ControlContractMismatchError(
            f"policy {policy_id!r} control contract differs: "
            + "; ".join(control_issues))

    runtime_checkpoint = Path(entry.checkpoint_path)
    if checkpoint_cache_root is not None:
        runtime_checkpoint = stage_checkpoint(
            source_path=entry.checkpoint_path,
            checkpoint_fingerprint=entry.checkpoint_hash,
            cache_root=checkpoint_cache_root,
            policy_id=entry.id,
        )
    from robo.policy.sampling_contract import sampling_contract, ExplicitNoisePolicy
    sampling = sampling_contract(entry.training_config) if deterministic_sampling else None
    identity = build_server_identity(
        policy_id=entry.id,
        checkpoint_path=entry.checkpoint_path,
        checkpoint_fingerprint=entry.checkpoint_hash,
        training_config=entry.training_config,
        openpi_root=openpi_root,
        openpi_commit=openpi_commit,
        port=port,
        runtime_checkpoint_path_value=runtime_checkpoint,
        sampling=sampling,
    )
    root = Path(identity["openpi"]["root"])
    import openpi
    import openpi_client
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as training_config

    _require_import_from(openpi, root, "src")
    _require_import_from(openpi_client, root, "packages/openpi-client/src")
    selected_config = training_config.get_config(entry.training_config)
    if selected_config.name != entry.training_config:
        raise PolicyRuntimeIdentityError(
            "OpenPI returned a different training config than requested")
    policy = policy_config.create_trained_policy(
        selected_config, str(runtime_checkpoint))
    metadata = dict(policy.metadata or {})
    if sampling is not None:
        if [selected_config.model.action_horizon, selected_config.model.action_dim] != sampling["shape"]:
            raise PolicyRuntimeIdentityError("loaded model sampling dimensions differ")
        policy = ExplicitNoisePolicy(policy, sampling)
    if IDENTITY_METADATA_KEY in metadata:
        raise PolicyRuntimeIdentityError(
            f"upstream policy metadata already defines {IDENTITY_METADATA_KEY!r}")
    metadata[IDENTITY_METADATA_KEY] = identity
    _install_identity_health_check(websocket_policy_server, identity)
    # The policy is fully restored before this receipt appears.  Readiness is
    # established by polling /healthz, whose payload must equal this receipt.
    _atomic_json(Path(identity_file), identity)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=port, metadata=metadata,
    ).serve_forever()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--openpi-root", required=True)
    parser.add_argument("--openpi-commit", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--identity-file", required=True)
    parser.add_argument("--checkpoint-cache-root")
    parser.add_argument("--deterministic-sampling", action="store_true")
    args = parser.parse_args(argv)
    serve(
        policy_id=args.policy_id,
        openpi_root=args.openpi_root,
        openpi_commit=args.openpi_commit,
        port=args.port,
        identity_file=args.identity_file,
        checkpoint_cache_root=args.checkpoint_cache_root,
        deterministic_sampling=args.deterministic_sampling,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "serve", "_install_identity_health_check"]
