"""robo.policy.clients: adapters normalizing each real policy-serving
backend onto robo.policy.control_contract.PolicyClient (one
environment-facing 8-dim action schema, explicit warmup(), schema-validated
__call__).

`CLIENT_KIND_TO_CLASS` maps a `robo.policy.registry.PolicyEntry.client_kind`
string to the class implementing it; `PolicyRegistry.make_client` uses this
rather than importing individual client modules by name, so adding a new
client kind is a one-line addition here plus a new module, not a change to
the registry itself. `"unavailable"` deliberately has no entry --
`PolicyRegistry.make_client` raises `PolicyUnavailableError` before this
mapping is ever consulted for an unavailable policy.
"""
from robo.policy.clients.pi05_client import Pi05PolicyClient
from robo.policy.clients.scripted_client import ScriptedPolicyClient

CLIENT_KIND_TO_CLASS = {
    "pi05_server": Pi05PolicyClient,
    "scripted": ScriptedPolicyClient,
}

__all__ = ["Pi05PolicyClient", "ScriptedPolicyClient", "CLIENT_KIND_TO_CLASS"]
