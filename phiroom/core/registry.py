"""Feature contracts are package resources, loaded without importing backends."""

from __future__ import annotations

import json
from importlib.resources import files


def catalog() -> dict[str, dict]:
    result = {}
    for path in sorted(files("phiroom.modules").iterdir(), key=lambda p: p.name):
        if not path.name.endswith(".json"):
            continue
        block = json.loads(path.read_text())
        name = block["id"]
        if name in result or path.name != name + ".json":
            raise ValueError(f"duplicate or mismatched module id: {name}")
        if (
            not block.get("actions")
            or not block.get("inputs")
            or not block.get("outputs")
        ):
            raise ValueError(f"incomplete feature contract: {name}")
        result[name] = block
    return result


def action(module: str, name: str) -> dict:
    modules = catalog()
    if module not in modules:
        raise ValueError(f"unknown module {module!r}; choose: {', '.join(modules)}")
    actions = modules[module]["actions"]
    if name not in actions:
        raise ValueError(
            f"unknown action {module}/{name}; choose: {', '.join(actions)}"
        )
    return actions[name]
