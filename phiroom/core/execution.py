"""Pure command planning and fail-fast sequential execution with operational receipts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from phiroom.core.registry import action
from phiroom.core.runtime import interpreter


@dataclass(frozen=True)
class Command:
    module: str
    action: str
    argv: tuple[str, ...]
    cwd: str
    env: dict[str, str]
    runtime: str
    needs_gpu: bool
    requires: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()


def plan(
    module: str,
    name: str,
    args: list[str],
    root: Path,
    cfg: dict,
    requires: list[str] = (),
    produces: list[str] = (),
) -> Command:
    if not isinstance(module, str) or not isinstance(name, str):
        raise ValueError("module and action must be strings")
    spec = action(module, name)
    kind = spec.get("kind", "python")
    runtime = spec.get("runtime", "main")
    target = spec["target"]
    if kind == "python":
        file = root.joinpath(*target.split(".")).with_suffix(".py")
        argv = [interpreter(runtime, root, cfg), "-m", target]
    elif kind == "shell":
        file = root / target
        argv = ["bash", str(file)]
    elif kind == "phiview":
        file = root / "integrations/phiview/physicalview/cli.py"
        argv = ["bash", str(root / "run/phiview.sh")]
    else:
        raise ValueError(f"unknown command kind: {kind}")
    if not file.is_file():
        raise ValueError(
            f"missing implementation: {file}; initialize submodules if needed"
        )
    if any(not isinstance(v, str) or "\0" in v for v in args):
        raise ValueError("backend arguments must be strings without NUL characters")
    exports = {"SIMANY_ROOT": str(root), "PYTHONNOUSERSITE": "1", **cfg.get("env", {})}
    exports["SIMANY_ROOT"] = str(root)
    # Legacy shell launchers and preflight consume these same aliases.
    if kind == "shell" or target == "simfactory.runner":
        from phiroom.core.runtime import RUNTIMES

        exports.update(
            {key: interpreter(alias, root, cfg) for alias, (key, _) in RUNTIMES.items()}
        )
        exports["SIMANY_CAMPAIGN_PY"] = interpreter("control", root, cfg)

    def paths(values):
        if any(not isinstance(v, str) or not v for v in values):
            raise ValueError("artifact paths must be nonempty strings")
        return tuple(str(root / Path(v).expanduser()) for v in values)

    return Command(
        module,
        name,
        tuple(argv + spec.get("args", []) + list(args)),
        str(root),
        exports,
        runtime,
        spec.get("gpu", False),
        paths(requires),
        paths(produces),
    )


def pipeline(path: Path, root: Path, cfg: dict) -> list[Command]:
    recipe = json.loads(path.read_text())
    if (
        not isinstance(recipe, dict)
        or set(recipe) != {"schema_version", "steps"}
        or recipe["schema_version"] != 1
    ):
        raise ValueError("pipeline needs schema_version=1 and steps")
    if not isinstance(recipe["steps"], list) or not recipe["steps"]:
        raise ValueError("pipeline needs at least one step")
    commands = []
    for step in recipe["steps"]:
        if not isinstance(step, dict) or set(step) - {
            "module",
            "action",
            "args",
            "requires",
            "produces",
        }:
            raise ValueError("invalid pipeline step keys")
        if not {"module", "action"} <= step.keys():
            raise ValueError("each step needs a module and action")
        for field in ("args", "requires", "produces"):
            if not isinstance(step.get(field, []), list):
                raise ValueError(f"step {field} must be a list")
        commands.append(
            plan(
                step["module"],
                step["action"],
                step.get("args", []),
                root,
                cfg,
                step.get("requires", []),
                step.get("produces", []),
            )
        )
    return commands


def execute(commands: list[Command], receipts: Path) -> tuple[int, Path]:
    attempt = receipts / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:12]
    )
    attempt.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema_version": 1,
        "kind": "operational_execution",
        "status": "running",
        "scientific_validation": "NOT_RUN",
        "steps": [],
    }
    path = attempt / "receipt.json"

    def save():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(path)

    save()
    code = 0
    for number, command in enumerate(commands):
        row = {**asdict(command), "status": "running", "returncode": None}
        receipt["steps"].append(row)
        save()
        try:
            if not shutil.which(command.argv[0]):
                raise ValueError(
                    f"missing executable for {command.runtime}: {command.argv[0]}"
                )
            missing = [p for p in command.requires if not Path(p).exists()]
            if missing:
                raise ValueError(f"required inputs missing: {missing}")
            existing = [p for p in command.produces if Path(p).exists()]
            if existing:
                raise ValueError(
                    f"expected outputs already exist; choose a fresh output location: {existing}"
                )
            env = {**os.environ, **command.env}
            env["PYTHONPATH"] = command.cwd + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
            )
            log = attempt / f"{number:03d}-{command.module}-{command.action}.log"
            row["log"] = str(log)
            print(f"Running {command.module}/{command.action}; log: {log}", flush=True)
            with log.open("w") as stream:
                completed = subprocess.run(
                    command.argv,
                    cwd=command.cwd,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            code = completed.returncode
            row["returncode"] = code
            if code:
                raise ValueError(f"backend exited {code}; see {log}")
            missing = [p for p in command.produces if not Path(p).exists()]
            if missing:
                raise ValueError(
                    f"backend exited zero but expected outputs are missing: {missing}"
                )
            row["status"] = "passed"
        except (OSError, ValueError, KeyboardInterrupt) as error:
            code = (
                130
                if isinstance(error, KeyboardInterrupt)
                else (code if code > 0 else 1)
            )
            row.update(
                status="interrupted" if code == 130 else "failed", error=str(error)
            )
            receipt["status"] = row["status"]
            save()
            break
        save()
    else:
        receipt["status"] = "passed"
    receipt["steps_not_run"] = len(commands) - len(receipt["steps"])
    save()
    return code, path
