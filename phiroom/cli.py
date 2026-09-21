"""List, plan, and run independent feature modules or explicit pipeline recipes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from phiroom import __version__
from phiroom.core.execution import execute, pipeline, plan
from phiroom.core.registry import catalog
from phiroom.core.runtime import load_config, source_root


def main(argv=None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in tokens:
        split = tokens.index("--")
        tokens, extra = tokens[:split], tokens[split + 1 :]
    parser = argparse.ArgumentParser(prog="phiroom", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "modules", help="show all feature contracts without loading GPU code"
    )
    describe = sub.add_parser(
        "describe", help="show one feature's actions and interfaces"
    )
    describe.add_argument("module")
    for name in ("plan", "run", "pipeline"):
        child = sub.add_parser(name)
        if name == "pipeline":
            child.add_argument("operation", choices=("plan", "run"))
            child.add_argument("recipe", type=Path)
        else:
            child.add_argument("module")
            child.add_argument("action")
            child.add_argument("--require", action="append", default=[])
            child.add_argument("--expect", action="append", default=[])
        child.add_argument("--root", help="full PhiRoom source checkout")
        child.add_argument(
            "--config", help="JSON runtime paths and pipeline environment"
        )
        child.add_argument("--receipts", default="outputs/module-runs")
    args = parser.parse_args(tokens)
    try:
        if args.command in {"modules", "describe"}:
            if extra:
                raise ValueError("backend arguments are only supported by plan/run")
            modules = catalog()
            if args.command == "describe" and args.module not in modules:
                raise ValueError(f"unknown module: {args.module}")
            print(
                json.dumps(
                    modules if args.command == "modules" else modules[args.module],
                    indent=2,
                )
            )
            return 0
        root, cfg = source_root(args.root), load_config(args.config)
        if args.command == "pipeline":
            if extra:
                raise ValueError("pipeline arguments belong in each recipe step")
            commands = pipeline(args.recipe, root, cfg)
            operation = args.operation
        else:
            commands = [
                plan(
                    args.module,
                    args.action,
                    extra,
                    root,
                    cfg,
                    args.require,
                    args.expect,
                )
            ]
            operation = args.command
        if operation == "plan":
            print(
                json.dumps(
                    {"schema_version": 1, "steps": [asdict(c) for c in commands]},
                    indent=2,
                )
            )
            return 0
        code, receipt = execute(commands, root / args.receipts)
        print(f"Receipt: {receipt}")
        return code
    except (OSError, ValueError) as error:
        print(f"phiroom: {error}", file=sys.stderr)
        return 2
