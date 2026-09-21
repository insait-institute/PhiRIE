#!/usr/bin/env bash
set -euo pipefail
: "${SR_BUILD_PYTHON:?host launcher Python interpreter required}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SR_BUILD_PYTHON" "$script_dir/build_isolated.py" "$@"
