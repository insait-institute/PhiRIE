#!/usr/bin/env bash
# Invoke ONLY as the command inside SR1's constructor_command allowlist.
set -euo pipefail
: "${SR_BUILD_PYTHON:?absolute isolated Python interpreter required}"
exec "$SR_BUILD_PYTHON" -m robo.roundtrip.build "$@"
