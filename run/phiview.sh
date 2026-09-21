#!/usr/bin/env bash
# Forward to the pinned PhiView CLI and its independent uv environment.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
viewer="$repo/integrations/phiview"
if [[ ! -f "$viewer/physicalview/cli.py" ]]; then
  printf '%s\n' 'PhiView is not initialized. Run: git submodule update --init --recursive' >&2
  exit 2
fi
if [[ $# -eq 0 ]]; then
  set -- blocks
fi
exec "${UV:-uv}" run --project "$viewer" --locked physicalview "$@"
