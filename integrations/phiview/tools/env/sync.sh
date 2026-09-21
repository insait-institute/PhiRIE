#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
profile=${1:-cpu}
uv_bin=${UV:-uv}
case "$profile" in
  cpu) exec "$uv_bin" sync --project "$repo" --locked ;;
  studio|inference|generation) exec "$uv_bin" sync --project "$repo/envs/$profile" --locked ;;
  *) echo "Usage: $0 cpu|studio|inference|generation" >&2; exit 2 ;;
esac
