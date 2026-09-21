#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PHIROOM_ROOT="$repo"
exec "${UV:-uv}" run --project "$repo/envs/control" --locked phiroom "$@"
