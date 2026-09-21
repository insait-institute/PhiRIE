#!/usr/bin/env bash
set -euo pipefail
PHIVIEW_PACKAGE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$PHIVIEW_PACKAGE/.envs/phiview-inference:$PHIVIEW_PACKAGE/.envs/phiview-sam3-source:$PHIVIEW_PACKAGE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/group/worldcept/hf_cache
export TORCH_HOME=/group/worldcept/torch_hub_cache
PHIVIEW_STUDIO_ENV=$(readlink -f "$PHIVIEW_PACKAGE/.envs/studio")
exec "$PHIVIEW_STUDIO_ENV/bin/python" "$@"
