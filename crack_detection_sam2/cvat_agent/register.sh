#!/usr/bin/env bash
# One-time: register the native function on app.cvat.ai. Prints the function-id to reuse in run_agent.sh.
# Requires CVAT_ACCESS_TOKEN exported (Personal Access Token from app.cvat.ai).
set -euo pipefail
: "${CVAT_ACCESS_TOKEN:?export CVAT_ACCESS_TOKEN with your app.cvat.ai PAT}"
CLI=/home/zzz90/research/cvat_agent_env/bin/cvat-cli
FUNC=/home/zzz90/research/crack_detection_sam2/cvat_agent/craq_crack_func.py

"$CLI" --server-host https://app.cvat.ai \
  function create-native "Heritage craq+crack" \
  --function-file "$FUNC"
