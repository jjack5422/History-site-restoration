#!/usr/bin/env bash
# Keep running while annotating. Usage: ./run_agent.sh <function-id>
set -euo pipefail
: "${CVAT_ACCESS_TOKEN:?export CVAT_ACCESS_TOKEN with your app.cvat.ai PAT}"
FUNC_ID="${1:?usage: run_agent.sh <function-id from register.sh>}"
CLI=/home/zzz90/research/cvat_agent_env/bin/cvat-cli
FUNC=/home/zzz90/research/crack_detection_sam2/cvat_agent/craq_crack_func.py

"$CLI" --server-host https://app.cvat.ai \
  function run-agent "$FUNC_ID" \
  --function-file "$FUNC"
