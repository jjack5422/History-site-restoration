#!/usr/bin/env bash
set -euo pipefail

cd /home/zzz90/research

export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1

PORT="${1:-7861}"

if /home/zzz90/research/sam2_env/bin/python - "$PORT" <<'PY'
import http.client
import sys

port = int(sys.argv[1])
try:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
    conn.request("HEAD", "/")
    resp = conn.getresponse()
except OSError:
    raise SystemExit(1)
else:
    print(f"Already running: http://127.0.0.1:{port}/")
    raise SystemExit(0 if resp.status < 500 else 1)
PY
then
  exit 0
fi

/home/zzz90/research/sam2_env/bin/python \
  /home/zzz90/research/crack_detection_sam2/interactive_refine_sam2.py \
  --host 127.0.0.1 \
  --port "$PORT"
