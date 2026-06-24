#!/usr/bin/env bash
set -e

# GPU / Blackwell sanity check.
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print(f"[unet] torch {torch.__version__}  cuda_runtime={torch.version.cuda}  cuda_available={ok}")
if ok:
    cc = torch.cuda.get_device_capability()
    print(f"[unet] device={torch.cuda.get_device_name(0)}  compute_capability={cc[0]}.{cc[1]}")
    if cc[0] < 12:
        print("[unet] WARNING: expected sm_120 (5090); got lower. Check driver/image.")
else:
    print("[unet] WARNING: CUDA not visible. Did you pass --gpus all ?")
PY

exec "$@"
