#!/usr/bin/env bash
set -e

# run_craq_fused.sh hardcodes PY=/home/zzz90/research/sam2_env/bin/python.
# We don't ship the host venv; point that path at the image's venv so the
# existing scripts run unchanged. (The host mount has no sam2_env/, so this
# write lands in the container layer / mounted tree without clashing.)
TARGET=/home/zzz90/research/sam2_env/bin
if [ ! -e "$TARGET/python" ]; then
    mkdir -p "$TARGET"
    ln -sf /opt/venv/bin/python "$TARGET/python"
fi

# Fail fast if the GPU isn't actually Blackwell-capable in this container.
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print(f"[entrypoint] torch {torch.__version__}  cuda_runtime={torch.version.cuda}  cuda_available={ok}")
if ok:
    cc = torch.cuda.get_device_capability()
    print(f"[entrypoint] device={torch.cuda.get_device_name(0)}  compute_capability={cc[0]}.{cc[1]}")
    if cc[0] < 12:
        print("[entrypoint] WARNING: expected sm_120 (5090); got lower. Check driver/image.")
else:
    print("[entrypoint] WARNING: CUDA not visible. Did you pass --gpus all ?")
PY

exec "$@"
