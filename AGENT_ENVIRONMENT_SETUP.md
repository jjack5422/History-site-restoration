# Agent Environment Setup

This document is for an automated AI agent setting up this repository on a new
Linux machine. Follow the steps in order. Do not commit downloaded datasets,
virtual environments, model checkpoints, or generated outputs.

The project uses two separate Python environments:

- `unet_env`: ResUNet / segmentation-models-pytorch pipeline.
- `sam2_env`: SAM2 refine / prompt-based pipeline.

The exact, reproducible setup is Docker-based. Local virtualenv setup is also
provided for machines where Docker is unavailable.

## 0. Repository Layout Expected by Scripts

Many scripts assume this checkout is available at:

```bash
/home/zzz90/research
```

On another machine, prefer cloning to that exact path:

```bash
mkdir -p /home/zzz90
cd /home/zzz90
git clone https://github.com/jjack5422/History-site-restoration.git research
cd /home/zzz90/research
```

If the user requires a different path, expect some scripts with absolute paths
to need patching. Search before running:

```bash
rg "/home/zzz90/research"
```

## 1. System Prerequisites

Minimum:

```bash
python3 --version        # prefer Python 3.12
git --version
nvidia-smi              # required for GPU training/inference
```

Recommended NVIDIA stack for the current lock files:

- Linux host.
- NVIDIA driver compatible with CUDA 12.8.
- Docker with NVIDIA Container Toolkit, if using Docker.

Install common system packages on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y \
  git git-lfs curl wget build-essential python3.12 python3.12-venv python3-pip \
  libgl1 libglib2.0-0
```

If `python3.12` is unavailable, use the system Python 3 version only after
confirming PyTorch wheels exist for that Python version.

## 2. External Dependencies Not Stored in This Repo

This Git repository intentionally excludes:

- `_data/`
- `*_env/`
- `runs/`, `results/`, `outputs/`, `preds/`
- `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.onnx`, `*.npy`
- the vendored SAM2 source tree

The agent must provision these separately when needed.

### 2.1 SAM2 Source

Clone Meta SAM2 next to this project code:

```bash
cd /home/zzz90/research
git clone https://github.com/facebookresearch/sam2.git segment-anything-2
```

The local code imports `sam2` as a Python package, so the SAM2 checkout must be
installed editable inside `sam2_env` later.

### 2.2 SAM2 Checkpoints

`crack_detection_sam2/model.py` expects SAM2 checkpoints at:

```bash
/home/zzz90/research/crack_detection_sam2/checkpoints/
```

Required filenames:

```text
sam2.1_hiera_tiny.pt
sam2.1_hiera_small.pt
sam2.1_hiera_base_plus.pt
sam2.1_hiera_large.pt
```

After cloning `segment-anything-2`, use the SAM2 repository's official checkpoint
download script if present:

```bash
cd /home/zzz90/research/segment-anything-2
find . -maxdepth 3 -type f -name '*download*ckpt*.sh' -o -name 'download_ckpts.sh'
```

If a script exists, run it, then copy or symlink the resulting `.pt` files:

```bash
mkdir -p /home/zzz90/research/crack_detection_sam2/checkpoints
find /home/zzz90/research/segment-anything-2 -type f -name 'sam2.1_hiera_*.pt' \
  -exec cp -n {} /home/zzz90/research/crack_detection_sam2/checkpoints/ \;
```

If the user's lab already has checkpoints, copy them into the same directory.
Do not commit them.

### 2.3 Project Checkpoints and Data

Training and inference scripts need external data and trained weights. Expected
locations used by current scripts include:

```text
/home/zzz90/research/_data/
/home/zzz90/research/crack_detection_unet/runs/<run-name>/best.pt
/home/zzz90/research/crack_detection_sam2/runs/<run-name>/best.pt
```

Ask the user for these files or mount/copy them from the lab storage. Keep them
outside Git.

## 3. Preferred Setup: Docker

Use Docker when the machine has a recent NVIDIA driver and
`nvidia-container-toolkit`.

Verify Docker GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu24.04 nvidia-smi
```

Build both images from the repository root:

```bash
cd /home/zzz90/research

docker build \
  -f crack_detection_sam2/docker/Dockerfile.unet \
  -t history-restoration-unet:cu128 \
  .

docker build \
  -f crack_detection_sam2/docker/Dockerfile \
  -t history-restoration-sam2:cu128 \
  .
```

Run the UNet container:

```bash
cd /home/zzz90/research
docker run --rm -it --gpus all --shm-size=16g \
  -v "$PWD":/home/zzz90/research \
  history-restoration-unet:cu128 bash
```

Run the SAM2 container:

```bash
cd /home/zzz90/research
docker run --rm -it --gpus all --shm-size=16g \
  -v "$PWD":/home/zzz90/research \
  history-restoration-sam2:cu128 bash
```

Inside each container, run the smoke checks in Section 5.

## 4. Alternative Setup: Local Virtualenvs

Use this only if Docker is unavailable. The lock files were generated from the
working lab environments and may include PyTorch CUDA wheels that are not always
available on every package index. If exact installation fails, install the
closest compatible PyTorch CUDA 12.8 build first, then install the remaining
packages.

### 4.1 Create `unet_env`

```bash
cd /home/zzz90/research
python3.12 -m venv unet_env
source unet_env/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Try the locked environment first.
python -m pip install -r crack_detection_sam2/docker/requirements.unet.lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --pre

# Install shared local package.
python -m pip install -e _lib

deactivate
```

If the exact nightly Torch wheels are unavailable, use:

```bash
source /home/zzz90/research/unet_env/bin/activate
python -m pip install --upgrade --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128
python -m pip install albumentations opencv-python opencv-python-headless \
  segmentation-models-pytorch timm matplotlib tqdm PyYAML scipy pillow \
  python-pptx xlsxwriter pytest
python -m pip install -e /home/zzz90/research/_lib
deactivate
```

### 4.2 Create `sam2_env`

```bash
cd /home/zzz90/research
python3.12 -m venv sam2_env
source sam2_env/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Try the locked environment first.
python -m pip install -r crack_detection_sam2/docker/requirements.lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128

# Install external and local editable packages.
python -m pip install -e segment-anything-2
python -m pip install -e _lib

# Optional: needed only for the interactive UI.
python -m pip install gradio

deactivate
```

If the exact Torch version is unavailable, use:

```bash
source /home/zzz90/research/sam2_env/bin/activate
python -m pip install --upgrade torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install albumentations opencv-python-headless hydra-core iopath \
  matplotlib tqdm PyYAML scipy scikit-image pillow timm \
  segmentation-models-pytorch python-pptx xlsxwriter pytest gradio
python -m pip install -e /home/zzz90/research/segment-anything-2
python -m pip install -e /home/zzz90/research/_lib
deactivate
```

## 5. Smoke Checks

Run from `/home/zzz90/research`.

### 5.1 Shared Package

```bash
source sam2_env/bin/activate
python - <<'PY'
import crackseg_common
from crackseg_common.metrics import ConfusionMeter
print("crackseg_common OK")
PY
deactivate
```

### 5.2 UNet Environment

```bash
source unet_env/bin/activate
python - <<'PY'
import torch
import segmentation_models_pytorch as smp
from crack_detection_unet.src.unet_model import build_resunet
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
model = build_resunet(num_classes=2)
print("unet OK", type(model).__name__)
PY
deactivate
```

### 5.3 SAM2 Environment

```bash
source sam2_env/bin/activate
python - <<'PY'
import torch
import sam2
from crack_detection_sam2.model import MODEL_VARIANTS
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("variants", sorted(MODEL_VARIANTS))
print("sam2 OK")
PY
deactivate
```

### 5.4 Checkpoint Presence

```bash
ls -lh /home/zzz90/research/crack_detection_sam2/checkpoints/sam2.1_hiera_*.pt
```

## 6. Common Commands After Setup

Train UNet:

```bash
cd /home/zzz90/research/crack_detection_unet
../unet_env/bin/python src/train.py \
  --tiles_root /home/zzz90/research/_data/craq_0-94_v1/tiles_512 \
  --split /home/zzz90/research/_data/craq_0-94_v1/tiles_512/group_split_stem.json \
  --fold 0 \
  --output_dir runs/unet-smoke
```

Run SAM2 refine training smoke:

```bash
cd /home/zzz90/research/crack_detection_sam2
../sam2_env/bin/python train_craq_promptrefine.py \
  --tiles_root /home/zzz90/research/_data/craq_0-94_v1/tiles_512 \
  --split /home/zzz90/research/_data/craq_0-94_v1/tiles_512/group_split_stem.json \
  --prob_dir /home/zzz90/research/_data/craq_0-94_v1/tiles_512/resunet_prob \
  --prompt_mode mask \
  --tversky_alpha 0.2 \
  --tversky_beta 0.8 \
  --epochs 1 \
  --batch_size 1 \
  --fold 0 \
  --output_dir runs/sam2-refine-smoke
```

Run interactive refine UI:

```bash
cd /home/zzz90/research
sam2_env/bin/python crack_detection_sam2/interactive_refine_sam2.py \
  --host 127.0.0.1 \
  --port 7861
```

## 7. Troubleshooting

`ModuleNotFoundError: sam2`

: Install SAM2 editable into `sam2_env`:

```bash
source /home/zzz90/research/sam2_env/bin/activate
python -m pip install -e /home/zzz90/research/segment-anything-2
```

`ModuleNotFoundError: crackseg_common`

: Install the shared local package into both envs:

```bash
source /home/zzz90/research/sam2_env/bin/activate
python -m pip install -e /home/zzz90/research/_lib
deactivate

source /home/zzz90/research/unet_env/bin/activate
python -m pip install -e /home/zzz90/research/_lib
deactivate
```

`FileNotFoundError: sam2.1_hiera_*.pt`

: Copy or symlink SAM2 checkpoints into:

```bash
/home/zzz90/research/crack_detection_sam2/checkpoints/
```

`CUDA not available`

: Check `nvidia-smi`, driver compatibility, and Docker `--gpus all`.

`no kernel image is available for execution`

: The installed PyTorch CUDA build is not compatible with the GPU. Reinstall a
CUDA 12.8-compatible PyTorch build for the target GPU.

`Permission denied` while writing `runs/`

: Ensure the repository directory is owned by the current user, or set
`--output_dir` to a writable location.

## 8. Git Hygiene Rules for Agents

Never commit these:

```text
*_env/
_data/
segment-anything-2/
crack_detection_sam2/checkpoints/
crack_detection_sam2/runs/
crack_detection_unet/runs/
*.pt
*.pth
*.ckpt
*.safetensors
*.onnx
*.npy
```

Before committing, run:

```bash
git status --short
find . -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.npy' \)
```

The second command should print nothing.
