# SAM2 craquelure refine - reproducible training image for RTX 5090 (Blackwell sm_120)
#
# Host requirements (already satisfied on the lab box):
#   - NVIDIA driver >= 570 (nvidia-smi works)
#   - nvidia-container-toolkit (docker run --gpus all works)
#
# Pins mirror the working local sam2_env: torch 2.11.0+cu128 / CUDA 12.8 / Python 3.12.
# Build context must be the research root (so segment-anything-2/ and _lib/ are visible):
#   docker build -f crack_detection_sam2/docker/Dockerfile -t craq-sam2:cu128 .

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    SAM2_BUILD_CUDA=0

# Ubuntu 24.04 ships Python 3.12 natively (matches local env).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        git libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv so we don't fight Ubuntu's externally-managed python.
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip setuptools wheel

# 1) torch trio from the cu128 index (Blackwell-capable build).
RUN pip install \
        torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

# 2) everything else; torch lines already satisfied. PyPI + cu128 fallback.
COPY crack_detection_sam2/docker/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install -r /tmp/requirements.lock.txt \
        --extra-index-url https://download.pytorch.org/whl/cu128

# 3) the two local editable packages (no git remote -> install from vendored source).
COPY _lib /opt/src/_lib
COPY segment-anything-2 /opt/src/segment-anything-2
RUN pip install -e /opt/src/_lib && \
    pip install -e /opt/src/segment-anything-2

COPY crack_detection_sam2/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /home/zzz90/research/crack_detection_sam2
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
