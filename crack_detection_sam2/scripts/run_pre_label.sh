#!/usr/bin/env bash
# End-to-end pre-label inference: SAM2 craq + ResUNet crack -> CVAT-importable palette mask.
# Two-pass because sam2_env and unet_env are mutually exclusive on package deps.
#
# Usage:
#   bash scripts/run_pre_label.sh <craq_ckpt> <crack_ckpt> [out_root]
#
# Example:
#   bash scripts/run_pre_label.sh \
#     outputs/expert_craq_v3_final_small/last.pt \
#     /home/zzz90/research/crack_detection_unet/outputs/expert_crack_v3_final/last.pt \
#     outputs/pre_label_v3

set -euo pipefail

CRAQ_CKPT="${1:?craq ckpt path required}"
CRACK_CKPT="${2:?crack ckpt path required}"
OUT_ROOT="${3:-outputs/pre_label_v3}"

SAM2_ROOT=/home/zzz90/research/crack_detection_sam2
UNET_ROOT=/home/zzz90/research/crack_detection_unet
SAM2_PY=/home/zzz90/research/sam2_env/bin/python
UNET_PY=/home/zzz90/research/unet_env/bin/python

IMG_DIR=$SAM2_ROOT/data/selected_slices
SKIP=$SAM2_ROOT/data/1-31test/ImageSets/Segmentation/default.txt

OUT=$SAM2_ROOT/$OUT_ROOT
mkdir -p "$OUT"

echo "=== [1/3] craquelure inference (SAM2-small) ==="
cd "$SAM2_ROOT"
"$SAM2_PY" predict_full.py \
  --ckpt "$CRAQ_CKPT" \
  --image_dir "$IMG_DIR" \
  --out_dir "$OUT/craq_raw" \
  --tile 512 --stride 256 \
  --batch_size 4 \
  --save_prob

echo "=== [2/3] crack inference (ResUNet resnet50) ==="
cd "$UNET_ROOT"
"$UNET_PY" predict_full.py \
  --ckpt "$CRACK_CKPT" \
  --image_dir "$IMG_DIR" \
  --out_dir "$OUT/crack_raw" \
  --tile 512 --stride 256 \
  --batch_size 4 \
  --save_prob

echo "=== [3/3] merge to VOC palette ==="
cd "$SAM2_ROOT"
"$SAM2_PY" scripts/merge_pre_label.py \
  --craq_prob_dir "$OUT/craq_raw/prob" \
  --crack_prob_dir "$OUT/crack_raw/prob" \
  --image_dir "$IMG_DIR" \
  --out_dir "$OUT/merged" \
  --skip_list "$SKIP" \
  --craq_thresh 0.5 \
  --crack_thresh 0.5 \
  --priority craq_over_crack

echo
echo "=== done. CVAT pre-label PNGs at: $OUT/merged/voc_palette ==="
