#!/usr/bin/env bash
# scripts/run_craq_fused.sh — C0 (baseline) 與 E1 (DINOv2 fused) 跨 5 fold
set -euo pipefail
PY=/home/zzz90/research/sam2_env/bin/python
ROOT=/home/zzz90/research/_data/craq_0-94_v1/tiles_512
SPLIT=$ROOT/group_split_stem.json
PROB=$ROOT/resunet_prob
DINO=$ROOT/dinov2_feat
DATE=2026-06-11
COMMON="--tiles_root $ROOT --split $SPLIT --prob_dir $PROB --prompt_mode mask \
  --tversky_alpha 0.2 --tversky_beta 0.8 --epochs 60 --batch_size 4 --base_lr 2e-4"

for k in 0 1 2 3 4; do
  echo "=== C0 baseline fold $k ==="
  $PY train_craq_promptrefine.py $COMMON --fold $k \
    --output_dir runs/craq-base-c0-fold${k}-${DATE}
  echo "=== E1 fused fold $k ==="
  $PY train_craq_promptrefine.py $COMMON --fold $k \
    --dino_feat_dir $DINO --dino_dim 384 \
    --output_dir runs/craq-fused-e1-fold${k}-${DATE}
done
