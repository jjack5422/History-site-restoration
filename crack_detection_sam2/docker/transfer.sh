#!/usr/bin/env bash
# transfer.sh - 把 research root 搬到實驗室 5090 機器(本 branch 無 git remote,用 rsync)
#
# 用法:
#   # 1. 先 dry-run 看會傳什麼(不實際傳):
#   DRY=1 ./transfer.sh labuser@lab-host /data/research
#   # 2. 確認後實傳:
#   ./transfer.sh labuser@lab-host /data/research
#
# 參數:
#   $1  DEST_HOST   實驗室機器的 ssh 目標,例如 labuser@192.168.1.50
#   $2  DEST_PATH   實驗室機器上的落地路徑(host 放哪都行,容器再掛到 /home/zzz90/research)
#
# 可調環境變數:
#   DRY=1          只做 rsync --dry-run,不實際傳輸
#   WITH_RUNS=1    連 crack_detection_sam2/runs/(~7.3G 舊輸出)一起傳,預設排除
#   SRC=...        來源 root,預設 /home/zzz90/research
#   SSH_PORT=22    ssh 連接埠

set -euo pipefail

DEST_HOST="${1:-}"
DEST_PATH="${2:-}"
SRC="${SRC:-/home/zzz90/research}"
SSH_PORT="${SSH_PORT:-22}"

if [[ -z "$DEST_HOST" || -z "$DEST_PATH" ]]; then
    echo "用法: [DRY=1] [WITH_RUNS=1] $0 <user@host> <dest_path>" >&2
    echo "例:  DRY=1 $0 labuser@192.168.1.50 /data/research" >&2
    exit 1
fi

# 來源結尾的 / 很重要:把 SRC 的「內容」搬進 DEST_PATH/,而非多包一層目錄。
SRC="${SRC%/}/"
DEST_PATH="${DEST_PATH%/}/"

# rsync 排除清單。
EXCLUDES=(
    --exclude='*_env/'                       # venv 不可攜,容器會重建
    --exclude='.pytest_cache/'
    --exclude='__pycache__/'
    --exclude='*.egg-info/'
    --exclude='.agent-trash/'
)
if [[ "${WITH_RUNS:-0}" != "1" ]]; then
    EXCLUDES+=( --exclude='crack_detection_sam2/runs/' )   # 7.3G 舊輸出,重訓會重生
fi

RSYNC_OPTS=( -avhP --human-readable --partial )
[[ "${DRY:-0}" == "1" ]] && RSYNC_OPTS+=( --dry-run )

echo "=== rsync ==="
echo "  from : $SRC"
echo "  to   : $DEST_HOST:$DEST_PATH"
echo "  runs : $([[ "${WITH_RUNS:-0}" == "1" ]] && echo included || echo EXCLUDED)"
echo "  mode : $([[ "${DRY:-0}" == "1" ]] && echo DRY-RUN || echo LIVE)"
echo

rsync "${RSYNC_OPTS[@]}" "${EXCLUDES[@]}" \
    -e "ssh -p ${SSH_PORT}" \
    "$SRC" "${DEST_HOST}:${DEST_PATH}"

if [[ "${DRY:-0}" == "1" ]]; then
    echo
    echo "(dry-run 完成,沒有實際傳輸。拿掉 DRY=1 才會真的傳。)"
    exit 0
fi

# 實傳後,遠端驗證重訓最小集合是否到位。
echo
echo "=== 遠端檢查重訓必要檔案 ==="
ssh -p "${SSH_PORT}" "${DEST_HOST}" "bash -s" <<EOF
set -e
cd "${DEST_PATH}"
check() { if [ -e "\$1" ]; then echo "  OK   \$1"; else echo "  MISS \$1"; fi; }
check segment-anything-2/checkpoints/sam2.1_hiera_large.pt
check _lib/crackseg_common
check _data/craq_0-94_v1/tiles_512/resunet_prob
check _data/craq_0-94_v1/tiles_512/dinov2_feat
check _data/craq_0-94_v1/tiles_512/group_split_stem.json
check _data/0-94
check crack_detection_sam2/scripts/run_craq_fused.sh
# 全流程(B)還需要 ResUNet 權重 + unet code 來重算 prompt 機率圖:
check crack_detection_unet/src/predict_full.py
check crack_detection_unet/runs/craq-resunet50-2026-06-10/best.pt
echo "  --- 大小 ---"
du -sh _data/craq_0-94_v1 segment-anything-2/checkpoints crack_detection_unet/runs 2>/dev/null || true
EOF

echo
echo "完成。下一步在實驗室機器上:"
echo "  cd ${DEST_PATH}"
echo "  docker build -f crack_detection_sam2/docker/Dockerfile -t craq-sam2:cu128 ."
