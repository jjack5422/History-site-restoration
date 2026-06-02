"""M1 smoke test: 載入 SAM2，對一張裂紋圖手動給點，驗證 build_large_model + sam_prompt_sam2 都通。

跑法（從 sepsam/ 目錄）:
    /home/zzz90/research/SepSAM2_env/bin/python scripts/m1_sam2_smoke.py \
        --image /home/zzz90/research/_data/selected_slices/MGLST-RH-M-A1-2_R1_C02.jpg \
        --cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
        --ckpt weights/sam2.1_hiera_base_plus.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.large_model import build_large_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    ap.add_argument("--ckpt", default="weights/sam2.1_hiera_base_plus.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/m1_smoke")
    args = ap.parse_args()

    img_path = Path(args.image)
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise SystemExit(f"image not found: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    print(f"image: {img_path.name}  shape=({h},{w},3)")

    hp = SimpleNamespace(
        sam_backend="sam2",
        sam2_cfg=args.cfg,
        sam2_ckpt=str(REPO / args.ckpt) if not Path(args.ckpt).is_absolute() else args.ckpt,
        device=args.device,
    )
    print(f"loading SAM2: cfg={hp.sam2_cfg} ckpt={hp.sam2_ckpt}")
    predictor, prompt_fn = build_large_model(hp)

    # 3 個手動點：圖中心 + 兩個對角
    pts = np.array([
        [w // 2, h // 2],
        [w // 4, h // 4],
        [3 * w // 4, 3 * h // 4],
    ], dtype=np.float32)
    print(f"prompt points (xy): {pts.tolist()}")

    mask, score = prompt_fn(predictor, rgb, pts)
    print(f"mask: shape={mask.shape} dtype={mask.dtype} "
          f"unique={np.unique(mask).tolist()} fg_pixels={int((mask > 0).sum())}")
    print(f"sam_score: {score:.4f}")

    # 疊圖輸出
    overlay = bgr.copy()
    mask_color = np.zeros_like(bgr)
    mask_color[mask > 0] = (0, 0, 255)  # red
    overlay = cv2.addWeighted(overlay, 1.0, mask_color, 0.5, 0)
    for x, y in pts.astype(int):
        cv2.circle(overlay, (int(x), int(y)), 8, (0, 255, 0), -1)
        cv2.circle(overlay, (int(x), int(y)), 8, (0, 0, 0), 2)
    out_path = out_dir / f"{img_path.stem}_m1.png"
    cv2.imwrite(str(out_path), overlay)
    cv2.imwrite(str(out_dir / f"{img_path.stem}_m1_mask.png"), mask)
    print(f"saved: {out_path}")
    print("M1 SMOKE PASSED ✓")


if __name__ == "__main__":
    main()
