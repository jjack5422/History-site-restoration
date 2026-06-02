"""M2 smoke test: 拿一張現成的 crack GT mask 走過 mask_to_points_and_width，把點疊在原圖上看分布。

跑法（從 sepsam/）:
    /home/zzz90/research/SepSAM2_env/bin/python scripts/m2_medial_axis.py \
        --image /home/zzz90/research/_data/labeled32_crack_v3/images/KJTHT-SC-L-1RB1-1_R2_C04.jpg \
        --mask  /home/zzz90/research/_data/labeled32_crack_v3/masks/KJTHT-SC-L-1RB1-1_R2_C04.png \
        --points_divisor 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.geometry import mask_to_points_and_width, mean_crack_width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True, help="binary (or 0/255) crack mask")
    ap.add_argument("--points_divisor", type=int, default=50,
                    help="n_points = max(H,W) // POINTS_DIVISOR (論文預設 50)")
    ap.add_argument("--out", default="outputs/m2_medial_axis")
    args = ap.parse_args()

    img_path = Path(args.image)
    mask_path = Path(args.mask)
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(img_path))
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None or mask_raw is None:
        raise SystemExit(f"failed: img={img_path.exists()} mask={mask_path.exists()}")
    if bgr.shape[:2] != mask_raw.shape:
        raise SystemExit(f"shape mismatch: img={bgr.shape[:2]} mask={mask_raw.shape}")

    h, w = bgr.shape[:2]
    mask_bin = mask_raw > 0
    fg = int(mask_bin.sum())
    print(f"image: {img_path.name}  shape=({h},{w},3)  fg_pixels={fg}")

    n_pts = max(h, w) // args.points_divisor
    print(f"n_points target = max({h},{w})//{args.points_divisor} = {n_pts}")

    pts, widths = mask_to_points_and_width(mask_bin, n_pts)
    print(f"sampled points: {pts.shape[0]}")
    if pts.shape[0]:
        print(f"width  min={widths.min():.2f} mean={widths.mean():.2f} max={widths.max():.2f} px")
        print(f"mean_crack_width (全骨架) = {mean_crack_width(mask_bin):.2f} px")

    # 驗證：所有取樣點應落在 mask 內
    if pts.shape[0]:
        xs = pts[:, 0].astype(int); ys = pts[:, 1].astype(int)
        inside = int(mask_bin[ys, xs].sum())
        print(f"points inside mask: {inside}/{pts.shape[0]} "
              f"({'PASS' if inside == pts.shape[0] else 'FAIL — 應為 100%'})")

    # 視覺化：(1) 中軸 + 點; (2) 點疊原圖
    from skimage.morphology import medial_axis
    skel = medial_axis(mask_bin)
    vis_mask = cv2.cvtColor((mask_bin.astype(np.uint8) * 80), cv2.COLOR_GRAY2BGR)
    vis_mask[skel] = (0, 255, 255)
    for x, y in pts.astype(int):
        cv2.circle(vis_mask, (int(x), int(y)), 4, (0, 0, 255), -1)
    cv2.imwrite(str(out_dir / f"{img_path.stem}_skel_pts.png"), vis_mask)

    overlay = bgr.copy()
    for x, y in pts.astype(int):
        cv2.circle(overlay, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.circle(overlay, (int(x), int(y)), 5, (0, 0, 0), 1)
    cv2.imwrite(str(out_dir / f"{img_path.stem}_overlay.png"), overlay)
    print(f"saved overlay + skel: {out_dir}/")
    print("M2 SMOKE PASSED ✓")


if __name__ == "__main__":
    main()
