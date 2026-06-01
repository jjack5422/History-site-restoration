"""
binary_mask_to_yolo_seg.py — 把二值 mask PNG 轉成 YOLOv8-Seg polygon label。

每個前景連通區做 findContours 取外輪廓，approxPolyDP 簡化，正規化到 [0,1]，
寫成 `cls x1 y1 x2 y2 ...`。預設單一類別索引 0。

範例：
    python scripts/binary_mask_to_yolo_seg.py \
        --images datasets/heritage_1_31test/images \
        --masks  datasets/heritage_1_31test/masks \
        --out    datasets/heritage_1_31test/labels \
        --epsilon 0.002 --min_area 64
"""
import argparse
import glob
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cls", type=int, default=0)
    ap.add_argument("--epsilon", type=float, default=0.002,
                    help="approxPolyDP epsilon as fraction of contour perimeter")
    ap.add_argument("--min_area", type=int, default=64,
                    help="忽略 <min_area 像素的連通區")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n_imgs = n_empty = n_poly = 0
    for ip in sorted(glob.glob(os.path.join(args.images, "*"))):
        stem = os.path.splitext(os.path.basename(ip))[0]
        mp = os.path.join(args.masks, stem + ".png")
        if not os.path.exists(mp):
            continue
        im = cv2.imread(ip)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if im is None or m is None:
            continue
        h, w = im.shape[:2]
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m = (m > 0).astype(np.uint8) * 255

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        lines = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < args.min_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, max(1.0, args.epsilon * peri), True)
            pts = approx.reshape(-1, 2)
            if pts.shape[0] < 3:
                continue
            xs = pts[:, 0].astype(np.float64) / float(w)
            ys = pts[:, 1].astype(np.float64) / float(h)
            xs = np.clip(xs, 0.0, 1.0)
            ys = np.clip(ys, 0.0, 1.0)
            coords = np.empty(pts.shape[0] * 2, dtype=np.float64)
            coords[0::2] = xs
            coords[1::2] = ys
            lines.append(str(args.cls) + " " + " ".join(f"{v:.6f}" for v in coords))
            n_poly += 1

        n_imgs += 1
        if not lines:
            n_empty += 1
        with open(os.path.join(args.out, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))

    print(f"wrote labels for {n_imgs} images → {args.out}")
    print(f"  empty (no polygon survived min_area): {n_empty}, total polygons: {n_poly}")


if __name__ == "__main__":
    main()
