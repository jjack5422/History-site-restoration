"""
yolo_seg_to_masks.py — 把 YOLO segmentation label（多邊形）轉成二值 mask PNG。

每張影像會掃描對應 .txt 內所有 polygon（任意類別都當前景）並 fillPoly 到單通道
mask，輸出尺寸 = 原圖尺寸。供 calibrate_sam_thresh.py / eval.py 用。

範例：
    python scripts/yolo_seg_to_masks.py \
        --images datasets/crack_seg/valid/images \
        --labels datasets/crack_seg/valid/labels \
        --out    datasets/crack_seg/valid/masks
"""
import argparse
import glob
import os

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def index_by_stem(folder, exts):
    out = {}
    for p in glob.glob(os.path.join(folder, "*")):
        ext = os.path.splitext(p)[1].lower()
        if ext in exts:
            out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


def polygons_from_label(txt_path):
    """每行: cls x1 y1 x2 y2 ... (xi,yi 已歸一化到 [0,1])。回傳 list[np.ndarray (K,2) float]."""
    polys = []
    if not os.path.exists(txt_path):
        return polys
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                coords = coords[:-1]
            poly = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
            if poly.shape[0] >= 3:
                polys.append(poly)
    return polys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    imgs = index_by_stem(args.images, IMG_EXTS)
    n_ok = 0
    n_empty = 0
    n_skip = 0
    for stem, ip in sorted(imgs.items()):
        im = cv2.imread(ip)
        if im is None:
            n_skip += 1
            continue
        h, w = im.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        polys = polygons_from_label(os.path.join(args.labels, stem + ".txt"))
        if polys:
            scaled = [np.round(p * [w, h]).astype(np.int32) for p in polys]
            cv2.fillPoly(mask, scaled, 255)
        else:
            n_empty += 1
        cv2.imwrite(os.path.join(args.out, stem + ".png"), mask)
        n_ok += 1
    print(f"wrote {n_ok} masks to {args.out}  (empty/no-label: {n_empty}, unreadable: {n_skip})")


if __name__ == "__main__":
    main()
