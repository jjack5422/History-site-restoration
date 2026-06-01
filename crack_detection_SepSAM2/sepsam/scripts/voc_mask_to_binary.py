"""
voc_mask_to_binary.py — 把 Pascal-VOC 樣式的彩色 SegmentationClass 過濾成二值前景 mask。

範例（1-31test 只留 crack+craquelure）：
    python scripts/voc_mask_to_binary.py \
        --src_masks  /home/zzz90/research/crack_detection_sam2/data/1-31test/SegmentationClass \
        --src_images /home/zzz90/research/crack_detection_sam2/data/selected_slices \
        --out_masks  datasets/heritage_1_31test/masks \
        --out_images datasets/heritage_1_31test/images \
        --keep crack:255,24,3 craquelure:102,255,102
"""
import argparse
import glob
import os
import shutil

import cv2
import numpy as np


def parse_keep(items):
    out = []
    for it in items:
        name, rgb = it.split(":")
        r, g, b = [int(x) for x in rgb.split(",")]
        out.append((name, (r, g, b)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_masks", required=True, help="VOC SegmentationClass 資料夾（PNG）")
    ap.add_argument("--src_images", required=True, help="對應原圖資料夾（JPG/PNG）")
    ap.add_argument("--out_masks", required=True)
    ap.add_argument("--out_images", required=True)
    ap.add_argument("--keep", nargs="+", required=True,
                    help="保留的類別，形式 name:R,G,B。可多個。")
    ap.add_argument("--image_exts", default=".jpg,.jpeg,.png")
    args = ap.parse_args()

    os.makedirs(args.out_masks, exist_ok=True)
    os.makedirs(args.out_images, exist_ok=True)
    keep = parse_keep(args.keep)
    keep_rgb = [k[1] for k in keep]
    print("keep:", keep)

    exts = tuple(e.strip().lower() for e in args.image_exts.split(","))
    img_by_stem = {}
    for p in glob.glob(os.path.join(args.src_images, "*")):
        ext = os.path.splitext(p)[1].lower()
        if ext in exts:
            img_by_stem[os.path.splitext(os.path.basename(p))[0]] = p

    n_ok = n_empty = n_skip = 0
    per_cls = {k[0]: 0 for k in keep}
    for mp in sorted(glob.glob(os.path.join(args.src_masks, "*.png"))):
        stem = os.path.splitext(os.path.basename(mp))[0]
        if stem not in img_by_stem:
            n_skip += 1
            continue
        m_bgr = cv2.imread(mp)
        if m_bgr is None:
            n_skip += 1
            continue
        rgb = m_bgr[..., ::-1]
        fg = np.zeros(rgb.shape[:2], np.uint8)
        for cls_name, c_rgb in keep:
            sel = np.all(rgb == np.array(c_rgb, dtype=np.uint8), axis=-1)
            if sel.any():
                per_cls[cls_name] += 1
            fg[sel] = 255
        if fg.max() == 0:
            n_empty += 1
        cv2.imwrite(os.path.join(args.out_masks, stem + ".png"), fg)

        # 複製原圖（保留副檔名）
        ip = img_by_stem[stem]
        ext = os.path.splitext(ip)[1]
        shutil.copy(ip, os.path.join(args.out_images, stem + ext))
        n_ok += 1

    print(f"wrote {n_ok} image/mask pairs (no-image stems skipped: {n_skip}, "
          f"empty-after-filter: {n_empty})")
    print("per-class image counts (any pixel present):", per_cls)


if __name__ == "__main__":
    main()
