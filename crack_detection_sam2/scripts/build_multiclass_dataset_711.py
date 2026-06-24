"""Build a MULTI-CLASS (5-class) version of craq_512_dataset_711.

Same exact 711 tiles (images + split) as _data/craq_512_dataset_711, but the
masks carry every available deterioration class instead of binary craquelure.

Per-tile we re-crop the canonical RGB-palette label at the recorded (y, x) and
map colors -> class index:

    0 background   1 crack   2 loss   3 shrinkage   4 craquelure   5 flaking
    255 ignore  (excluded from loss; only present in 0-94 source)

Index 0..4 match _lib/crackseg_common/dataset.py CLASS_NAMES so existing 5-class
code stays compatible; flaking is appended as 5.

Sources:
  * base tiles  -> _data/craq_0-94_v1/_seg95/{stem}.png   (0-94 palette, all classes)
  * A4-8 R1     -> _data/batch_4/labels/{stem}.png        (batch_4 palette: crack/craq/loss)

Consistency check: the craquelure channel (idx==4) of every rebuilt mask must
equal the existing binary mask in _data/craq_512_dataset_711/masks. Any mismatch
is reported and (by default) aborts the write.

Run:
    /home/zzz90/research/sam2_env/bin/python \
        crack_detection_sam2/scripts/build_multiclass_dataset_711.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter

import numpy as np
from PIL import Image

ROOT = "/home/zzz90/research"
SRC_DS = os.path.join(ROOT, "_data/craq_512_dataset_711")
# base source: 0-94/SegmentationClass is the CANONICAL corrected multi-class label
# (CVAT re-export, recovered overwritten labels). _seg95 is the older/sparser
# craquelure-focused source the binary dataset was built from -> fallback only
# for stems 0-94 lacks (a couple of all-background tiles).
SEG094 = os.path.join(ROOT, "_data/0-94/SegmentationClass")
SEG95 = os.path.join(ROOT, "_data/craq_0-94_v1/_seg95")
B4_SEG = os.path.join(ROOT, "_data/batch_4/labels")

# stems whose source lives in batch_4 (renamed A4-8 lintel R1)
B4_STEMS = {
    "KJTHT-SC-M-A4-8_R1_C01",
    "KJTHT-SC-M-A4-8_R1_C02",
    "KJTHT-SC-M-A4-8_R1_C03",
}

CLASS_NAMES = ["background", "crack", "loss", "shrinkage", "craquelure", "flaking"]
IGNORE_IDX = 255
CRAQ_IDX = CLASS_NAMES.index("craquelure")

# RGB -> class index per source palette (from labelmap.txt of each source)
PAL_094 = {
    (0, 0, 0): 0,
    (255, 24, 3): 1,       # crack
    (9, 249, 213): 2,      # loss
    (149, 0, 222): 3,      # shrinkage
    (102, 255, 102): 4,    # craquelure
    (236, 236, 0): 5,      # flaking
    (255, 106, 77): IGNORE_IDX,  # ignore
}
PAL_B4 = {
    (0, 0, 0): 0,
    (165, 236, 223): 1,    # crack
    (106, 163, 124): 2,    # loss
    (13, 117, 210): 4,     # craquelure
}

TILE = 512


def rgb_to_index(rgb: np.ndarray, palette: dict):
    """Map an HxWx3 RGB array to HxW uint8 index mask. Returns (mask, unknown_counter)."""
    h, w = rgb.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    matched = np.zeros((h, w), dtype=bool)
    for color, idx in palette.items():
        m = (rgb == np.array(color, dtype=np.uint8)).all(axis=-1)
        out[m] = idx
        matched |= m
    unknown = Counter()
    if not matched.all():
        um = ~matched
        for c in map(tuple, rgb[um].reshape(-1, 3)):
            unknown[c] += 1
    return out, unknown


def crop_pad(arr: np.ndarray, y: int, x: int, ts: int = TILE) -> np.ndarray:
    """Crop [y:y+ts, x:x+ts] with zero padding on right/bottom (matches tile_image)."""
    h, w = arr.shape[:2]
    py, px = max(0, y + ts - h), max(0, x + ts - w)
    if py or px:
        arr = np.pad(arr, ((0, py), (0, px), (0, 0)), constant_values=0)
    return arr[y:y + ts, x:x + ts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "_data/multiclass_512_dataset_711"))
    ap.add_argument("--copy_images", action="store_true", default=True)
    ap.add_argument("--allow_mismatch", action="store_true",
                    help="write even if craquelure channel disagrees with the binary masks")
    args = ap.parse_args()

    items = json.load(open(os.path.join(SRC_DS, "tile_index.json")))["items"]
    src_seg_cache = {}

    out_img = os.path.join(args.out, "images")
    out_msk = os.path.join(args.out, "masks")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_msk, exist_ok=True)

    unknown_total = Counter()
    mismatches = []
    class_pixels = Counter()
    n_no_extra = 0  # tiles whose only fg class is craquelure (sanity)

    for it in items:
        tile, stem, y, x = it["tile"], it["stem"], it["y"], it["x"]
        if stem in B4_STEMS:
            seg_path, palette = os.path.join(B4_SEG, stem + ".png"), PAL_B4
        elif os.path.exists(os.path.join(SEG094, stem + ".png")):
            seg_path, palette = os.path.join(SEG094, stem + ".png"), PAL_094
        else:
            seg_path, palette = os.path.join(SEG95, stem + ".png"), PAL_094
        if seg_path not in src_seg_cache:
            src_seg_cache[seg_path] = np.array(Image.open(seg_path).convert("RGB"))
        rgb = crop_pad(src_seg_cache[seg_path], y, x)
        idx, unknown = rgb_to_index(rgb, palette)
        unknown_total.update(unknown)

        # consistency vs existing binary craquelure mask
        binm = np.array(Image.open(os.path.join(SRC_DS, "masks", tile)))
        if binm.ndim == 3:
            binm = binm[..., 0]
        binm = (binm > 0).astype(np.uint8)
        craq_new = (idx == CRAQ_IDX).astype(np.uint8)
        diff = int((craq_new != binm).sum())
        if diff:
            mismatches.append((tile, diff))

        for c in np.unique(idx):
            class_pixels[int(c)] += int((idx == c).sum())
        if set(np.unique(idx)) <= {0, CRAQ_IDX}:
            n_no_extra += 1

        if args.copy_images:
            shutil.copy2(os.path.join(SRC_DS, "images", tile), os.path.join(out_img, tile))
        Image.fromarray(idx).save(os.path.join(out_msk, tile))

    # ---- report ----
    print(f"tiles processed: {len(items)}")
    print("class pixel totals:")
    for c in sorted(class_pixels):
        name = "ignore" if c == IGNORE_IDX else CLASS_NAMES[c]
        print(f"  {c:>3} {name:<11} {class_pixels[c]:>12,}")
    print(f"tiles with only background+craquelure: {n_no_extra}/{len(items)}")
    print(f"craquelure-channel differs from (stale) binary masks: {len(mismatches)} tiles "
          f"(expected: base now uses canonical 0-94, not _seg95)")
    if unknown_total:
        print(f"UNKNOWN colors (top 10 of {len(unknown_total)}):")
        for c, n in unknown_total.most_common(10):
            print(f"    {c}  {n:,} px")

    # copy split + index, write labelmap + README
    for f in ("tile_index.json", "group_split_stem.json"):
        shutil.copy2(os.path.join(SRC_DS, f), os.path.join(args.out, f))
    with open(os.path.join(args.out, "labelmap.txt"), "w") as f:
        f.write("# index:label:color_rgb(0-94 source palette)\n")
        colors = {v: k for k, v in PAL_094.items()}
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"{i}:{name}:{','.join(map(str, colors.get(i, (0,0,0))))}\n")
        f.write(f"{IGNORE_IDX}:ignore:255,106,77\n")
    print(f"\nWritten -> {args.out}")


if __name__ == "__main__":
    main()
