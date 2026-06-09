#!/usr/bin/env python
"""Package merged voc_palette class masks into CVAT 'Segmentation mask 1.1' import layout.

Output structure (mirrors a CVAT VOC segmentation-mask export):
    <out>/labelmap.txt
    <out>/ImageSets/Segmentation/default.txt
    <out>/SegmentationClass/<stem>.png   RGB, class colors from labelmap
    <out>/SegmentationObject/<stem>.png   RGB, per-connected-component instance colors (VOC colormap)
and a zip at <out>.zip ready to upload to CVAT (Import annotations -> Segmentation mask 1.1).
"""
from __future__ import annotations
import argparse
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from scipy.ndimage import label as cc_label
except Exception:  # pragma: no cover
    cc_label = None


def voc_colormap(n=256):
    cmap = []
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap.append((r, g, b))
    return cmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc_dir", required=True, help="merged/voc_palette dir (RGB class PNGs)")
    ap.add_argument("--labelmap", required=True, help="source labelmap.txt to copy")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--zip", action="store_true", help="also produce <out_dir>.zip")
    ap.add_argument("--instance_per_class", action="store_true",
                    help="treat each class as ONE instance (one object per class) instead of "
                         "one instance per connected component")
    args = ap.parse_args()

    voc_dir = Path(args.voc_dir)
    out = Path(args.out_dir)
    seg_cls = out / "SegmentationClass"
    seg_obj = out / "SegmentationObject"
    imgsets = out / "ImageSets" / "Segmentation"
    for d in (seg_cls, seg_obj, imgsets):
        d.mkdir(parents=True, exist_ok=True)

    cmap = voc_colormap()
    pngs = sorted(voc_dir.glob("*.png"))
    stems = []
    for p in pngs:
        stem = p.stem
        stems.append(stem)
        cls = Image.open(p).convert("RGB")
        cls.save(seg_cls / f"{stem}.png")
        arr = np.array(cls)
        fg = np.any(arr != 0, axis=2)
        if args.instance_per_class:
            # one instance per distinct class color -> all pixels of a class are a single object
            obj = np.zeros_like(arr)
            colors = [tuple(c) for c in np.unique(arr.reshape(-1, 3), axis=0) if tuple(c) != (0, 0, 0)]
            for idx, c in enumerate(colors, start=1):
                obj[np.all(arr == c, axis=2)] = cmap[idx % 256]
        elif cc_label is not None and fg.any():
            # instance mask: one instance per connected component over any-foreground pixels
            lab, n = cc_label(fg)
            # vectorized instance coloring via colormap LUT (O(pixels), not O(instances))
            lut = np.array([cmap[i % 256] for i in range(n + 1)], dtype=np.uint8)
            lut[0] = (0, 0, 0)
            obj = lut[lab]
        else:
            obj = np.zeros_like(arr)
        Image.fromarray(obj, mode="RGB").save(seg_obj / f"{stem}.png")

    (imgsets / "default.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")
    shutil.copy(args.labelmap, out / "labelmap.txt")
    print(f"packaged {len(stems)} masks -> {out}")

    if args.zip:
        zip_path = str(out) + ".zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(out):
                for f in files:
                    fp = Path(root) / f
                    zf.write(fp, fp.relative_to(out))
        print(f"zip -> {zip_path}")


if __name__ == "__main__":
    main()
