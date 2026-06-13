"""Clip the high-recall craquelure draft to human-drawn region polygons.

Fast path of the semi-auto workflow (see
docs/superpowers/specs/2026-06-14-craquelure-semiauto-annotation-design.md).

Input regions come from CVAT polygons (export "CVAT for images 1.1") drawn on
the full or a downscaled copy of the image. Two region kinds:
  clean   labels {craq_clean, clean, craquelure, region}  -> aggressive draft
  overlap labels {craq_overlap, overlap, content}         -> model-only draft

Final craquelure = (clean_region & aggressive_draft) | (overlap_region & model_draft)
Everything outside the regions (frame/relief/title/blank) is dropped.

Outputs full-res:
  mask_craq.png    binary 255 mask
  seg_voc.png      Pascal VOC colour (craquelure = 102,255,102) for CVAT import / GT
  overlay_final.png (+ _preview)

Run with sam2_env:
  sam2_env/bin/python region_clip.py \
      --image /home/zzz90/research/_data/image/KJTHT-SC-L-1RB1-1.jpg \
      --prob  .../resunet/prob/KJTHT-SC-L-1RB1-1.npy \
      --ridge .../draft/ridge_u8.png \
      --regions regions.xml \
      --out_dir .../final/KJTHT-SC-L-1RB1-1
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import remove_small_objects, closing, disk

Image.MAX_IMAGE_PIXELS = None

CRAQ_RGB = (102, 255, 102)
CLEAN_LABELS = {"craq_clean", "clean", "craquelure", "region"}
OVERLAP_LABELS = {"craq_overlap", "overlap", "content"}


def parse_cvat_polygons(xml_path, image_stem):
    """Return (list[(label, Nx2 pts)], (xml_w, xml_h)) for the matching image."""
    root = ET.parse(xml_path).getroot()
    for im in root.iter("image"):
        name = im.get("name", "")
        stem = Path(name).stem
        if stem != image_stem:
            continue
        w, h = int(im.get("width")), int(im.get("height"))
        polys = []
        for pg in im.iter("polygon"):
            label = pg.get("label", "").strip().lower()
            pts = [tuple(map(float, p.split(","))) for p in pg.get("points").split(";")]
            polys.append((label, np.array(pts, dtype=np.float32)))
        return polys, (w, h)
    raise SystemExit(f"no <image> matching stem '{image_stem}' in {xml_path}")


def fill_regions(polys, labels, scale, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)
    hit = False
    for label, pts in polys:
        if label in labels:
            cv2.fillPoly(mask, [np.round(pts * scale).astype(np.int32)], 1)
            hit = True
    return mask.astype(bool), hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prob", required=True)
    ap.add_argument("--ridge", required=True, help="ridge_u8.png from prelabel_draft.py")
    ap.add_argument("--regions", required=True, help="CVAT for images 1.1 XML")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--p_lo", type=float, default=0.15, help="model thr in clean (high recall)")
    ap.add_argument("--p_hi", type=float, default=0.25, help="model thr in overlap (conservative)")
    ap.add_argument("--ridge_pct", type=float, default=90.0, help="ridge percentile in clean")
    ap.add_argument("--min_area", type=int, default=8)
    ap.add_argument("--close", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    rgb = np.array(Image.open(args.image).convert("RGB"))
    H, W = rgb.shape[:2]

    prob = np.load(args.prob)
    craq = prob[1] if prob.ndim == 3 else prob
    if craq.shape != (H, W):
        craq = cv2.resize(craq, (W, H), interpolation=cv2.INTER_LINEAR)

    ridge = np.array(Image.open(args.ridge).convert("L")).astype(np.float32) / 255.0
    if ridge.shape != (H, W):
        ridge = cv2.resize(ridge, (W, H), interpolation=cv2.INTER_LINEAR)

    polys, (xw, xh) = parse_cvat_polygons(args.regions, stem)
    scale = W / xw
    if abs(scale - H / xh) > 0.02:
        print(f"[warn] non-uniform scale x={W/xw:.3f} y={H/xh:.3f}; using x")
    print(f"image {W}x{H}  regions drawn at {xw}x{xh}  scale={scale:.3f}  polys={len(polys)}")

    clean_region, has_clean = fill_regions(polys, CLEAN_LABELS, scale, H, W)
    overlap_region, has_overlap = fill_regions(polys, OVERLAP_LABELS, scale, H, W)
    if not (has_clean or has_overlap):
        raise SystemExit("no polygons matched clean/overlap label sets; check labels in XML")

    r_thr = np.percentile(ridge, args.ridge_pct)
    aggressive = (craq >= args.p_lo) | (ridge >= r_thr)
    conservative = craq >= args.p_hi

    final = (clean_region & aggressive) | (overlap_region & conservative)
    if args.min_area > 0:
        final = remove_small_objects(final, min_size=args.min_area)
    if args.close > 0:
        final = closing(final, disk(args.close))

    cov = 100.0 * final.sum() / (H * W)
    print(f"clean_region%={100*clean_region.mean():.2f} overlap_region%={100*overlap_region.mean():.2f} "
          f"final_craq%={cov:.2f}")

    # binary + VOC colour + overlay
    Image.fromarray((final.astype(np.uint8) * 255)).save(out / "mask_craq.png")
    voc = np.zeros((H, W, 3), dtype=np.uint8)
    voc[final] = CRAQ_RGB
    Image.fromarray(voc).save(out / "seg_voc.png")

    ov = rgb.copy()
    ov[final] = (0.45 * ov[final] + 0.55 * np.array((255, 0, 0))).astype(np.uint8)
    Image.fromarray(ov).save(out / "overlay_final.png")
    if W > 2000:
        s = 2000 / W
        Image.fromarray(cv2.resize(ov, (2000, int(H * s)), interpolation=cv2.INTER_AREA)).save(
            out / "overlay_final_preview.png")
    print(f"wrote final to {out}")


if __name__ == "__main__":
    main()
