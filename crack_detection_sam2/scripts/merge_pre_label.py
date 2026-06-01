"""Merge per-class binary expert outputs into a CVAT-compatible 5-class palette pre-label.

Each expert's `predict_full.py --save_prob` writes `prob/{stem}.npy` of shape `(2, H, W)`
(softmax over [bg, target_class]). This script reads both, applies thresholds, merges with
priority (craquelure overrides crack in overlap — matches existing project policy), and writes:

    {out_dir}/voc_palette/{stem}.png   RGB PNG using CVAT labelmap colors
    {out_dir}/binary_crack/{stem}.png  uint8 0/1
    {out_dir}/binary_craq/{stem}.png   uint8 0/1
    {out_dir}/overlay/{stem}.png       image blended with class colors
    {out_dir}/summary.json             per-image positive-pixel counts

By default skips the already-labeled tile stems listed in --skip_list (default.txt format).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


PALETTE = {
    "background": (0, 0, 0),
    "crack": (255, 24, 3),
    "craquelure": (102, 255, 102),
    "flaking": (236, 236, 0),
    "loss": (9, 249, 213),
    "shrinkage": (149, 0, 222),
}


def load_prob_positive(npy_path: str) -> np.ndarray:
    """Load (2,H,W) probability NPY, return positive-class prob (H,W) float32."""
    arr = np.load(npy_path)
    if arr.ndim != 3 or arr.shape[0] != 2:
        raise ValueError(
            f"expect (2,H,W) prob array, got shape={arr.shape} from {npy_path}"
        )
    return arr[1].astype(np.float32)


def read_skip_list(path: str | None) -> set[str]:
    if path is None or not os.path.isfile(path):
        return set()
    with open(path) as f:
        return {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}


def overlay_mask(img: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    fg = (mask_rgb.sum(axis=-1) > 0)
    out = img.copy()
    if out.dtype != np.uint8:
        out = out.astype(np.uint8)
    out[fg] = (alpha * mask_rgb[fg] + (1 - alpha) * out[fg]).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--craq_prob_dir", required=True,
                        help="dir of (2,H,W) prob .npy from craq predict_full.py")
    parser.add_argument("--crack_prob_dir", required=True,
                        help="dir of (2,H,W) prob .npy from crack predict_full.py")
    parser.add_argument("--image_dir", required=True,
                        help="dir of source images (1024x1024 jpg); used for overlay + size")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--skip_list", default=None,
                        help="optional path to default.txt of already-labeled stems to skip")
    parser.add_argument("--craq_thresh", type=float, default=0.5)
    parser.add_argument("--crack_thresh", type=float, default=0.5)
    parser.add_argument("--min_blob_px", type=int, default=0,
                        help="optional: drop predicted blobs smaller than this many pixels per class")
    parser.add_argument("--priority", choices=["craq_over_crack", "crack_over_craq", "by_prob"],
                        default="craq_over_crack",
                        help="overlap resolution. project policy = craq overrides crack.")
    args = parser.parse_args()

    out = Path(args.out_dir)
    for sub in ("voc_palette", "binary_crack", "binary_craq", "overlay"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    skip = read_skip_list(args.skip_list)
    print(f"skip_list size: {len(skip)} (already-labeled stems)")

    # find pairs (must exist in both prob dirs)
    craq_files = {Path(f).stem: os.path.join(args.craq_prob_dir, f)
                  for f in os.listdir(args.craq_prob_dir) if f.endswith(".npy")}
    crack_files = {Path(f).stem: os.path.join(args.crack_prob_dir, f)
                   for f in os.listdir(args.crack_prob_dir) if f.endswith(".npy")}
    stems = sorted(set(craq_files) & set(crack_files))
    missing_craq = sorted(set(crack_files) - set(craq_files))
    missing_crack = sorted(set(craq_files) - set(crack_files))
    if missing_craq:
        print(f"[warn] missing craq prob for {len(missing_craq)} stems (e.g. {missing_craq[:3]})")
    if missing_crack:
        print(f"[warn] missing crack prob for {len(missing_crack)} stems (e.g. {missing_crack[:3]})")

    process = [s for s in stems if s not in skip]
    skipped = [s for s in stems if s in skip]
    print(f"will process {len(process)} stems; skipped {len(skipped)} already-labeled")

    crack_rgb = np.array(PALETTE["crack"], dtype=np.uint8)
    craq_rgb = np.array(PALETTE["craquelure"], dtype=np.uint8)

    summary = []
    for stem in tqdm(process):
        craq_p = load_prob_positive(craq_files[stem])
        crack_p = load_prob_positive(crack_files[stem])

        craq_mask = craq_p > args.craq_thresh
        crack_mask = crack_p > args.crack_thresh

        if args.min_blob_px > 0:
            try:
                from scipy import ndimage as ndi
                for m in (craq_mask, crack_mask):
                    lab, n = ndi.label(m)
                    if n == 0:
                        continue
                    counts = np.bincount(lab.ravel())
                    keep = counts >= args.min_blob_px
                    keep[0] = False
                    m[:] = keep[lab]
            except ImportError:
                pass  # scipy missing → skip blob filter

        H, W = craq_mask.shape

        if args.priority == "craq_over_crack":
            # crack first, craq overrides
            voc = np.zeros((H, W, 3), dtype=np.uint8)
            voc[crack_mask] = crack_rgb
            voc[craq_mask] = craq_rgb
        elif args.priority == "crack_over_craq":
            voc = np.zeros((H, W, 3), dtype=np.uint8)
            voc[craq_mask] = craq_rgb
            voc[crack_mask] = crack_rgb
        else:  # by_prob — whichever has higher prob wins
            voc = np.zeros((H, W, 3), dtype=np.uint8)
            both = craq_mask & crack_mask
            craq_wins = both & (craq_p >= crack_p)
            crack_wins = both & (crack_p > craq_p)
            voc[craq_mask & ~both] = craq_rgb
            voc[crack_mask & ~both] = crack_rgb
            voc[craq_wins] = craq_rgb
            voc[crack_wins] = crack_rgb

        # save outputs
        Image.fromarray(voc, mode="RGB").save(out / "voc_palette" / f"{stem}.png")
        Image.fromarray(craq_mask.astype(np.uint8)).save(out / "binary_craq" / f"{stem}.png")
        Image.fromarray(crack_mask.astype(np.uint8)).save(out / "binary_crack" / f"{stem}.png")

        # overlay on source image (find jpg/png variants)
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(args.image_dir, stem + ext)
            if os.path.isfile(p):
                img_path = p
                break
        if img_path is not None:
            img = np.array(Image.open(img_path).convert("RGB"))
            if img.shape[:2] != (H, W):
                # resize voc to image dims (nearest)
                voc_resized = np.array(
                    Image.fromarray(voc).resize((img.shape[1], img.shape[0]), Image.NEAREST)
                )
                ov = overlay_mask(img, voc_resized)
            else:
                ov = overlay_mask(img, voc)
            Image.fromarray(ov).save(out / "overlay" / f"{stem}.png")

        summary.append({
            "stem": stem,
            "craq_pixels": int(craq_mask.sum()),
            "crack_pixels": int(crack_mask.sum()),
            "craq_max_prob": float(craq_p.max()),
            "crack_max_prob": float(crack_p.max()),
            "craq_mean_prob": float(craq_p.mean()),
            "crack_mean_prob": float(crack_p.mean()),
        })

    payload = {
        "args": vars(args),
        "n_processed": len(process),
        "n_skipped_labeled": len(skipped),
        "skipped_stems_sample": skipped[:10],
        "per_image": summary,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(payload, f, indent=2)

    # aggregate stats
    if summary:
        craq_px = np.array([r["craq_pixels"] for r in summary])
        crack_px = np.array([r["crack_pixels"] for r in summary])
        print()
        print("=== aggregate ===")
        print(f"  craq pixels per image  mean={craq_px.mean():.0f}  median={int(np.median(craq_px))}  >0 in {(craq_px>0).sum()}/{len(summary)}")
        print(f"  crack pixels per image mean={crack_px.mean():.0f}  median={int(np.median(crack_px))}  >0 in {(crack_px>0).sum()}/{len(summary)}")
    print(f"outputs at: {out}")


if __name__ == "__main__":
    main()
