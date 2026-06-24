"""Build a hard-negative-augmented craquelure tile dataset.

Hard negatives are mined from the model's own errors: regions where the stage-1
ResUNet craq prob fires but the (corrected /0-94) GT craquelure is 0 -- i.e. the
painted-content false positives (figures, decorative motifs). Tiles dominated by
such FP and containing ~no real craquelure are tagged `hard_neg` and kept at
ratio 1.0 (never subsampled), instead of being thrown away as ordinary
background. Masks stay standard binary craq (hard-neg tiles are all-background);
only the tile *sampling* changes.

Outputs (mirrors build_binary_datasets layout):
    {out_root}/tiles_{tsize}/images/*.png
    {out_root}/tiles_{tsize}/masks/*.png            (uint8 0/1)
    {out_root}/tiles_{tsize}/tile_index.json        (+ hard_neg, fp_pixels)
    {out_root}/tiles_{tsize}/group_split_stem.json  (reuses seed=42 stem folds)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from crackseg_common.data_utils import tile_image  # noqa: E402

CRAQ_RGB = (102, 255, 102)
import re

STEM_RE = re.compile(r"_R\d+_C\d+$")


def stem_group(stem):
    return STEM_RE.sub("", stem)


def kfold_loso(groups, n_splits=4, seed=42):
    rng = np.random.default_rng(seed)
    g = sorted(groups)
    rng.shuffle(g)
    folds = [[] for _ in range(n_splits)]
    for i, name in enumerate(g):
        folds[i % n_splits].append(name)
    return folds


def clean_specks(mask, min_area):
    if min_area <= 0:
        return mask
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    areas = np.bincount(lab.ravel())
    keep = areas >= min_area
    keep[0] = False
    return keep[lab]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_dir", default="../_data/_seg_src_craq_0-94_v2")
    ap.add_argument("--image_dir", default="../_data/_img_src_craq_0-94_v2")
    ap.add_argument("--prob_dir",
                    default="../crack_detection_unet/runs/predict-batch1-resunet-2026-06-10/prob")
    ap.add_argument("--out_root", default="../_data/craq_0-94_v2_hn")
    ap.add_argument("--tile_size", type=int, default=224)
    ap.add_argument("--stride", type=int, default=112)
    ap.add_argument("--tau_fp", type=float, default=0.5,
                    help="prob threshold for counting a pixel as a (false) positive")
    ap.add_argument("--fp_min_area", type=int, default=64,
                    help="drop FP components smaller than this (full-res px) before tiling")
    ap.add_argument("--hardneg_fp_frac", type=float, default=0.01,
                    help="a fg-free tile is hard_neg if fp_pixels/tile_area >= this")
    ap.add_argument("--bg_std_threshold", type=float, default=5.0)
    ap.add_argument("--bg_keep_ratio", type=float, default=0.15)
    ap.add_argument("--n_splits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = os.path.abspath(args.out_root)
    tiles_root = os.path.join(out_root, f"tiles_{args.tile_size}")
    timg = os.path.join(tiles_root, "images")
    tmsk = os.path.join(tiles_root, "masks")
    for d in (timg, tmsk):
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    files = sorted(f for f in os.listdir(args.seg_dir) if f.lower().endswith(".png"))
    tile_area = args.tile_size * args.tile_size
    hardneg_min_px = int(args.hardneg_fp_frac * tile_area)

    tile_index = []
    n_total = n_pos = n_hardneg = n_plainbg = n_drop_blank = n_drop_bg = 0
    fp_total = 0

    for fname in tqdm(files, desc="hardneg build"):
        stem = os.path.splitext(fname)[0]
        seg = np.array(Image.open(os.path.join(args.seg_dir, fname)).convert("RGB"))
        gt = np.all(seg == np.array(CRAQ_RGB), axis=-1)

        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(args.image_dir, stem + ext)
            if os.path.isfile(p):
                img_path = p
                break
        if img_path is None:
            print(f"[warn] no image for {stem}, skip")
            continue
        img = np.array(Image.open(img_path).convert("RGB"))

        prob_path = os.path.join(args.prob_dir, stem + ".npy")
        if not os.path.isfile(prob_path):
            print(f"[warn] no prob for {stem}, skip")
            continue
        prob = np.load(prob_path)[1].astype(np.float32)
        fp_map = (prob > args.tau_fp) & (~gt)
        fp_map = clean_specks(fp_map, args.fp_min_area)
        fp_total += int(fp_map.sum())

        img_tiles, coords, _ = tile_image(img, tile_size=args.tile_size, stride=args.stride, pad_value=0)
        gt_tiles, _, _ = tile_image(gt.astype(np.uint8), tile_size=args.tile_size, stride=args.stride, pad_value=0)
        fp_tiles, _, _ = tile_image(fp_map.astype(np.uint8), tile_size=args.tile_size, stride=args.stride, pad_value=0)

        for img_t, gt_t, fp_t, (y, x) in zip(img_tiles, gt_tiles, fp_tiles, coords):
            n_total += 1
            fg = int(gt_t.sum())
            fpx = int(fp_t.sum())
            has_fg = fg > 0
            tile_std = float(img_t.astype(np.float32).std())
            hard_neg = (not has_fg) and (fpx >= hardneg_min_px)

            if has_fg:
                n_pos += 1
            elif hard_neg:
                n_hardneg += 1            # kept at ratio 1.0
            else:
                if tile_std < args.bg_std_threshold:
                    n_drop_blank += 1
                    continue
                if rng.random() > args.bg_keep_ratio:
                    n_drop_bg += 1
                    continue
                n_plainbg += 1

            tile_name = f"{stem}__y{y:05d}_x{x:05d}.png"
            Image.fromarray(img_t).save(os.path.join(timg, tile_name))
            Image.fromarray(gt_t.astype(np.uint8)).save(os.path.join(tmsk, tile_name))
            tile_index.append({
                "tile": tile_name, "stem": stem, "y": int(y), "x": int(x),
                "has_fg": has_fg, "hard_neg": bool(hard_neg),
                "fg_pixels": fg, "fp_pixels": fpx, "tile_std": tile_std,
            })

    summary = {
        "target_class": "craquelure", "mode": "hard_negative_mining",
        "seg_dir": os.path.abspath(args.seg_dir), "image_dir": os.path.abspath(args.image_dir),
        "prob_dir": os.path.abspath(args.prob_dir),
        "n_source_labels": len(files), "tile_size": args.tile_size, "stride": args.stride,
        "tau_fp": args.tau_fp, "fp_min_area": args.fp_min_area,
        "hardneg_fp_frac": args.hardneg_fp_frac, "hardneg_min_px": hardneg_min_px,
        "bg_std_threshold": args.bg_std_threshold, "bg_keep_ratio": args.bg_keep_ratio, "seed": args.seed,
        "total_tiles_scanned": n_total, "kept_positive": n_pos, "kept_hard_neg": n_hardneg,
        "kept_plain_bg": n_plainbg, "dropped_blank": n_drop_blank, "dropped_bg_subsample": n_drop_bg,
        "fp_pixels_mined_total": fp_total,
    }
    with open(os.path.join(tiles_root, "tile_index.json"), "w") as f:
        json.dump({"summary": summary, "items": tile_index}, f, indent=2)

    groups_by_stem = defaultdict(list)
    for it in tile_index:
        groups_by_stem[stem_group(it["stem"])].append(it)
    groups = sorted(groups_by_stem.keys())
    n_eff = min(args.n_splits, len(groups))
    folds_groups = kfold_loso(groups, n_splits=n_eff, seed=args.seed)
    folds = []
    for k in range(n_eff):
        val_groups = sorted(folds_groups[k])
        train_groups = sorted(g for j, gs in enumerate(folds_groups) if j != k for g in gs)
        val_items = [it for g in val_groups for it in groups_by_stem[g]]
        train_items = [it for g in train_groups for it in groups_by_stem[g]]
        folds.append({
            "fold": k, "val_groups": val_groups, "train_groups": train_groups,
            "n_train_tiles": len(train_items), "n_val_tiles": len(val_items),
            "n_train_fg_tiles": sum(1 for it in train_items if it["has_fg"]),
            "n_train_hardneg_tiles": sum(1 for it in train_items if it["hard_neg"]),
            "n_val_fg_tiles": sum(1 for it in val_items if it["has_fg"]),
            "train": [it["tile"] for it in train_items],
            "val": [it["tile"] for it in val_items],
        })
    with open(os.path.join(tiles_root, "group_split_stem.json"), "w") as f:
        json.dump({"tiles_root": tiles_root, "group_by": "stem", "n_splits": n_eff,
                   "seed": args.seed, "groups": groups, "folds": folds}, f, indent=2)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for fd in folds:
        print(f"fold {fd['fold']}: train={fd['n_train_tiles']} "
              f"(fg={fd['n_train_fg_tiles']}, hardneg={fd['n_train_hardneg_tiles']})  "
              f"val={fd['n_val_tiles']} (fg={fd['n_val_fg_tiles']})  val_groups={fd['val_groups']}")


if __name__ == "__main__":
    main()
