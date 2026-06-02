"""Build per-class binary segmentation datasets from CVAT VOC-style RGB-palette masks.

Pipeline:
    selected_slices/{stem}.jpg  +  1-31test/SegmentationClass/{stem}.png (RGB palette)
        -> for each target class:
            data/labeled32_{class}_v3/images/{stem}.jpg                (raw 1024x1024)
            data/labeled32_{class}_v3/masks/{stem}.png                 (uint8 0/1)
            data/labeled32_{class}_v3/tiles_{tsize}/images/*.png       (1024 -> tsize)
            data/labeled32_{class}_v3/tiles_{tsize}/masks/*.png
            data/labeled32_{class}_v3/tiles_{tsize}/tile_index.json
            data/labeled32_{class}_v3/tiles_{tsize}/group_split_stem.json   (4-fold LOSO by stem)

Defaults mirror existing tile_pairs.py (tile=512, stride=256, bg_std=5, bg_keep=0.15, seed=42).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from crackseg_common.data_utils import tile_image  # noqa: E402


PALETTE = {
    "background": (0, 0, 0),
    "crack": (255, 24, 3),
    "craquelure": (102, 255, 102),
    "flaking": (236, 236, 0),
    "loss": (9, 249, 213),
    "shrinkage": (149, 0, 222),
}

STEM_RE = re.compile(r"_R\d+_C\d+$")


def stem_group(stem: str) -> str:
    return STEM_RE.sub("", stem)


def palette_mask_to_binary(rgb_mask: np.ndarray, target_rgb: tuple) -> np.ndarray:
    """Return uint8 mask: 1 where pixel equals target_rgb, else 0."""
    if rgb_mask.ndim != 3 or rgb_mask.shape[-1] not in (3, 4):
        raise ValueError(f"expect RGB mask, got shape={rgb_mask.shape}")
    rgb = rgb_mask[..., :3]
    target = np.array(target_rgb, dtype=rgb.dtype).reshape(1, 1, 3)
    return (rgb == target).all(axis=-1).astype(np.uint8)


def list_label_files(seg_dir: str) -> list[str]:
    return sorted(
        f for f in os.listdir(seg_dir) if f.lower().endswith(".png")
    )


def pair_image(stem: str, image_dir: str, exts=(".jpg", ".jpeg", ".png")) -> str:
    for ext in exts:
        p = os.path.join(image_dir, stem + ext)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"no image for stem={stem} in {image_dir}")


def kfold_loso(groups, n_splits=4, seed=42):
    rng = np.random.default_rng(seed)
    g = sorted(groups)
    rng.shuffle(g)
    folds = [[] for _ in range(n_splits)]
    for i, name in enumerate(g):
        folds[i % n_splits].append(name)
    return folds


def build_class(
    target_class: str,
    target_rgb: tuple,
    seg_dir: str,
    image_dir: str,
    out_root: str,
    tile_size: int,
    stride: int,
    bg_std_threshold: float,
    bg_keep_ratio: float,
    n_splits: int,
    seed: int,
):
    out_root = os.path.abspath(out_root)
    img_out = os.path.join(out_root, "images")
    msk_out = os.path.join(out_root, "masks")
    tiles_root = os.path.join(out_root, f"tiles_{tile_size}")
    timg_out = os.path.join(tiles_root, "images")
    tmsk_out = os.path.join(tiles_root, "masks")
    for d in (img_out, msk_out, timg_out, tmsk_out):
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(seed)
    files = list_label_files(seg_dir)

    pos_pixels_per_image = []
    tile_index = []
    n_total = n_kept_fg = n_kept_bg = n_drop_blank = n_drop_bg_subsample = 0

    for fname in tqdm(files, desc=f"build {target_class}"):
        stem = os.path.splitext(fname)[0]
        rgb_mask = np.array(Image.open(os.path.join(seg_dir, fname)).convert("RGB"))
        bin_mask = palette_mask_to_binary(rgb_mask, target_rgb)
        pos_px = int(bin_mask.sum())
        pos_pixels_per_image.append((stem, pos_px))

        img_path = pair_image(stem, image_dir)
        img = np.array(Image.open(img_path).convert("RGB"))
        if img.shape[:2] != bin_mask.shape[:2]:
            print(
                f"[warn] shape mismatch skip {stem} img={img.shape} msk={bin_mask.shape}"
            )
            continue

        # 1024 level outputs
        shutil.copy2(img_path, os.path.join(img_out, stem + ".jpg"))
        Image.fromarray(bin_mask).save(os.path.join(msk_out, stem + ".png"))

        # tile to tile_size with stride
        img_tiles, coords, _ = tile_image(img, tile_size=tile_size, stride=stride, pad_value=0)
        msk_tiles, _, _ = tile_image(bin_mask, tile_size=tile_size, stride=stride, pad_value=0)

        for img_t, msk_t, (y, x) in zip(img_tiles, msk_tiles, coords):
            n_total += 1
            has_fg = bool((msk_t > 0).any())
            tile_std = float(img_t.astype(np.float32).std())
            if not has_fg:
                if tile_std < bg_std_threshold:
                    n_drop_blank += 1
                    continue
                if rng.random() > bg_keep_ratio:
                    n_drop_bg_subsample += 1
                    continue
            tile_name = f"{stem}__y{y:05d}_x{x:05d}.png"
            Image.fromarray(img_t).save(os.path.join(timg_out, tile_name))
            Image.fromarray(msk_t).save(os.path.join(tmsk_out, tile_name))
            tile_index.append({
                "tile": tile_name,
                "stem": stem,
                "y": int(y),
                "x": int(x),
                "has_fg": has_fg,
                "tile_std": tile_std,
                "fg_pixels": int(msk_t.sum()),
            })
            if has_fg:
                n_kept_fg += 1
            else:
                n_kept_bg += 1

    # tile-level class pixel counts (binary)
    bg_px = sum(tile_size * tile_size - it["fg_pixels"] for it in tile_index)
    fg_px = sum(it["fg_pixels"] for it in tile_index)

    summary = {
        "target_class": target_class,
        "target_rgb": list(target_rgb),
        "image_dir": os.path.abspath(image_dir),
        "seg_dir": os.path.abspath(seg_dir),
        "n_source_labels": len(files),
        "tile_size": tile_size,
        "stride": stride,
        "bg_std_threshold": bg_std_threshold,
        "bg_keep_ratio": bg_keep_ratio,
        "seed": seed,
        "total_tiles": n_total,
        "kept_foreground": n_kept_fg,
        "kept_background_sampled": n_kept_bg,
        "dropped_blank": n_drop_blank,
        "dropped_background_subsample": n_drop_bg_subsample,
        "tile_pixel_counts": {"bg": bg_px, target_class: fg_px},
        "image_positive_pixels": dict(pos_pixels_per_image),
    }
    with open(os.path.join(tiles_root, "tile_index.json"), "w") as f:
        json.dump({"summary": summary, "items": tile_index}, f, indent=2)

    # 4-fold leave-one-stem-out group split
    groups_by_stem = defaultdict(list)
    for it in tile_index:
        groups_by_stem[stem_group(it["stem"])].append(it)
    groups = sorted(groups_by_stem.keys())
    n_eff = min(n_splits, len(groups))
    folds_groups = kfold_loso(groups, n_splits=n_eff, seed=seed)

    folds = []
    for k in range(n_eff):
        val_groups = sorted(folds_groups[k])
        train_groups = sorted(g for j, gs in enumerate(folds_groups) if j != k for g in gs)
        val_items = [it for g in val_groups for it in groups_by_stem[g]]
        train_items = [it for g in train_groups for it in groups_by_stem[g]]
        folds.append({
            "fold": k,
            "val_groups": val_groups,
            "train_groups": train_groups,
            "n_train_tiles": len(train_items),
            "n_val_tiles": len(val_items),
            "n_train_fg_tiles": sum(1 for it in train_items if it["has_fg"]),
            "n_val_fg_tiles": sum(1 for it in val_items if it["has_fg"]),
            "train": [it["tile"] for it in train_items],
            "val": [it["tile"] for it in val_items],
        })

    split_payload = {
        "tiles_root": tiles_root,
        "group_by": "stem",
        "n_splits": n_eff,
        "seed": seed,
        "groups": groups,
        "folds": folds,
    }
    with open(os.path.join(tiles_root, "group_split_stem.json"), "w") as f:
        json.dump(split_payload, f, indent=2)

    return summary, folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seg_dir",
        default=os.path.join(os.path.dirname(PROJECT_ROOT), "_data/1-31test/SegmentationClass"),
    )
    parser.add_argument(
        "--image_dir",
        default=os.path.join(os.path.dirname(PROJECT_ROOT), "_data/selected_slices"),
    )
    parser.add_argument(
        "--out_root_template",
        default=os.path.join(os.path.dirname(PROJECT_ROOT), "_data/labeled32_{class}_v3"),
        help="{class} 會替換為 crack/craq",
    )
    parser.add_argument("--classes", nargs="+", default=["crack", "craquelure"])
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--bg_std_threshold", type=float, default=5.0)
    parser.add_argument("--bg_keep_ratio", type=float, default=0.15)
    parser.add_argument("--n_splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    short = {"crack": "crack", "craquelure": "craq"}
    overall = {}
    for cls in args.classes:
        if cls not in PALETTE:
            raise SystemExit(f"unknown class: {cls}; known={list(PALETTE)}")
        if cls == "background":
            raise SystemExit("won't build dataset for background")
        out_root = args.out_root_template.format(class_=short.get(cls, cls), **{"class": short.get(cls, cls)})
        # support {class} placeholder via plain replace as well for safety
        out_root = args.out_root_template.replace("{class}", short.get(cls, cls))
        print(f"\n=== building {cls} -> {out_root} ===")
        summary, folds = build_class(
            target_class=cls,
            target_rgb=PALETTE[cls],
            seg_dir=args.seg_dir,
            image_dir=args.image_dir,
            out_root=out_root,
            tile_size=args.tile_size,
            stride=args.stride,
            bg_std_threshold=args.bg_std_threshold,
            bg_keep_ratio=args.bg_keep_ratio,
            n_splits=args.n_splits,
            seed=args.seed,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        for fd in folds:
            print(
                f"fold {fd['fold']}: train={fd['n_train_tiles']} (fg={fd['n_train_fg_tiles']})  "
                f"val={fd['n_val_tiles']} (fg={fd['n_val_fg_tiles']})  val_groups={fd['val_groups']}"
            )
        overall[cls] = summary

    print("\n=== overall ===")
    print(json.dumps({k: v["tile_pixel_counts"] for k, v in overall.items()}, indent=2))


if __name__ == "__main__":
    main()
