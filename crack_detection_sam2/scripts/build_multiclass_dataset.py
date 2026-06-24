"""Build a multiclass tile segmentation dataset from VOC-style RGB masks.

The output mask values are class ids:
    0 background
    1 crack
    2 loss
    3 shrinkage
    4 craquelure
    5 flaking
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
from crackseg_common.data_utils import tile_image  # noqa: E402


CLASS_NAMES = ["background", "crack", "loss", "shrinkage", "craquelure", "flaking"]
PALETTE = {
    "background": (0, 0, 0),
    "crack": (255, 24, 3),
    "loss": (9, 249, 213),
    "shrinkage": (149, 0, 222),
    "craquelure": (102, 255, 102),
    "flaking": (236, 236, 0),
}
STEM_RE = re.compile(r"_R\d+_C\d+$")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def stem_group(stem: str) -> str:
    return STEM_RE.sub("", stem)


def build_image_lookup(image_root: Path) -> dict[str, Path]:
    lookup = {}
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            lookup.setdefault(path.stem, path)
    return lookup


def rgb_to_class_mask(rgb_mask: np.ndarray) -> np.ndarray:
    if rgb_mask.ndim != 3 or rgb_mask.shape[-1] not in (3, 4):
        raise ValueError(f"expect RGB/RGBA mask, got shape={rgb_mask.shape}")
    rgb = rgb_mask[..., :3]
    out = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for class_id, name in enumerate(CLASS_NAMES[1:], start=1):
        color = np.array(PALETTE[name], dtype=rgb.dtype).reshape(1, 1, 3)
        out[(rgb == color).all(axis=-1)] = class_id
    return out


def kfold_groups(groups: list[str], n_splits: int, seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    shuffled = list(sorted(groups))
    rng.shuffle(shuffled)
    folds = [[] for _ in range(n_splits)]
    for idx, group in enumerate(shuffled):
        folds[idx % n_splits].append(group)
    return folds


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg_dir", default=str(RESEARCH_ROOT / "_data/0-94/SegmentationClass"))
    parser.add_argument("--image_root", default=str(RESEARCH_ROOT / "_data/selected_slices"))
    parser.add_argument("--out_root", default=str(RESEARCH_ROOT / "_data/multiclass_512_dataset_0-94"))
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--bg_std_threshold", type=float, default=5.0)
    parser.add_argument("--bg_keep_ratio", type=float, default=0.15)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seg_dir = Path(args.seg_dir)
    image_root = Path(args.image_root)
    out_root = Path(args.out_root)
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"output exists and is not empty: {out_root}")

    img_out = out_root / "images"
    mask_out = out_root / "masks"
    tiles_root = out_root / f"tiles_{args.tile_size}"
    tile_img_out = tiles_root / "images"
    tile_mask_out = tiles_root / "masks"
    for path in (img_out, mask_out, tile_img_out, tile_mask_out):
        path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    image_lookup = build_image_lookup(image_root)
    seg_files = sorted(seg_dir.glob("*.png"))
    if not seg_files:
        raise SystemExit(f"no segmentation masks found in {seg_dir}")

    tile_index = []
    missing_images = []
    class_pixels_full = {name: 0 for name in CLASS_NAMES}
    class_pixels_tiles = {name: 0 for name in CLASS_NAMES}
    n_total_tiles = 0
    n_kept_fg = 0
    n_kept_bg = 0
    n_drop_blank = 0
    n_drop_bg_subsample = 0

    for seg_path in tqdm(seg_files, desc="build multiclass"):
        stem = seg_path.stem
        image_path = image_lookup.get(stem)
        if image_path is None:
            missing_images.append(stem)
            continue

        rgb_mask = np.array(Image.open(seg_path).convert("RGB"))
        class_mask = rgb_to_class_mask(rgb_mask)
        image = np.array(Image.open(image_path).convert("RGB"))
        if image.shape[:2] != class_mask.shape[:2]:
            print(f"[warn] shape mismatch skip {stem}: image={image.shape[:2]} mask={class_mask.shape[:2]}")
            continue

        shutil.copy2(image_path, img_out / f"{stem}{image_path.suffix.lower()}")
        Image.fromarray(class_mask).save(mask_out / f"{stem}.png")
        for class_id, name in enumerate(CLASS_NAMES):
            class_pixels_full[name] += int((class_mask == class_id).sum())

        image_tiles, coords, _ = tile_image(image, tile_size=args.tile_size, stride=args.stride, pad_value=0)
        mask_tiles, _, _ = tile_image(class_mask, tile_size=args.tile_size, stride=args.stride, pad_value=0)
        for img_tile, mask_tile, (y, x) in zip(image_tiles, mask_tiles, coords):
            n_total_tiles += 1
            has_fg = bool((mask_tile > 0).any())
            tile_std = float(img_tile.astype(np.float32).std())
            if not has_fg:
                if tile_std < args.bg_std_threshold:
                    n_drop_blank += 1
                    continue
                if rng.random() > args.bg_keep_ratio:
                    n_drop_bg_subsample += 1
                    continue

            tile_name = f"{stem}__y{y:05d}_x{x:05d}.png"
            Image.fromarray(img_tile).save(tile_img_out / tile_name)
            Image.fromarray(mask_tile).save(tile_mask_out / tile_name)
            pixel_counts = {name: int((mask_tile == class_id).sum()) for class_id, name in enumerate(CLASS_NAMES)}
            for name, count in pixel_counts.items():
                class_pixels_tiles[name] += count
            tile_index.append({
                "tile": tile_name,
                "stem": stem,
                "group": stem_group(stem),
                "y": int(y),
                "x": int(x),
                "has_fg": has_fg,
                "tile_std": tile_std,
                "class_pixels": pixel_counts,
            })
            if has_fg:
                n_kept_fg += 1
            else:
                n_kept_bg += 1

    groups_by_stem = defaultdict(list)
    for item in tile_index:
        groups_by_stem[item["group"]].append(item)
    groups = sorted(groups_by_stem)
    n_eff = min(args.n_splits, len(groups))
    folds_groups = kfold_groups(groups, n_eff, args.seed)
    folds = []
    for fold_idx in range(n_eff):
        val_groups = sorted(folds_groups[fold_idx])
        train_groups = sorted(group for idx, fold in enumerate(folds_groups) if idx != fold_idx for group in fold)
        val_items = [item for group in val_groups for item in groups_by_stem[group]]
        train_items = [item for group in train_groups for item in groups_by_stem[group]]
        folds.append({
            "fold": fold_idx,
            "val_groups": val_groups,
            "train_groups": train_groups,
            "n_train_tiles": len(train_items),
            "n_val_tiles": len(val_items),
            "n_train_fg_tiles": sum(1 for item in train_items if item["has_fg"]),
            "n_val_fg_tiles": sum(1 for item in val_items if item["has_fg"]),
            "train": [item["tile"] for item in train_items],
            "val": [item["tile"] for item in val_items],
        })

    summary = {
        "class_names": CLASS_NAMES,
        "class_ids": {name: idx for idx, name in enumerate(CLASS_NAMES)},
        "palette_rgb": {name: list(PALETTE[name]) for name in CLASS_NAMES},
        "seg_dir": str(seg_dir.resolve()),
        "image_root": str(image_root.resolve()),
        "tile_size": args.tile_size,
        "stride": args.stride,
        "bg_std_threshold": args.bg_std_threshold,
        "bg_keep_ratio": args.bg_keep_ratio,
        "seed": args.seed,
        "n_source_masks": len(seg_files),
        "n_missing_images": len(missing_images),
        "missing_images": missing_images,
        "total_tiles_before_filter": n_total_tiles,
        "kept_tiles": len(tile_index),
        "kept_foreground": n_kept_fg,
        "kept_background_sampled": n_kept_bg,
        "dropped_blank": n_drop_blank,
        "dropped_background_subsample": n_drop_bg_subsample,
        "full_image_pixel_counts": class_pixels_full,
        "tile_pixel_counts": class_pixels_tiles,
    }
    write_json(tiles_root / "tile_index.json", {"summary": summary, "items": tile_index})
    write_json(tiles_root / "group_split_stem.json", {
        "tiles_root": str(tiles_root.resolve()),
        "group_by": "stem",
        "n_splits": n_eff,
        "seed": args.seed,
        "groups": groups,
        "folds": folds,
    })

    readme = f"""# {out_root.name}

Multiclass 512x512 tile segmentation dataset built from `_data/0-94/SegmentationClass`.

## Classes

| id | class | RGB in source mask |
|---:|---|---|
""" + "\n".join(
        f"| {idx} | {name} | `{PALETTE[name]}` |" for idx, name in enumerate(CLASS_NAMES)
    ) + f"""

## Contents

```text
images/            source images copied from {image_root}
masks/             full-size uint8 class-id masks
tiles_{args.tile_size}/images/  {len(tile_index)} tiled RGB PNG images
tiles_{args.tile_size}/masks/   {len(tile_index)} tiled uint8 class-id masks
tiles_{args.tile_size}/tile_index.json
tiles_{args.tile_size}/group_split_stem.json
```

Background-only tiles are filtered with `bg_std_threshold={args.bg_std_threshold}` and sampled with `bg_keep_ratio={args.bg_keep_ratio}`.
Group splits are by source stem with seed `{args.seed}`.
"""
    (out_root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
