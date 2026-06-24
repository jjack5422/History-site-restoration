"""Crop craquelure-labeled 512 tiles from batch_4 large-resolution labels and
merge them into craq_0-94_v1/tiles_512 as TRAIN-ONLY data.

Motivation: the deterioration model misses white craquelure (e.g. top-center of
KJTHT-SC-M-A4-8) because those regions were skipped during slicing. batch_4 has
3 usable large-res labels (02._R1_C01/C02/C03). We tile only the craquelure-
labeled regions and add them so every fold trains on them.

Key rules (per user):
  * Only craquelure (RGB 13,117,210 in batch_4 palette) -> foreground=1.
  * KEEP ONLY tiles that contain craquelure. Unlabeled area is NOT a clean
    negative (it was skipped because similar tiles already exist in training),
    so pure-background tiles are dropped entirely (no bg sampling).
  * All new tiles go into folds[*]["train"] for every fold; never val.

Idempotent: a pristine copy of tile_index.json / group_split_stem.json is saved
to *.orig.bak on first run and used as the base every run, so re-running cannot
double-append.

Run:
    /home/zzz90/research/sam2_env/bin/python \
        crack_detection_sam2/scripts/add_batch4_craq_tiles.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/zzz90/research/_lib")
from crackseg_common.data_utils import tile_image  # noqa: E402

# batch_4 CVAT palette (from _data/batch_4/labelmap.txt)
CRAQUELURE_RGB = (13, 117, 210)

DEFAULT_SOURCES = ["02._R1_C01", "02._R1_C02", "02._R1_C03"]
DEFAULT_IMG_DIR = "/home/zzz90/research/_data/batch_4/images"
DEFAULT_SEG_DIR = "/home/zzz90/research/_data/batch_4/labels"
DEFAULT_TILES_ROOT = "/home/zzz90/research/_data/craq_0-94_v1/tiles_512"
STEM_PREFIX = "batch4-"  # marks tiles added by this script (for idempotency)


def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def craquelure_binary(rgb_mask: np.ndarray) -> np.ndarray:
    target = np.array(CRAQUELURE_RGB, dtype=np.uint8)
    return (rgb_mask == target).all(axis=-1).astype(np.uint8)


def pristine_load(path: str):
    """Load a JSON, snapshotting to *.orig.bak on first run; always read from
    the pristine snapshot so the script is idempotent."""
    bak = path + ".orig.bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    with open(bak) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default=DEFAULT_IMG_DIR)
    ap.add_argument("--seg_dir", default=DEFAULT_SEG_DIR)
    ap.add_argument("--tiles_root", default=DEFAULT_TILES_ROOT)
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--min_fg_pixels", type=int, default=32,
                    help="drop tiles whose craquelure pixel count is below this")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    img_out = os.path.join(args.tiles_root, "images")
    msk_out = os.path.join(args.tiles_root, "masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(msk_out, exist_ok=True)

    index_path = os.path.join(args.tiles_root, "tile_index.json")
    split_path = os.path.join(args.tiles_root, "group_split_stem.json")
    tile_index = pristine_load(index_path)
    split = pristine_load(split_path)

    new_items = []
    new_tiles = []
    groups = set()

    for src in args.sources:
        img_path = os.path.join(args.img_dir, src + ".jpg")
        seg_path = os.path.join(args.seg_dir, src + ".png")
        img = load_rgb(img_path)
        msk = craquelure_binary(load_rgb(seg_path))
        if img.shape[:2] != msk.shape[:2]:
            raise SystemExit(f"shape mismatch {src}: img {img.shape} msk {msk.shape}")

        stem = STEM_PREFIX + src
        group = stem.rsplit("_R", 1)[0] if "_R" in stem else stem
        groups.add(group)

        img_tiles, coords, _ = tile_image(img, tile_size=args.size,
                                          stride=args.stride, pad_value=0)
        msk_tiles, _, _ = tile_image(msk, tile_size=args.size,
                                     stride=args.stride, pad_value=0)

        kept = 0
        for img_t, msk_t, (y, x) in zip(img_tiles, msk_tiles, coords):
            fg = int((msk_t > 0).sum())
            if fg < args.min_fg_pixels:
                continue
            tile_name = f"{stem}__y{y:05d}_x{x:05d}.png"
            if not args.dry_run:
                Image.fromarray(img_t).save(os.path.join(img_out, tile_name))
                Image.fromarray(msk_t.astype(np.uint8)).save(
                    os.path.join(msk_out, tile_name))
            new_items.append({
                "tile": tile_name, "stem": stem, "y": int(y), "x": int(x),
                "has_fg": True,
                "tile_std": float(img_t.astype(np.float32).std()),
                "fg_pixels": fg,
            })
            new_tiles.append(tile_name)
            kept += 1
        total = len(coords)
        print(f"[{src}] {img.shape[1]}x{img.shape[0]}  tiles={total}  "
              f"kept_craquelure={kept}  dropped={total - kept}")

    print(f"\nTOTAL new craquelure tiles: {len(new_tiles)}  groups={sorted(groups)}")

    if args.dry_run:
        print("dry_run: nothing written.")
        return

    # --- append to tile_index (pristine base already loaded) ---
    tile_index["items"].extend(new_items)
    summ = tile_index.setdefault("summary", {})
    summ["batch4_added"] = {
        "n_tiles": len(new_tiles), "sources": args.sources,
        "target_class": "craquelure", "target_rgb": list(CRAQUELURE_RGB),
        "policy": "labeled tiles only (no bg), all into train",
    }
    with open(index_path, "w") as f:
        json.dump(tile_index, f, ensure_ascii=False, indent=2)

    # --- add new tiles to TRAIN of every fold; never val ---
    for fd in split["folds"]:
        fd["train"].extend(new_tiles)
        fd["n_train_tiles"] = len(fd["train"])
        fd["n_train_fg_tiles"] = fd.get("n_train_fg_tiles", 0) + len(new_tiles)
        for g in groups:
            if g not in fd["train_groups"]:
                fd["train_groups"].append(g)
    for g in groups:
        if g not in split["groups"]:
            split["groups"].append(g)
    with open(split_path, "w") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    print(f"Updated:\n  {index_path} (items {len(tile_index['items'])})")
    print(f"  {split_path} (+{len(new_tiles)} train tiles in all "
          f"{len(split['folds'])} folds)")
    print("Pristine backups: *.orig.bak (re-running is idempotent).")


if __name__ == "__main__":
    main()
