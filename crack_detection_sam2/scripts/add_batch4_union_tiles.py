"""Add batch4 (3 panels) crack∪craquelure tiles into the crackcraq union dataset, train-only.

Mirrors add_batch4_craq_tiles.py but foreground = crack OR craquelure under the batch_4
NEW palette. Keeps ONLY tiles containing crack/craquelure (unlabeled batch4 area is not a
clean negative); appends to crackcraq_0-94_v1 tile_index + every fold's train list.

  /home/zzz90/research/sam2_env/bin/python crack_detection_sam2/scripts/add_batch4_union_tiles.py
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
import numpy as np
from PIL import Image
sys.path.insert(0, "/home/zzz90/research/_lib")
from crackseg_common.data_utils import tile_image

# batch_4 NEW palette (from _data/batch_4/labelmap.txt)
CRACK_RGB = (165, 236, 223)
CRAQUELURE_RGB = (13, 117, 210)
DEFAULT_SOURCES = ["02._R1_C01", "02._R1_C02", "02._R1_C03"]
IMG_DIR = "/home/zzz90/research/_data/batch_4/images"
SEG_DIR = "/home/zzz90/research/_data/batch_4/labels"
TILES_ROOT = "/home/zzz90/research/_data/crackcraq_0-94_v1/tiles_512"
STEM_PREFIX = "batch4-"


def union_mask(rgb):
    rgb = rgb[..., :3]
    m = np.zeros(rgb.shape[:2], np.uint8)
    for c in (CRACK_RGB, CRAQUELURE_RGB):
        m |= (rgb == np.array(c, rgb.dtype).reshape(1, 1, 3)).all(-1).astype(np.uint8)
    return m


def pristine_load(path):
    bak = path + ".orig.bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    return json.load(open(bak))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_root", default=TILES_ROOT)
    ap.add_argument("--seg_dir", default=SEG_DIR)
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--min_fg_pixels", type=int, default=32)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    img_out = os.path.join(args.tiles_root, "images")
    msk_out = os.path.join(args.tiles_root, "masks")
    os.makedirs(img_out, exist_ok=True); os.makedirs(msk_out, exist_ok=True)
    index_path = os.path.join(args.tiles_root, "tile_index.json")
    split_path = os.path.join(args.tiles_root, "group_split_stem.json")
    tile_index = pristine_load(index_path)
    split = pristine_load(split_path)

    new_items, new_tiles, groups = [], [], set()
    for src in args.sources:
        img = np.array(Image.open(os.path.join(IMG_DIR, src + ".jpg")).convert("RGB"))
        msk = union_mask(np.array(Image.open(os.path.join(args.seg_dir, src + ".png")).convert("RGB")))
        if img.shape[:2] != msk.shape[:2]:
            raise SystemExit(f"shape mismatch {src}")
        stem = STEM_PREFIX + src
        group = stem.rsplit("_R", 1)[0] if "_R" in stem else stem
        groups.add(group)
        it_, coords, _ = tile_image(img, tile_size=args.size, stride=args.stride, pad_value=0)
        mt_, _, _ = tile_image(msk, tile_size=args.size, stride=args.stride, pad_value=0)
        kept = 0
        for img_t, msk_t, (y, x) in zip(it_, mt_, coords):
            fg = int((msk_t > 0).sum())
            if fg < args.min_fg_pixels:
                continue
            name = f"{stem}__y{y:05d}_x{x:05d}.png"
            if not args.dry_run:
                Image.fromarray(img_t).save(os.path.join(img_out, name))
                Image.fromarray(msk_t.astype(np.uint8)).save(os.path.join(msk_out, name))
            new_items.append({"tile": name, "stem": stem, "y": int(y), "x": int(x),
                              "has_fg": True, "tile_std": float(img_t.astype(np.float32).std()),
                              "fg_pixels": fg})
            new_tiles.append(name); kept += 1
        print(f"[{src}] tiles={len(coords)} kept_union={kept}")
    print(f"TOTAL new crack∪craquelure tiles: {len(new_tiles)} groups={sorted(groups)}")
    if args.dry_run:
        print("dry_run: nothing written."); return

    tile_index["items"].extend(new_items)
    tile_index.setdefault("summary", {})["batch4_union_added"] = {
        "n_tiles": len(new_tiles), "sources": args.sources,
        "palette": {"crack": list(CRACK_RGB), "craquelure": list(CRAQUELURE_RGB)},
        "policy": "labeled tiles only, all into train"}
    json.dump(tile_index, open(index_path, "w"), ensure_ascii=False, indent=2)

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
    json.dump(split, open(split_path, "w"), ensure_ascii=False, indent=2)
    print(f"appended {len(new_tiles)} train tiles to all {len(split['folds'])} folds (idempotent via *.orig.bak)")


if __name__ == "__main__":
    main()
