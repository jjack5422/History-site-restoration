"""Build a binary (background / target-class) tiles_512 dataset from a multiclass
index-mask dataset (0 bg / 1 crack / 2 loss / 3 shrinkage / 4 craquelure / 5 flaking / 255 ignore).

For each target class id, write <dest>/masks/<tile> where pixel==target_id -> 1 else 0
(all other classes AND 255 ignore collapse to background). images/ is a directory
symlink back to the source images (tiles shared). tile_index.json is copied verbatim.

Also writes allval_split.json: a single fold with train == val == ALL tiles, so the
trainer can pick best.pt by full-set metric while still training on all data.
"""
import argparse, json, os, shutil
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="multiclass dataset root (has images/ masks/ tile_index.json)")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--target_id", type=int, required=True, help="multiclass index to map -> 1")
    ap.add_argument("--target_name", required=True, help="positive class name (e.g. craquelure)")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)
    dst_masks = os.path.join(dest, "masks")
    os.makedirs(dst_masks, exist_ok=True)

    # images: directory symlink (tiles are shared, no need to duplicate)
    dst_images = os.path.join(dest, "images")
    if not os.path.islink(dst_images) and not os.path.exists(dst_images):
        os.symlink(os.path.join(src, "images"), dst_images)

    # tile_index.json copy
    shutil.copy(os.path.join(src, "tile_index.json"), os.path.join(dest, "tile_index.json"))
    # keep reference group split too
    if os.path.exists(os.path.join(src, "group_split_stem.json")):
        shutil.copy(os.path.join(src, "group_split_stem.json"),
                    os.path.join(dest, "group_split_stem.json"))

    ti = json.load(open(os.path.join(src, "tile_index.json")))
    names = [it["tile"] for it in ti["items"]]

    src_masks = os.path.join(src, "masks")
    pos_total = 0
    pix_total = 0
    n_pos_tiles = 0
    for n in names:
        m = np.array(Image.open(os.path.join(src_masks, n)))
        if m.ndim == 3:
            m = m[..., 0]
        b = (m == args.target_id).astype(np.uint8)
        Image.fromarray(b).save(os.path.join(dst_masks, n))
        s = int(b.sum())
        pos_total += s
        pix_total += b.size
        if s > 0:
            n_pos_tiles += 1

    # all-data split: train == val == ALL tiles (best.pt selectable, trains on everything)
    split = {
        "tiles_root": dest,
        "note": "all-data: train==val==ALL; best.pt = best full-set epoch (val=train, optimistic)",
        "folds": [{"train": names, "val": names, "val_groups": ["ALL"]}],
    }
    json.dump(split, open(os.path.join(dest, "allval_split.json"), "w"), indent=2)

    print(f"[{args.target_name}] dest={dest}")
    print(f"  tiles={len(names)}  positive_tiles={n_pos_tiles}  "
          f"pos_px={pos_total:,} ({100*pos_total/max(pix_total,1):.3f}% of pixels)")


if __name__ == "__main__":
    main()
