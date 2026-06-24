"""Generate a no-fold 'train on everything' split for a tiles dataset.

Reads <tiles_root>/tile_index.json and writes <tiles_root>/nofold_all_train.json with a single
fold whose train and val both contain every tile (final-expert style: train on all, keep last.pt).
Both train.py and crack_detection_unet/src/train.py read folds[fold]["train"]/["val"] tile lists.
"""
from __future__ import annotations

import argparse
import json
import os


def build_nofold(tile_index):
    tiles = [it["tile"] for it in tile_index["items"]]
    return {
        "n_splits": 1,
        "group_by": "stem",
        "folds": [{"fold": 0, "train": tiles, "val": tiles}],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_root", required=True)
    args = ap.parse_args()
    with open(os.path.join(args.tiles_root, "tile_index.json")) as f:
        idx = json.load(f)
    out = build_nofold(idx)
    path = os.path.join(args.tiles_root, "nofold_all_train.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}: {len(out['folds'][0]['train'])} tiles (train==val)")


if __name__ == "__main__":
    main()
