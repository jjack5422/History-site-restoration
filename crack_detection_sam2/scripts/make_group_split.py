import argparse
import json
import os
import re
import sys
from collections import defaultdict
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dataset import load_tile_index


PANEL_RE = re.compile(r"_R\d+_C\d+$")


def panel_key(stem):
    """把 stem 去掉 _R*_C* 變成 panel 等級的 group key。"""
    return PANEL_RE.sub("", stem)


def site_key(stem):
    """取 stem 第一段 (例如 KJTHT-SC-L-1RB1-1 / MGLST-DT-1L-A2-1) 當 site key。"""
    return PANEL_RE.sub("", stem)


def kfold_groups(groups, n_splits=4, seed=42):
    rng = np.random.default_rng(seed)
    g = list(groups)
    rng.shuffle(g)
    folds = [[] for _ in range(n_splits)]
    for i, name in enumerate(g):
        folds[i % n_splits].append(name)
    return folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_root", default=os.path.join(PROJECT_ROOT, "data/tiles_512"))
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data/tiles_512/group_split.json"))
    parser.add_argument("--n_splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group_by", choices=["panel", "stem"], default="panel",
                        help="panel: 把 _R*_C* 去掉聚成 panel 級 group; stem: 用整個 stem (每張原圖)")
    args = parser.parse_args()

    idx = load_tile_index(args.tiles_root)
    items = idx["items"]
    if not items:
        raise SystemExit("tile_index.json 沒有 items")

    if args.group_by == "panel":
        get_group = panel_key
    else:
        get_group = lambda s: s

    grouped = defaultdict(list)
    for it in items:
        grouped[get_group(it["stem"])].append(it)

    groups = sorted(grouped.keys())
    print(f"groups({args.group_by})={len(groups)}: {groups}")

    n_splits = args.n_splits
    if n_splits > len(groups):
        print(f"[warn] n_splits={n_splits} > n_groups={len(groups)}，clamp 成 {len(groups)} (LOPO)")
        n_splits = len(groups)

    folds_groups = kfold_groups(groups, n_splits=n_splits, seed=args.seed)

    folds = []
    for k in range(n_splits):
        val_groups = folds_groups[k]
        train_groups = [g for j, gs in enumerate(folds_groups) if j != k for g in gs]

        val_items = [it for g in val_groups for it in grouped[g]]
        train_items = [it for g in train_groups for it in grouped[g]]

        folds.append({
            "fold": k,
            "val_groups": sorted(val_groups),
            "train_groups": sorted(train_groups),
            "n_train_tiles": len(train_items),
            "n_val_tiles": len(val_items),
            "train": [it["tile"] for it in train_items],
            "val": [it["tile"] for it in val_items],
        })
        print(f"fold {k}: train_groups={len(train_groups)} train_tiles={len(train_items)} "
              f"val_groups={len(val_groups)} val_tiles={len(val_items)} val={val_groups}")

    out_payload = {
        "tiles_root": args.tiles_root,
        "group_by": args.group_by,
        "n_splits": n_splits,
        "seed": args.seed,
        "groups": groups,
        "folds": folds,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"輸出: {args.out}")


if __name__ == "__main__":
    main()
