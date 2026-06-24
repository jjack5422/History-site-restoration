import argparse
import json
import os
import sys
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from crackseg_common.data_utils import list_images, split_train_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs/default.yaml"))
    parser.add_argument("--src", default=None)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(PROJECT_ROOT), "_data/split.json"))
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    src = args.src or cfg["project"]["tiled_dir"]
    val_ratio = args.val_ratio if args.val_ratio is not None else cfg["split"]["val_ratio"]
    seed = args.seed if args.seed is not None else cfg["split"]["seed"]

    items = [os.path.basename(p) for p in list_images(src)]
    if not items:
        raise SystemExit(f"{src} 沒有圖")
    train, val = split_train_val(items, val_ratio=val_ratio, seed=seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"src": src, "train": train, "val": val,
                   "val_ratio": val_ratio, "seed": seed}, f, indent=2)
    print(f"train={len(train)}, val={len(val)} -> {args.out}")


if __name__ == "__main__":
    main()
