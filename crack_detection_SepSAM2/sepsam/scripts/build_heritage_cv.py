"""
build_heritage_cv.py — 把 heritage_ft 的全部 32 張 1024 crop（train+val 來源合併）
切成 512/stride256 tile 成單一 flat pool，再依「原圖 panel」分 4 個 fold（LOPO）。

panel key：tile 檔名 {crop}_y{y}_x{x}，crop = PANEL_R*_C*。
  先去 _y*_x* 得 crop，再去 _R*_C* 得 panel。
  4 panels（KJTHT-SC-L-1RB1-1 / -M-2LB1-2 / -M-2RB1-4 / -L-A4-4）→ 剛好 4 folds。

每個 fold 留 1 panel（8 crop→72 tile）當 val、其餘 3 panel（24 crop→216 tile）train。

輸出：
  datasets/heritage_ft_cv/{images,labels}/          全部 tile 共用 pool
  datasets/heritage_ft_cv/fold{k}/{train,val}.txt   各 fold 的影像清單（絕對路徑）
  datasets/heritage_ft_cv/folds.json                fold 摘要
  configs/data_heritage_cv_fold{k}.yaml             各 fold 的 ultralytics data yaml

範例：
  python scripts/build_heritage_cv.py --src datasets/heritage_ft \
      --pool datasets/heritage_ft_cv --tile 512 --stride 256 --n_splits 4 --seed 42
"""
import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from make_heritage_tiles import (  # noqa: E402
    index_by_stem,
    mask_to_yolo_lines,
    tile_origins,
)
from src.preprocess import clahe_bgr  # noqa: E402

# crop stem = PANEL_R\d+_C\d+ ；tile = crop + _y\d+_x\d+
TILE_SUFFIX_RE = re.compile(r"_y\d+_x\d+$")
PANEL_RE = re.compile(r"_R\d+_C\d+$")


def crop_to_panel(crop_stem):
    """KJTHT-SC-L-1RB1-1_R2_C04 -> KJTHT-SC-L-1RB1-1"""
    return PANEL_RE.sub("", crop_stem)


def tile_to_panel(tile_stem):
    """KJTHT-SC-L-1RB1-1_R2_C04_y0_x256 -> KJTHT-SC-L-1RB1-1"""
    return crop_to_panel(TILE_SUFFIX_RE.sub("", tile_stem))


def kfold_groups(groups, n_splits, seed):
    rng = np.random.default_rng(seed)
    g = list(groups)
    rng.shuffle(g)
    folds = [[] for _ in range(n_splits)]
    for i, name in enumerate(g):
        folds[i % n_splits].append(name)
    return folds


def build_pool(src, splits, pool, tile, stride, min_area, epsilon, keep_empty,
               clahe=False, clahe_clip=2.0, clahe_grid=8):
    """把 src/{split}/{images,masks} 全部切成 tile 寫進單一 flat pool。回傳 tile stem list。"""
    out_img = os.path.join(pool, "images")
    out_lbl = os.path.join(pool, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    tile_stems = []
    n_pos = n_empty = n_inst = 0
    for sp in splits:
        idir = os.path.join(src, sp, "images")
        mdir = os.path.join(src, sp, "masks")
        imgs = index_by_stem(idir)
        masks = index_by_stem(mdir)
        stems = sorted(set(imgs) & set(masks))
        for stem in stems:
            img = cv2.imread(imgs[stem])
            msk = cv2.imread(masks[stem], cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                print(f"[skip] 讀檔失敗: {stem}")
                continue
            if msk.shape != img.shape[:2]:
                msk = cv2.resize(msk, (img.shape[1], img.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            msk = (msk > 127).astype(np.uint8) * 255
            if clahe:
                img = clahe_bgr(img, clip=clahe_clip, grid=clahe_grid)
            H, W = img.shape[:2]
            for y0 in tile_origins(H, tile, stride):
                for x0 in tile_origins(W, tile, stride):
                    it = img[y0:y0 + tile, x0:x0 + tile]
                    mt = msk[y0:y0 + tile, x0:x0 + tile]
                    lines = mask_to_yolo_lines(mt, tile, min_area, epsilon)
                    has = len(lines) > 0
                    if not has and not keep_empty:
                        continue
                    name = f"{stem}_y{y0}_x{x0}"
                    cv2.imwrite(os.path.join(out_img, name + ".jpg"), it,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    with open(os.path.join(out_lbl, name + ".txt"), "w", encoding="utf-8") as f:
                        f.write("\n".join(lines))
                    tile_stems.append(name)
                    n_inst += len(lines)
                    n_pos += int(has)
                    n_empty += int(not has)
    print(f"[pool] tiles={len(tile_stems)} (pos={n_pos} empty={n_empty}) instances={n_inst}")
    return sorted(tile_stems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/heritage_ft")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--pool", default="datasets/heritage_ft_cv")
    ap.add_argument("--configs_dir", default="configs")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--min-area", type=float, default=8.0, dest="min_area")
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--keep-empty", action="store_true", default=True, dest="keep_empty")
    ap.add_argument("--drop-empty", dest="keep_empty", action="store_false")
    ap.add_argument("--n_splits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--clahe-clip", type=float, default=2.0, dest="clahe_clip")
    ap.add_argument("--clahe-grid", type=int, default=8, dest="clahe_grid")
    args = ap.parse_args()

    # 路徑全轉絕對，方便 ultralytics 從任何 cwd 解析
    pool = os.path.abspath(args.pool)
    src = os.path.abspath(args.src)
    configs_dir = os.path.abspath(args.configs_dir)
    img_dir = os.path.join(pool, "images")

    tile_stems = build_pool(src, args.splits, pool, args.tile, args.stride,
                            args.min_area, args.epsilon, args.keep_empty,
                            clahe=args.clahe, clahe_clip=args.clahe_clip,
                            clahe_grid=args.clahe_grid)
    if not tile_stems:
        raise SystemExit("沒有產生任何 tile")

    # group by panel
    grouped = {}
    for st in tile_stems:
        grouped.setdefault(tile_to_panel(st), []).append(st)
    panels = sorted(grouped)
    print(f"panels({len(panels)}): " + ", ".join(f"{p}({len(grouped[p])})" for p in panels))

    n_splits = args.n_splits
    if n_splits > len(panels):
        print(f"[warn] n_splits={n_splits} > n_panels={len(panels)}，clamp 成 {len(panels)}")
        n_splits = len(panels)

    fold_panels = kfold_groups(panels, n_splits, args.seed)

    folds_meta = []
    for k in range(n_splits):
        val_panels = sorted(fold_panels[k])
        train_panels = sorted(p for j, ps in enumerate(fold_panels) if j != k for p in ps)
        val_tiles = [st for p in val_panels for st in grouped[p]]
        train_tiles = [st for p in train_panels for st in grouped[p]]

        fdir = os.path.join(pool, f"fold{k}")
        os.makedirs(fdir, exist_ok=True)
        train_txt = os.path.join(fdir, "train.txt")
        val_txt = os.path.join(fdir, "val.txt")
        with open(train_txt, "w") as f:
            f.write("\n".join(os.path.join(img_dir, st + ".jpg") for st in sorted(train_tiles)) + "\n")
        with open(val_txt, "w") as f:
            f.write("\n".join(os.path.join(img_dir, st + ".jpg") for st in sorted(val_tiles)) + "\n")

        yaml_path = os.path.join(configs_dir, f"data_heritage_cv_fold{k}.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(
                f"# heritage 4-fold panel-grouped CV — fold {k}（由 build_heritage_cv.py 產生）\n"
                f"# val panel: {', '.join(val_panels)}\n"
                f"# train {len(train_tiles)} tiles / val {len(val_tiles)} tiles\n"
                f"path: {pool}\n"
                f"train: {train_txt}\n"
                f"val: {val_txt}\n"
                f"nc: 1\n"
                f"names:\n"
                f"  - crack\n"
            )

        folds_meta.append({
            "fold": k,
            "val_panels": val_panels,
            "train_panels": train_panels,
            "n_train_tiles": len(train_tiles),
            "n_val_tiles": len(val_tiles),
            "data_yaml": yaml_path,
        })
        print(f"fold {k}: val={val_panels} train_tiles={len(train_tiles)} val_tiles={len(val_tiles)}")

    payload = {
        "pool": pool,
        "src": src,
        "tile": args.tile,
        "stride": args.stride,
        "keep_empty": args.keep_empty,
        "n_splits": n_splits,
        "seed": args.seed,
        "group_by": "panel",
        "panels": panels,
        "folds": folds_meta,
    }
    with open(os.path.join(pool, "folds.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n完成 → pool={pool}  folds.json + {n_splits} 個 data_heritage_cv_fold*.yaml")


if __name__ == "__main__":
    main()
