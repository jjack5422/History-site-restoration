"""Build a binary dataset where foreground = UNION of crack + craquelure (one class).

Mirrors scripts/build_binary_datasets.py layout (tiles_512/{images,masks} + tile_index.json
+ group_split_stem.json) so train.py / train_craq_promptrefine.py consume it unchanged.

Source: _data/0-94/SegmentationClass (canonical multi-class RGB palette, 181 labels);
foreground pixel = crack(255,24,3) OR craquelure(102,255,102). Source jpgs are searched
across selected_slices/* and _data/image.

  /home/zzz90/research/sam2_env/bin/python crack_detection_sam2/scripts/build_union_dataset.py
"""
import argparse, glob, json, os, re, shutil, sys
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm
sys.path.insert(0, "/home/zzz90/research/_lib")
from crackseg_common.data_utils import tile_image

CRACK = (255, 24, 3)
CRAQUELURE = (102, 255, 102)
STEM_RE = re.compile(r"_R\d+_C\d+$")


def stem_group(s): return STEM_RE.sub("", s)


def union_mask(rgb):
    rgb = rgb[..., :3]
    m = np.zeros(rgb.shape[:2], np.uint8)
    for c in (CRACK, CRAQUELURE):
        m |= (rgb == np.array(c, rgb.dtype).reshape(1, 1, 3)).all(-1).astype(np.uint8)
    return m


def find_jpg(stem, srcdirs):
    for d in srcdirs:
        p = os.path.join(d, stem + ".jpg")
        if os.path.isfile(p):
            return p
    return None


def kfold_loso(groups, n, seed=42):
    rng = np.random.default_rng(seed); g = sorted(groups); rng.shuffle(g)
    folds = [[] for _ in range(n)]
    for i, name in enumerate(g):
        folds[i % n].append(name)
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_dir", default="/home/zzz90/research/_data/0-94/SegmentationClass")
    ap.add_argument("--src_dirs", nargs="+",
                    default=sorted(glob.glob("/home/zzz90/research/_data/selected_slices/*"))
                    + ["/home/zzz90/research/_data/image"])
    ap.add_argument("--out_root", default="/home/zzz90/research/_data/crackcraq_0-94_v1")
    ap.add_argument("--tile_size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--bg_std_threshold", type=float, default=5.0)
    ap.add_argument("--bg_keep_ratio", type=float, default=0.15)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tiles_root = os.path.join(args.out_root, f"tiles_{args.tile_size}")
    timg = os.path.join(tiles_root, "images"); tmsk = os.path.join(tiles_root, "masks")
    for d in (timg, tmsk): os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    files = sorted(f for f in os.listdir(args.seg_dir) if f.lower().endswith(".png"))
    index = []
    n_total = n_fg = n_bg = n_blank = n_sub = n_noimg = 0
    for fname in tqdm(files, desc="build crackcraq"):
        stem = os.path.splitext(fname)[0]
        bm = union_mask(np.array(Image.open(os.path.join(args.seg_dir, fname)).convert("RGB")))
        jp = find_jpg(stem, args.src_dirs)
        if jp is None:
            n_noimg += 1; continue
        img = np.array(Image.open(jp).convert("RGB"))
        if img.shape[:2] != bm.shape[:2]:
            print(f"[warn] shape mismatch skip {stem}"); continue
        it_, coords, _ = tile_image(img, tile_size=args.tile_size, stride=args.stride, pad_value=0)
        mt_, _, _ = tile_image(bm, tile_size=args.tile_size, stride=args.stride, pad_value=0)
        for img_t, msk_t, (y, x) in zip(it_, mt_, coords):
            n_total += 1
            has_fg = bool((msk_t > 0).any()); std = float(img_t.astype(np.float32).std())
            if not has_fg:
                if std < args.bg_std_threshold: n_blank += 1; continue
                if rng.random() > args.bg_keep_ratio: n_sub += 1; continue
            name = f"{stem}__y{y:05d}_x{x:05d}.png"
            Image.fromarray(img_t).save(os.path.join(timg, name))
            Image.fromarray(msk_t).save(os.path.join(tmsk, name))
            index.append({"tile": name, "stem": stem, "y": int(y), "x": int(x),
                          "has_fg": has_fg, "tile_std": std, "fg_pixels": int(msk_t.sum())})
            n_fg += has_fg; n_bg += (not has_fg)

    summary = {"target_class": "crackcraq", "target_rgb_union": [list(CRACK), list(CRAQUELURE)],
               "seg_dir": args.seg_dir, "n_source_labels": len(files), "n_no_image": n_noimg,
               "tile_size": args.tile_size, "stride": args.stride, "seed": args.seed,
               "total_tiles": n_total, "kept_foreground": n_fg, "kept_background_sampled": n_bg,
               "dropped_blank": n_blank, "dropped_background_subsample": n_sub}
    json.dump({"summary": summary, "items": index},
              open(os.path.join(tiles_root, "tile_index.json"), "w"), ensure_ascii=False, indent=2)

    by = defaultdict(list)
    for it in index: by[stem_group(it["stem"])].append(it)
    groups = sorted(by); n_eff = min(args.n_splits, len(groups))
    fg = kfold_loso(groups, n_eff, args.seed)
    folds = []
    for k in range(n_eff):
        vg = sorted(fg[k]); tg = sorted(g for j, gs in enumerate(fg) if j != k for g in gs)
        vi = [it for g in vg for it in by[g]]; ti = [it for g in tg for it in by[g]]
        folds.append({"fold": k, "val_groups": vg, "train_groups": tg,
                      "n_train_tiles": len(ti), "n_val_tiles": len(vi),
                      "n_train_fg_tiles": sum(it["has_fg"] for it in ti),
                      "n_val_fg_tiles": sum(it["has_fg"] for it in vi),
                      "train": [it["tile"] for it in ti], "val": [it["tile"] for it in vi]})
    json.dump({"tiles_root": tiles_root, "group_by": "stem", "n_splits": n_eff,
               "seed": args.seed, "groups": groups, "folds": folds},
              open(os.path.join(tiles_root, "group_split_stem.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"groups={len(groups)} folds={n_eff} -> {tiles_root}")


if __name__ == "__main__":
    main()
