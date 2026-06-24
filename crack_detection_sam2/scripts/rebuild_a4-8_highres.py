"""Rebuild the high-res A4-8 panel tiles in craq_0-94_v1 from the 8 thinned panels.

State: the high-res A4-8 panels were renamed batch4-02._R1_C0x -> KJTHT-SC-M-A4-8_R1/R2_C0x
and merged into the KJTHT-SC-M-A4-8 group. This script:
  1. removes existing high-res A4-8 tiles (KJTHT-SC-M-A4-8_R1_* and _R2_*) from
     tiles_512/{images,masks}, tiles_512_corrgt/masks, and tile_index  (keeps R4/R5/R6 = orig 0-94).
  2. tiles the 8 thinned panels (labels_thin, craquelure ~7.8px, stride 512, keep-fg) into
     KJTHT-SC-M-A4-8_R{r}_C{c}__y_x.png  (+ corrgt copy).
  3. regenerates group_split_stem.json (5-fold by stem, seed42) + demo_holdout + nofold.

  /home/zzz90/research/sam2_env/bin/python crack_detection_sam2/scripts/rebuild_a4-8_highres.py [--dry_run]
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict
import numpy as np
from PIL import Image
sys.path.insert(0, "/home/zzz90/research/_lib")
from crackseg_common.data_utils import tile_image
Image.MAX_IMAGE_PIXELS = None

CRAQ = (13, 117, 210)
V1 = "/home/zzz90/research/_data/craq_0-94_v1/tiles_512"
IMG_DIR = "/home/zzz90/research/_data/batch_4/images"
THIN_DIR = "/home/zzz90/research/_data/batch_4/labels_thin"
PANELS = [f"KJTHT-SC-M-A4-8_R{r}_C{c:02d}" for r in (1, 2) for c in range(1, 5)]
HIRES_RE = re.compile(r"^KJTHT-SC-M-A4-8_R[12]_C\d+__")   # high-res panel tiles to remove
STEM_RE = re.compile(r"_R\d+_C\d+$")
DEMO = {"KJTHT-SC-L-1RB1-1", "KJTHT-SC-M-A4-8", "KJTHT-SC-R-A4-3"}


def grp(t): return STEM_RE.sub("", t.split("__")[0])


def kfold(groups, n=5, seed=42):
    rng = np.random.default_rng(seed); g = sorted(groups); rng.shuffle(g)
    f = [[] for _ in range(n)]
    for i, x in enumerate(g): f[i % n].append(x)
    return f


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--size", type=int, default=512); ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--min_fg", type=int, default=32); args = ap.parse_args()

    ti = json.load(open(f"{V1}/tile_index.json"))
    old = [it for it in ti["items"] if HIRES_RE.match(it["tile"])]
    print(f"removing {len(old)} existing high-res A4-8 tiles (R1/R2)")

    new_items = []
    for f in PANELS:
        img = np.array(Image.open(f"{IMG_DIR}/{f}.jpg").convert("RGB"))
        craq = (np.array(Image.open(f"{THIN_DIR}/{f}.png").convert("RGB")) == np.array(CRAQ)).all(-1).astype(np.uint8)
        it_, coords, _ = tile_image(img, tile_size=args.size, stride=args.stride, pad_value=0)
        mt_, _, _ = tile_image(craq, tile_size=args.size, stride=args.stride, pad_value=0)
        kept = 0
        for img_t, msk_t, (y, x) in zip(it_, mt_, coords):
            fg = int((msk_t > 0).sum())
            if fg < args.min_fg: continue
            name = f"{f}__y{y:05d}_x{x:05d}.png"
            if not args.dry_run:
                Image.fromarray(img_t).save(f"{V1}/images/{name}")
                Image.fromarray(msk_t).save(f"{V1}/masks/{name}")
                Image.fromarray(msk_t).save(f"{V1}/../tiles_512_corrgt/masks/{name}")
            new_items.append({"tile": name, "stem": f, "y": int(y), "x": int(x),
                              "has_fg": True, "tile_std": float(img_t.astype(np.float32).std()), "fg_pixels": fg})
            kept += 1
        print(f"  {f}: kept {kept}")
    print(f"new high-res A4-8 tiles: {len(new_items)}")
    if args.dry_run:
        print("dry_run: nothing written."); return

    # delete old high-res tile files (R1/R2) from disk
    for it in old:
        for d in (f"{V1}/images", f"{V1}/masks", f"{V1}/../tiles_512_corrgt/masks"):
            p = f"{d}/{it['tile']}"
            if os.path.exists(p): os.remove(p)

    # update index
    keep = [it for it in ti["items"] if not HIRES_RE.match(it["tile"])]
    ti["items"] = keep + new_items
    ti.setdefault("summary", {})["a4-8_highres_thinned"] = {"n_panels": len(PANELS), "n_tiles": len(new_items), "craq_width_px": 7.8}
    json.dump(ti, open(f"{V1}/tile_index.json", "w"), ensure_ascii=False, indent=2)

    # regenerate splits from new index
    allt = [it["tile"] for it in ti["items"]]
    by = defaultdict(list)
    for t in allt: by[grp(t)].append(t)
    groups = sorted(by)
    fg = kfold(groups)
    folds = []
    for k in range(5):
        vg = sorted(fg[k]); tg = sorted(g for j, gs in enumerate(fg) if j != k for g in gs)
        vi = [t for g in vg for t in by[g]]; tr = [t for g in tg for t in by[g]]
        folds.append({"fold": k, "val_groups": vg, "train_groups": tg, "n_train_tiles": len(tr), "n_val_tiles": len(vi),
                      "n_train_fg_tiles": len(tr), "n_val_fg_tiles": len(vi), "train": tr, "val": vi})
    json.dump({"tiles_root": V1, "group_by": "stem", "n_splits": 5, "seed": 42, "groups": groups, "folds": folds},
              open(f"{V1}/group_split_stem.json", "w"), ensure_ascii=False, indent=2)
    # demo_holdout + nofold
    tr = [t for t in allt if grp(t) not in DEMO]; va = [t for t in allt if grp(t) in DEMO]
    json.dump({"tiles_root": V1, "group_by": "stem", "n_splits": 1, "seed": 42, "groups": groups,
               "folds": [{"fold": 0, "val_groups": sorted(DEMO), "train_groups": [g for g in groups if g not in DEMO],
                          "n_train_tiles": len(tr), "n_val_tiles": len(va), "n_train_fg_tiles": len(tr), "n_val_fg_tiles": len(va),
                          "train": tr, "val": va}]}, open(f"{V1}/demo_holdout_split.json", "w"), ensure_ascii=False, indent=2)
    json.dump({"tiles_root": V1, "group_by": "stem", "n_splits": 1, "seed": 42, "groups": groups,
               "folds": [{"fold": 0, "val_groups": [], "train_groups": groups, "n_train_tiles": len(allt), "n_val_tiles": 0,
                          "train": allt, "val": []}]}, open(f"{V1}/nofold_all_train.json", "w"), ensure_ascii=False, indent=2)
    # which fold holds out A4-8
    for fd in folds:
        if "KJTHT-SC-M-A4-8" in fd["val_groups"]:
            print(f"A4-8 held out in fold{fd['fold']}  (train {fd['n_train_tiles']} / val {fd['n_val_tiles']})")
    print(f"total tiles now: {len(allt)}")


if __name__ == "__main__":
    main()
