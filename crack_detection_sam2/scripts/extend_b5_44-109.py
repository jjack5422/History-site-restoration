"""Append batch5 frames 44-109 (default.txt[44:110]) to multiclass_512_dataset.

Context: 0-43 (default.txt[:44]) was already added by extend_multiclass_711.py.
This appends the next slice. batch5 was re-exported to _data/0-109_batch5 (its
SegmentationClass filename order is scrambled by the CVAT import -> we index by
default.txt, which is the authoritative order; first 44 lines are identical to the
old _data/batch5_0-43 export, verified).

Append-only & in-place on _data/multiclass_512_dataset:
  - tile each new frame (512/stride256; mural frames are 512^2 -> 1 tile)
  - same fg/bg-sampling rule as the base build (BG_STD=5, BG_KEEP=0.15, seed42)
  - write images/, masks/ (index) + masks_color/ (rgb view)
  - regenerate group_split_stem.json (groups unchanged -> base folds stable)
  - update tile_index.json summary, rewrite README

Run:
  /home/zzz90/research/sam2_env/bin/python crack_detection_sam2/scripts/extend_b5_44-109.py
"""
from __future__ import annotations
import json, os, re, shutil
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, "/home/zzz90/research/_lib")
from crackseg_common.data_utils import tile_image  # noqa: E402

ROOT = "/home/zzz90/research"
DS = f"{ROOT}/_data/multiclass_512_dataset"
SEG_B5 = f"{ROOT}/_data/0-109_batch5/SegmentationClass"
B5_DEFAULT = f"{ROOT}/_data/0-109_batch5/ImageSets/Segmentation/default.txt"
SLICE = (44, 110)  # default.txt[44:110] = CVAT idx 44-109
IMG_ROOTS = [f"{ROOT}/_data/selected_slices", f"{ROOT}/_data/batch_4/images",
             f"{ROOT}/_data/multiclass_512_dataset_0-94/images"]
TILE, STRIDE = 512, 256
BG_STD, BG_KEEP, SEED = 5.0, 0.15, 42
PAL = {(0,0,0):0,(255,24,3):1,(9,249,213):2,(149,0,222):3,(102,255,102):4,(236,236,0):5,(255,106,77):255}
IDX2RGB = {v:k for k,v in PAL.items()}
CLASS_NAMES = ["background","crack","loss","shrinkage","craquelure","flaking"]
STEM_RE = re.compile(r"_R\d+_C\d+$")
stem_group = lambda s: STEM_RE.sub("", s)


def img_lookup():
    L = {}
    for r in IMG_ROOTS:
        for p in Path(r).rglob("*"):
            if p.is_file() and p.suffix.lower() in (".jpg",".jpeg",".png"):
                L.setdefault(p.stem, p)
    return L


def rgb_to_index(rgb):
    out = np.zeros(rgb.shape[:2], np.uint8); matched = np.zeros(rgb.shape[:2], bool)
    for c, i in PAL.items():
        m = (rgb == np.array(c, np.uint8)).all(-1); out[m] = i; matched |= m
    unk = Counter(map(tuple, rgb[~matched].reshape(-1,3))) if not matched.all() else Counter()
    return out, unk


def index_to_rgb(idx):
    out = np.zeros((*idx.shape, 3), np.uint8)
    for i, c in IDX2RGB.items():
        out[idx == i] = c
    return out


def main():
    L = img_lookup()
    os.makedirs(f"{DS}/masks_color", exist_ok=True)
    ti = json.load(open(f"{DS}/tile_index.json"))
    items = list(ti["items"])
    present = {it["tile"] for it in items}

    lines = [l.strip() for l in open(B5_DEFAULT) if l.strip()][SLICE[0]:SLICE[1]]
    frames = [s for s in lines if os.path.exists(f"{SEG_B5}/{s}.png")]
    nomask = [s for s in lines if not os.path.exists(f"{SEG_B5}/{s}.png")]
    print(f"default.txt[{SLICE[0]}:{SLICE[1]}] = {len(lines)} frames, "
          f"{len(frames)} with mask png, {len(nomask)} unannotated -> background")

    rng = np.random.default_rng(SEED)
    cls_px = Counter(); unk_total = Counter(); added = 0; added_bg = 0; skipped_dup = 0

    for stem in frames:
        if stem not in L:
            print(f"[skip] no raw image: {stem}"); continue
        img = np.array(Image.open(L[stem]).convert("RGB"))
        idx_full, unk = rgb_to_index(np.array(Image.open(f"{SEG_B5}/{stem}.png").convert("RGB")))
        unk_total.update(unk)
        if img.shape[:2] != idx_full.shape[:2]:
            print(f"[skip] size mismatch {stem}"); continue
        it_t, coords, _ = tile_image(img, TILE, STRIDE, 0)
        mt_t, _, _ = tile_image(idx_full, TILE, STRIDE, 0)
        for im, mk, (y,x) in zip(it_t, mt_t, coords):
            has_fg = bool(((mk>0)&(mk!=255)).any())
            tstd = float(im.astype(np.float32).std())
            if not has_fg:
                if tstd < BG_STD or rng.random() > BG_KEEP:
                    continue
            name = f"{stem}__y{y:05d}_x{x:05d}.png"
            if name in present:
                skipped_dup += 1; continue
            Image.fromarray(im).save(f"{DS}/images/{name}")
            Image.fromarray(mk).save(f"{DS}/masks/{name}")
            Image.fromarray(index_to_rgb(mk)).save(f"{DS}/masks_color/{name}")
            items.append({"tile":name,"stem":stem,"y":int(y),"x":int(x),
                          "has_fg":has_fg,"tile_std":tstd,
                          "fg_pixels":int(((mk>0)&(mk!=255)).sum()),"source":"batch5"})
            present.add(name)
            for c in np.unique(mk): cls_px[int(c)] += int((mk==c).sum())
            added += 1

    # unannotated mural frames = clean negatives -> all-background mask (no bg sampling,
    # user-requested: include them all as background; valuable FP-domain negatives)
    for stem in nomask:
        if stem not in L:
            print(f"[skip] no raw image: {stem}"); continue
        img = np.array(Image.open(L[stem]).convert("RGB"))
        it_t, coords, _ = tile_image(img, TILE, STRIDE, 0)
        for im, (y,x) in zip(it_t, coords):
            name = f"{stem}__y{y:05d}_x{x:05d}.png"
            if name in present:
                skipped_dup += 1; continue
            mk = np.zeros(im.shape[:2], np.uint8)
            tstd = float(im.astype(np.float32).std())
            Image.fromarray(im).save(f"{DS}/images/{name}")
            Image.fromarray(mk).save(f"{DS}/masks/{name}")
            Image.fromarray(index_to_rgb(mk)).save(f"{DS}/masks_color/{name}")
            items.append({"tile":name,"stem":stem,"y":int(y),"x":int(x),
                          "has_fg":False,"tile_std":tstd,
                          "fg_pixels":0,"source":"batch5","unannotated_bg":True})
            present.add(name)
            cls_px[0] += int(mk.size)
            added_bg += 1

    # regenerate 5-fold GroupKFold by stem-group (seed 42) over full item list
    groups = defaultdict(list)
    for it in items: groups[stem_group(it["stem"])].append(it)
    gnames = sorted(groups); rng2 = np.random.default_rng(SEED)
    shuf = list(gnames); rng2.shuffle(shuf)
    folds_g = [shuf[i::5] for i in range(5)]
    folds = []
    for k in range(5):
        val_g = sorted(folds_g[k]); tr_g = sorted(g for j in range(5) if j!=k for g in folds_g[j])
        val = [it["tile"] for g in val_g for it in groups[g]]
        tr = [it["tile"] for g in tr_g for it in groups[g]]
        folds.append({"fold":k,"val_groups":val_g,"train_groups":tr_g,
                      "n_train_tiles":len(tr),"n_val_tiles":len(val),"train":tr,"val":val})
    json.dump({"tiles_root":DS,"group_by":"stem","n_splits":5,"seed":SEED,
               "groups":gnames,"folds":folds},
              open(f"{DS}/group_split_stem.json","w"), ensure_ascii=False, indent=2)

    src_count = Counter(it.get("source","base") for it in items)
    json.dump({"summary":{"base":"multiclass_512_dataset_711",
               "sources":dict(src_count),"total_tiles":len(items),
               "last_append":{"batch5_default_slice":list(SLICE),
                              "added_fg_tiles":added,"added_bg_tiles":added_bg}},
               "items":items}, open(f"{DS}/tile_index.json","w"), ensure_ascii=False, indent=2)

    print(f"added {added} fg + {added_bg} bg tiles (dup-skipped {skipped_dup}) -> total {len(items)}")
    print("sources:", dict(src_count))
    print("new-tile class pixel totals:")
    for c in sorted(cls_px):
        nm = "ignore" if c==255 else CLASS_NAMES[c]
        print(f"  {c:>3} {nm:<11} {cls_px[c]:>12,}")
    if unk_total:
        print("UNKNOWN colors:", unk_total.most_common(8))
    print(f"-> {DS}")


if __name__ == "__main__":
    main()
