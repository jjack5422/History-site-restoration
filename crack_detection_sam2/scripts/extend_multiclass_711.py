"""Extend multiclass_512_dataset_711 with two supplements, append-only:
  A. 0-111 minus 0-94  = 17 newly-labeled frames (CVAT idx 95-111)
  B. batch5 frames 0-43 = default.txt[:44] (door-god mural; CVAT alpha order)

Output = new dataset = 711 tiles + supplement tiles, with regenerated
group_split (5-fold GroupKFold by stem-group, seed 42), index, masks_color, README.
Index map matches multiclass_512_dataset_711:
  0 bg 1 crack 2 loss 3 shrinkage 4 craquelure 5 flaking  255 ignore

Run:
  /home/zzz90/research/sam2_env/bin/python crack_detection_sam2/scripts/extend_multiclass_711.py
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
BASE = f"{ROOT}/_data/multiclass_512_dataset_711"
OUT = f"{ROOT}/_data/multiclass_512_dataset_711_0-111b5"
SEG_0111 = f"{ROOT}/_data/0-111/SegmentationClass"
SEG_B5 = f"{ROOT}/_data/batch5_0-43/SegmentationClass"
B5_DEFAULT = f"{ROOT}/_data/batch5_0-43/ImageSets/Segmentation/default.txt"
IMG_ROOTS = [f"{ROOT}/_data/selected_slices", f"{ROOT}/_data/batch_4/images",
             f"{ROOT}/_data/multiclass_512_dataset_0-94/images"]
TILE, STRIDE = 512, 256
BG_STD, BG_KEEP, SEED = 5.0, 0.15, 42
PAL = {(0,0,0):0,(255,24,3):1,(9,249,213):2,(149,0,222):3,(102,255,102):4,(236,236,0):5,(255,106,77):255}
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


def supplement_frames():
    # A. 0-111 newly-labeled vs 0-94 (foreground-diff == CVAT 95-111)
    def fgset(d):
        s=set()
        for f in os.listdir(d):
            if f.endswith(".png") and (np.array(Image.open(f"{d}/{f}").convert("RGB")).sum(-1)>0).any():
                s.add(os.path.splitext(f)[0])
        return s
    new17 = sorted(fgset(SEG_0111) - fgset(f"{ROOT}/_data/0-94/SegmentationClass"))
    # B. batch5 default.txt[:44] that have a mask png
    lines = [l.strip() for l in open(B5_DEFAULT) if l.strip()][:44]
    b5 = [s for s in lines if os.path.exists(f"{SEG_B5}/{s}.png")]
    return [(s, SEG_0111) for s in new17] + [(s, SEG_B5) for s in b5]


def main():
    L = img_lookup()
    os.makedirs(f"{OUT}/images", exist_ok=True)
    os.makedirs(f"{OUT}/masks", exist_ok=True)
    # 1. copy base 711
    for sub in ("images","masks"):
        for f in os.listdir(f"{BASE}/{sub}"):
            shutil.copy2(f"{BASE}/{sub}/{f}", f"{OUT}/{sub}/{f}")
    base_items = json.load(open(f"{BASE}/tile_index.json"))["items"]
    items = list(base_items)
    rng = np.random.default_rng(SEED)
    cls_px = Counter(); unk_total = Counter(); added = 0
    src_count = Counter()

    for stem, seg_dir in supplement_frames():
        if stem not in L:
            print(f"[skip] no raw image: {stem}"); continue
        img = np.array(Image.open(L[stem]).convert("RGB"))
        idx_full, unk = rgb_to_index(np.array(Image.open(f"{seg_dir}/{stem}.png").convert("RGB")))
        unk_total.update(unk)
        if img.shape[:2] != idx_full.shape[:2]:
            print(f"[skip] size mismatch {stem}"); continue
        it_t, coords, _ = tile_image(img, TILE, STRIDE, 0)
        mt_t, _, _ = tile_image(idx_full, TILE, STRIDE, 0)
        src = "0-111" if "0-111" in seg_dir else "batch5"
        for im, mk, (y,x) in zip(it_t, mt_t, coords):
            has_fg = bool(((mk>0)&(mk!=255)).any())
            tstd = float(im.astype(np.float32).std())
            if not has_fg:
                if tstd < BG_STD or rng.random() > BG_KEEP:
                    continue
            name = f"{stem}__y{y:05d}_x{x:05d}.png"
            Image.fromarray(im).save(f"{OUT}/images/{name}")
            Image.fromarray(mk).save(f"{OUT}/masks/{name}")
            items.append({"tile":name,"stem":stem,"y":int(y),"x":int(x),
                          "has_fg":has_fg,"tile_std":tstd,
                          "fg_pixels":int(((mk>0)&(mk!=255)).sum()),"source":src})
            for c in np.unique(mk): cls_px[int(c)] += int((mk==c).sum())
            added += 1; src_count[src]+=1

    # 2. regenerate 5-fold GroupKFold by stem-group (seed 42)
    groups = defaultdict(list)
    for it in items: groups[stem_group(it["stem"])].append(it)
    gnames = sorted(groups); rng2 = np.random.default_rng(SEED)
    shuf = list(gnames); rng2.shuffle(shuf)
    folds_g = [shuf[i::5] for i in range(5)]  # deterministic 5-way
    folds = []
    for k in range(5):
        val_g = sorted(folds_g[k]); tr_g = sorted(g for j in range(5) if j!=k for g in folds_g[j])
        val = [it["tile"] for g in val_g for it in groups[g]]
        tr = [it["tile"] for g in tr_g for it in groups[g]]
        folds.append({"fold":k,"val_groups":val_g,"train_groups":tr_g,
                      "n_train_tiles":len(tr),"n_val_tiles":len(val),"train":tr,"val":val})
    json.dump({"tiles_root":OUT,"group_by":"stem","n_splits":5,"seed":SEED,
               "groups":gnames,"folds":folds},
              open(f"{OUT}/group_split_stem.json","w"), ensure_ascii=False, indent=2)
    json.dump({"summary":{"base":"multiclass_512_dataset_711","added_tiles":added,
               "supplement_sources":dict(src_count),"total_tiles":len(items)},
               "items":items}, open(f"{OUT}/tile_index.json","w"), ensure_ascii=False, indent=2)
    shutil.copy2(f"{BASE}/labelmap.txt", f"{OUT}/labelmap.txt")

    print(f"base 711 + added {added} (src {dict(src_count)}) = {len(items)} tiles")
    print("class pixel totals:")
    for c in sorted(cls_px):
        nm = "ignore" if c==255 else CLASS_NAMES[c]
        print(f"  {c:>3} {nm:<11} {cls_px[c]:>12,}")
    if unk_total:
        print("UNKNOWN colors:", unk_total.most_common(8))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
