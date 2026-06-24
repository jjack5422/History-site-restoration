"""Rename batch4 '02.' tiles (= upper lintel of KJTHT-SC-M-A4-8) to the A4-8 stem
everywhere, and MERGE them into the A4-8 group in every split (fixes the latent
leakage where A4-8 base was in val but batch4-A4-8 was in train: fold4 of
group_split_stem and fold0 of demo_holdout).

Idempotent-ish: re-running after success is a no-op for files (nothing matches).
"""
import os, json, shutil, sys

ROOT = "/home/zzz90/research/_data"
A48 = "KJTHT-SC-M-A4-8"
DS = f"{ROOT}/craq_0-94_v1/tiles_512"
COR = f"{ROOT}/craq_0-94_v1/tiles_512_corrgt"
NEW = f"{ROOT}/craq_512_dataset_711"
B4 = f"{ROOT}/batch_4"

# ---- 1. rename files in data dirs (substring batch4-02. -> A48) ----
DATA_DIRS = [
    f"{DS}/images", f"{DS}/masks", f"{DS}/resunet_prob/prob",
    f"{DS}/resunet_prob_demoholdout/prob", f"{DS}/resunet_prob_demoholdout_thin/prob",
    f"{COR}/masks", f"{COR}/fp_weight",
    f"{NEW}/images", f"{NEW}/masks",
]
n_files = 0
for d in DATA_DIRS:
    if not os.path.isdir(d):
        print("  (skip missing)", d); continue
    cnt = 0
    for fn in os.listdir(d):
        if "batch4-02." in fn:
            new = fn.replace("batch4-02.", A48)
            assert not os.path.exists(os.path.join(d, new)), f"collision {new} in {d}"
            os.rename(os.path.join(d, fn), os.path.join(d, new)); cnt += 1
    n_files += cnt
    print(f"  {d}: renamed {cnt}")

# source batch_4: files starting '02.' -> A48 (jpg images + png labels)
for sub in ("images", "labels"):
    d = f"{B4}/{sub}"
    if not os.path.isdir(d):
        continue
    cnt = 0
    for fn in os.listdir(d):
        if fn.startswith("02."):
            new = A48 + fn[3:]            # drop '02.' (3 chars), prepend stem
            assert not os.path.exists(os.path.join(d, new)), f"collision {new}"
            os.rename(os.path.join(d, fn), os.path.join(d, new)); cnt += 1
    n_files += cnt
    print(f"  {d}: renamed {cnt}")
print(f"total files renamed: {n_files}")


# ---- 2. JSON updates ----
def backup(p):
    b = p + ".prerename.bak"
    if not os.path.exists(b):
        shutil.copy2(p, b)


def rename_str(s):
    return s.replace("batch4-02.", A48)


def fix_tile_index(p):
    backup(p)
    d = json.load(open(p))
    for it in d["items"]:
        it["tile"] = rename_str(it["tile"])
        it["stem"] = rename_str(it["stem"])
    json.dump(d, open(p, "w"), ensure_ascii=False, indent=0)
    print("  tile_index updated:", p)


def fix_split(p, move_to_val_folds, tile_index_path):
    """rename tiles; drop 'batch4-02.' group; for folds in move_to_val_folds move
    the batch4 tiles from train to val. Recompute per-fold count metadata."""
    backup(p)
    hasfg = {it["tile"]: it["has_fg"] for it in json.load(open(tile_index_path))["items"]}
    d = json.load(open(p))
    if "groups" in d:
        d["groups"] = [g for g in d["groups"] if g != "batch4-02."]
        if A48 not in d["groups"]:
            d["groups"].append(A48)
    for f in d["folds"]:
        fi = f.get("fold")
        b4_train = {rename_str(t) for t in f["train"] if "batch4-02." in t}
        tr = [rename_str(t) for t in f["train"]]
        va = [rename_str(t) for t in f["val"]]
        for k in ("train_groups", "val_groups"):
            if k in f:
                f[k] = [g for g in f[k] if g != "batch4-02."]
        if fi in move_to_val_folds:
            tr = [t for t in tr if t not in b4_train]
            va = va + sorted(b4_train)
            if A48 not in f.get("val_groups", []):
                f.setdefault("val_groups", []).append(A48)
        f["train"], f["val"] = tr, va
        if "n_train_tiles" in f:
            f["n_train_tiles"] = len(tr); f["n_val_tiles"] = len(va)
            f["n_train_fg_tiles"] = sum(hasfg.get(t, False) for t in tr)
            f["n_val_fg_tiles"] = sum(hasfg.get(t, False) for t in va)
    json.dump(d, open(p, "w"), ensure_ascii=False, indent=0)
    print(f"  split updated: {p}  (moved->val folds {move_to_val_folds})")


# group_split_stem: fold4 = A4-8 in val -> move batch4 to val there
fix_tile_index(f"{DS}/tile_index.json")
fix_split(f"{DS}/group_split_stem.json", {4}, f"{DS}/tile_index.json")
# demo_holdout: fold0 val = demo stems incl A4-8 -> move batch4 to val
fix_split(f"{DS}/demo_holdout_split.json", {0}, f"{DS}/tile_index.json")
# nofold: everything train, no val -> just rename
fix_split(f"{DS}/nofold_all_train.json", set(), f"{DS}/tile_index.json")
# new folder copies
fix_tile_index(f"{NEW}/tile_index.json")
fix_split(f"{NEW}/group_split_stem.json", {4}, f"{NEW}/tile_index.json")
print("DONE")
