"""build_craq_cv.py — 由 labeled32_craq_v3 group_split 寫各 fold YOLO 清單 + data yaml,
並複製 val 的 images/masks 到 craqfold{k}/{val_images,val_masks}(供 calibrate/eval 的 dir 介面)。
標籤需先用 binary_mask_to_yolo_seg.py 產到 <tiles>/labels。"""
import argparse, json, os, shutil

DEF_TILES = "/home/zzz90/research/crack_detection_SepSAM2/sepsam/datasets/craq_cv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default=DEF_TILES)
    ap.add_argument("--split", default=None)
    ap.add_argument("--configs_dir", default="configs")
    ap.add_argument("--yaml_prefix", default="data_craq_cv")
    args = ap.parse_args()

    tiles = os.path.abspath(args.tiles)
    split = args.split or os.path.join(tiles, "group_split_stem.json")
    img_dir = os.path.join(tiles, "images")
    msk_dir = os.path.join(tiles, "masks")
    cfg_dir = os.path.abspath(args.configs_dir)
    os.makedirs(cfg_dir, exist_ok=True)

    payload = json.load(open(split, encoding="utf-8"))
    for fd in payload["folds"]:
        k = fd["fold"]
        def img_of(name):
            return os.path.join(img_dir, os.path.splitext(name)[0] + ".png")
        train = [img_of(n) for n in fd["train"]]
        val = [img_of(n) for n in fd["val"]]

        fdir = os.path.join(tiles, f"craqfold{k}")
        vi = os.path.join(fdir, "val_images"); vm = os.path.join(fdir, "val_masks")
        os.makedirs(vi, exist_ok=True); os.makedirs(vm, exist_ok=True)
        for n in fd["val"]:
            stem = os.path.splitext(n)[0]
            shutil.copyfile(os.path.join(img_dir, stem + ".png"), os.path.join(vi, stem + ".png"))
            shutil.copyfile(os.path.join(msk_dir, stem + ".png"), os.path.join(vm, stem + ".png"))

        tr = os.path.join(fdir, "train.txt"); va = os.path.join(fdir, "val.txt")
        with open(tr, "w") as f: f.write("\n".join(train) + "\n")
        with open(va, "w") as f: f.write("\n".join(val) + "\n")

        yml = os.path.join(cfg_dir, f"{args.yaml_prefix}_fold{k}.yaml")
        with open(yml, "w", encoding="utf-8") as f:
            f.write(f"# craquelure CV fold {k} (val_groups={fd.get('val_groups')})\n"
                    f"path: {tiles}\ntrain: {tr}\nval: {va}\nnc: 1\nnames:\n  - craquelure\n")
        print(f"fold{k}: train={len(train)} val={len(val)} val_groups={fd.get('val_groups')} -> {yml}")


if __name__ == "__main__":
    main()
