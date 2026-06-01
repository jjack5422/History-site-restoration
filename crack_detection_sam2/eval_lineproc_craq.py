"""eval_lineproc_craq.py — 讀各 method 的 preds + GT,raw 與 post-proc 各算 vanilla-F1/tolerant-F1/clDice,出對照表。"""
import argparse, glob, os
import cv2, numpy as np
from lineproc import cc_filter, skeleton_centerline, cldice, tolerant_f1

TILES = "data/labeled32_craq_v3/tiles_512"
METHODS = ["denseseg", "sepsam_yolo", "promptsam2_oracle", "promptsam2_yolo"]
DENSE_VANILLA = {"denseseg": 0.634, "sepsam_yolo": 0.541,
                 "promptsam2_oracle": 0.554, "promptsam2_yolo": 0.500}


def vanilla_f1(p, g):
    p = p.astype(bool); g = g.astype(bool)
    tp = (p & g).sum(); fp = (p & ~g).sum(); fn = (~p & g).sum()
    if (tp + fp + fn) == 0:
        return 1.0
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return 0.0 if (pr + rc) == 0 else float(2 * pr * rc / (pr + rc))


def eval_method(method, k, min_area, tol, postproc):
    pd = f"preds/{method}/fold{k}"
    vm = os.path.join(TILES, f"craqfold{k}", "val_masks")
    van = tolm = cld = 0.0; n = 0
    for pp in sorted(glob.glob(os.path.join(pd, "*.png"))):
        st = os.path.splitext(os.path.basename(pp))[0]
        pred = cv2.imread(pp, 0) > 0
        gt = cv2.imread(os.path.join(vm, st + ".png"), 0) > 0
        if postproc:
            pred = cc_filter(pred, min_area)
            pred = skeleton_centerline(pred, width=3)
        van += vanilla_f1(pred, gt); tolm += tolerant_f1(pred, gt, tol); cld += cldice(pred, gt); n += 1
    return {"van": van / max(n, 1), "tol": tolm / max(n, 1), "cld": cld / max(n, 1), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--min_area", type=int, default=64)
    ap.add_argument("--tol", type=int, default=3)
    args = ap.parse_args()
    no2 = [k for k in args.folds if k != 2]
    for postproc in [False, True]:
        tag = "post-proc" if postproc else "raw"
        print(f"\n==== {tag} (min_area={args.min_area}, tol={args.tol}) — 排除fold2 平均 ====")
        print(f"{'method':22} {'vanilla':>8} {'tolerant':>9} {'clDice':>8}  (舊vanilla)")
        for m in METHODS:
            per = {k: eval_method(m, k, args.min_area, args.tol, postproc) for k in args.folds}
            def mean(key, folds): return sum(per[k][key] for k in folds) / max(len(folds), 1)
            print(f"{m:22} {mean('van',no2):>8.3f} {mean('tol',no2):>9.3f} {mean('cld',no2):>8.3f}  ({DENSE_VANILLA[m]})")


if __name__ == "__main__":
    main()
