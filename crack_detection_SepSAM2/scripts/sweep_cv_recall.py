"""sweep_cv_recall.py — Stage 1:在 4-fold CV val 上掃 conf/iou/max_det,
量化免費槓桿對 box/mask P/R/F1 的影響,輸出 csv + markdown 表。

用法:
  python scripts/sweep_cv_recall.py --folds 0 1 2 3
"""
import argparse, csv, itertools, os


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--conf", type=float, nargs="+", default=[0.25, 0.10, 0.05, 0.02])
    ap.add_argument("--iou", type=float, nargs="+", default=[0.7, 0.5, 0.4])
    ap.add_argument("--max_det", type=int, nargs="+", default=[300, 1000])
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--run_root", default="runs/segment/runs")
    ap.add_argument("--ckpt_prefix", default="sepsam_agent_heritage_cv",
                    help="run 目錄前綴;CLAHE 版傳 sepsam_agent_heritage_cv_clahe,過採樣版傳 sepsam_agent_heritage_cv_os")
    ap.add_argument("--out_csv", default="runs/cv_recall_sweep.csv")
    ap.add_argument("--out_md", default="runs/cv_recall_sweep.md")
    args = ap.parse_args()

    from ultralytics.nn.tasks import C2f_SEA  # noqa: F401  (註冊)
    from ultralytics import YOLO

    rows = []
    for conf, iou, md in itertools.product(args.conf, args.iou, args.max_det):
        per_fold = []
        for k in args.folds:
            w = f"{args.run_root}/{args.ckpt_prefix}_fold{k}/weights/best.pt"
            data = f"configs/data_heritage_cv_fold{k}.yaml"
            m = YOLO(w)
            r = m.val(data=data, conf=conf, iou=iou, max_det=md, imgsz=args.imgsz,
                      batch=16, device="0", plots=False, verbose=False, split="val")
            per_fold.append({
                "fold": k,
                "boxP": float(r.box.mp), "boxR": float(r.box.mr),
                "maskP": float(r.seg.mp), "maskR": float(r.seg.mr),
            })
        for pf in per_fold:
            pf["boxF1"] = f1(pf["boxP"], pf["boxR"])
            pf["maskF1"] = f1(pf["maskP"], pf["maskR"])
        def mean(key, folds=None):
            sel = [p for p in per_fold if folds is None or p["fold"] in folds]
            return sum(p[key] for p in sel) / max(len(sel), 1)
        no2 = [k for k in args.folds if k != 2]
        rows.append({
            "conf": conf, "iou": iou, "max_det": md,
            "maskR_4f": mean("maskR"), "maskF1_4f": mean("maskF1"),
            "maskP_4f": mean("maskP"),
            "maskR_no2": mean("maskR", no2), "maskF1_no2": mean("maskF1", no2),
            "boxR_4f": mean("boxR"), "boxF1_4f": mean("boxF1"),
        })
        print(f"conf={conf} iou={iou} max_det={md} -> "
              f"maskR_4f={rows[-1]['maskR_4f']:.3f} maskF1_4f={rows[-1]['maskF1_4f']:.3f} "
              f"maskF1_no2={rows[-1]['maskF1_no2']:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader(); wri.writerows(rows)
    with open(args.out_md, "w") as f:
        f.write("| conf | iou | max_det | maskR_4f | maskF1_4f | maskP_4f | maskR_no2 | maskF1_no2 |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['conf']} | {r['iou']} | {r['max_det']} | {r['maskR_4f']:.3f} | "
                    f"{r['maskF1_4f']:.3f} | {r['maskP_4f']:.3f} | {r['maskR_no2']:.3f} | "
                    f"{r['maskF1_no2']:.3f} |\n")
    print(f"\nwritten {args.out_csv} / {args.out_md}  ({len(rows)} combos)")


if __name__ == "__main__":
    main()
