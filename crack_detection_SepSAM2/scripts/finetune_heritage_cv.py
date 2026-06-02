"""
finetune_heritage_cv.py — 在 build_heritage_cv.py 產生的 4-fold panel-grouped tile 資料上，
依序 fine-tune SEA Agent（每 fold 一個 run）。imgsz=512（tile 尺度，對齊切片），
augmentation 對齊 finetune_heritage.py / sam2 augment.py。

範例：
    python scripts/finetune_heritage_cv.py --epochs 100 --imgsz 512 --batch 16
    python scripts/finetune_heritage_cv.py --folds 0 2   # 只跑指定 fold
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_ckpt",
                    default="runs/segment/runs/sepsam_agent_v8n_200ep/weights/best.pt")
    ap.add_argument("--configs_dir", default="configs")
    ap.add_argument("--data_prefix", default="data_heritage_cv",
                    help="data yaml 檔名前綴;craquelure 用 data_craq_cv")
    ap.add_argument("--n_splits", type=int, default=4)
    ap.add_argument("--folds", type=int, nargs="+", default=None,
                    help="只跑指定 fold（預設全跑）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name_prefix", default="sepsam_agent_heritage_cv")
    ap.add_argument("--patience", type=int, default=30)
    args = ap.parse_args()

    try:
        from ultralytics.nn.tasks import C2f_SEA  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"C2f_SEA 未註冊 — 請先執行 `python sea_setup.py`。({e})")

    from ultralytics import YOLO

    folds = args.folds if args.folds is not None else list(range(args.n_splits))
    summary = []
    for k in folds:
        data_yaml = os.path.join(args.configs_dir, f"{args.data_prefix}_fold{k}.yaml")
        if not os.path.isfile(data_yaml):
            raise SystemExit(f"找不到 {data_yaml} — 請先跑 build_heritage_cv.py")
        run_name = f"{args.name_prefix}_fold{k}"
        print(f"\n========== fold {k}  data={data_yaml}  name={run_name} ==========")

        model = YOLO(args.init_ckpt)
        model.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project="runs",
            name=run_name,
            patience=args.patience,
            # 對齊 sam2 augment.py / finetune_heritage.py
            fliplr=0.5,
            flipud=0.5,
            degrees=15.0,
            translate=0.05,
            scale=0.10,
            shear=0.0,
            perspective=0.0,
            hsv_h=0.015,
            hsv_s=0.25,
            hsv_v=0.25,
            mosaic=1.0,
            close_mosaic=15,
            mixup=0.10,
            copy_paste=0.10,
            erasing=0.30,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,
            cos_lr=True,
            warmup_epochs=2.0,
            single_cls=True,
            verbose=True,
        )
        # 收 best 的 val 指標
        m = {}
        try:
            res = model.metrics
            if res is not None:
                m = {
                    "mAP50(M)": float(getattr(res.seg, "map50", float("nan"))),
                    "mAP50-95(M)": float(getattr(res.seg, "map", float("nan"))),
                    "mAP50(B)": float(getattr(res.box, "map50", float("nan"))),
                }
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 無法取得 metrics: {e}")
        summary.append({"fold": k, "run": f"runs/{run_name}", "metrics": m})
        print(f"fold {k} 完成 → runs/{run_name}/weights/best.pt  metrics={m}")

    print("\n========== 4-fold 摘要 ==========")
    for s in summary:
        print(f"fold {s['fold']}: {s['metrics']}")
    # mean
    keys = set().union(*[s["metrics"].keys() for s in summary]) if summary else set()
    for key in sorted(keys):
        vals = [s["metrics"][key] for s in summary if key in s["metrics"]]
        vals = [v for v in vals if v == v]  # 過濾 nan
        if vals:
            print(f"mean {key} = {sum(vals) / len(vals):.4f}  (n={len(vals)})")

    with open("runs/heritage_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("摘要 → runs/heritage_cv_summary.json")


if __name__ == "__main__":
    main()
