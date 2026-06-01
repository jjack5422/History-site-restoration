"""
finetune_heritage.py — 從現有 SEA Agent ckpt 在 1-31test 歷史劣化資料上 fine-tune。
augmentation 對齊 sam2 augment.py：HFlip/VFlip + Affine(±15°/±5%/±10%)
+ HSV + mosaic/mixup 強化（YOLO 內建）。

範例：
    python scripts/finetune_heritage.py --epochs 100 --imgsz 1024 --batch 2
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_ckpt",
                    default="runs/segment/runs/sepsam_agent_v8n_200ep/weights/best.pt")
    ap.add_argument("--data", default="configs/data_heritage_ft.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="sepsam_agent_heritage_ft")
    ap.add_argument("--patience", type=int, default=30)
    args = ap.parse_args()

    try:
        from ultralytics.nn.tasks import C2f_SEA  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"C2f_SEA 未註冊 — 請先執行 `python sea_setup.py`。({e})")

    from ultralytics import YOLO

    model = YOLO(args.init_ckpt)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs",
        name=args.name,
        patience=args.patience,
        # 對齊 sam2 augment.py
        fliplr=0.5,         # HorizontalFlip 0.5
        flipud=0.5,         # VerticalFlip 0.5（ultralytics 預設 0.0，需顯式打開）
        degrees=15.0,       # Affine rotate ±15°
        translate=0.05,     # Affine translate ±5%
        scale=0.10,         # Affine scale ±10%
        shear=0.0,
        perspective=0.0,
        hsv_h=0.015,        # HueSatVal hue ±15 ≈ 0.04 of 360°；保守
        hsv_s=0.25,         # sat ±25%
        hsv_v=0.25,         # val ±25%（≈ BrightnessContrast）
        # YOLO 強化（小資料集關鍵）
        mosaic=1.0,
        close_mosaic=15,    # 最後 15 epoch 關閉 mosaic 收尾
        mixup=0.10,
        copy_paste=0.10,
        erasing=0.30,       # 近似 GaussNoise 對 patch 的擾動
        # 訓練設定
        optimizer="AdamW",
        lr0=0.001,          # fine-tune 用較小 lr
        lrf=0.01,
        weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=2.0,
        single_cls=True,
        verbose=True,
    )
    print(f"\n完成。權重在 runs/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
