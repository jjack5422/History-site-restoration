"""
train_agent.py — 訓練 SepSAM 的 Agent（YOLOv8-Seg + SEA）。SAM 不在此訓練。
前置：先執行 `python sea_setup.py`（讓 ultralytics 認得 C2f_SEA）。

範例：
    python scripts/train_agent.py --epochs 500 --batch 16 --imgsz 416 --device 0
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="configs/yolov8n-seg-sea.yaml")
    ap.add_argument("--data", default="configs/data_crackseg.yaml")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="sepsam_agent")
    args = ap.parse_args()

    # 確認 SEA 已註冊
    try:
        from ultralytics.nn.tasks import C2f_SEA  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"C2f_SEA 未註冊 — 請先執行 `python sea_setup.py`。({e})")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs",
        name=args.name,
    )
    print(f"\n完成。權重在 runs/{args.name}/weights/best.pt")
    print("把該路徑填回 configs/cmc.yaml 的 agent_ckpt。")


if __name__ == "__main__":
    main()
