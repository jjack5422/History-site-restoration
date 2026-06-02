"""
m7_ablation_sea.py — 有/無 SEA 對照（論文 Fig. 13）。
比較兩個 agent ckpt 在：
  (1) val 集 mAP / loss 曲線（畫圖、列峰值/末值）
  (2) 下游 CMC 完整流程在 test 集上的 P/R/F1/IoU

範例：
    python scripts/m7_ablation_sea.py \
        --sea_run runs/segment/runs/sepsam_agent_v8n_200ep \
        --base_run runs/segment/runs/sepsam_agent_v8n_noSEA_200ep \
        --images datasets/crack_seg/test/images \
        --masks  datasets/crack_seg/test/masks
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace  # noqa: E402

from src.agent import Agent  # noqa: E402
from src.cmc import cmc_predict  # noqa: E402
from src.large_model import build_large_model  # noqa: E402
from src.metrics import aggregate, prf_iou  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def index_by_stem(folder):
    out = {}
    for p in glob.glob(os.path.join(folder, "*")):
        if os.path.splitext(p)[1].lower() in IMG_EXTS:
            out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


def read_curves(run_dir):
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        return None
    rows = []
    with open(csv_path, "r") as f:
        for r in csv.DictReader(f):
            rows.append({k.strip(): v for k, v in r.items()})
    return rows


def col(rows, key):
    return np.asarray([float(r[key]) for r in rows if key in r and r[key] != ""], dtype=float)


def summarize_curves(label, rows):
    if not rows:
        print(f"{label}: results.csv missing")
        return
    map50 = col(rows, "metrics/mAP50(M)")
    map5095 = col(rows, "metrics/mAP50-95(M)")
    val_box = col(rows, "val/box_loss")
    val_seg = col(rows, "val/seg_loss")
    train_seg = col(rows, "train/seg_loss")
    print(f"\n[{label}]  epochs={len(rows)}")
    print(f"  mAP50(M)  peak={map50.max():.4f} @ep{int(map50.argmax())+1}   "
          f"final={map50[-1]:.4f}")
    print(f"  mAP50-95(M) peak={map5095.max():.4f} @ep{int(map5095.argmax())+1}   "
          f"final={map5095[-1]:.4f}")
    print(f"  val/seg_loss min={val_seg.min():.4f} @ep{int(val_seg.argmin())+1}   "
          f"final={val_seg[-1]:.4f}   last-min gap={val_seg[-1]-val_seg.min():.4f}")
    print(f"  train/seg_loss final={train_seg[-1]:.4f}   "
          f"val-train gap (final)={val_seg[-1]-train_seg[-1]:.4f}")


def eval_cmc(ckpt, hp_dict, images_dir, masks_dir, limit=0):
    hp = SimpleNamespace(**hp_dict)
    hp.agent_ckpt = ckpt
    from ultralytics.nn.tasks import C2f_SEA  # noqa: F401
    agent = Agent(hp.agent_ckpt, device=hp.device)
    predictor, prompt_fn = build_large_model(hp)
    imgs = index_by_stem(images_dir); gts = index_by_stem(masks_dir)
    stems = sorted(set(imgs) & set(gts))
    if limit > 0: stems = stems[:limit]
    records = []
    for stem in stems:
        bgr = cv2.imread(imgs[stem]); gt = cv2.imread(gts[stem], cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt is None: continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred = cmc_predict(rgb, agent, predictor, prompt_fn, hp)
        records.append(prf_iou(pred, gt > 0))
    return aggregate(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cmc.yaml")
    ap.add_argument("--sea_run",  required=True)
    ap.add_argument("--base_run", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks",  required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print("=== val curve summary ===")
    summarize_curves("SEA",     read_curves(args.sea_run))
    summarize_curves("no-SEA",  read_curves(args.base_run))

    hp_dict = yaml.safe_load(open(args.config, encoding="utf-8"))
    sea_ckpt  = os.path.join(args.sea_run,  "weights", "best.pt")
    base_ckpt = os.path.join(args.base_run, "weights", "best.pt")

    if os.path.exists(sea_ckpt) and os.path.exists(base_ckpt):
        print("\n=== downstream CMC eval ===")
        sea  = eval_cmc(sea_ckpt,  hp_dict, args.images, args.masks, args.limit)
        base = eval_cmc(base_ckpt, hp_dict, args.images, args.masks, args.limit)
        print(f"SEA     n={sea['n']:4d}  P={sea['P']:.4f}  R={sea['R']:.4f}  "
              f"F1={sea['F1']:.4f}  IoU={sea['IoU']:.4f}")
        print(f"no-SEA  n={base['n']:4d}  P={base['P']:.4f}  R={base['R']:.4f}  "
              f"F1={base['F1']:.4f}  IoU={base['IoU']:.4f}")
        print(f"ΔF1 (SEA - no-SEA) = {sea['F1']-base['F1']:+.4f}   "
              f"ΔIoU = {sea['IoU']-base['IoU']:+.4f}")
    else:
        print(f"\nskip downstream eval (missing ckpt): "
              f"sea={os.path.exists(sea_ckpt)} base={os.path.exists(base_ckpt)}")


if __name__ == "__main__":
    main()
