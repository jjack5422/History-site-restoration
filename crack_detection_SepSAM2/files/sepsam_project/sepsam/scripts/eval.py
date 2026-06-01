"""
eval.py — 在 mask 格式的評估集上跑 CMC 並計算 P/R/F1/IoU。
影像與 GT mask 以「檔名主檔名（stem）」配對。

範例：
    python scripts/eval.py --images datasets/cfd/images --masks datasets/cfd/masks
"""
import argparse
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
        ext = os.path.splitext(p)[1].lower()
        if ext in IMG_EXTS:
            out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cmc.yaml")
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--limit", type=int, default=0, help=">0 時只評估前 N 張")
    args = ap.parse_args()

    hp = SimpleNamespace(**yaml.safe_load(open(args.config, encoding="utf-8")))

    try:
        from ultralytics.nn.tasks import C2f_SEA  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"C2f_SEA 未註冊 — 請先執行 `python sea_setup.py`。({e})")

    agent = Agent(hp.agent_ckpt, device=hp.device)
    predictor, prompt_fn = build_large_model(hp)

    imgs = index_by_stem(args.images)
    gts = index_by_stem(args.masks)
    stems = sorted(set(imgs) & set(gts))
    if args.limit > 0:
        stems = stems[: args.limit]
    if not stems:
        raise SystemExit("找不到可配對的 (image, mask)；請確認檔名主檔名一致。")

    records = []
    for stem in stems:
        bgr = cv2.imread(imgs[stem])
        gt = cv2.imread(gts[stem], cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred = cmc_predict(rgb, agent, predictor, prompt_fn, hp)
        records.append(prf_iou(pred, gt > 0))

    res = aggregate(records)
    print(f"\nEvaluated {res['n']} images ({hp.sam_backend} backend)")
    print(f"Precision: {res['P']:.4f}")
    print(f"Recall:    {res['R']:.4f}")
    print(f"F1:        {res['F1']:.4f}")
    print(f"IoU:       {res['IoU']:.4f}")
    print("\n注意：論文指出 GT 人工標註偏差會低估指標；建議搭配 infer.py --dump-steps 的質化圖判讀。")


if __name__ == "__main__":
    main()
