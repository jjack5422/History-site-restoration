"""
m7_ablation_cmc.py — M7 有/無 CMC 對照（Fig. 15 趨勢）。
對同一張影像同時記錄三種預測的 P/R/F1/IoU：
  - YOLO-only：Agent 草稿（CMC 1st round 輸出）
  - SAM-only：完全採用 contour-filtered SAM mask（CMC 2/3 round 輸出，不做衝突分析）
  - CMC：完整四輪流程（最終決策）

範例：
    python scripts/m7_ablation_cmc.py \
        --images datasets/crack_seg/test/images \
        --masks  datasets/crack_seg/test/masks
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
from src.cmc import conflict_ratio  # noqa: E402
from src.filters import contour_filter  # noqa: E402
from src.geometry import mask_to_points_and_width  # noqa: E402
from src.large_model import build_large_model  # noqa: E402
from src.metrics import aggregate, prf_iou  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def index_by_stem(folder):
    out = {}
    for p in glob.glob(os.path.join(folder, "*")):
        if os.path.splitext(p)[1].lower() in IMG_EXTS:
            out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cmc.yaml")
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    hp = SimpleNamespace(**yaml.safe_load(open(args.config, encoding="utf-8")))
    from ultralytics.nn.tasks import C2f_SEA  # noqa: F401

    agent = Agent(hp.agent_ckpt, device=hp.device)
    predictor, prompt_fn = build_large_model(hp)

    imgs = index_by_stem(args.images)
    gts = index_by_stem(args.masks)
    stems = sorted(set(imgs) & set(gts))
    if args.limit > 0:
        stems = stems[: args.limit]

    rec_yolo, rec_sam, rec_cmc = [], [], []
    n_sam_picked = 0
    for stem in stems:
        bgr = cv2.imread(imgs[stem])
        gt = cv2.imread(gts[stem], cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gtb = gt > 0

        mask_yolo, yolo_conf = agent.predict(rgb, conf=hp.YOLO_CONF_1)
        n_pts = max(rgb.shape[:2]) // hp.POINTS_DIVISOR
        pts, _ = mask_to_points_and_width(mask_yolo > 0, n_pts)
        m_raw, score = prompt_fn(predictor, rgb, pts)
        m_sam = contour_filter(m_raw, yolo_conf, hp.YOLO_CONF_2)
        cr = conflict_ratio(m_sam, mask_yolo)
        accept = (cr < hp.CONFLICTION_RATIO) and (score > hp.SAM_THRESH)
        m_cmc = m_sam if accept else mask_yolo
        if accept:
            n_sam_picked += 1

        rec_yolo.append(prf_iou(mask_yolo, gtb))
        rec_sam.append(prf_iou(m_sam, gtb))
        rec_cmc.append(prf_iou(m_cmc, gtb))

    def line(name, rec):
        r = aggregate(rec)
        return (f"{name:10s}  n={r['n']:4d}  P={r['P']:.4f}  R={r['R']:.4f}  "
                f"F1={r['F1']:.4f}  IoU={r['IoU']:.4f}")

    print(f"\nDataset: {args.images}")
    print(f"sam_backend={hp.sam_backend}  SAM_THRESH={hp.SAM_THRESH}  "
          f"CONFLICTION_RATIO={hp.CONFLICTION_RATIO}  YOLO_CONF_1={hp.YOLO_CONF_1}")
    print("-" * 70)
    print(line("YOLO-only", rec_yolo))
    print(line("SAM-only ", rec_sam))
    print(line("CMC      ", rec_cmc))
    n_total = len(rec_cmc)
    print(f"\nCMC 採用 SAM 的比例：{n_sam_picked}/{n_total} = "
          f"{100.0*n_sam_picked/max(n_total,1):.1f}%")


if __name__ == "__main__":
    main()
