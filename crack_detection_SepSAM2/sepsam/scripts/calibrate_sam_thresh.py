"""
calibrate_sam_thresh.py — 在驗證集上掃描 SAM_THRESH（與可選的 CONFLICTION_RATIO），
挑選使平均 F1 最佳者。換成 SAM2 後務必執行（附錄 D.5）。

範例：
    python scripts/calibrate_sam_thresh.py --images datasets/cfd/images --masks datasets/cfd/masks --limit 80
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
from src.metrics import prf_iou  # noqa: E402

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
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--sweep-conflict", action="store_true",
                    help="同時掃描 CONFLICTION_RATIO")
    ap.add_argument("--sweep-conf2", action="store_true",
                    help="同時掃描 YOLO_CONF_2（contour_filter 的高信心門檻）")
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
    stems = sorted(set(imgs) & set(gts))[: args.limit]
    if not stems:
        raise SystemExit("找不到可配對的 (image, mask)。")

    # 1) 先一次跑完，快取每張的 (sam_score, m_raw, yolo_conf, mask_yolo, gt)。
    #    注意：快取「未過濾」的 SAM raw mask 與 YOLO 信心，contour_filter（受 YOLO_CONF_2 影響）
    #    與 conflict_ratio 留到掃描迴圈內重算，才能聯掃 YOLO_CONF_2。
    cache = []
    for stem in stems:
        bgr = cv2.imread(imgs[stem])
        gt = cv2.imread(gts[stem], cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask_yolo, yolo_conf = agent.predict(rgb, conf=hp.YOLO_CONF_1)
        n = max(rgb.shape[:2]) // hp.POINTS_DIVISOR
        pts, _ = mask_to_points_and_width(mask_yolo > 0, n)
        m_raw, score = prompt_fn(predictor, rgb, pts)
        cache.append((score, m_raw, yolo_conf, mask_yolo, gt > 0))
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    print(f"cached {len(cache)} samples; sam_score range "
          f"[{min(c[0] for c in cache):.3f}, {max(c[0] for c in cache):.3f}]")

    thr_grid = np.round(np.arange(0.60, 0.96, 0.05), 2)
    conf_grid = ([round(float(x), 2) for x in np.arange(1.0, 2.6, 0.25)]
                 if args.sweep_conflict else [hp.CONFLICTION_RATIO])
    conf2_grid = ([0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
                  if args.sweep_conf2 else [hp.YOLO_CONF_2])

    # 對每個 YOLO_CONF_2 候選預先算好 (m_sam, conflict)，避免在最內層重複 findContours。
    best = None
    for c2 in conf2_grid:
        per_img = []  # [(score, conflict, m_sam, m_yolo, gt), ...]
        for score, m_raw, yolo_conf, m_yolo, gt in cache:
            m_sam = contour_filter(m_raw, yolo_conf, c2)
            per_img.append((score, conflict_ratio(m_sam, m_yolo), m_sam, m_yolo, gt))
        for cr in conf_grid:
            for thr in thr_grid:
                f1s = []
                for score, conf, m_sam, m_yolo, gt in per_img:
                    chosen = m_sam if (conf < cr and score > thr) else m_yolo
                    f1s.append(prf_iou(chosen, gt)[2])
                mean_f1 = float(np.mean(f1s)) if f1s else 0.0
                if best is None or mean_f1 > best[3]:
                    best = (float(thr), float(cr), float(c2), mean_f1)

    print(f"\nBEST → SAM_THRESH={best[0]:.2f}  CONFLICTION_RATIO={best[1]:.2f}  "
          f"YOLO_CONF_2={best[2]:.2f}  mean F1={best[3]:.4f}")
    print("把上述值（依需要）寫回對應的 configs/*.yaml。")


if __name__ == "__main__":
    main()
