"""
eval_heritage_before_after.py — 比較 fine-tune 前/後 SEA Agent
在 heritage 1-31test 全集（32 張）與 val piece（8 張）上的：
  - YOLO-only / SAM-only / CMC 的 P/R/F1/IoU
  - CMC 採用 SAM 的比例
配對輸出，方便看 domain adaptation 提升量。
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


def run_on(ckpt, hp_dict, images_dir, masks_dir):
    hp = SimpleNamespace(**hp_dict)
    hp.agent_ckpt = ckpt
    from ultralytics.nn.tasks import C2f_SEA  # noqa: F401
    agent = Agent(hp.agent_ckpt, device=hp.device)
    predictor, prompt_fn = build_large_model(hp)
    imgs = index_by_stem(images_dir); gts = index_by_stem(masks_dir)
    stems = sorted(set(imgs) & set(gts))
    rec_y, rec_s, rec_c = [], [], []
    n_sam = 0
    for stem in stems:
        bgr = cv2.imread(imgs[stem]); gt = cv2.imread(gts[stem], cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt is None: continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); gtb = gt > 0
        mask_y, yc = agent.predict(rgb, conf=hp.YOLO_CONF_1, iou=getattr(hp, "YOLO_IOU", 0.7))
        n_pts = max(rgb.shape[:2]) // hp.POINTS_DIVISOR
        pts, _ = mask_to_points_and_width(mask_y > 0, n_pts)
        m_raw, score = prompt_fn(predictor, rgb, pts)
        m_s = contour_filter(m_raw, yc, hp.YOLO_CONF_2)
        cr = conflict_ratio(m_s, mask_y)
        accept = (cr < hp.CONFLICTION_RATIO) and (score > hp.SAM_THRESH)
        m_c = m_s if accept else mask_y
        if accept: n_sam += 1
        rec_y.append(prf_iou(mask_y, gtb))
        rec_s.append(prf_iou(m_s, gtb))
        rec_c.append(prf_iou(m_c, gtb))
    return aggregate(rec_y), aggregate(rec_s), aggregate(rec_c), n_sam, len(rec_c)


def fmt(name, r):
    return (f"  {name:10s} P={r['P']:.4f} R={r['R']:.4f} "
            f"F1={r['F1']:.4f} IoU={r['IoU']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cmc.yaml")
    ap.add_argument("--ckpt_before", required=True)
    ap.add_argument("--ckpt_after",  required=True)
    ap.add_argument("--sets", nargs="+", required=True,
                    help="形式 name:images_dir:masks_dir，可多個。")
    args = ap.parse_args()

    hp_dict = yaml.safe_load(open(args.config, encoding="utf-8"))
    print(f"sam_backend={hp_dict['sam_backend']}  "
          f"SAM_THRESH={hp_dict['SAM_THRESH']}  "
          f"CONFLICTION_RATIO={hp_dict['CONFLICTION_RATIO']}  "
          f"YOLO_CONF_1={hp_dict['YOLO_CONF_1']}")

    for ent in args.sets:
        name, im, mk = ent.split(":")
        print(f"\n=== {name} ===")
        for label, ck in (("BEFORE (Roboflow)", args.ckpt_before),
                          ("AFTER  (heritage ft)", args.ckpt_after)):
            ry, rs, rc, n_sam, n_tot = run_on(ck, hp_dict, im, mk)
            print(f"[{label}]  n={n_tot}  CMC 採用 SAM={n_sam}/{n_tot}")
            print(fmt("YOLO-only", ry))
            print(fmt("SAM-only ", rs))
            print(fmt("CMC      ", rc))


if __name__ == "__main__":
    main()
