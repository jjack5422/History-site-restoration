"""eval_craq_cv.py — 逐 fold 用該 fold 的 craquelure agent + cmc_craq.yaml 在 fold val 上跑 CMC,
輸出 YOLO-only / SAM-only / CMC 的 P/R/F1/IoU,並與 dense-seg expert 對照。"""
import argparse, glob, os, sys
import cv2, numpy as np, yaml
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import Agent
from src.cmc import conflict_ratio
from src.filters import contour_filter
from src.geometry import mask_to_points_and_width
from src.large_model import build_large_model
from src.metrics import prf_iou, aggregate

TILES = "/home/zzz90/research/crack_detection_SepSAM2/sepsam/datasets/craq_cv"
# dense-seg expert(+CLAHE)per-fold craqF1 對照
DENSE = {0: 0.634, 1: 0.614, 2: 0.040, 3: 0.655}


def eval_fold(k, hp_dict):
    hp = SimpleNamespace(**hp_dict)
    hp.agent_ckpt = f"runs/segment/runs/sepsam_agent_craq_cv_fold{k}/weights/best.pt"
    from ultralytics.nn.tasks import C2f_SEA  # noqa: F401
    agent = Agent(hp.agent_ckpt, device=hp.device)
    predictor, prompt_fn = build_large_model(hp)
    vi = os.path.join(TILES, f"craqfold{k}", "val_images")
    vm = os.path.join(TILES, f"craqfold{k}", "val_masks")
    stems = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(vi, "*.png")))
    ry, rs, rc = [], [], []
    n_sam = 0
    for st in stems:
        bgr = cv2.imread(os.path.join(vi, st + ".png"))
        gt = cv2.imread(os.path.join(vm, st + ".png"), cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); gtb = gt > 0
        mask_y, yc = agent.predict(rgb, conf=hp.YOLO_CONF_1)
        n_pts = max(rgb.shape[:2]) // hp.POINTS_DIVISOR
        pts, _ = mask_to_points_and_width(mask_y > 0, n_pts)
        m_raw, score = prompt_fn(predictor, rgb, pts)
        m_s = contour_filter(m_raw, yc, hp.YOLO_CONF_2)
        cr = conflict_ratio(m_s, mask_y)
        accept = (cr < hp.CONFLICTION_RATIO) and (score > hp.SAM_THRESH)
        m_c = m_s if accept else mask_y
        n_sam += int(accept)
        ry.append(prf_iou(mask_y, gtb)); rs.append(prf_iou(m_s, gtb)); rc.append(prf_iou(m_c, gtb))
    return aggregate(ry), aggregate(rs), aggregate(rc), n_sam, len(rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cmc_craq.yaml")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()
    hp_dict = yaml.safe_load(open(args.config, encoding="utf-8"))
    rows = []
    for k in args.folds:
        ry, rs, rc, n_sam, n = eval_fold(k, hp_dict)
        rows.append((k, ry, rs, rc, n_sam, n))
        print(f"fold{k} n={n} SAM採用={n_sam}/{n} | "
              f"YOLO F1={ry['F1']:.3f} | SAM F1={rs['F1']:.3f} | CMC F1={rc['F1']:.3f} "
              f"| dense-seg={DENSE.get(k,'?')}", flush=True)
    def mean(key, idx, folds=None):
        sel = [r for r in rows if folds is None or r[0] in folds]
        return sum(r[idx][key] for r in sel) / max(len(sel), 1)
    no2 = [k for k in args.folds if k != 2]
    print("\n==== 4-fold 平均 ====")
    print(f"YOLO-only F1={mean('F1',1):.3f}  SAM-only F1={mean('F1',2):.3f}  CMC F1={mean('F1',3):.3f}")
    print(f"排除fold2: YOLO F1={mean('F1',1,no2):.3f}  CMC F1={mean('F1',3,no2):.3f}  (dense-seg≈0.634/0.614/0.655)")


if __name__ == "__main__":
    main()
