"""eval_sepsam_crack_cv.py — SepSAM 完整 CMC 在 crack 4-fold(heritage_ft_cv)評估。
逐 fold 用該 fold agent,對 val tile 跑 cmc_predict(YOLO草稿→中軸點→SAM2→過濾→決策),
報 YOLO-only / SAM-only / CMC 的 P/R/F1/IoU + SAM 採用率,並存每張 val tile 的 TP/FN/FP 疊圖。
GT 由 YOLO-seg polygon label 柵格化(heritage_ft_cv 無 binary mask)。
用法:SepSAM2_env / cwd=crack_detection_SepSAM2/sepsam
  python scripts/eval_sepsam_crack_cv.py --folds 0 1 2 3 --conf 0.10 --iou 0.5
"""
import argparse, os, sys
import cv2, numpy as np, yaml
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import Agent              # noqa: E402
from src.cmc import cmc_predict          # noqa: E402
from src.large_model import build_large_model  # noqa: E402
from src.metrics import prf_iou, aggregate     # noqa: E402

DATA = "datasets/heritage_ft_cv"


def rasterize(label_path, h, w):
    m = np.zeros((h, w), np.uint8)
    if not os.path.isfile(label_path):
        return m
    for ln in open(label_path):
        v = ln.split()
        if len(v) < 7:
            continue
        pts = np.array(v[1:], float).reshape(-1, 2) * [w, h]
        cv2.fillPoly(m, [pts.astype(np.int32)], 1)
    return m


def overlay(bgr, pred, gt):
    o = bgr.copy()
    P = np.asarray(pred) > 0
    G = np.asarray(gt) > 0
    o[P & G] = (255, 150, 0)    # TP 藍
    o[(~P) & G] = (0, 255, 0)   # FN 綠(漏抓)
    o[P & (~G)] = (0, 0, 255)   # FP 紅(誤報)
    return cv2.addWeighted(bgr, 0.5, o, 0.5, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cmc_heritage.yaml")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--save_dir", default="outputs/sepsam_crack_cv")
    args = ap.parse_args()

    from ultralytics.nn.tasks import C2f_SEA  # noqa: F401
    hp_d = yaml.safe_load(open(args.config, encoding="utf-8"))
    hp_d["YOLO_CONF_1"] = args.conf
    hp_d["YOLO_IOU"] = args.iou

    rows = []
    for k in args.folds:
        hp = SimpleNamespace(**hp_d)
        hp.agent_ckpt = f"runs/segment/runs/sepsam_agent_heritage_cv_fold{k}/weights/best.pt"
        agent = Agent(hp.agent_ckpt, device=hp.device)
        predictor, prompt_fn = build_large_model(hp)
        imgs = [l.strip() for l in open(f"{DATA}/fold{k}/val.txt") if l.strip()]
        od = f"{args.save_dir}/fold{k}"
        os.makedirs(od, exist_ok=True)
        ry, rs, rc = [], [], []
        n_sam = 0
        for ip in imgs:
            st = os.path.splitext(os.path.basename(ip))[0]
            bgr = cv2.imread(ip)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = bgr.shape[:2]
            gt = rasterize(f"{DATA}/labels/{st}.txt", h, w)
            final, info = cmc_predict(rgb, agent, predictor, prompt_fn, hp, return_intermediates=True)
            ry.append(prf_iou(info["draft"], gt > 0))
            rs.append(prf_iou(info["sam_filtered"], gt > 0))
            rc.append(prf_iou(final, gt > 0))
            n_sam += int(info["decision"] == "sam")
            cv2.imwrite(f"{od}/{st}.png", overlay(bgr, final, gt))
        ay, as_, ac = aggregate(ry), aggregate(rs), aggregate(rc)
        rows.append((k, ay, as_, ac, n_sam, len(imgs)))
        print(f"fold{k} n={len(imgs)} SAM採用={n_sam}/{len(imgs)} | "
              f"YOLO F1={ay['F1']:.3f} R={ay['R']:.3f} | SAM F1={as_['F1']:.3f} | "
              f"CMC F1={ac['F1']:.3f} IoU={ac['IoU']:.3f} | imgs->{od}", flush=True)

    no2 = [r for r in rows if r[0] != 2]
    def mF1(idx, rr): return sum(r[idx]["F1"] for r in rr) / max(len(rr), 1)
    print("\n==== 平均 ====")
    print(f"4-fold     : YOLO {mF1(1,rows):.3f} | SAM {mF1(2,rows):.3f} | CMC {mF1(3,rows):.3f}")
    print(f"排除fold2  : YOLO {mF1(1,no2):.3f} | SAM {mF1(2,no2):.3f} | CMC {mF1(3,no2):.3f}")


if __name__ == "__main__":
    main()
