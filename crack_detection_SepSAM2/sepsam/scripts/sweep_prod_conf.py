"""sweep_prod_conf.py — 在生產 agent 上對 heritage 全圖評估集掃 conf×iou(YOLO-only),
找最佳 conf/iou(heritage domain CMC≡YOLO-only)。

用法:
  python scripts/sweep_prod_conf.py \
      --ckpt runs/segment/runs/sepsam_agent_heritage_ft-2/weights/best.pt \
      --images datasets/heritage_1_31test/images --masks datasets/heritage_1_31test/masks
"""
import argparse, glob, itertools, os, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import Agent
from src.metrics import prf_iou, aggregate

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def index_by_stem(folder):
    out = {}
    for p in glob.glob(os.path.join(folder, "*")):
        if os.path.splitext(p)[1].lower() in IMG_EXTS:
            out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--conf", type=float, nargs="+", default=[0.05, 0.08, 0.10, 0.12, 0.15])
    ap.add_argument("--iou", type=float, nargs="+", default=[0.7, 0.5])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from ultralytics.nn.tasks import C2f_SEA  # noqa: F401
    agent = Agent(args.ckpt, device=args.device)
    imgs = index_by_stem(args.images); gts = index_by_stem(args.masks)
    stems = sorted(set(imgs) & set(gts))
    print(f"n={len(stems)} images")
    print(f"{'conf':>6} {'iou':>5} {'P':>7} {'R':>7} {'F1':>7} {'IoU':>7}")
    best = None
    for conf, iou in itertools.product(args.conf, args.iou):
        recs = []
        for st in stems:
            bgr = cv2.imread(imgs[st]); gt = cv2.imread(gts[st], cv2.IMREAD_GRAYSCALE)
            if bgr is None or gt is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask, _ = agent.predict(rgb, conf=conf, iou=iou)
            recs.append(prf_iou(mask, gt > 0))
        a = aggregate(recs)
        print(f"{conf:>6} {iou:>5} {a['P']:>7.4f} {a['R']:>7.4f} {a['F1']:>7.4f} {a['IoU']:>7.4f}", flush=True)
        if best is None or a["F1"] > best[2]["F1"]:
            best = (conf, iou, a)
    print(f"\nBEST F1: conf={best[0]} iou={best[1]} -> "
          f"P={best[2]['P']:.4f} R={best[2]['R']:.4f} F1={best[2]['F1']:.4f} IoU={best[2]['IoU']:.4f}")


if __name__ == "__main__":
    main()
