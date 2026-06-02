"""dump_preds_sepsam_yolo.py — craq YOLO agent(conf0.05/iou0.5)union mask,
存到 crack_detection_sam2 的 preds/sepsam_yolo/fold{k}/{stem}.png。"""
import argparse, glob, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import Agent

TILES = "/home/zzz90/research/crack_detection_SepSAM2/sepsam/datasets/craq_cv"
PREDS = "/home/zzz90/research/crack_detection_sam2/preds/sepsam_yolo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--conf", type=float, default=0.05)
    args = ap.parse_args()
    from ultralytics.nn.tasks import C2f_SEA  # noqa
    for k in args.folds:
        agent = Agent(f"runs/segment/runs/sepsam_agent_craq_cv_fold{k}/weights/best.pt", device="cuda")
        vi = os.path.join(TILES, f"craqfold{k}", "val_images")
        od = os.path.join(PREDS, f"fold{k}"); os.makedirs(od, exist_ok=True)
        for p in sorted(glob.glob(os.path.join(vi, "*.png"))):
            st = os.path.splitext(os.path.basename(p))[0]
            rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            mask, _ = agent.predict(rgb, conf=args.conf, iou=0.5)
            cv2.imwrite(os.path.join(od, st + ".png"), mask)
        print(f"sepsam_yolo fold{k}: dumped -> {od}", flush=True)


if __name__ == "__main__":
    main()
