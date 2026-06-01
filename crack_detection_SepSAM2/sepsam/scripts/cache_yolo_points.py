"""cache_yolo_points.py — 用 fold-k craquelure YOLO agent 對該 fold val tile 取中軸點,
存 <tiles>/craqfold{k}/yolo_points.json: {stem: [[x,y],...]}(供 crack_detection_sam2 的 eval 載入)。"""
import argparse, glob, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import Agent
from src.geometry import mask_to_points_and_width

TILES = "/home/zzz90/research/crack_detection_SepSAM2/sepsam/datasets/craq_cv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--points_divisor", type=int, default=50)
    args = ap.parse_args()
    from ultralytics.nn.tasks import C2f_SEA  # noqa
    for k in args.folds:
        ck = f"runs/segment/runs/sepsam_agent_craq_cv_fold{k}/weights/best.pt"
        agent = Agent(ck, device="cuda")
        vi = os.path.join(TILES, f"craqfold{k}", "val_images")
        out = {}
        for p in sorted(glob.glob(os.path.join(vi, "*.png"))):
            st = os.path.splitext(os.path.basename(p))[0]
            bgr = cv2.imread(p); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask, _ = agent.predict(rgb, conf=args.conf, iou=0.5)
            n = max(rgb.shape[:2]) // args.points_divisor
            pts, _ = mask_to_points_and_width(mask > 0, n)
            out[st] = pts.tolist()
        dst = os.path.join(TILES, f"craqfold{k}", "yolo_points.json")
        json.dump(out, open(dst, "w"))
        print(f"fold{k}: {len(out)} tiles -> {dst}")


if __name__ == "__main__":
    main()
