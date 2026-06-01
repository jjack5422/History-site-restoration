"""dump_preds_promptsam2.py — 把 PromptedSAM2 oracle/yolo 預測存成 preds/{method}/fold{k}/{stem}.png。"""
import argparse, glob, json, os
import cv2, numpy as np, torch
from model_prompted_sam2 import PromptedSAM2Seg
from gt_points import gt_points
from eval_promptsam2_craq import load_img, predict

TILES = "data/labeled32_craq_v3/tiles_512"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--n_points", type=int, default=10)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for k in args.folds:
        model = PromptedSAM2Seg(variant="small", image_size=args.image_size, device=dev).to(dev)
        model.load_state_dict(torch.load(f"outputs/promptsam2_craq_fold{k}/best.pt", map_location=dev)["model"])
        model.eval()
        vi = os.path.join(TILES, f"craqfold{k}", "val_images")
        vm = os.path.join(TILES, f"craqfold{k}", "val_masks")
        yp = json.load(open(os.path.join(TILES, f"craqfold{k}", "yolo_points.json")))
        for method, getpts in [("promptsam2_oracle", "oracle"), ("promptsam2_yolo", "yolo")]:
            od = f"preds/{method}/fold{k}"; os.makedirs(od, exist_ok=True)
            for p in sorted(glob.glob(os.path.join(vi, "*.png"))):
                st = os.path.splitext(os.path.basename(p))[0]
                img_t, hw = load_img(p, args.image_size)
                if getpts == "oracle":
                    gt = cv2.imread(os.path.join(vm, st + ".png"), 0)
                    pts, _ = gt_points(gt > 0, args.n_points); pts = pts.tolist()
                else:
                    pts = yp.get(st, [])
                pred = predict(model, img_t, pts, args.image_size, dev, hw)
                cv2.imwrite(os.path.join(od, st + ".png"), (pred.astype(np.uint8) * 255))
            print(f"{method} fold{k}: dumped -> {od}", flush=True)


if __name__ == "__main__":
    main()
