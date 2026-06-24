"""dump_preds_denseseg.py — dense-seg craquelure expert 在 raw val 上的預測,
存 preds/denseseg/fold{k}/{stem}.png。CLAHE 只是 train aug,val 不套。"""
import argparse, glob, os
import cv2, numpy as np, torch
from crackseg_common.dataset import set_class_names
from crackseg_common.augment import val_transforms
from model_seg import SAM2SemSeg

TILES = "data/labeled32_craq_v3/tiles_512"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--run_prefix", default="outputs/expert_craq_v3_clahe")
    ap.add_argument("--image_size", type=int, default=512)
    args = ap.parse_args()
    set_class_names(["background", "craquelure"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tfm = val_transforms(image_size=args.image_size)
    for k in args.folds:
        model = SAM2SemSeg(variant="small", num_classes=2, device=dev).to(dev)
        ck = torch.load(f"{args.run_prefix}_fold{k}_small/best.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        model.eval()
        vi = os.path.join(TILES, f"craqfold{k}", "val_images")
        od = f"preds/denseseg/fold{k}"; os.makedirs(od, exist_ok=True)
        for p in sorted(glob.glob(os.path.join(vi, "*.png"))):
            st = os.path.splitext(os.path.basename(p))[0]
            rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            out = tfm(image=rgb, mask=np.zeros((h, w), np.uint8))
            x = out["image"].unsqueeze(0).to(dev)
            with torch.no_grad():
                logits = model(x)
            pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            if pred.shape != (h, w):
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(od, st + ".png"), ((pred == 1).astype(np.uint8) * 255))
        print(f"denseseg fold{k}: dumped -> {od}", flush=True)


if __name__ == "__main__":
    main()
