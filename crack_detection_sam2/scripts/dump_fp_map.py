"""載入 baseline best.pt,在指定 fold 的 val tiles 上 dump FP(pred & ~gt)紅色 overlay,
目視確認 false positive 是否集中在彩繪/背景紋理。mask-prompt 模式。"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_prompted_sam2 import PromptedSAM2Seg

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tiles_root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--prob_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev)
    model = PromptedSAM2Seg(variant=ck["args"]["variant"], image_size=ck["args"]["image_size"],
                            mask_prompt_size=ck["args"].get("mask_prompt_size"), device=dev).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    mask_hw = tuple(model.sam_prompt_encoder.mask_input_size)

    va = json.load(open(args.split))["folds"][args.fold]["val"]
    timg = Path(args.tiles_root) / "images"; tmsk = Path(args.tiles_root) / "masks"
    prob_d = Path(args.prob_dir) / "prob"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    n = 0
    for name in va:
        if n >= args.n:
            break
        ip = timg / name
        pp = prob_d / (Path(name).stem + ".npy")
        if not (ip.exists() and pp.exists()):
            continue
        img = np.array(Image.open(ip).convert("RGB"))
        gt = (np.array(Image.open(tmsk / name)) > 0)
        prob = np.load(pp)[1].astype(np.float32)
        x = torch.from_numpy(img).float().div_(255).permute(2, 0, 1)
        x = ((x - MEAN) / STD).unsqueeze(0).to(dev)
        p = np.clip(prob, 1e-4, 1 - 1e-4)
        logit = torch.from_numpy(np.log(p / (1 - p)))[None, None].float()
        pm = F.interpolate(logit, size=mask_hw, mode="bilinear", align_corners=False).to(dev)
        coords = torch.zeros(1, 1, 2, device=dev); labels = -torch.ones(1, 1, dtype=torch.long, device=dev)
        with torch.no_grad():
            out_logits = model(x, coords, labels, pm)
        pred = (out_logits.squeeze().cpu().numpy() > 0)
        ov = img.copy()
        ov[pred & ~gt] = [255, 0, 0]      # FP 紅
        ov[pred & gt] = [0, 255, 0]       # TP 綠
        ov[~pred & gt] = [0, 0, 255]      # FN 藍
        blend = (0.5 * img + 0.5 * ov).astype(np.uint8)
        Image.fromarray(blend).save(out / f"{Path(name).stem}_fp.png")
        n += 1
    print(f"dumped {n} overlays to {out}")


if __name__ == "__main__":
    main()
