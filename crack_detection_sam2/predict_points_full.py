"""Full-image sliding-window inference for the POINTS-prompt PromptedSAM2Seg.

Mirrors predict_refine_full.py but, per tile, samples +/- points from the ResUNet
craq prob (via craq_prompt_sampling.sample_points_from_prob) instead of feeding a
dense mask prompt. Used to render the points-prompt qualitative overlay.

    python predict_points_full.py --ckpt runs/<points>/best.pt \
        --image_dir /tmp/imgs --prob_dir <resunet prob dir> --out_dir runs/<out>
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from model_prompted_sam2 import PromptedSAM2Seg
from craq_prompt_sampling import sample_points_from_prob
from predict_refine_full import (gaussian_window, sliding_coords, pad_to_min,
                                 normalize_tile, overlay)

Image.MAX_IMAGE_PIXELS = None


@torch.no_grad()
def predict_points(model, img, prob_craq, device, tile=512, stride=384,
                   batch_size=4, n_pos=3, n_neg=3, use_amp=True):
    H0, W0 = img.shape[:2]
    img_p = pad_to_min(img, tile, True)
    prob_p = pad_to_min(prob_craq, tile, False)
    H, W = img_p.shape[:2]
    win = gaussian_window(tile)
    out_canvas = np.zeros((H, W), np.float32)
    weight_canvas = np.zeros((H, W), np.float32)
    buf_img, buf_c, buf_l, pos = [], [], [], []

    def flush():
        if not buf_img:
            return
        x = torch.stack(buf_img, 0).to(device, non_blocking=True)
        coords = torch.stack(buf_c, 0).to(device)         # (B,P,2) x,y
        labels = torch.stack(buf_l, 0).to(device)         # (B,P)
        if use_amp and device == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x, coords, labels, None)
        else:
            logits = model(x, coords, labels, None)
        p = torch.sigmoid(logits.float().squeeze(1)).cpu().numpy()
        for pi, (y, x_) in zip(p, pos):
            out_canvas[y:y + tile, x_:x_ + tile] += pi * win
            weight_canvas[y:y + tile, x_:x_ + tile] += win
        buf_img.clear(); buf_c.clear(); buf_l.clear(); pos.clear()

    for idx, (y, x) in enumerate(sliding_coords(H, W, tile, stride)):
        buf_img.append(normalize_tile(img_p[y:y + tile, x:x + tile]))
        ptile = prob_p[y:y + tile, x:x + tile]
        c, l = sample_points_from_prob(ptile, n_pos=n_pos, n_neg=n_neg,
                                       size=tile, seed=idx)
        c = c[:, ::-1].copy()                              # (y,x) -> (x,y)
        buf_c.append(torch.from_numpy(c).float())
        buf_l.append(torch.from_numpy(l))
        pos.append((y, x))
        if len(buf_img) >= batch_size:
            flush()
    flush()
    weight_canvas = np.maximum(weight_canvas, 1e-6)
    out_canvas /= weight_canvas
    return out_canvas[:H0, :W0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--prob_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--variant", default="small")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--save_prob", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir); (out / "overlay").mkdir(parents=True, exist_ok=True)
    if args.save_prob:
        (out / "prob").mkdir(parents=True, exist_ok=True)

    model = PromptedSAM2Seg(variant=args.variant, image_size=args.image_size, device=device).to(device)
    payload = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(sd); model.eval()
    print(f"loaded {args.ckpt} val_iou={payload.get('val', {}).get('craq_iou') if isinstance(payload, dict) else '?'}")

    for ip in sorted(Path(args.image_dir).glob("*.jpg")):
        stem = ip.stem
        pp = Path(args.prob_dir) / f"{stem}.npy"
        if not pp.is_file():
            print(f"  [skip] no prob {stem}"); continue
        img = np.array(Image.open(ip).convert("RGB"))
        prob = np.load(pp)
        prob_craq = prob[1].astype(np.float32) if prob.ndim == 3 else prob.astype(np.float32)
        op = predict_points(model, img, prob_craq, device, tile=args.tile,
                            stride=args.stride, batch_size=args.batch_size)
        mask = op > args.thr
        Image.fromarray(overlay(img, mask)).save(out / "overlay" / f"{stem}.png")
        if args.save_prob:
            np.save(out / "prob" / f"{stem}.npy", op.astype(np.float32))
        print(f"  {stem} done  fg={mask.mean()*100:.2f}%")
    print(f"done -> {out/'overlay'}")


if __name__ == "__main__":
    main()
