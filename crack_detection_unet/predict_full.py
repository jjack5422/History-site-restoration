"""ResUNet (smp) sliding-window 推論, 與 sam2 版的 predict_full.py 介面對齊。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

SAM2_ROOT = "/home/zzz90/research/crack_detection_sam2"
if SAM2_ROOT not in sys.path:
    sys.path.insert(0, SAM2_ROOT)

from augment import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
import dataset as _dataset  # noqa: E402
from dataset import set_class_names  # noqa: E402
from metrics import ConfusionMeter, format_metrics  # noqa: E402

from unet_model import build_resunet


_DEFAULT_PALETTE = {
    "background": (0, 0, 0),
    "crack":      (255, 0, 0),
    "loss":       (0, 255, 255),
    "shrinkage":  (255, 255, 0),
    "craquelure": (255, 0, 255),
}
_FALLBACK = [(0, 255, 0), (255, 128, 0), (128, 0, 255), (0, 128, 255), (255, 255, 255)]


def build_class_rgb(class_names):
    rgb = []
    fb = iter(_FALLBACK)
    for n in class_names:
        rgb.append(_DEFAULT_PALETTE.get(n, next(fb, (200, 200, 200))))
    return np.array(rgb, dtype=np.uint8)


CLASS_NAMES = _dataset.CLASS_NAMES
NUM_CLASSES = _dataset.NUM_CLASSES
CLASS_RGB = build_class_rgb(CLASS_NAMES)


def load_image_rgb(path): return np.array(Image.open(path).convert("RGB"))
def load_label(path):
    arr = np.array(Image.open(path))
    return (arr[..., 0] if arr.ndim == 3 else arr).astype(np.uint8)


def gaussian_window(tile_size, sigma_ratio=0.125):
    sigma = max(1.0, tile_size * sigma_ratio)
    coords = np.arange(tile_size) - (tile_size - 1) / 2.0
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    w = np.outer(g, g).astype(np.float32)
    w /= w.max()
    return w


def sliding_coords(H, W, tile, stride):
    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if (H - tile) % stride != 0 or H < tile:
        ys.append(max(0, H - tile))
    if (W - tile) % stride != 0 or W < tile:
        xs.append(max(0, W - tile))
    coords, seen = [], set()
    for y in ys:
        for x in xs:
            if (y, x) not in seen:
                seen.add((y, x)); coords.append((y, x))
    return coords


def pad_to_min(img, tile):
    h, w = img.shape[:2]
    pad_h = max(0, tile - h); pad_w = max(0, tile - w)
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    if img.ndim == 3:
        out = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
    else:
        out = np.pad(img, ((0, pad_h), (0, pad_w)), constant_values=0)
    return out, (pad_h, pad_w)


def normalize_tile(tile_uint8, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    x = torch.from_numpy(tile_uint8).float().div_(255.0).permute(2, 0, 1)
    m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (x - m) / s


@torch.no_grad()
def predict_full(model, img, device, tile=512, stride=384,
                 batch_size=4, tta_flip=False, use_amp=True):
    H0, W0 = img.shape[:2]
    img_p, _ = pad_to_min(img, tile)
    H, W = img_p.shape[:2]
    coords = sliding_coords(H, W, tile, stride)
    win = gaussian_window(tile)
    prob_canvas = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    weight_canvas = np.zeros((H, W), dtype=np.float32)
    buffer_tiles, buffer_pos = [], []

    def flush():
        if not buffer_tiles:
            return
        x = torch.stack(buffer_tiles, dim=0).to(device, non_blocking=True)
        if use_amp and device == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                if tta_flip:
                    logits = logits + torch.flip(model(torch.flip(x, dims=[-1])), dims=[-1])
                    logits = logits + torch.flip(model(torch.flip(x, dims=[-2])), dims=[-2])
                    logits = logits / 3.0
        else:
            logits = model(x)
            if tta_flip:
                logits = logits + torch.flip(model(torch.flip(x, dims=[-1])), dims=[-1])
                logits = logits + torch.flip(model(torch.flip(x, dims=[-2])), dims=[-2])
                logits = logits / 3.0
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        for p, (y, x_) in zip(probs, buffer_pos):
            prob_canvas[:, y:y + tile, x_:x_ + tile] += p * win[None, :, :]
            weight_canvas[y:y + tile, x_:x_ + tile] += win
        buffer_tiles.clear(); buffer_pos.clear()

    for (y, x) in coords:
        buffer_tiles.append(normalize_tile(img_p[y:y + tile, x:x + tile]))
        buffer_pos.append((y, x))
        if len(buffer_tiles) >= batch_size:
            flush()
    flush()

    weight_canvas = np.maximum(weight_canvas, 1e-6)
    prob_canvas /= weight_canvas[None, :, :]
    return prob_canvas[:, :H0, :W0]


def colorize_label(label):
    return CLASS_RGB[label.clip(0, NUM_CLASSES - 1)]


def overlay(img, label, alpha=0.5):
    color = colorize_label(label)
    fg = label > 0
    out = img.copy()
    out[fg] = (alpha * color[fg] + (1 - alpha) * img[fg]).astype(np.uint8)
    return out


def load_model_from_ckpt(ckpt_path, device):
    global CLASS_NAMES, NUM_CLASSES, CLASS_RGB
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = payload.get("args", {})
    encoder = args.get("encoder", "resnet50")
    cls_str = args.get("class_names", "background,craquelure")
    names = [s.strip() for s in cls_str.split(",") if s.strip()]
    set_class_names(names)
    CLASS_NAMES = _dataset.CLASS_NAMES
    NUM_CLASSES = _dataset.NUM_CLASSES
    CLASS_RGB = build_class_rgb(CLASS_NAMES)
    print(f"ckpt encoder={encoder} class_names={CLASS_NAMES}")
    model = build_resunet(encoder=encoder, encoder_weights=None,
                          num_classes=NUM_CLASSES).to(device)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--mask_dir", default=None)
    parser.add_argument("--out_dir", default="outputs/predict")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tta_flip", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--save_prob", action="store_true")
    args = parser.parse_args()

    if (args.image is None) == (args.image_dir is None):
        raise SystemExit("請指定 --image 或 --image_dir 其中之一")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    for sub in ("label", "color", "overlay"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    if args.save_prob:
        (out_dir / "prob").mkdir(parents=True, exist_ok=True)

    model, payload = load_model_from_ckpt(args.ckpt, device)
    print(f"loaded ckpt={args.ckpt} epoch={payload.get('epoch')} val={payload.get('val', {}).get('miou')}")

    if args.image:
        items = [args.image]
    else:
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        items = sorted([os.path.join(args.image_dir, f)
                        for f in os.listdir(args.image_dir)
                        if f.lower().endswith(exts)])
    print(f"images={len(items)} tile={args.tile} stride={args.stride} tta_flip={args.tta_flip}")

    overall = ConfusionMeter(NUM_CLASSES)
    per_image_rows = []

    for path in tqdm(items):
        stem = Path(path).stem
        img = load_image_rgb(path)
        prob = predict_full(model, img, device,
                            tile=args.tile, stride=args.stride,
                            batch_size=args.batch_size, tta_flip=args.tta_flip,
                            use_amp=not args.no_amp)
        label = prob.argmax(0).astype(np.uint8)

        Image.fromarray(label).save(out_dir / "label" / f"{stem}.png")
        Image.fromarray(colorize_label(label)).save(out_dir / "color" / f"{stem}.png")
        Image.fromarray(overlay(img, label)).save(out_dir / "overlay" / f"{stem}.png")
        if args.save_prob:
            np.save(out_dir / "prob" / f"{stem}.npy", prob.astype(np.float32))

        if args.mask_dir is not None:
            gt_path = os.path.join(args.mask_dir, stem + ".png")
            if os.path.isfile(gt_path):
                gt = load_label(gt_path)
                if gt.shape == label.shape:
                    per_meter = ConfusionMeter(NUM_CLASSES)
                    per_meter.update(label, gt)
                    overall.update(label, gt)
                    res = per_meter.compute(class_names=CLASS_NAMES, ignore_index=0)
                    per_image_rows.append({
                        "image": stem,
                        "miou": res["miou"],
                        "mdice": res["mdice"],
                        "pixel_acc": res["pixel_accuracy"],
                        **{f"iou_{k}": v["iou"] for k, v in res["per_class"].items()},
                    })
                else:
                    print(f"[warn] shape mismatch {stem}: pred={label.shape} gt={gt.shape}")

    if args.mask_dir is not None and per_image_rows:
        res = overall.compute(class_names=CLASS_NAMES, ignore_index=0)
        print("=== overall ===")
        print(format_metrics(res))
        with open(out_dir / "overall_metrics.json", "w") as f:
            json.dump(res, f, indent=2)
        keys = list(per_image_rows[0].keys())
        with open(out_dir / "per_image_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in per_image_rows:
                w.writerow(r)
        print(f"輸出: {out_dir}")


if __name__ == "__main__":
    main()
