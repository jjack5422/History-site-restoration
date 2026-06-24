"""對單張 (或一批) 原圖跑 sliding-window 推論, Gaussian 加權 stitch, 輸出 semantic mask 與彩色 overlay。

Example:
    python predict_full.py \\
        --ckpt outputs/stem_fold0_small/best.pt \\
        --image_dir merged_4class_mask_semantic/images \\
        --out_dir outputs/predict_stem_fold0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from crackseg_common.augment import IMAGENET_MEAN, IMAGENET_STD
import crackseg_common.dataset as _dataset
from crackseg_common.dataset import set_class_names
from crackseg_common.metrics import ConfusionMeter, format_metrics
from model_seg import SAM2SemSeg


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


def load_image_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_label(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def gaussian_window(tile_size: int, sigma_ratio: float = 0.125) -> np.ndarray:
    sigma = max(1.0, tile_size * sigma_ratio)
    coords = np.arange(tile_size) - (tile_size - 1) / 2.0
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    w = np.outer(g, g).astype(np.float32)
    w /= w.max()
    return w  # [tile, tile]


def sliding_coords(H: int, W: int, tile: int, stride: int) -> List[Tuple[int, int]]:
    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if (H - tile) % stride != 0 or H < tile:
        ys.append(max(0, H - tile))
    if (W - tile) % stride != 0 or W < tile:
        xs.append(max(0, W - tile))
    coords = []
    seen = set()
    for y in ys:
        for x in xs:
            if (y, x) not in seen:
                seen.add((y, x))
                coords.append((y, x))
    return coords


def pad_to_min(img: np.ndarray, tile: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = img.shape[:2]
    pad_h = max(0, tile - h)
    pad_w = max(0, tile - w)
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    if img.ndim == 3:
        out = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
    else:
        out = np.pad(img, ((0, pad_h), (0, pad_w)), constant_values=0)
    return out, (pad_h, pad_w)


def normalize_tile(tile_uint8: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    x = torch.from_numpy(tile_uint8).float().div_(255.0).permute(2, 0, 1)
    m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (x - m) / s


@torch.no_grad()
def predict_full(model: SAM2SemSeg, img: np.ndarray, device: str,
                 tile: int = 512, stride: int = 384,
                 batch_size: int = 4, tta_flip: bool = False,
                 use_amp: bool = True) -> np.ndarray:
    """回傳 [num_classes, H, W] float32 的 logits-after-softmax 機率圖 (Gaussian 加權平均)。"""
    H0, W0 = img.shape[:2]
    img_p, (pad_h, pad_w) = pad_to_min(img, tile)
    H, W = img_p.shape[:2]
    coords = sliding_coords(H, W, tile, stride)

    win = gaussian_window(tile)  # [tile, tile]
    prob_canvas = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    weight_canvas = np.zeros((H, W), dtype=np.float32)

    # batch tiles for speed
    buffer_tiles = []
    buffer_pos = []

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
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()  # [B,C,H,W]
        for p, (y, x_) in zip(probs, buffer_pos):
            prob_canvas[:, y:y + tile, x_:x_ + tile] += p * win[None, :, :]
            weight_canvas[y:y + tile, x_:x_ + tile] += win
        buffer_tiles.clear()
        buffer_pos.clear()

    for (y, x) in coords:
        tile_img = img_p[y:y + tile, x:x + tile]
        buffer_tiles.append(normalize_tile(tile_img))
        buffer_pos.append((y, x))
        if len(buffer_tiles) >= batch_size:
            flush()
    flush()

    weight_canvas = np.maximum(weight_canvas, 1e-6)
    prob_canvas /= weight_canvas[None, :, :]
    return prob_canvas[:, :H0, :W0]  # 去除 padding


def colorize_label(label: np.ndarray) -> np.ndarray:
    return CLASS_RGB[label.clip(0, NUM_CLASSES - 1)]


def apply_priority(label: np.ndarray,
                   priority_class_name: str,
                   target_class_name: str,
                   dilate: int = 0) -> np.ndarray:
    """把 priority_class 區域(可選膨脹 N px) 內的 target_class 像素改判為 priority_class。

    對齊 dataset 的 priority overwrite 規則 (e.g. craquelure overwrites crack)。
    dilate=0 時只覆蓋完全重疊處 (對 argmax 輸出無效, 因每像素只一類);
    dilate>=1 時以方形 structuring element 擴張。
    """
    if priority_class_name not in CLASS_NAMES or target_class_name not in CLASS_NAMES:
        return label
    pri_id = CLASS_NAMES.index(priority_class_name)
    tgt_id = CLASS_NAMES.index(target_class_name)
    pri_mask = (label == pri_id)
    if dilate > 0 and pri_mask.any():
        try:
            from scipy.ndimage import binary_dilation
            pri_mask = binary_dilation(pri_mask, iterations=dilate)
        except ImportError:
            k = dilate
            pad = np.pad(pri_mask, k, mode="constant")
            acc = np.zeros_like(pri_mask)
            for dy in range(-k, k + 1):
                for dx in range(-k, k + 1):
                    acc |= pad[k + dy:k + dy + pri_mask.shape[0],
                               k + dx:k + dx + pri_mask.shape[1]]
            pri_mask = acc
    out = label.copy()
    out[pri_mask & (label == tgt_id)] = pri_id
    return out


def overlay(img: np.ndarray, label: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    color = colorize_label(label)
    fg = label > 0
    out = img.copy()
    out[fg] = (alpha * color[fg] + (1 - alpha) * img[fg]).astype(np.uint8)
    return out


def load_model_from_ckpt(ckpt_path: str, device: str) -> Tuple[SAM2SemSeg, dict]:
    global CLASS_NAMES, NUM_CLASSES, CLASS_RGB
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = payload.get("args", {})
    variant = args.get("variant", "small")
    freeze_trunk = args.get("freeze_trunk", True)
    freeze_neck = args.get("freeze_neck", False)
    cls_str = args.get("class_names")
    if cls_str:
        names = [s.strip() for s in cls_str.split(",") if s.strip()]
        set_class_names(names)
        CLASS_NAMES = _dataset.CLASS_NAMES
        NUM_CLASSES = _dataset.NUM_CLASSES
        CLASS_RGB = build_class_rgb(CLASS_NAMES)
        print(f"ckpt class_names={CLASS_NAMES}")
    model = SAM2SemSeg(variant=variant, num_classes=NUM_CLASSES,
                       freeze_trunk=freeze_trunk, freeze_neck=freeze_neck,
                       device=device).to(device)
    state = payload["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] load state_dict missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--image", default=None,
                        help="單張影像; 與 --image_dir 二擇一")
    parser.add_argument("--image_dir", default=None,
                        help="批次資料夾; 會掃 jpg/png")
    parser.add_argument("--mask_dir", default=None,
                        help="若提供, 對每張原圖額外算 metrics")
    parser.add_argument("--out_dir", default="outputs/predict")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tta_flip", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--save_prob", action="store_true",
                        help="同時存每類機率 npy")
    parser.add_argument("--craq_dilate", type=int, default=0,
                        help="craquelure 區域膨脹 N px, 內部 crack 像素改判 craquelure (對齊 label priority)。0=關閉。")
    args = parser.parse_args()

    if (args.image is None) == (args.image_dir is None):
        raise SystemExit("請指定 --image 或 --image_dir 其中之一")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    (out_dir / "label").mkdir(parents=True, exist_ok=True)
    (out_dir / "color").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlay").mkdir(parents=True, exist_ok=True)
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
        if args.craq_dilate > 0 and "craquelure" in CLASS_NAMES and "crack" in CLASS_NAMES:
            label = apply_priority(label, "craquelure", "crack",
                                   dilate=args.craq_dilate)

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
