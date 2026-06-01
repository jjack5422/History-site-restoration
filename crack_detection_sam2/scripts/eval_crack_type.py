"""Evaluate merged crack+craquelure as a single 'crack_type' binary class.

4-fold cross-validation: for each fold, load the held-out crack expert (ResUNet)
and craquelure expert (SAM2), merge their predictions (union), and evaluate
against merged GT from CVAT VOC palette masks.

Usage:
    python scripts/eval_crack_type.py
    python scripts/eval_crack_type.py --save_overlay
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

UNET_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "crack_detection_unet")
sys.path.insert(0, UNET_ROOT)

from augment import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from metrics import ConfusionMeter, format_metrics  # noqa: E402

CRACK_RGB = (255, 24, 3)
CRAQ_RGB = (102, 255, 102)

FOLD_VAL_GROUPS = [
    "KJTHT-SC-M-2RB1-4",
    "KJTHT-SC-M-2LB1-2",
    "KJTHT-SC-L-A4-4",
    "KJTHT-SC-L-1RB1-1",
]


def load_image_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def build_merged_gt(seg_path):
    rgb = np.array(Image.open(seg_path).convert("RGB"))
    crack = np.all(rgb == np.array(CRACK_RGB, dtype=np.uint8), axis=-1)
    craq = np.all(rgb == np.array(CRAQ_RGB, dtype=np.uint8), axis=-1)
    gt = np.zeros(rgb.shape[:2], dtype=np.uint8)
    gt[crack | craq] = 1
    return gt


def images_for_group(image_dir, group):
    out = []
    for f in sorted(os.listdir(image_dir)):
        if f.startswith(group) and f.lower().endswith((".jpg", ".jpeg", ".png")):
            out.append(os.path.join(image_dir, f))
    return out


def sliding_predict(model, img, device, tile=512, stride=384, batch_size=4):
    """Run sliding-window inference, return prob [C, H, W]."""
    import torch.nn.functional as F

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    H0, W0 = img.shape[:2]
    pad_h = max(0, tile - H0)
    pad_w = max(0, tile - W0)
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
    H, W = img.shape[:2]

    sigma = max(1.0, tile * 0.125)
    coords_1d = np.arange(tile) - (tile - 1) / 2.0
    g = np.exp(-(coords_1d ** 2) / (2 * sigma ** 2))
    win = np.outer(g, g).astype(np.float32)
    win /= win.max()

    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if (H - tile) % stride != 0 or H < tile:
        ys.append(max(0, H - tile))
    if (W - tile) % stride != 0 or W < tile:
        xs.append(max(0, W - tile))
    all_coords = list({(y, x) for y in ys for x in xs})

    num_classes = None
    prob_canvas = None
    weight_canvas = np.zeros((H, W), dtype=np.float32)

    buf_tiles, buf_pos = [], []

    def flush():
        nonlocal prob_canvas, num_classes
        if not buf_tiles:
            return
        x = torch.stack(buf_tiles, dim=0).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        if prob_canvas is None:
            num_classes = probs.shape[1]
            prob_canvas = np.zeros((num_classes, H, W), dtype=np.float32)
        for p, (y, x_) in zip(probs, buf_pos):
            prob_canvas[:, y:y + tile, x_:x_ + tile] += p * win[None, :, :]
            weight_canvas[y:y + tile, x_:x_ + tile] += win
        buf_tiles.clear()
        buf_pos.clear()

    for (y, x) in all_coords:
        t = torch.from_numpy(img[y:y + tile, x:x + tile]).float().div_(255.0).permute(2, 0, 1)
        t = (t - mean) / std
        buf_tiles.append(t)
        buf_pos.append((y, x))
        if len(buf_tiles) >= batch_size:
            flush()
    flush()

    weight_canvas = np.maximum(weight_canvas, 1e-6)
    prob_canvas /= weight_canvas[None, :, :]
    return prob_canvas[:, :H0, :W0]


def load_sam2_model(ckpt_path, device):
    from model_seg import SAM2SemSeg
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = payload.get("args", {})
    variant = args.get("variant", "small")
    cls_str = args.get("class_names", "background,craquelure")
    names = [s.strip() for s in cls_str.split(",") if s.strip()]
    num_classes = len(names)
    model = SAM2SemSeg(variant=variant, num_classes=num_classes,
                       freeze_trunk=True, freeze_neck=False, device=device).to(device)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model, names


def load_unet_model(ckpt_path, device):
    from unet_model import build_resunet
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = payload.get("args", {})
    encoder = args.get("encoder", "resnet50")
    cls_str = args.get("class_names", "background,crack")
    names = [s.strip() for s in cls_str.split(",") if s.strip()]
    num_classes = len(names)
    model = build_resunet(encoder=encoder, encoder_weights=None,
                          num_classes=num_classes).to(device)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model, names


def overlay_binary(img, mask, color=(255, 80, 80), alpha=0.5):
    out = img.copy()
    fg = mask > 0
    c = np.array(color, dtype=np.uint8)
    out[fg] = (alpha * c + (1 - alpha) * out[fg]).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",
                        default=os.path.join(PROJECT_ROOT, "data/selected_slices"))
    parser.add_argument("--seg_dir",
                        default=os.path.join(PROJECT_ROOT, "data/1-31test/SegmentationClass"))
    parser.add_argument("--craq_ckpt_pattern",
                        default=os.path.join(PROJECT_ROOT,
                                             "outputs/expert_craq_v3_fold{fold}_small/best.pt"))
    parser.add_argument("--crack_ckpt_pattern",
                        default=os.path.join(os.path.dirname(PROJECT_ROOT),
                                             "crack_detection_unet/outputs/expert_crack_v3_fold{fold}_resnet50/best.pt"))
    parser.add_argument("--out_dir",
                        default=os.path.join(PROJECT_ROOT, "outputs/eval_crack_type_4fold"))
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="probability threshold for positive class")
    parser.add_argument("--erode", type=int, default=0,
                        help="morphological erosion iterations on merged mask")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlay:
        (out_dir / "overlay").mkdir(exist_ok=True)
        (out_dir / "pred_mask").mkdir(exist_ok=True)
        (out_dir / "gt_mask").mkdir(exist_ok=True)

    overall_meter = ConfusionMeter(2)
    per_image_rows = []
    class_names = ["background", "crack_type"]

    for fold, val_group in enumerate(FOLD_VAL_GROUPS):
        craq_ckpt = args.craq_ckpt_pattern.format(fold=fold)
        crack_ckpt = args.crack_ckpt_pattern.format(fold=fold)
        if not os.path.isfile(craq_ckpt) or not os.path.isfile(crack_ckpt):
            print(f"[skip] fold {fold}: ckpt not found")
            continue

        print(f"\n=== Fold {fold} | val_group={val_group} ===")
        craq_model, _ = load_sam2_model(craq_ckpt, device)
        crack_model, _ = load_unet_model(crack_ckpt, device)

        val_images = images_for_group(args.image_dir, val_group)
        print(f"  images: {len(val_images)}")

        for img_path in tqdm(val_images, desc=f"fold{fold}"):
            stem = Path(img_path).stem
            seg_path = os.path.join(args.seg_dir, stem + ".png")
            if not os.path.isfile(seg_path):
                continue

            img = load_image_rgb(img_path)
            gt = build_merged_gt(seg_path)

            with torch.no_grad():
                craq_prob = sliding_predict(craq_model, img, device,
                                           tile=args.tile, stride=args.stride,
                                           batch_size=args.batch_size)
                crack_prob = sliding_predict(crack_model, img, device,
                                            tile=args.tile, stride=args.stride,
                                            batch_size=args.batch_size)

            craq_fg = craq_prob[1] > args.threshold
            crack_fg = crack_prob[1] > args.threshold
            merged_pred = (craq_fg | crack_fg).astype(np.uint8)

            if args.erode > 0:
                from scipy.ndimage import binary_erosion
                merged_pred = binary_erosion(merged_pred, iterations=args.erode).astype(np.uint8)

            meter = ConfusionMeter(2)
            meter.update(merged_pred, gt)
            overall_meter.update(merged_pred, gt)
            res = meter.compute(class_names=class_names, ignore_index=0)

            per_image_rows.append({
                "fold": fold,
                "val_group": val_group,
                "image": stem,
                "iou": res["per_class"]["crack_type"]["iou"],
                "dice": res["per_class"]["crack_type"]["dice"],
                "precision": res["per_class"]["crack_type"]["precision"],
                "recall": res["per_class"]["crack_type"]["recall"],
                "gt_pixels": res["per_class"]["crack_type"]["gt_pixels"],
            })

            if args.save_overlay:
                ov = overlay_binary(img, merged_pred)
                Image.fromarray(ov).save(out_dir / "overlay" / f"{stem}.png")
                Image.fromarray(merged_pred * 255).save(out_dir / "pred_mask" / f"{stem}.png")
                Image.fromarray(gt * 255).save(out_dir / "gt_mask" / f"{stem}.png")

        del craq_model, crack_model
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Overall (4-fold cross-validated, merged crack+craquelure)")
    print("=" * 60)
    res = overall_meter.compute(class_names=class_names, ignore_index=0)
    print(format_metrics(res))

    with open(out_dir / "overall_metrics.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    if per_image_rows:
        keys = list(per_image_rows[0].keys())
        with open(out_dir / "per_image_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in per_image_rows:
                w.writerow(r)

    print(f"\noutput: {out_dir}")


if __name__ == "__main__":
    main()
