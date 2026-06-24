"""Full-image sliding-window inference for the ResUNet->SAM2 mask-prompt refiner.

Stage 2 of the refine pipeline. For each image it needs the ResUNet craq prob
map (channel 1 of the .npy dumped by crack_detection_unet predict_full --save_prob).
It slides `tile`x`tile` windows over the image, builds a dense mask prompt from
the matching ResUNet prob tile (logit -> resize to the prompt encoder's
mask_input_size), runs PromptedSAM2Seg, sigmoids, stitches a full-res craq prob
map and writes a red overlay. Overlay only; no GT.

Example:
    python predict_refine_full.py \
        --ckpt runs/2026-06-10-craq-sam2prompt-mask/best.pt \
        --image_dir /home/zzz90/research/_data/selected_slices/batch_1 \
        --prob_dir  ../crack_detection_unet/runs/2026-06-10-predict-batch1-resunet/prob \
        --out_dir   runs/2026-06-10-predict-batch1-refine \
        --tile 512 --stride 384
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None  # heritage scans can exceed PIL's bomb limit

from crackseg_common.augment import IMAGENET_MEAN, IMAGENET_STD
from model_prompted_sam2 import PromptedSAM2Seg


def gaussian_window(tile, sigma_ratio=0.125):
    sigma = max(1.0, tile * sigma_ratio)
    c = np.arange(tile) - (tile - 1) / 2.0
    g = np.exp(-(c ** 2) / (2 * sigma ** 2))
    w = np.outer(g, g).astype(np.float32)
    return w / w.max()


def sliding_coords(H, W, tile, stride):
    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if (H - tile) % stride != 0 or H < tile:
        ys.append(max(0, H - tile))
    if (W - tile) % stride != 0 or W < tile:
        xs.append(max(0, W - tile))
    return [(y, x) for y in sorted(set(ys)) for x in sorted(set(xs))]


def pad_to_min(arr, tile, is_img):
    h, w = arr.shape[:2]
    ph, pw = max(0, tile - h), max(0, tile - w)
    if ph or pw:
        pad = ((0, ph), (0, pw), (0, 0)) if is_img else ((0, ph), (0, pw))
        arr = np.pad(arr, pad, mode="reflect")
    return arr


def normalize_tile(tile_uint8):
    x = torch.from_numpy(tile_uint8).float().div_(255.0).permute(2, 0, 1)
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - m) / s


@torch.no_grad()
def predict_full(model, img, prob_craq, device, mask_hw, tile=512, stride=384,
                 batch_size=4, use_amp=True):
    H0, W0 = img.shape[:2]
    img_p = pad_to_min(img, tile, True)
    prob_p = pad_to_min(prob_craq, tile, False)
    H, W = img_p.shape[:2]
    win = gaussian_window(tile)
    out_canvas = np.zeros((H, W), dtype=np.float32)
    weight_canvas = np.zeros((H, W), dtype=np.float32)
    buf_img, buf_pm, pos = [], [], []

    def flush():
        if not buf_img:
            return
        x = torch.stack(buf_img, 0).to(device, non_blocking=True)
        pm = torch.stack(buf_pm, 0).to(device)                       # (B,1,mh,mw)
        n = x.shape[0]
        coords = torch.zeros(n, 1, 2, device=device)
        labels = -torch.ones(n, 1, dtype=torch.long, device=device)  # padding point
        if use_amp and device == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x, coords, labels, pm)
        else:
            logits = model(x, coords, labels, pm)
        p = torch.sigmoid(logits.float().squeeze(1)).cpu().numpy()   # (B,tile,tile)
        for pi, (y, x_) in zip(p, pos):
            out_canvas[y:y + tile, x_:x_ + tile] += pi * win
            weight_canvas[y:y + tile, x_:x_ + tile] += win
        buf_img.clear(); buf_pm.clear(); pos.clear()

    for (y, x) in sliding_coords(H, W, tile, stride):
        buf_img.append(normalize_tile(img_p[y:y + tile, x:x + tile]))
        ptile = prob_p[y:y + tile, x:x + tile]
        pc = np.clip(ptile, 1e-4, 1 - 1e-4)
        logit = np.log(pc / (1 - pc)).astype(np.float32)
        lt = torch.from_numpy(logit)[None, None]                     # (1,1,tile,tile)
        pm = F.interpolate(lt, size=mask_hw, mode="bilinear", align_corners=False)[0]
        buf_pm.append(pm)
        pos.append((y, x))
        if len(buf_img) >= batch_size:
            flush()
    flush()
    weight_canvas = np.maximum(weight_canvas, 1e-6)
    out_canvas /= weight_canvas
    return out_canvas[:H0, :W0]


def overlay(img, mask, color=(255, 0, 0), alpha=0.5):
    out = img.copy()
    out[mask] = (alpha * np.array(color) + (1 - alpha) * img[mask]).astype(np.uint8)
    return out


def morph_filter(mask, min_area=150, max_fill=0.55, blob_max_area=3000):
    """Enforce craquelure morphology: keep thin, connected, network-like components;
    drop isolated small specks (area < min_area) and round/solid decorative blobs
    (high bbox fill ratio, i.e. area/bbox_area > max_fill, when smaller than
    blob_max_area). Returns a filtered bool mask. min_area<=0 disables filtering."""
    if min_area <= 0:
        return mask
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    areas = np.bincount(lab.ravel())
    objs = ndimage.find_objects(lab)
    keep = np.zeros(n + 1, dtype=bool)
    n_small = n_blob = 0
    for i in range(1, n + 1):
        a = int(areas[i])
        if a < min_area:
            n_small += 1
            continue
        sl = objs[i - 1]
        bbox = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
        fill = a / max(bbox, 1)
        if fill > max_fill and a < blob_max_area:
            n_blob += 1
            continue
        keep[i] = True
    print(f"    morph_filter: {n} comps -> kept {int(keep.sum())} "
          f"(dropped small={n_small}, blob={n_blob})")
    return keep[lab]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--prob_dir", required=True, help="ResUNet prob/*.npy dir (channel 1 = craq)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--variant", default="small")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--save_prob", action="store_true", help="save stitched sigmoid craq prob npy")
    ap.add_argument("--min_area", type=int, default=0,
                    help="post-filter: drop connected components smaller than this many px; 0=off "
                         "(also writes overlay_filtered/)")
    ap.add_argument("--max_fill", type=float, default=0.55,
                    help="post-filter: drop round/solid blobs with area/bbox fill ratio above this")
    ap.add_argument("--blob_max_area", type=int, default=3000,
                    help="post-filter: only apply the fill-ratio drop to blobs smaller than this area")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    (out / "overlay").mkdir(parents=True, exist_ok=True)
    if args.save_prob:
        (out / "prob").mkdir(parents=True, exist_ok=True)
    if args.min_area > 0:
        (out / "overlay_filtered").mkdir(parents=True, exist_ok=True)

    model = PromptedSAM2Seg(variant=args.variant, image_size=args.image_size, device=device).to(device)
    payload = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    mask_hw = tuple(model.sam_prompt_encoder.mask_input_size)
    print(f"loaded {args.ckpt} epoch={payload.get('epoch')} "
          f"val_iou={payload.get('val', {}).get('craq_iou')} mask_input_size={mask_hw}")

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    items = sorted(os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                   if f.lower().endswith(exts))
    print(f"images={len(items)} tile={args.tile} stride={args.stride}")

    missing = 0
    for i, path in enumerate(items):
        stem = Path(path).stem
        prob_path = Path(args.prob_dir) / f"{stem}.npy"
        if not prob_path.is_file():
            missing += 1
            print(f"  [skip] no prob for {stem}")
            continue
        img = np.array(Image.open(path).convert("RGB"))
        prob_craq = np.load(prob_path)[1].astype(np.float32)         # (H,W)
        out_prob = predict_full(model, img, prob_craq, device, mask_hw,
                                tile=args.tile, stride=args.stride, batch_size=args.batch_size)
        mask = out_prob > args.thr
        Image.fromarray(overlay(img, mask)).save(out / "overlay" / f"{stem}.png")
        if args.min_area > 0:
            mask_f = morph_filter(mask, args.min_area, args.max_fill, args.blob_max_area)
            Image.fromarray(overlay(img, mask_f)).save(out / "overlay_filtered" / f"{stem}.png")
        if args.save_prob:
            np.save(out / "prob" / f"{stem}.npy", out_prob.astype(np.float32))
        if (i + 1) % 20 == 0 or (i + 1) == len(items):
            print(f"  {i + 1}/{len(items)}")
    print(f"done -> {out / 'overlay'}  (skipped {missing} without prob)")


if __name__ == "__main__":
    main()
