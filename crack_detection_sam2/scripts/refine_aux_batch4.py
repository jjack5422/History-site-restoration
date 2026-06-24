"""Wire the batch4 craquelure tiles into the REFINE pipeline.

The refine trainer (train_craq_promptrefine.py) reads, per tile:
  * image  : tiles_512_corrgt/images/<n>.png      (symlink -> tiles_512/images)  [already present]
  * mask   : tiles_512_corrgt/masks/<n>.png        (corrected GT, 0/1)
  * prob   : tiles_512/resunet_prob/prob/<stem>.npy (ResUNet 2-ch softmax, craq=ch1)
  * fpw    : tiles_512_corrgt/fp_weight/<n>.png     (mined FP mask, 0/1) [only used by fpw3/5]

This script generates the missing mask / prob / fpw for the batch4-* tiles so they
are no longer silently dropped by the trainer's exists_only() filter.

  * corrgt mask = the fresh batch4 binary label itself (it IS the corrected GT).
  * prob        = stage-1 ResUNet (craq-resunet50 fold0) softmax per 512 tile.
  * fpw         = (prob_craq > tau_fp) & (~gt), specks < fp_min_area removed.
                  Matches the existing fp_weight set (reverse-engineered tau_fp=0.25).

Idempotent: only (re)writes batch4-* aux files.

Run:
    /home/zzz90/research/unet_env/bin/python \
        crack_detection_sam2/scripts/refine_aux_batch4.py
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

sys.path.insert(0, "/home/zzz90/research/crack_detection_unet/src")
from unet_model import build_resunet  # noqa: E402

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)

V1 = "/home/zzz90/research/_data/craq_0-94_v1"
DEFAULT_CKPT = "/home/zzz90/research/crack_detection_unet/runs/craq-resunet50-2026-06-10/best.pt"


def clean_specks(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0 or not mask.any():
        return mask
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    areas = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = areas >= min_area
    return keep[lab]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--tiles_root", default=os.path.join(V1, "tiles_512"))
    ap.add_argument("--corrgt_root", default=os.path.join(V1, "tiles_512_corrgt"))
    ap.add_argument("--prefix", default="batch4-")
    ap.add_argument("--tau_fp", type=float, default=0.25)
    ap.add_argument("--fp_min_area", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    img_dir = os.path.join(args.tiles_root, "images")
    base_msk_dir = os.path.join(args.tiles_root, "masks")
    prob_dir = os.path.join(args.tiles_root, "resunet_prob", "prob")
    corr_msk_dir = os.path.join(args.corrgt_root, "masks")
    fpw_dir = os.path.join(args.corrgt_root, "fp_weight")
    for d in (prob_dir, corr_msk_dir, fpw_dir):
        os.makedirs(d, exist_ok=True)

    tiles = sorted(os.path.basename(p) for p in
                   glob.glob(os.path.join(base_msk_dir, args.prefix + "*.png")))
    if not tiles:
        raise SystemExit(f"no {args.prefix}* tiles in {base_msk_dir}; run add_batch4_craq_tiles.py first")
    print(f"{len(tiles)} {args.prefix}* tiles")

    # 1) corrected GT mask = the fresh binary label.
    # NEVER clobber an existing corrgt mask: the original 530 tiles carry a
    # *corrected* GT that differs from the base mask. Only fill in missing ones
    # (i.e. the new batch4 tiles). This makes a `--prefix ''` prob-regen safe.
    n_copy = 0
    for n in tiles:
        dst = os.path.join(corr_msk_dir, n)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(base_msk_dir, n), dst)
            n_copy += 1
    print(f"[corrgt] filled {n_copy} missing masks -> {corr_msk_dir} "
          f"({len(tiles) - n_copy} already present, left untouched)")

    # 2) ResUNet stage-1 prob
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    encoder = ck.get("args", {}).get("encoder", "resnet50")
    model = build_resunet(encoder=encoder, encoder_weights=None, num_classes=2).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)

    n_fp = 0
    for i in range(0, len(tiles), args.batch_size):
        chunk = tiles[i:i + args.batch_size]
        imgs = [np.array(Image.open(os.path.join(img_dir, n)).convert("RGB")) for n in chunk]
        x = torch.stack([torch.from_numpy(a).float().div_(255).permute(2, 0, 1) for a in imgs])
        x = ((x.to(device) - mean) / std)
        with torch.no_grad():
            prob = F.softmax(model(x).float(), dim=1).cpu().numpy()  # (B,2,512,512)
        for n, p in zip(chunk, prob):
            stem = os.path.splitext(n)[0]
            np.save(os.path.join(prob_dir, stem + ".npy"), p.astype(np.float32))
            gt = np.array(Image.open(os.path.join(corr_msk_dir, n))) > 0
            fp = clean_specks((p[1] > args.tau_fp) & (~gt), args.fp_min_area)
            Image.fromarray(fp.astype(np.uint8)).save(os.path.join(fpw_dir, n))
            n_fp += int(fp.sum())
    print(f"[prob] wrote {len(tiles)} npy -> {prob_dir}")
    print(f"[fpw ] wrote {len(tiles)} masks (tau_fp={args.tau_fp}) -> {fpw_dir}; total FP px={n_fp}")
    print("Done. Refine trainer will now include batch4-* tiles (train-only).")


if __name__ == "__main__":
    main()
