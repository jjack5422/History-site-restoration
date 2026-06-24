"""Threshold sweep for a trained ResUNet->SAM2 refine checkpoint on fold0 val.

Runs the val set once, caches per-pixel sigmoid prob + GT (as TP/FP/FN counts
accumulated per threshold), then reports precision/recall/IoU/F1 at each threshold.
Flags the lowest threshold whose recall >= --target_recall and its precision, so we
can check whether refine dominates ResUNet (R>=0.80 with P>0.62).

Example:
    python threshold_sweep_eval.py \
        --ckpt runs/2026-06-10-craq-sam2prompt-tversky37/best.pt \
        --tiles_root /home/.../tiles_512 \
        --split /home/.../tiles_512/group_split_stem.json --fold 0 \
        --prob_dir /home/.../tiles_512/resunet_prob
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from model_prompted_sam2 import PromptedSAM2Seg
from train_craq_promptrefine import PromptTileDS, build_prompts, load_split, exists_only


@torch.no_grad()
def sweep(model, loader, device, mask_hw, thrs):
    tp = np.zeros(len(thrs)); fp = np.zeros(len(thrs)); fn = np.zeros(len(thrs))
    for batch in loader:
        img = batch["image"].to(device)
        gt = (batch["mask"] > 0.5).to(device)
        c, l, pm = build_prompts(batch, "mask", device, mask_hw)
        prob = torch.sigmoid(model(img, c, l, pm).squeeze(1))
        g = gt.bool()
        for i, t in enumerate(thrs):
            pred = prob > t
            tp[i] += int((pred & g).sum())
            fp[i] += int((pred & ~g).sum())
            fn[i] += int((~pred & g).sum())
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tiles_root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--prob_dir", required=True)
    ap.add_argument("--variant", default="small")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--mask_prompt_size", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--target_recall", type=float, default=0.80)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, va = load_split(args.split, args.fold)
    va = exists_only(args.tiles_root, args.prob_dir, va)
    loader = DataLoader(PromptTileDS(args.tiles_root, args.prob_dir, va, train=False),
                        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = PromptedSAM2Seg(variant=args.variant, image_size=args.image_size,
                            mask_prompt_size=args.mask_prompt_size, device=device).to(device)
    payload = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    mask_hw = tuple(model.sam_prompt_encoder.mask_input_size)
    print(f"ckpt={args.ckpt} val_tiles={len(va)} mask_hw={mask_hw}")

    thrs = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
    tp, fp, fn = sweep(model, loader, device, mask_hw, thrs)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
    iou = tp / np.maximum(tp + fp + fn, 1); f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-8)

    print(f"{'thr':>5} {'P':>7} {'R':>7} {'IoU':>7} {'F1':>7}")
    for i, t in enumerate(thrs):
        print(f"{t:5.2f} {prec[i]:7.4f} {rec[i]:7.4f} {iou[i]:7.4f} {f1[i]:7.4f}")

    print("\n--- vs ResUNet (R=0.804, P=0.620) ---")
    hit = [i for i, t in enumerate(thrs) if rec[i] >= args.target_recall]
    if hit:
        i = hit[0]  # highest threshold (least FP) that still reaches target recall
        print(f"R>={args.target_recall}: thr={thrs[i]:.2f} -> R={rec[i]:.4f} P={prec[i]:.4f} "
              f"IoU={iou[i]:.4f}  {'DOMINATES (P>0.62)' if prec[i] > 0.62 else 'P below 0.62'}")
    else:
        print(f"no threshold reaches recall {args.target_recall}; max recall={rec.max():.4f} "
              f"@thr={thrs[int(rec.argmax())]:.2f}")


if __name__ == "__main__":
    main()
