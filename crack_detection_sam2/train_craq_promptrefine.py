"""SAM2 prompt-refine training for craquelure.

Two-stage pipeline stage 2: a ResUNet craquelure probability map (produced offline
by predict_full.py --save_prob) is fed to SAM2 (frozen image encoder, trainable
prompt encoder + mask decoder) either as a dense mask prompt or as sampled point
prompts. Output is a refined craquelure mask.

  --prompt_mode mask    : ResUNet prob -> resize to mask_input_size -> prev_mask
  --prompt_mode points  : sample +/- points from ResUNet prob -> point prompt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_prompted_sam2 import PromptedSAM2Seg
from craq_prompt_sampling import sample_points_from_prob

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PromptTileDS(Dataset):
    def __init__(self, tiles_root, prob_dir, names, train=False, dino_dir=None):
        self.timg = Path(tiles_root) / "images"
        self.tmsk = Path(tiles_root) / "masks"
        self.prob = Path(prob_dir) / "prob"
        self.dino = Path(dino_dir) if dino_dir else None
        self.names = names
        self.train = train

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        img = np.array(Image.open(self.timg / n).convert("RGB"))
        msk = (np.array(Image.open(self.tmsk / n)) > 0).astype(np.float32)
        prob = np.load(self.prob / (Path(n).stem + ".npy"))[1].astype(np.float32)  # craq channel
        dino = None
        if self.dino is not None:
            dino = np.load(self.dino / (Path(n).stem + ".npy")).astype(np.float32)  # [C,gh,gw]
        flip = self.train and np.random.rand() < 0.5
        if flip:
            img = img[:, ::-1].copy(); msk = msk[:, ::-1].copy(); prob = prob[:, ::-1].copy()
            if dino is not None:
                dino = dino[..., ::-1].copy()  # flip width
        x = torch.from_numpy(img).float().div_(255).permute(2, 0, 1)
        m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        out = {"image": (x - m) / s, "mask": torch.from_numpy(msk),
               "prob": torch.from_numpy(prob), "name": n}
        if dino is not None:
            out["dino"] = torch.from_numpy(dino)
        return out


def build_prompts(batch, mode, device, mask_hw, size=512):
    probs = batch["prob"]            # (B,H,W) in [0,1]
    B = probs.shape[0]
    if mode == "mask":
        # logit-ise prob then resize to the prompt encoder's mask_input_size
        p = probs.clamp(1e-4, 1 - 1e-4)
        logit = torch.log(p / (1 - p)).unsqueeze(1)
        pm = F.interpolate(logit, size=mask_hw, mode="bilinear", align_corners=False).to(device)
        coords = torch.zeros(B, 1, 2, device=device)
        labels = -torch.ones(B, 1, dtype=torch.long, device=device)  # padding point
        return coords, labels, pm
    # points mode
    cs, ls = [], []
    for b in range(B):
        c, l = sample_points_from_prob(probs[b].numpy(), n_pos=3, n_neg=3, size=size, seed=b)
        c = c[:, ::-1].copy()        # (y,x) -> (x,y) for SAM2
        cs.append(torch.from_numpy(c).float())
        ls.append(torch.from_numpy(l))
    return torch.stack(cs).to(device), torch.stack(ls).to(device), None


def dice_bce_loss(logits, target, alpha=0.5, beta=0.5):
    # logits (B,1,H,W), target (B,H,W) float 0/1.
    # alpha=beta=0.5 -> Dice (symmetric). alpha<beta -> Tversky penalising FN
    # more (recall-leaning). loss = 0.5*BCE + 0.5*(1 - Tversky).
    bce = F.binary_cross_entropy_with_logits(logits.squeeze(1), target)
    p = torch.sigmoid(logits.squeeze(1))
    tp = (p * target).sum((1, 2))
    fp = (p * (1 - target)).sum((1, 2))
    fn = ((1 - p) * target).sum((1, 2))
    tversky = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
    return 0.5 * bce + 0.5 * (1 - tversky).mean()


def run_model(model, img, dino, c, l, pm):
    return model(img, dino, c, l, pm) if dino is not None else model(img, c, l, pm)


@torch.no_grad()
def evaluate(model, loader, mode, device, mask_hw):
    model.eval()
    tp = fp = fn = 0
    for batch in loader:
        img = batch["image"].to(device)
        gt = (batch["mask"] > 0.5).to(device)
        dino = batch["dino"].to(device) if "dino" in batch else None
        c, l, pm = build_prompts(batch, mode, device, mask_hw)
        logits = run_model(model, img, dino, c, l, pm)
        pred = logits.squeeze(1) > 0
        tp += int((pred & gt).sum()); fp += int((pred & ~gt).sum()); fn += int((~pred & gt).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"craq_iou": iou, "craq_f1": f1, "precision": prec, "recall": rec}


def load_split(path, fold):
    fd = json.load(open(path))["folds"][fold]
    return fd["train"], fd["val"]


def exists_only(tiles_root, prob_dir, names, dino_dir=None):
    img = Path(tiles_root) / "images"; prob = Path(prob_dir) / "prob"
    dino = Path(dino_dir) if dino_dir else None
    keep = []
    for n in names:
        if not ((img / n).exists() and (prob / (Path(n).stem + ".npy")).exists()):
            continue
        if dino is not None and not (dino / (Path(n).stem + ".npy")).exists():
            continue
        keep.append(n)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--prob_dir", required=True)
    ap.add_argument("--prompt_mode", choices=["mask", "points"], required=True)
    ap.add_argument("--variant", default="small")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--mask_prompt_size", type=int, default=None,
                    help="higher-res mask prompt (e.g. 256); default 4*embed=128")
    ap.add_argument("--tversky_alpha", type=float, default=0.5,
                    help="FP weight; <beta = recall-leaning (default 0.5 = Dice)")
    ap.add_argument("--tversky_beta", type=float, default=0.5, help="FN weight")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--base_lr", type=float, default=2e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--output_dir", default="runs/sam2prompt")
    ap.add_argument("--dino_feat_dir", default=None,
                    help="給定則啟用 DINOv2 fusion (FusedPromptedSAM2Seg);不給為 baseline")
    ap.add_argument("--dino_dim", type=int, default=384)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)

    tr_names, va_names = load_split(args.split, args.fold)
    tr_names = exists_only(args.tiles_root, args.prob_dir, tr_names, args.dino_feat_dir)
    va_names = exists_only(args.tiles_root, args.prob_dir, va_names, args.dino_feat_dir)
    print(f"mode={args.prompt_mode} train={len(tr_names)} val={len(va_names)} "
          f"dino={'on' if args.dino_feat_dir else 'off'}")

    tr = DataLoader(PromptTileDS(args.tiles_root, args.prob_dir, tr_names, train=True,
                                 dino_dir=args.dino_feat_dir),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True, drop_last=True)
    va = DataLoader(PromptTileDS(args.tiles_root, args.prob_dir, va_names, train=False,
                                 dino_dir=args.dino_feat_dir),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=True)

    if args.dino_feat_dir:
        from model_fused_sam2 import FusedPromptedSAM2Seg
        model = FusedPromptedSAM2Seg(variant=args.variant, image_size=args.image_size,
                                     dino_dim=args.dino_dim,
                                     mask_prompt_size=args.mask_prompt_size, device=device).to(device)
    else:
        model = PromptedSAM2Seg(variant=args.variant, image_size=args.image_size,
                                mask_prompt_size=args.mask_prompt_size, device=device).to(device)
    mask_hw = tuple(model.sam_prompt_encoder.mask_input_size)
    print(f"mask_input_size={mask_hw}")
    opt = torch.optim.AdamW(model.param_groups(args.base_lr, encoder_lr_mult=0.1), lr=args.base_lr,
                            weight_decay=1e-4)
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    total_steps = max(1, args.epochs * len(tr)); warmup = max(1, 2 * len(tr))

    log = {"args": vars(args), "history": []}
    best = -1.0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        for it, batch in enumerate(tr):
            step = ep * len(tr) + it
            scale = (step / warmup) if step < warmup else 0.5 * (1 + math.cos(
                math.pi * (step - warmup) / max(1, total_steps - warmup)))
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * scale
            img = batch["image"].to(device)
            target = (batch["mask"] > 0.5).float().to(device)
            dino = batch["dino"].to(device) if "dino" in batch else None
            c, l, pm = build_prompts(batch, args.prompt_mode, device, mask_hw)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = run_model(model, img, dino, c, l, pm)
                    loss = dice_bce_loss(logits.float(), target,
                                         alpha=args.tversky_alpha, beta=args.tversky_beta)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                scaler.step(opt); scaler.update()
            else:
                logits = run_model(model, img, dino, c, l, pm)
                loss = dice_bce_loss(logits.float(), target)
                loss.backward(); opt.step()
            run += float(loss.detach()); nb += 1
        ev = evaluate(model, va, args.prompt_mode, device, mask_hw)
        ev["train_loss"] = run / max(nb, 1)
        print(f"ep{ep+1}/{args.epochs} loss={ev['train_loss']:.4f} craq_iou={ev['craq_iou']:.4f} "
              f"f1={ev['craq_f1']:.4f} P={ev['precision']:.3f} R={ev['recall']:.3f} {time.time()-t0:.1f}s",
              flush=True)
        log["history"].append({"epoch": ep + 1, **ev})
        json.dump(log, open(out / "log.json", "w"), indent=2)
        torch.save({"epoch": ep + 1, "model": model.state_dict(), "args": vars(args), "val": ev},
                   out / "last.pt")
        if ev["craq_iou"] > best:
            best = ev["craq_iou"]
            torch.save({"epoch": ep + 1, "model": model.state_dict(), "args": vars(args), "val": ev},
                       out / "best.pt")
            json.dump({"best_epoch": ep + 1, **ev}, open(out / "metrics.json", "w"), indent=2)
            print(f"  [best] craq_iou={best:.4f}")
    print(f"done mode={args.prompt_mode} best_craq_iou={best:.4f}")


if __name__ == "__main__":
    main()
