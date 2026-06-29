"""Train SAM2 decoder-only (no prompt) for binary segmentation.

Uses SAM2DecoderSeg (model_decoder_seg.py): drops the prompt encoder entirely
and reuses the pretrained SAM2 mask decoder as a segmentation head. Same output
口徑 as train_prompt.py ([B,1,H,W] binary logits + BCE+Dice), so the loss / loop /
checkpoint logic is identical; only the model build + a couple of args differ.

Example (craquelure expert, no prompt):
    python train_decoder.py \
        --tiles_root data/labeled32_craq_v3/tiles_512 \
        --split data/labeled32_craq_v3/tiles_512/group_split_stem.json \
        --fold 0 --variant small --epochs 80 --batch_size 4 \
        --dense_mode learnable --num_queries 0 \
        --class_names background,craquelure \
        --output_dir outputs/decoder_craq_fold0_small
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from crackseg_common.augment import train_transforms, val_transforms
import crackseg_common.dataset as _dataset
from crackseg_common.dataset import TileSegDataset, compute_class_weights, load_tile_index, set_class_names
from model_decoder_seg import SAM2DecoderSeg, count_params


# --------------- loss ---------------

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = logits.sigmoid()
        inter = (prob * target).sum(dim=(1, 2, 3))
        union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()


class BinaryCEDiceLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, pos_weight=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.dice = BinaryDiceLoss()

    def forward(self, logits, target_idx):
        # logits: [B, 1, H, W], target_idx: [B, H, W] with values 0/1
        target = (target_idx > 0).float().unsqueeze(1)  # [B, 1, H, W]
        ce = self.bce(logits, target)
        dice = self.dice(logits, target)
        loss = self.ce_weight * ce + self.dice_weight * dice
        return loss, {"ce": float(ce.detach()), "dice": float(dice.detach())}


# --------------- eval ---------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        msk = batch["mask"].to(device, non_blocking=True)
        logits = model(img)  # [B, 1, H, W]
        pred = (logits.squeeze(1) > 0).long()
        gt = (msk > 0).long()
        tp += ((pred == 1) & (gt == 1)).sum().item()
        fp += ((pred == 1) & (gt == 0)).sum().item()
        fn += ((pred == 0) & (gt == 1)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}


# --------------- train ---------------

def cosine_with_warmup(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train_one_epoch(model, loader, optimizer, scaler, criterion, device,
                    epoch, total_epochs, total_steps, warmup_steps,
                    base_lrs, log_interval=20):
    model.train()
    running = {"loss": 0.0, "ce": 0.0, "dice": 0.0, "n": 0}
    step_offset = epoch * len(loader)
    t0 = time.time()
    for it, batch in enumerate(loader):
        global_step = step_offset + it
        scale = cosine_with_warmup(global_step, total_steps, warmup_steps)
        for g, lr in zip(optimizer.param_groups, base_lrs):
            g["lr"] = lr * scale

        img = batch["image"].to(device, non_blocking=True)
        msk = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(img)
                loss, parts = criterion(logits, msk)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(img)
            loss, parts = criterion(logits, msk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=5.0)
            optimizer.step()

        bs = img.size(0)
        running["loss"] += float(loss.detach()) * bs
        running["ce"] += parts["ce"] * bs
        running["dice"] += parts["dice"] * bs
        running["n"] += bs

        if (it + 1) % log_interval == 0 or (it + 1) == len(loader):
            n = running["n"]
            print(f"  ep{epoch+1}/{total_epochs} it{it+1}/{len(loader)} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} "
                  f"loss={running['loss']/n:.4f} ce={running['ce']/n:.4f} "
                  f"dice={running['dice']/n:.4f} "
                  f"{(time.time()-t0):.1f}s", flush=True)
    n = max(1, running["n"])
    return {"loss": running["loss"]/n, "ce": running["ce"]/n, "dice": running["dice"]/n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--variant", default="small")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_queries", type=int, default=0,
                        help="變體 B:>0 加 learnable query tokens 當 sparse;0 = 純空 prompt")
    parser.add_argument("--dense_mode", default="learnable",
                        choices=["zero", "learnable", "image"],
                        help="dense_prompt 來源:zero / learnable 全域 param / image-conditioned conv")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--base_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ce_weight", type=float, default=0.5)
    parser.add_argument("--dice_weight", type=float, default=0.5)
    parser.add_argument("--class_weight_mode", default="median_freq",
                        choices=["median_freq", "inv_sqrt", "none"])
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/decoder_run")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--class_names", default=None)
    args = parser.parse_args()

    if args.class_names:
        names = [s.strip() for s in args.class_names.split(",") if s.strip()]
        set_class_names(names)

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # data
    tile_index = load_tile_index(args.tiles_root)
    with open(args.split) as f:
        payload = json.load(f)
    folds = payload["folds"]
    fd = folds[args.fold]
    train_names, val_names = fd["train"], fd["val"]

    by_name = {it["tile"]: it for it in tile_index["items"]}
    train_items = [by_name[n] for n in train_names if n in by_name]
    val_items = [by_name[n] for n in val_names if n in by_name]
    print(f"fold={args.fold} train={len(train_items)} val={len(val_items)}")

    train_ds = TileSegDataset(args.tiles_root, train_items,
                              transforms=train_transforms(image_size=args.image_size))
    val_ds = TileSegDataset(args.tiles_root, val_items,
                            transforms=val_transforms(image_size=args.image_size))
    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size), shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    # pos_weight from class frequency
    pos_weight = None
    if args.class_weight_mode != "none":
        cw, counts = compute_class_weights(train_items, args.tiles_root,
                                           num_classes=_dataset.NUM_CLASSES,
                                           mode=args.class_weight_mode)
        if len(counts) >= 2 and counts[1] > 0:
            pos_weight = float(counts[0]) / float(counts[1])
            pos_weight = min(pos_weight, 100.0)
        print(f"pixel counts: {counts.tolist()}, pos_weight={pos_weight}")

    # model
    model = SAM2DecoderSeg(
        variant=args.variant, image_size=args.image_size,
        num_queries=args.num_queries, dense_mode=args.dense_mode, device=device,
    ).to(device)
    total, trainable = count_params(model)
    print(f"SAM2DecoderSeg variant={args.variant} num_queries={args.num_queries} "
          f"dense_mode={args.dense_mode} "
          f"total={total/1e6:.1f}M trainable={trainable/1e6:.2f}M")

    groups = model.param_groups(base_lr=args.base_lr)
    base_lrs = [g["lr"] for g in groups]
    optimizer = torch.optim.AdamW(groups, lr=args.base_lr, weight_decay=args.weight_decay)

    criterion = BinaryCEDiceLoss(
        ce_weight=args.ce_weight, dice_weight=args.dice_weight,
        pos_weight=pos_weight,
    ).to(device)
    scaler = None if args.no_amp or device != "cuda" else torch.amp.GradScaler("cuda")

    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = max(1, args.warmup_epochs * len(train_loader))

    log = {"args": vars(args), "history": []}
    best_iou = -1.0

    for epoch in range(args.epochs):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, criterion,
                             device, epoch, args.epochs, total_steps, warmup_steps,
                             base_lrs, log_interval=args.log_interval)

        ev = evaluate(model, val_loader, device)
        print(f"[val] ep{epoch+1} IoU={ev['iou']:.4f} F1={ev['f1']:.4f} "
              f"P={ev['precision']:.4f} R={ev['recall']:.4f}", flush=True)

        log["history"].append({"epoch": epoch + 1, "train": tr, "val": ev})
        with open(out_dir / "log.json", "w") as f:
            json.dump(log, f, indent=2)

        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args), "val": ev,
        }, out_dir / "last.pt")

        if ev["iou"] > best_iou:
            best_iou = ev["iou"]
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "args": vars(args), "val": ev,
            }, out_dir / "best.pt")
            print(f"[best] ep{epoch+1} IoU={best_iou:.4f}")

    print(f"done best_iou={best_iou:.4f}")


if __name__ == "__main__":
    main()
