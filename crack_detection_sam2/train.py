"""訓練 SAM2 + FPN seg head 的多類別語意分割。

Example:
    python train.py \\
        --tiles_root data/tiles_512 \\
        --split data/tiles_512/group_split_stem.json \\
        --fold 0 \\
        --variant small \\
        --epochs 60 \\
        --batch_size 4 \\
        --output_dir outputs/run_stem_fold0_small
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
from torch.utils.data import DataLoader

from augment import train_transforms, val_transforms
import dataset as _dataset
from dataset import (TileSegDataset, compute_class_weights, load_tile_index,
                     set_class_names)
from losses import CEDiceLoss
from metrics import ConfusionMeter, format_metrics
from model_seg import SAM2SemSeg, count_params
from model_seg_full_fpn import SAM2SemSegFullFPN


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(path: str, fold: int):
    with open(path) as f:
        payload = json.load(f)
    folds = payload["folds"]
    if fold < 0 or fold >= len(folds):
        raise ValueError(f"fold={fold} 超出範圍 0..{len(folds)-1}")
    fd = folds[fold]
    return fd["train"], fd["val"], payload, fd


def items_from_index(tile_index, names):
    by_name = {it["tile"]: it for it in tile_index["items"]}
    return [by_name[n] for n in names if n in by_name]


def cosine_with_warmup(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    meter = ConfusionMeter(num_classes)
    total_loss = 0.0
    n_batches = 0
    ce_loss = nn.CrossEntropyLoss()
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        msk = batch["mask"].to(device, non_blocking=True)
        logits = model(img)
        loss = ce_loss(logits, msk)
        pred = logits.argmax(1)
        meter.update(pred, msk)
        total_loss += float(loss.detach())
        n_batches += 1
    res = meter.compute(class_names=_dataset.CLASS_NAMES, ignore_index=0)
    res["val_ce_loss"] = total_loss / max(1, n_batches)
    return res


def train_one_epoch(model, loader, optimizer, scaler, criterion, device,
                    epoch, total_epochs, total_steps, warmup_steps,
                    base_lrs, log_interval=10):
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
                  f"loss={running['loss']/n:.4f} ce={running['ce']/n:.4f} dice={running['dice']/n:.4f} "
                  f"{(time.time()-t0):.1f}s", flush=True)
    n = max(1, running["n"])
    return {"loss": running["loss"]/n, "ce": running["ce"]/n, "dice": running["dice"]/n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_root", default="data/tiles_512")
    parser.add_argument("--split", default="data/tiles_512/group_split_stem.json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--variant", default="small",
                        choices=["tiny", "small", "base", "large"])
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--base_lr", type=float, default=3e-4)
    parser.add_argument("--encoder_lr_mult", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ce_weight", type=float, default=0.5)
    parser.add_argument("--dice_weight", type=float, default=0.5)
    parser.add_argument("--cldice_weight", type=float, default=0.0,
                        help="centerline Dice loss weight; 0 = off")
    parser.add_argument("--cldice_iters", type=int, default=3,
                        help="soft skeletonize iterations")
    parser.add_argument("--focal_weight", type=float, default=0.0,
                        help="focal loss weight; 0 = off")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--class_weight_mode", default="median_freq",
                        choices=["median_freq", "inv_sqrt", "none"])
    parser.add_argument("--freeze_trunk", action="store_true", default=True)
    parser.add_argument("--no_freeze_trunk", dest="freeze_trunk", action="store_false")
    parser.add_argument("--freeze_neck", action="store_true", default=False)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/run")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--class_names", default=None,
                        help="逗號分隔的類別名稱 (含背景), e.g. 'background,crack,craquelure'。"
                             "未指定則用 dataset.py 預設 5 類。")
    parser.add_argument("--head_type", default="simple",
                        choices=["simple", "full_fpn"],
                        help="simple = FPNSegHead (concat); full_fpn = Semantic FPN (add)")
    args = parser.parse_args()

    if args.class_names:
        names = [s.strip() for s in args.class_names.split(",") if s.strip()]
        set_class_names(names)
        print(f"override CLASS_NAMES={names} (NUM_CLASSES={_dataset.NUM_CLASSES})")
    CLASS_NAMES = _dataset.CLASS_NAMES
    NUM_CLASSES = _dataset.NUM_CLASSES

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # data
    tile_index = load_tile_index(args.tiles_root)
    train_names, val_names, payload, fold_info = load_split(args.split, args.fold)
    train_items = items_from_index(tile_index, train_names)
    val_items = items_from_index(tile_index, val_names)
    print(f"split={args.split} fold={args.fold} train={len(train_items)} val={len(val_items)}")
    print(f"val_groups={fold_info.get('val_groups')}")

    train_ds = TileSegDataset(args.tiles_root, train_items,
                              transforms=train_transforms(image_size=args.image_size))
    val_ds = TileSegDataset(args.tiles_root, val_items,
                            transforms=val_transforms(image_size=args.image_size))
    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size), shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    # class weights from train items
    if args.class_weight_mode == "none":
        class_weights = None
    else:
        cw, counts = compute_class_weights(train_items, args.tiles_root,
                                           num_classes=NUM_CLASSES,
                                           mode=args.class_weight_mode)
        class_weights = cw.to(device)
        print(f"class pixel counts (train): {counts.tolist()}")
        print(f"class weights ({args.class_weight_mode}): {[round(float(v),4) for v in cw]}")

    # model
    ModelCls = SAM2SemSegFullFPN if args.head_type == "full_fpn" else SAM2SemSeg
    model = ModelCls(variant=args.variant, num_classes=NUM_CLASSES,
                     freeze_trunk=args.freeze_trunk, freeze_neck=args.freeze_neck,
                     device=device).to(device)
    total, trainable = count_params(model)
    print(f"model={ModelCls.__name__} variant={args.variant} "
          f"total={total/1e6:.1f}M trainable={trainable/1e6:.2f}M")

    # optimizer
    groups = model.param_groups(base_lr=args.base_lr, encoder_lr_mult=args.encoder_lr_mult)
    base_lrs = [g["lr"] for g in groups]
    optimizer = torch.optim.AdamW(groups, lr=args.base_lr,
                                  weight_decay=args.weight_decay)

    # loss / scaler
    criterion = CEDiceLoss(num_classes=NUM_CLASSES, class_weights=class_weights,
                           ce_weight=args.ce_weight, dice_weight=args.dice_weight,
                           cldice_weight=args.cldice_weight, cldice_iters=args.cldice_iters,
                           focal_weight=args.focal_weight, focal_gamma=args.focal_gamma,
                           ignore_index_in_dice=0).to(device)
    scaler = None if args.no_amp or device != "cuda" else torch.amp.GradScaler("cuda")

    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = max(1, args.warmup_epochs * len(train_loader))

    log = {"args": vars(args), "history": []}
    best_miou = -1.0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    for epoch in range(args.epochs):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, criterion,
                             device, epoch, args.epochs,
                             total_steps, warmup_steps, base_lrs,
                             log_interval=args.log_interval)
        if len(val_loader) > 0:
            ev = evaluate(model, val_loader, device, NUM_CLASSES)
            print(f"[val] ep{epoch+1} mIoU={ev['miou']:.4f} mDice={ev['mdice']:.4f} "
                  f"pixel_acc={ev['pixel_accuracy']:.4f} ce={ev['val_ce_loss']:.4f}")
            print(format_metrics(ev))
            miou = ev["miou"]
        else:
            ev = {"miou": float("nan")}
            miou = -1.0

        log["history"].append({"epoch": epoch + 1, "train": tr, "val": ev})
        with open(out_dir / "log.json", "w") as f:
            json.dump(log, f, indent=2)

        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "val": ev,
        }, last_path)

        if not math.isnan(miou) and miou > best_miou:
            best_miou = miou
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "args": vars(args),
                "val": ev,
                "best_miou": best_miou,
            }, best_path)
            print(f"[best] ep{epoch+1} mIoU={best_miou:.4f} -> {best_path}")

    print(f"訓練結束 best_miou={best_miou:.4f}")


if __name__ == "__main__":
    main()
