"""Train a click-promptable SAM2 expert (binary segmentation) with SAM-style iterative clicks.

Reuses PromptedSAM2Seg (model_prompted_sam2.py), the loss/schedule from train_prompt.py, and the
shared dataset/augment. Each tile is annotated by simulating `n_clicks` clicks: a positive seed,
then correction points sampled from the error region. Loss is averaged over clicks; eval reports
IoU@{1,3,5} clicks and NoC@0.8.

Example (craquelure expert, fold 0):
    /home/zzz90/research/sam2_env/bin/python train_click.py \
        --tiles_root /home/zzz90/research/_data/labeled32_craq_v3/tiles_512 \
        --split /home/zzz90/research/_data/labeled32_craq_v3/tiles_512/group_split_stem.json \
        --fold 0 --variant small --image_size 512 --epochs 80 --batch_size 2 --n_clicks 8 \
        --class_names background,craquelure \
        --output_dir runs/2026-06-09-click-craq-4fold/fold0
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from crackseg_common.augment import train_transforms, val_transforms
import crackseg_common.dataset as _dataset
from crackseg_common.dataset import TileSegDataset, compute_class_weights, load_tile_index, set_class_names
from model_prompted_sam2 import PromptedSAM2Seg
from train_prompt import BinaryCEDiceLoss, cosine_with_warmup
from click_sampling import mask_iou, sample_initial_point, sample_correction_point


def batch_point_tensors(pts, labs, device):
    """pts: list (len B) of lists of (row, col); labs: same shape of int labels.
    All rows have equal length N. Returns coords [B,N,2] as (x,y) float, labels [B,N] int32."""
    B = len(pts); N = len(pts[0])
    coords = torch.zeros(B, N, 2)
    labels = torch.zeros(B, N, dtype=torch.int32)
    for b in range(B):
        for i, (r, c) in enumerate(pts[b]):
            coords[b, i, 0] = float(c)   # x = col
            coords[b, i, 1] = float(r)   # y = row
            labels[b, i] = int(labs[b][i])
    return coords.to(device), labels.to(device)


def _init_points(gt_np, rng):
    B = gt_np.shape[0]
    pts = [[] for _ in range(B)]; labs = [[] for _ in range(B)]
    for b in range(B):
        (r, c), l = sample_initial_point(gt_np[b], rng)
        pts[b].append((r, c)); labs[b].append(l)
    return pts, labs


def _append_correction(pts, labs, pred_np, gt_np, rng):
    for b in range(pred_np.shape[0]):
        nxt = sample_correction_point(pred_np[b], gt_np[b], rng)
        if nxt is None:                       # already perfect -> repeat last click (uniform N)
            pts[b].append(pts[b][-1]); labs[b].append(labs[b][-1])
        else:
            (r, c), l = nxt; pts[b].append((r, c)); labs[b].append(l)


def clicks_train_loss(model, img, gt, n_clicks, rng, criterion):
    """Run n_clicks iterations on a batch, return loss averaged over clicks (graph retained)."""
    enc = model.encode_image(img)
    gt_np = (gt > 0).cpu().numpy()
    pts, labs = _init_points(gt_np, rng)
    prev = None
    total = 0.0
    for k in range(n_clicks):
        coords, labels = batch_point_tensors(pts, labs, img.device)
        masks, low = model.decode(enc, coords, labels, prev_mask=prev)
        loss, _ = criterion(masks, gt)
        total = total + loss
        # detach: truncated BPTT (SAM-style). Each click's loss backprops only through its own
        # decode; gradients do not flow across clicks. Intentional -- do not remove the detach.
        prev = low.detach()
        if k < n_clicks - 1:
            pred_np = (masks.detach().squeeze(1) > 0).cpu().numpy()  # CPU sync each click: scipy click sampling needs numpy
            _append_correction(pts, labs, pred_np, gt_np, rng)
    return total / n_clicks


def aggregate_click_metrics(per_click_iou, reached, n_clicks, iou_target=0.8):
    """reached: per-sample click count at which IoU first reached iou_target, or None if never.
    Keys are labeled for the default iou_target=0.8 (the only value the trainer uses)."""
    def at(k):
        k = min(k, n_clicks)
        vals = per_click_iou[k - 1]
        return float(np.mean(vals)) if vals else 0.0
    noc = [(r if r is not None else n_clicks) for r in reached]
    return {
        "iou@1": at(1), "iou@3": at(3), "iou@5": at(5),
        "noc@0.8": float(np.mean(noc)) if noc else float(n_clicks),
        "cap_rate": float(np.mean([r is None for r in reached])) if reached else 1.0,
    }


@torch.no_grad()
def evaluate_clicks(model, loader, device, n_clicks, iou_target=0.8, seed=0):
    model.eval()
    rng = np.random.default_rng(seed)
    per_click_iou = [[] for _ in range(n_clicks)]
    reached_all = []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        gt = batch["mask"].to(device, non_blocking=True)
        B = img.size(0)
        enc = model.encode_image(img)
        gt_np = (gt > 0).cpu().numpy()
        pts, labs = _init_points(gt_np, rng)
        prev = None
        reached = [None] * B
        for k in range(n_clicks):
            coords, labels = batch_point_tensors(pts, labs, device)
            masks, low = model.decode(enc, coords, labels, prev_mask=prev)
            prev = low
            pred_np = (masks.squeeze(1) > 0).cpu().numpy()
            for b in range(B):
                iou = mask_iou(pred_np[b], gt_np[b])
                per_click_iou[k].append(iou)
                if reached[b] is None and iou >= iou_target:
                    reached[b] = k + 1
            if k < n_clicks - 1:
                _append_correction(pts, labs, pred_np, gt_np, rng)
        reached_all.extend(reached)
    return aggregate_click_metrics(per_click_iou, reached_all, n_clicks, iou_target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiles_root", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--variant", default="small")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--n_clicks", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--base_lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ce_weight", type=float, default=0.5)
    p.add_argument("--dice_weight", type=float, default=0.5)
    p.add_argument("--class_weight_mode", default="median_freq",
                   choices=["median_freq", "inv_sqrt", "none"])
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_items", type=int, default=0, help="0 = all; >0 caps for sanity runs")
    p.add_argument("--output_dir", default="outputs/click_run")
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--class_names", default=None)
    args = p.parse_args()

    if args.class_names:
        set_class_names([s.strip() for s in args.class_names.split(",") if s.strip()])

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    tile_index = load_tile_index(args.tiles_root)
    with open(args.split) as f:
        fd = json.load(f)["folds"][args.fold]
    by_name = {it["tile"]: it for it in tile_index["items"]}
    train_items = [by_name[n] for n in fd["train"] if n in by_name]
    val_items = [by_name[n] for n in fd["val"] if n in by_name]
    if args.max_train_items > 0:
        train_items = train_items[:args.max_train_items]
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

    pos_weight = None
    if args.class_weight_mode != "none":
        _, counts = compute_class_weights(train_items, args.tiles_root,
                                          num_classes=_dataset.NUM_CLASSES,
                                          mode=args.class_weight_mode)
        if len(counts) >= 2 and counts[1] > 0:
            pos_weight = min(float(counts[0]) / float(counts[1]), 100.0)
        print(f"pixel counts: {counts.tolist()}, pos_weight={pos_weight}")

    model = PromptedSAM2Seg(variant=args.variant, image_size=args.image_size, device=device).to(device)
    groups = model.param_groups(base_lr=args.base_lr)
    base_lrs = [g["lr"] for g in groups]
    optimizer = torch.optim.AdamW(groups, lr=args.base_lr, weight_decay=args.weight_decay)
    criterion = BinaryCEDiceLoss(ce_weight=args.ce_weight, dice_weight=args.dice_weight,
                                 pos_weight=pos_weight).to(device)
    scaler = None if args.no_amp or device != "cuda" else torch.amp.GradScaler("cuda")

    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = max(1, args.warmup_epochs * len(train_loader))
    log = {"args": vars(args), "history": []}
    best = -1.0
    rng = np.random.default_rng(args.seed)

    for epoch in range(args.epochs):
        model.train()
        run_loss = 0.0; n = 0; t0 = time.time()
        for it, batch in enumerate(train_loader):
            scale = cosine_with_warmup(epoch * len(train_loader) + it, total_steps, warmup_steps)
            for g, lr in zip(optimizer.param_groups, base_lrs):
                g["lr"] = lr * scale
            img = batch["image"].to(device, non_blocking=True)
            gt = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = clicks_train_loss(model, img, gt, args.n_clicks, rng, criterion)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss = clicks_train_loss(model, img, gt, args.n_clicks, rng, criterion)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                optimizer.step()
            run_loss += float(loss.detach()) * img.size(0); n += img.size(0)
            if (it + 1) % args.log_interval == 0 or (it + 1) == len(train_loader):
                print(f"  ep{epoch+1}/{args.epochs} it{it+1}/{len(train_loader)} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e} loss={run_loss/max(1,n):.4f} "
                      f"{(time.time()-t0):.1f}s", flush=True)

        ev = evaluate_clicks(model, val_loader, device, args.n_clicks, seed=args.seed)
        print(f"[val] ep{epoch+1} IoU@1={ev['iou@1']:.3f} IoU@3={ev['iou@3']:.3f} "
              f"IoU@5={ev['iou@5']:.3f} NoC@0.8={ev['noc@0.8']:.2f} cap={ev['cap_rate']:.2f}", flush=True)
        log["history"].append({"epoch": epoch + 1, "train_loss": run_loss / max(1, n), "val": ev})
        with open(out_dir / "log.json", "w") as f:
            json.dump(log, f, indent=2)
        torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                    "args": vars(args), "val": ev}, out_dir / "last.pt")
        if ev["iou@5"] > best:
            best = ev["iou@5"]
            torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                        "args": vars(args), "val": ev}, out_dir / "best.pt")
            print(f"[best] ep{epoch+1} IoU@5={best:.4f}")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"best_iou@5": best, "last_val": log["history"][-1]["val"]}, f, indent=2)
    print(f"done best_iou@5={best:.4f}")


if __name__ == "__main__":
    main()
