"""Train DADNet for binary craquelure segmentation on 224x224 chunks.

Self-contained (no crackseg_common dependency) so dadnet_env stays isolated.
Reads the same tiles_root + group_split_stem.json (fold0 = holdout) as the other models.
Paper defaults: Adam lr 1e-4, batch 16, cross-entropy loss, 224x224.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dadnet_model import DADNet

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ChunkDS(Dataset):
    def __init__(self, tiles_root, names, train=False):
        self.img = Path(tiles_root) / "images"
        self.msk = Path(tiles_root) / "masks"
        self.names = names
        self.train = train

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        img = np.array(Image.open(self.img / n).convert("RGB"))
        msk = (np.array(Image.open(self.msk / n)) > 0).astype(np.int64)
        if self.train:
            if np.random.rand() < 0.5:
                img = img[:, ::-1].copy(); msk = msk[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img = img[::-1].copy(); msk = msk[::-1].copy()
        x = torch.from_numpy(img).float().div_(255).permute(2, 0, 1)
        m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (x - m) / s, torch.from_numpy(msk)


def load_split(path, fold):
    payload = json.load(open(path))
    fd = payload["folds"][fold]
    return fd["train"], fd["val"]


def exists_only(tiles_root, names):
    img = Path(tiles_root) / "images"
    return [n for n in names if (img / n).exists()]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    for x, m in loader:
        x = x.to(device); m = m.to(device)
        pred = model(x).argmax(1)
        gt = (m == 1)
        pr = (pred == 1)
        tp += int((pr & gt).sum()); fp += int((pr & ~gt).sum()); fn += int((~pr & gt).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"craq_iou": iou, "craq_f1": f1, "precision": prec, "recall": rec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--output_dir", default="runs/dadnet")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)

    tr_names, va_names = load_split(args.split, args.fold)
    tr_names = exists_only(args.tiles_root, tr_names)
    va_names = exists_only(args.tiles_root, va_names)
    print(f"train={len(tr_names)} val={len(va_names)}")

    tr = DataLoader(ChunkDS(args.tiles_root, tr_names, train=True), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    va = DataLoader(ChunkDS(args.tiles_root, va_names, train=False), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = DADNet(num_classes=2, k=7, dilation=7, pretrained=not args.no_pretrained).to(device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"DADNet params={n_par:.1f}M")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    log = {"args": vars(args), "history": []}
    best_iou = -1.0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        for x, m in tr:
            x = x.to(device); m = m.to(device)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = crit(model(x), m)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss = crit(model(x), m); loss.backward(); opt.step()
            run += float(loss.detach()); nb += 1
        ev = evaluate(model, va, device)
        ev["train_loss"] = run / max(nb, 1)
        print(f"ep{ep+1}/{args.epochs} loss={ev['train_loss']:.4f} "
              f"craq_iou={ev['craq_iou']:.4f} f1={ev['craq_f1']:.4f} "
              f"P={ev['precision']:.3f} R={ev['recall']:.3f} {time.time()-t0:.1f}s", flush=True)
        log["history"].append({"epoch": ep + 1, **ev})
        json.dump(log, open(out / "log.json", "w"), indent=2)
        torch.save({"epoch": ep + 1, "model": model.state_dict(), "args": vars(args), "val": ev},
                   out / "last.pt")
        if ev["craq_iou"] > best_iou:
            best_iou = ev["craq_iou"]
            torch.save({"epoch": ep + 1, "model": model.state_dict(), "args": vars(args), "val": ev},
                       out / "best.pt")
            json.dump({"best_epoch": ep + 1, **ev}, open(out / "metrics.json", "w"), indent=2)
            print(f"  [best] craq_iou={best_iou:.4f}")
    print(f"done best_craq_iou={best_iou:.4f}")


if __name__ == "__main__":
    main()
