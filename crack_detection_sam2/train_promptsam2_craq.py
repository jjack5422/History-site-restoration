"""train_promptsam2_craq.py — 4-fold 訓練 PromptedSAM2Seg(craquelure),GT 中軸點當 prompt。"""
import argparse, json, math, os, time
import numpy as np, torch
from torch.utils.data import DataLoader
from augment import train_transforms, val_transforms
import dataset as _dataset
from dataset import TileSegDataset, compute_class_weights, load_tile_index, set_class_names
from train_prompt import BinaryCEDiceLoss
from model_prompted_sam2 import PromptedSAM2Seg
from gt_points import gt_points

TILES = "data/labeled32_craq_v3/tiles_512"
SPLIT = "data/labeled32_craq_v3/tiles_512/group_split_stem.json"


def build_prompts(masks, n_points, image_size, device):
    per = []
    arr = masks.detach().cpu().numpy()
    for i in range(arr.shape[0]):
        pts, labs = gt_points(arr[i] > 0, n_points)
        if pts.shape[0] == 0:
            pts = np.array([[image_size / 2, image_size / 2]], np.float32)
            labs = np.ones(1, np.int64)
        per.append((pts, labs))
    maxk = max(p[0].shape[0] for p in per)
    B = len(per)
    coords = np.zeros((B, maxk, 2), np.float32)
    labels = -np.ones((B, maxk), np.int64)
    for i, (pts, labs) in enumerate(per):
        k = pts.shape[0]; coords[i, :k] = pts; labels[i, :k] = labs
    return (torch.from_numpy(coords).to(device), torch.from_numpy(labels).to(device))


def cosine_with_warmup(step, total, warm):
    if step < warm:
        return step / max(1, warm)
    prog = (step - warm) / max(1, total - warm)
    return 0.5 * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, loader, n_points, image_size, device):
    model.eval(); tp = fp = fn = 0
    for batch in loader:
        img = batch["image"].to(device); msk = batch["mask"].to(device)
        coords, labels = build_prompts(msk, n_points, image_size, device)
        logits = model(img, coords, labels)
        pred = (logits.squeeze(1) > 0).long(); gt = (msk > 0).long()
        tp += ((pred == 1) & (gt == 1)).sum().item()
        fp += ((pred == 1) & (gt == 0)).sum().item()
        fn += ((pred == 0) & (gt == 1)).sum().item()
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"iou": tp / max(tp + fp + fn, 1), "f1": 2 * p * r / max(p + r, 1e-8),
            "precision": p, "recall": r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--variant", default="small")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--n_points", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--base_lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_prefix", default="outputs/promptsam2_craq")
    args = ap.parse_args()
    set_class_names(["background", "craquelure"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42); np.random.seed(42)

    tile_index = load_tile_index(TILES)
    by_name = {it["tile"]: it for it in tile_index["items"]}
    payload = json.load(open(SPLIT))
    msk_dir = os.path.join(TILES, "masks")

    def has_fg(tile_name):
        import cv2
        stem = os.path.splitext(tile_name)[0]
        m = cv2.imread(os.path.join(msk_dir, stem + ".png"), 0)
        return m is not None and (m > 0).any()

    for k in args.folds:
        fd = payload["folds"][k]
        tr_items = [by_name[n] for n in fd["train"] if n in by_name and has_fg(n)]
        va_items = [by_name[n] for n in fd["val"] if n in by_name]
        print(f"fold{k}: train_fg={len(tr_items)} val={len(va_items)}", flush=True)
        tr_ds = TileSegDataset(TILES, tr_items, transforms=train_transforms(image_size=args.image_size))
        va_ds = TileSegDataset(TILES, va_items, transforms=val_transforms(image_size=args.image_size))
        tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
        va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        cw, counts = compute_class_weights(tr_items, TILES, num_classes=_dataset.NUM_CLASSES, mode="median_freq")
        pos_w = min(float(counts[0]) / max(float(counts[1]), 1), 100.0) if len(counts) >= 2 else None
        model = PromptedSAM2Seg(variant=args.variant, image_size=args.image_size, device=device).to(device)
        groups = model.param_groups(args.base_lr); base_lrs = [g["lr"] for g in groups]
        opt = torch.optim.AdamW(groups, lr=args.base_lr, weight_decay=1e-4)
        crit = BinaryCEDiceLoss(0.5, 0.5, pos_weight=pos_w).to(device)
        scaler = torch.amp.GradScaler("cuda")
        total = max(1, args.epochs * len(tr)); warm = max(1, args.warmup_epochs * len(tr))
        out = f"{args.out_prefix}_fold{k}"; os.makedirs(out, exist_ok=True)
        best = -1; log = {"history": []}
        for ep in range(args.epochs):
            model.train(); t0 = time.time(); run = 0; nb = 0
            for it, batch in enumerate(tr):
                gs = ep * len(tr) + it; sc = cosine_with_warmup(gs, total, warm)
                for g, lr in zip(opt.param_groups, base_lrs): g["lr"] = lr * sc
                img = batch["image"].to(device); msk = batch["mask"].to(device)
                coords, labels = build_prompts(msk, args.n_points, args.image_size, device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(img, coords, labels)
                    loss, _ = crit(logits, msk)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                scaler.step(opt); scaler.update()
                run += float(loss.detach()); nb += 1
            ev = evaluate(model, va, args.n_points, args.image_size, device)
            print(f"fold{k} ep{ep+1}/{args.epochs} loss={run/max(nb,1):.4f} "
                  f"valF1={ev['f1']:.4f} IoU={ev['iou']:.4f} {(time.time()-t0):.1f}s", flush=True)
            log["history"].append({"epoch": ep + 1, "val": ev})
            json.dump(log, open(os.path.join(out, "log.json"), "w"))
            if ev["iou"] > best:
                best = ev["iou"]
                torch.save({"model": model.state_dict(), "epoch": ep + 1, "val": ev,
                            "args": vars(args)}, os.path.join(out, "best.pt"))
        print(f"fold{k} done best_iou={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
