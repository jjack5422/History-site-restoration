"""聚合 C0 vs E1 的 5-fold metrics.json (mean±std),並對每個 best.pt 做 threshold sweep。
baseline 與 fused 共用:依 ckpt args 內有無 dino_feat_dir 自動切模型。"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_prompted_sam2 import PromptedSAM2Seg

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def agg(prefix):
    ious, ps, rs = [], [], []
    for k in range(5):
        mp = Path(f"runs/{prefix}-fold{k}-2026-06-11/metrics.json")
        if not mp.exists():
            print(f"  missing {mp}"); continue
        d = json.load(open(mp))
        ious.append(d["craq_iou"]); ps.append(d["precision"]); rs.append(d["recall"])
    f = lambda a: (float(np.mean(a)), float(np.std(a)))
    return {"n": len(ious), "iou": f(ious), "precision": f(ps), "recall": f(rs)}


def sweep(ckpt, tiles_root, split, fold, prob_dir, dino_dir, thresholds):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location=dev)
    use_dino = bool(ck["args"].get("dino_feat_dir"))
    if use_dino:
        from model_fused_sam2 import FusedPromptedSAM2Seg
        model = FusedPromptedSAM2Seg(variant=ck["args"]["variant"], image_size=ck["args"]["image_size"],
                                     dino_dim=ck["args"].get("dino_dim", 384),
                                     mask_prompt_size=ck["args"].get("mask_prompt_size"), device=dev).to(dev)
    else:
        model = PromptedSAM2Seg(variant=ck["args"]["variant"], image_size=ck["args"]["image_size"],
                                mask_prompt_size=ck["args"].get("mask_prompt_size"), device=dev).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    mask_hw = tuple(model.sam_prompt_encoder.mask_input_size)
    va = json.load(open(split))["folds"][fold]["val"]
    timg = Path(tiles_root) / "images"; tmsk = Path(tiles_root) / "masks"; prob_d = Path(prob_dir) / "prob"
    dino_p = Path(dino_dir) if (use_dino and dino_dir) else None
    stats = {t: [0, 0, 0] for t in thresholds}  # tp,fp,fn
    for name in va:
        ip = timg / name; pp = prob_d / (Path(name).stem + ".npy")
        if not (ip.exists() and pp.exists()):
            continue
        img = np.array(Image.open(ip).convert("RGB"))
        gt = (np.array(Image.open(tmsk / name)) > 0)
        prob = np.load(pp)[1].astype(np.float32)
        x = ((torch.from_numpy(img).float().div_(255).permute(2, 0, 1) - MEAN) / STD).unsqueeze(0).to(dev)
        p = np.clip(prob, 1e-4, 1 - 1e-4)
        pm = F.interpolate(torch.from_numpy(np.log(p / (1 - p)))[None, None].float(),
                           size=mask_hw, mode="bilinear", align_corners=False).to(dev)
        coords = torch.zeros(1, 1, 2, device=dev); labels = -torch.ones(1, 1, dtype=torch.long, device=dev)
        with torch.no_grad():
            if use_dino:
                dino = torch.from_numpy(np.load(dino_p / (Path(name).stem + ".npy")).astype(np.float32))[None].to(dev)
                logits = model(x, dino, coords, labels, pm)
            else:
                logits = model(x, coords, labels, pm)
        prob_pred = torch.sigmoid(logits.squeeze()).cpu().numpy()
        for t in thresholds:
            pred = prob_pred > t
            stats[t][0] += int((pred & gt).sum()); stats[t][1] += int((pred & ~gt).sum()); stats[t][2] += int((~pred & gt).sum())
    res = {}
    for t, (tp, fp, fn) in stats.items():
        res[t] = {"iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--prob_dir", required=True)
    ap.add_argument("--dino_dir", required=True)
    ap.add_argument("--prefixes", default="craq-base-c0,craq-fused-e1",
                    help="comma-sep run-dir prefixes to aggregate+sweep")
    args = ap.parse_args()
    prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]
    print("== 5-fold aggregate (logit>0, train-time metric) ==")
    for prefix in prefixes:
        print(f"{prefix}:", json.dumps(agg(prefix), indent=2))
    thr = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
    for prefix in prefixes:
        print(f"== threshold sweep {prefix} (per-fold) ==")
        for k in range(5):
            ck = Path(f"runs/{prefix}-fold{k}-2026-06-11/best.pt")
            if not ck.exists():
                print(f"  fold{k} missing"); continue
            r = sweep(str(ck), args.tiles_root, args.split, k, args.prob_dir, args.dino_dir, thr)
            best_t = max(r, key=lambda t: r[t]["iou"])
            print(f"  fold{k} best_thr={best_t} iou={r[best_t]['iou']:.4f} "
                  f"P={r[best_t]['precision']:.3f} R={r[best_t]['recall']:.3f}")


if __name__ == "__main__":
    main()
