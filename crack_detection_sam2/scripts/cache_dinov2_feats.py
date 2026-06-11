"""離線把 tiles_512 每張影像的 DINOv2 (reg4, S/14) patch token map cache 成 [C,37,37] fp16 .npy。
與 resunet_prob 平行;訓練時只讀此 cache,不跑 DINOv2。"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import timm
import torch
from PIL import Image

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="tiles_512/images 目錄")
    ap.add_argument("--out", required=True, help="輸出 dinov2_feat 目錄")
    ap.add_argument("--model", default="vit_small_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=518)  # 37*14
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = timm.create_model(args.model, pretrained=True, num_classes=0).eval().to(dev)
    npref = m.num_prefix_tokens
    g = args.size // 14
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    imgs = sorted(Path(args.images).glob("*.png")) + sorted(Path(args.images).glob("*.jpg"))
    print(f"model={args.model} prefix={npref} grid={g}x{g} n_imgs={len(imgs)}")
    done = 0
    for f in imgs:
        dst = out / (f.stem + ".npy")
        if dst.exists():
            continue
        im = Image.open(f).convert("RGB").resize((args.size, args.size), Image.BILINEAR)
        x = torch.from_numpy(np.array(im)).float().div_(255).permute(2, 0, 1)
        x = ((x - MEAN) / STD).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.amp.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            t = m.forward_features(x)            # [1, npref + g*g, C]
        t = t[:, npref:, :].float()             # [1, g*g, C]
        fm = t.reshape(1, g, g, -1).permute(0, 3, 1, 2).squeeze(0).cpu().numpy()  # [C,g,g]
        np.save(dst, fm.astype(np.float16))
        done += 1
        if done % 50 == 0:
            print(f"  {done} cached", flush=True)
    print(f"done. cached {done}, total existing {len(list(out.glob('*.npy')))}")


if __name__ == "__main__":
    main()
