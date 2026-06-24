"""Dump per-tile ResUNet craquelure prob (2ch softmax) for every tile in a tiles_512
images dir, using a given ResUNet ckpt, into <out_dir>/prob/<stem>.npy.

Used to regenerate the refine prompt source from a specific ResUNet (e.g. a held-out
model) WITHOUT overwriting the deployment resunet_prob.

  /home/zzz90/research/unet_env/bin/python crack_detection_sam2/scripts/dump_tile_prob.py \
      --ckpt <resunet>/best.pt \
      --images_dir /home/zzz90/research/_data/craq_0-94_v1/tiles_512/images \
      --out_dir /home/zzz90/research/_data/craq_0-94_v1/tiles_512/resunet_prob_demoholdout
"""
import argparse, glob, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, "/home/zzz90/research/crack_detection_unet/src")
from unet_model import build_resunet

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()
    out = os.path.join(args.out_dir, "prob"); os.makedirs(out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    enc = ck.get("args", {}).get("encoder", "resnet50")
    m = build_resunet(encoder=enc, encoder_weights=None, num_classes=2).to(dev)
    m.load_state_dict(ck["model"], strict=False); m.eval()
    mean, std = MEAN.to(dev), STD.to(dev)
    tiles = sorted(glob.glob(os.path.join(args.images_dir, "*.png")))
    print(f"{len(tiles)} tiles -> {out}")
    for i in range(0, len(tiles), args.batch_size):
        chunk = tiles[i:i + args.batch_size]
        x = torch.stack([torch.from_numpy(np.array(Image.open(p).convert("RGB"))).float().div_(255).permute(2, 0, 1) for p in chunk])
        x = ((x.to(dev) - mean) / std)
        with torch.no_grad():
            p = F.softmax(m(x).float(), 1).cpu().numpy()
        for path, pp in zip(chunk, p):
            np.save(os.path.join(out, os.path.splitext(os.path.basename(path))[0] + ".npy"), pp.astype(np.float32))
    print("done")


if __name__ == "__main__":
    main()
