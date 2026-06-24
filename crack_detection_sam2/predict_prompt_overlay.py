"""Generate overlay visualizations from SAM2 Prompt model checkpoints.

Usage:
    python predict_prompt_overlay.py \
        --ckpt outputs/prompt_craq_fold0_small/best.pt \
        --tiles_root data/labeled32_craq_v3/tiles_512 \
        --split data/labeled32_craq_v3/tiles_512/group_split_stem.json \
        --fold 0 \
        --out_dir outputs/prompt_craq_overlay
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from crackseg_common.augment import val_transforms
from crackseg_common.dataset import TileSegDataset, load_tile_index, set_class_names
from model_prompt_seg import SAM2PromptSeg


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tiles_root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--variant", default="small")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_points", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--class_names", default="background,craquelure")
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    args = parser.parse_args()

    if args.class_names:
        set_class_names([s.strip() for s in args.class_names.split(",")])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    (out_dir / "overlay").mkdir(parents=True, exist_ok=True)
    (out_dir / "mask").mkdir(parents=True, exist_ok=True)

    # Load model
    model = SAM2PromptSeg(
        variant=args.variant, image_size=args.image_size,
        num_points=args.num_points, device=device,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.ckpt} (epoch {ckpt.get('epoch', '?')})")

    # Load val tiles
    tile_index = load_tile_index(args.tiles_root)
    with open(args.split) as f:
        payload = json.load(f)
    val_names = payload["folds"][args.fold]["val"]
    by_name = {it["tile"]: it for it in tile_index["items"]}
    val_items = [by_name[n] for n in val_names if n in by_name]

    tfm = val_transforms(image_size=args.image_size)
    ds = TileSegDataset(args.tiles_root, val_items, transforms=tfm)

    # Color for overlay: green for craquelure
    color = np.array([0, 255, 0], dtype=np.uint8)

    for i in range(len(ds)):
        sample = ds[i]
        img_t = sample["image"].unsqueeze(0).to(device)
        tile_name = val_items[i]["tile"]

        logits = model(img_t)  # [1, 1, H, W]
        pred = (logits.squeeze(0).squeeze(0) > args.threshold).cpu().numpy()

        # Read original image for overlay
        img_path = Path(args.tiles_root) / "images" / val_items[i]["tile"]
        orig = np.array(Image.open(img_path).convert("RGB"))
        h, w = orig.shape[:2]

        # Resize prediction to original size if needed
        if pred.shape != (h, w):
            pred_resized = np.array(
                Image.fromarray(pred.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
            ) > 127
        else:
            pred_resized = pred

        # Create overlay
        overlay = orig.copy()
        mask_rgb = np.zeros_like(orig)
        mask_rgb[pred_resized] = color
        alpha = args.overlay_alpha
        overlay[pred_resized] = (
            (1 - alpha) * orig[pred_resized].astype(float) +
            alpha * mask_rgb[pred_resized].astype(float)
        ).astype(np.uint8)

        Image.fromarray(overlay).save(out_dir / "overlay" / f"{tile_name}.png")
        Image.fromarray((pred_resized * 255).astype(np.uint8)).save(
            out_dir / "mask" / f"{tile_name}.png"
        )

    print(f"saved {len(ds)} overlays to {out_dir}/overlay/")


if __name__ == "__main__":
    main()
