"""Inference helper for click-promptable SAM2 experts.

Single entry point consumed by the CVAT Nuclio interactor (sub-project 2). Same contract names
as the existing predict_full.py modules (`load_model_from_ckpt`). v0 assumes `img` is already at
the model working resolution (a tile); the CVAT-side coordinate mapping for arbitrary-size frames
is handled in sub-project 2.
"""
from __future__ import annotations

import numpy as np
import torch

from model_prompted_sam2 import PromptedSAM2Seg
from crackseg_common.augment import val_transforms


def load_model_from_ckpt(ckpt, device):
    payload = torch.load(ckpt, map_location=device)
    a = payload.get("args", {})
    model = PromptedSAM2Seg(variant=a.get("variant", "small"),
                            image_size=a.get("image_size", 512), device=device)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, payload


@torch.no_grad()
def predict_click(model, img, pos_points, neg_points, prev_mask, device, image_size=512):
    """img: HxWx3 uint8 RGB ndarray. pos/neg_points: lists of (x, y) in img pixel space.
    Returns (mask: bool HxW at working resolution, low_res_logits: tensor [1,1,128,128])."""
    if img.shape[0] != image_size or img.shape[1] != image_size:
        raise ValueError(
            f"predict_click v0 expects a {image_size}x{image_size} tile; got {tuple(img.shape[:2])}. "
            "Click coords are not rescaled, so a non-working-resolution image would mis-place points. "
            "Arbitrary-frame resolution/coordinate mapping is sub-project 2.")

    tf = val_transforms(image_size=image_size)
    out = tf(image=img, mask=np.zeros(img.shape[:2], np.uint8))
    x = out["image"].unsqueeze(0).to(device)

    all_pts = list(pos_points) + list(neg_points)   # each already (x, y)
    labs = [1] * len(pos_points) + [0] * len(neg_points)
    if all_pts:
        coords = torch.tensor([[[float(x), float(y)] for (x, y) in all_pts]], device=device)  # (x, y)
        labels = torch.tensor([labs], dtype=torch.int32, device=device)
    else:
        coords = torch.zeros(1, 0, 2, device=device)
        labels = torch.zeros(1, 0, dtype=torch.int32, device=device)

    enc = model.encode_image(x)
    masks, low = model.decode(enc, coords, labels, prev_mask=prev_mask)
    mask = (masks[0, 0] > 0).cpu().numpy()
    return mask, low
