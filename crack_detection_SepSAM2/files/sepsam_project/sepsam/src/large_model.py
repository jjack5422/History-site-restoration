"""
large_model.py — 凍結大模型後端（SAM v1 / SAM2）統一介面（附錄 D）。
build_large_model(hp) 依 hp.sam_backend 回傳 (predictor, prompt_fn)。
兩個 prompt_fn 介面完全相同：prompt_fn(predictor, image_rgb, pts) -> (mask uint8 0/255, sam_score float)
"""
import contextlib

import numpy as np
import torch


def build_large_model(hp):
    """依 hp.sam_backend（'v1' 或 'sam2'）回傳 (predictor, prompt_fn)。"""
    backend = getattr(hp, "sam_backend", "v1")
    device = getattr(hp, "device", "cuda")

    if backend == "v1":
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry[hp.sam_model_type](checkpoint=hp.sam_ckpt).to(device)
        sam.eval()
        return SamPredictor(sam), sam_prompt_v1

    if backend == "sam2":
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2(hp.sam2_cfg, hp.sam2_ckpt, device=device)
        return SAM2ImagePredictor(model), sam_prompt_sam2

    raise ValueError(f"unknown sam_backend: {backend!r} (use 'v1' or 'sam2')")


def sam_prompt_v1(predictor, image_rgb, pts):
    """SAM v1：回傳 (mask uint8 0/255, sam_score float)。"""
    if pts.shape[0] == 0:
        h, w = image_rgb.shape[:2]
        return np.zeros((h, w), np.uint8), 0.0
    predictor.set_image(image_rgb)
    labels = np.ones(pts.shape[0], dtype=np.int64)
    masks, scores, _ = predictor.predict(
        point_coords=pts, point_labels=labels, multimask_output=False
    )
    return (masks[0].astype(np.uint8) * 255), float(scores[0])


def sam_prompt_sam2(predictor, image_rgb, pts):
    """SAM2：介面與 v1 完全相同。"""
    if pts.shape[0] == 0:
        h, w = image_rgb.shape[:2]
        return np.zeros((h, w), np.uint8), 0.0
    use_cuda = torch.cuda.is_available()
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        predictor.set_image(image_rgb)
        labels = np.ones(pts.shape[0], dtype=np.int64)
        masks, scores, _ = predictor.predict(
            point_coords=pts, point_labels=labels, multimask_output=False
        )
    return (masks[0].astype(np.uint8) * 255), float(scores[0])
