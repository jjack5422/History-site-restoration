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


def _combine_points(pts, neg_pts):
    """正點(label=1) + 可選負點(label=0)合併成 (coords, labels)。"""
    coords = np.asarray(pts, np.float32)
    labels = np.ones(coords.shape[0], dtype=np.int64)
    if neg_pts is not None and len(neg_pts) > 0:
        neg = np.asarray(neg_pts, np.float32)
        coords = np.concatenate([coords, neg], axis=0)
        labels = np.concatenate([labels, np.zeros(neg.shape[0], dtype=np.int64)])
    return coords, labels


def _predict_kwargs(pts, neg_pts, mask_input):
    """組 predictor.predict 的 kwargs。支援 point-only / mask-only / 兩者皆有。"""
    kw = {"multimask_output": False}
    if pts is not None and len(pts) > 0:
        kw["point_coords"], kw["point_labels"] = _combine_points(pts, neg_pts)
    if mask_input is not None:
        kw["mask_input"] = mask_input
    return kw


def sam_prompt_v1(predictor, image_rgb, pts, neg_pts=None, mask_input=None):
    """SAM v1：回傳 (mask uint8 0/255, sam_score float)。支援點 prompt(可加負點)或 mask_input prompt。"""
    has_pts = pts is not None and len(pts) > 0
    if not has_pts and mask_input is None:
        h, w = image_rgb.shape[:2]
        return np.zeros((h, w), np.uint8), 0.0
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(**_predict_kwargs(pts, neg_pts, mask_input))
    return (masks[0].astype(np.uint8) * 255), float(scores[0])


def sam_prompt_sam2(predictor, image_rgb, pts, neg_pts=None, mask_input=None):
    """SAM2：介面與 v1 相同。支援點 prompt(可加負點)或 mask_input prompt(YOLO 草稿當 mask)。"""
    has_pts = pts is not None and len(pts) > 0
    if not has_pts and mask_input is None:
        h, w = image_rgb.shape[:2]
        return np.zeros((h, w), np.uint8), 0.0
    use_cuda = torch.cuda.is_available()
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(**_predict_kwargs(pts, neg_pts, mask_input))
    return (masks[0].astype(np.uint8) * 255), float(scores[0])
