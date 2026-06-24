import os
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

MODEL_VARIANTS = {
    "tiny":  ("configs/sam2.1/sam2.1_hiera_t.yaml",  "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml",  "sam2.1_hiera_small.pt"),
    "base":  ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml",  "sam2.1_hiera_large.pt"),
}


def build_sam2_model(variant="large", device=None, mode="eval"):
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"variant 必須是 {list(MODEL_VARIANTS)}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_name, ckpt_name = MODEL_VARIANTS[variant]
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    model = build_sam2(cfg_name, ckpt_path, device=device, mode=mode)
    return model


def build_image_predictor(variant="large", device=None):
    model = build_sam2_model(variant=variant, device=device, mode="eval")
    return SAM2ImagePredictor(model)
