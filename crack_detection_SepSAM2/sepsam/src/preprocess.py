"""preprocess.py — 共用確定性影像前處理。train tile 建構與 CMC 推論端共用,
確保 CLAHE 在訓練與推論一致(避免 domain 不一致)。"""
import cv2
import numpy as np


def clahe_rgb(img_rgb, clip=2.0, grid=8):
    """對 RGB 影像在 LAB 的 L 通道套 CLAHE(確定性)。

    Args:
        img_rgb: HxWx3 uint8 RGB
        clip:    clipLimit
        grid:    tileGridSize 邊長(grid x grid)
    Returns:
        HxWx3 uint8 RGB
    """
    arr = np.asarray(img_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"clahe_rgb expects HxWx3, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"clahe_rgb expects uint8, got {arr.dtype}")
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def clahe_bgr(img_bgr, clip=2.0, grid=8):
    """BGR 版包裝(cv2.imread 預設 BGR)。

    Args:
        img_bgr: HxWx3 uint8 BGR
    Returns:
        HxWx3 uint8 BGR
    """
    rgb = cv2.cvtColor(np.asarray(img_bgr), cv2.COLOR_BGR2RGB)
    out = clahe_rgb(rgb, clip=clip, grid=grid)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
