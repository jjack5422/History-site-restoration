import numpy as np
from .appearance import render


def compose(base_rgb, geo_mask, profile, cfg, rng):
    """回傳 (渲染後影像 RGB uint8, GT mask uint8 {0,1})。

    GT 取 blur 前的二值 geo_mask, 與外觀解耦以保證像素級精確。
    """
    mask = (geo_mask > 0).astype(np.uint8)
    img = render(base_rgb, geo_mask, profile, cfg, rng)
    return img, mask
