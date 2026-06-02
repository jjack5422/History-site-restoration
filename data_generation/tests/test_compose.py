import numpy as np
from synthgen.compose import compose


def test_mask_is_pre_blur_binary():
    base = np.full((64, 64, 3), 200, np.uint8)
    geo = np.zeros((64, 64), np.uint8)
    geo[30:33, 10:50] = 1
    prof = {"dL": [-30.0, 0.0], "da": [0.0, 0.0], "db": [0.0, 0.0], "n": 1}
    cfg = {"min_contrast": 12, "erosion": 0, "blur_sigma": 2}
    img, mask = compose(base, geo, prof, cfg, np.random.default_rng(0))
    assert img.shape == base.shape and img.dtype == np.uint8
    assert mask.dtype == np.uint8 and set(np.unique(mask)).issubset({0, 1})
    # mask 等於原 geo(blur 不影響 GT)
    assert np.array_equal(mask, (geo > 0).astype(np.uint8))
    # 影像在裂縫處變暗
    assert img[31, 30].mean() < 190
