import os, sys
import cv2
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import clahe_rgb


def test_shape_dtype_preserved():
    img = (np.random.default_rng(0).integers(90, 110, (64, 64, 3))).astype(np.uint8)
    out = clahe_rgb(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_deterministic():
    img = (np.random.default_rng(1).integers(90, 110, (64, 64, 3))).astype(np.uint8)
    a = clahe_rgb(img)
    b = clahe_rgb(img)
    assert np.array_equal(a, b)


def _L_std(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    return cv2.split(lab)[0].astype(float).std()


def test_contrast_increases_on_low_contrast():
    # 低對比影像(值集中在 100..110)→ CLAHE 後 L 通道 std 應上升
    img = (np.random.default_rng(2).integers(100, 110, (128, 128, 3))).astype(np.uint8)
    out = clahe_rgb(img, clip=3.0, grid=8)
    assert _L_std(out) > _L_std(img)


if __name__ == "__main__":
    test_shape_dtype_preserved()
    test_deterministic()
    test_contrast_increases_on_low_contrast()
    print("OK test_preprocess")
