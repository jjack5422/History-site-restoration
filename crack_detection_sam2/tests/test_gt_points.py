import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gt_points import gt_points


def test_empty_mask_returns_empty():
    m = np.zeros((64, 64), np.uint8)
    pts, labs = gt_points(m, 10)
    assert pts.shape == (0, 2) and labs.shape == (0,)


def test_points_lie_on_foreground_and_positive():
    m = np.zeros((64, 64), np.uint8)
    m[30:34, 5:60] = 1  # 一條水平帶
    pts, labs = gt_points(m, 8)
    assert 1 <= pts.shape[0] <= 8
    assert (labs == 1).all()
    for x, y in pts.astype(int):
        assert m[y, x] == 1  # (x,y) 順序,落在前景


if __name__ == "__main__":
    test_empty_mask_returns_empty()
    test_points_lie_on_foreground_and_positive()
    print("OK test_gt_points")
