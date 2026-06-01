import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lineproc import cc_filter, skeleton_centerline, cldice, tolerant_f1


def _line(h=64, w=64, row=32):
    m = np.zeros((h, w), np.uint8); m[row-1:row+2, 5:60] = 1; return m


def test_cc_filter_removes_small():
    m = _line()
    m[2:4, 2:4] = 1
    out = cc_filter(m, min_area=20)
    assert out[2, 2] == 0
    assert out[32, 30] == 1


def test_skeleton_centerline_thins_then_dilates():
    m = np.zeros((64, 64), np.uint8); m[28:36, 5:60] = 1
    out = skeleton_centerline(m, width=3)
    col = out[:, 30]
    assert 1 <= col.sum() <= 5
    assert out.dtype == bool or out.max() <= 1


def test_cldice_identical_is_one_disjoint_zero():
    m = _line().astype(bool)
    assert cldice(m, m) > 0.99
    other = np.zeros_like(m); other[5:8, 5:60] = 1
    assert cldice(m, other.astype(bool)) < 0.2


def test_tolerant_f1_shifted_line_high_vanilla_low():
    m = _line().astype(bool)
    shifted = np.roll(m, 1, axis=0)
    assert tolerant_f1(m, shifted, tol=3) > 0.9
    inter = (m & shifted).sum(); van = 2*inter/(m.sum()+shifted.sum())
    assert van < tolerant_f1(m, shifted, tol=3)


def test_empty_cases():
    z = np.zeros((16, 16), bool)
    assert cldice(z, z) == 1.0 and tolerant_f1(z, z) == 1.0
    nz = z.copy(); nz[5, 5] = True
    assert cldice(nz, z) == 0.0 and tolerant_f1(nz, z) == 0.0


if __name__ == "__main__":
    for f in [test_cc_filter_removes_small, test_skeleton_centerline_thins_then_dilates,
              test_cldice_identical_is_one_disjoint_zero, test_tolerant_f1_shifted_line_high_vanilla_low,
              test_empty_cases]:
        f()
    print("OK test_lineproc")
