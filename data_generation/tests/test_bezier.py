import numpy as np
from synthgen.geometry.bezier import cubic_bezier


def test_endpoints_match():
    p0, p1, p2, p3 = (0, 0), (10, 0), (20, 10), (30, 30)
    pts = cubic_bezier(p0, p1, p2, p3, n=50)
    assert pts.shape == (50, 2)
    assert np.allclose(pts[0], p0)
    assert np.allclose(pts[-1], p3)


def test_straight_line_midpoint():
    # 控制點落在直線上 -> 結果為直線, 中點為兩端中點
    pts = cubic_bezier((0, 0), (1, 1), (2, 2), (3, 3), n=4)
    assert np.allclose(pts[:, 0], pts[:, 1])  # x==y 全程
