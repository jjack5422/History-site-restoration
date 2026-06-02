import numpy as np


def cubic_bezier(p0, p1, p2, p3, n):
    """回傳 cubic Bézier 取樣點 (n,2) float。p* 為 (x,y)。"""
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    p0, p1, p2, p3 = map(lambda p: np.asarray(p, dtype=np.float64), (p0, p1, p2, p3))
    return ((1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2
            + t ** 3 * p3)
