import numpy as np
import cv2
from .bezier import cubic_bezier


def _draw_curve(mask, p0, p3, rng, taper_alpha, taper_sigma):
    """沿一條 Bézier 曲線畫 tapered(兩端細中段粗)裂縫。"""
    size = mask.shape[0]
    # 內控制點 = 端點連線附近加高斯擾動
    jitter = size * 0.08
    p1 = np.array(p0) + rng.normal(0, jitter, 2)
    p2 = np.array(p3) + rng.normal(0, jitter, 2)
    length = np.linalg.norm(np.array(p3) - np.array(p0))
    n = int(np.clip(length, 80, 180))
    pts = cubic_bezier(p0, p1, p2, p3, n)
    for i, (x, y) in enumerate(pts):
        t = i / max(n - 1, 1)
        r = rng.normal(taper_alpha * (1.0 - abs(t - 0.5) * 2.0), taper_sigma)
        r = max(int(round(r)), 0)
        cv2.circle(mask, (int(round(x)), int(round(y))), r, 1, -1)
    return pts


def generate(size, params, rng):
    """回傳 (size,size) uint8 {0,1} 的稀疏 crack mask。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    lo, hi = params["n_curves"]
    n_curves = int(rng.integers(lo, hi + 1))
    blo, bhi = params["branch_p"]
    for _ in range(n_curves):
        p0 = rng.uniform(0, size, 2)
        p3 = rng.uniform(0, size, 2)
        pts = _draw_curve(mask, p0, p3, rng,
                          params["taper_alpha"], params["taper_sigma"])
        # 機率從曲線中段長出一條分支
        if rng.uniform() < rng.uniform(blo, bhi):
            mid = pts[len(pts) // 2]
            direction = pts[-1] - pts[0]
            ang = rng.uniform(-np.pi / 3, np.pi / 3)
            rot = np.array([[np.cos(ang), -np.sin(ang)],
                            [np.sin(ang), np.cos(ang)]])
            end = mid + rot @ direction * rng.uniform(0.2, 0.5)
            _draw_curve(mask, mid, end, rng,
                        params["taper_alpha"] * 0.7, params["taper_sigma"])
    return (mask > 0).astype(np.uint8)
