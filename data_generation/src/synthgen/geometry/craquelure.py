import numpy as np
import cv2
from scipy.spatial import Voronoi


def generate(size, params, rng):
    """以 Voronoi cell 邊界生成 craquelure 網狀 mask, 回傳 (size,size) uint8 {0,1}。

    edge_w 設定 cv2.line 繪製寬度;之後施加 1px gap-closing dilation 以接合
    Voronoi 頂點處的對角縫隙,故有效線寬約為 edge_w+1。
    """
    clo, chi = params["cell_px"]
    cell = float(rng.uniform(clo, chi))
    jitter = params["jitter"]
    edge_w = int(params["edge_w"])
    break_p = params["break_p"]

    # 在略大於影像的範圍撒抖動格點種子(避免邊界 cell 缺失)
    step = cell
    coords = np.arange(-cell, size + cell, step)
    gx, gy = np.meshgrid(coords, coords)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    pts += rng.normal(0, cell * jitter, pts.shape)

    vor = Voronoi(pts)
    mask = np.zeros((size, size), dtype=np.uint8)
    for a, b in vor.ridge_vertices:
        if a < 0 or b < 0:
            continue  # 跳過延伸到無窮遠的脊
        if rng.uniform() < break_p:
            continue  # 隨機斷裂模擬不連續
        pa = vor.vertices[a]
        pb = vor.vertices[b]
        cv2.line(mask,
                 (int(round(pa[0])), int(round(pa[1]))),
                 (int(round(pb[0])), int(round(pb[1]))),
                 1, edge_w, lineType=cv2.LINE_8)
    # 2x2 膨脹填補對角接縫, 確保 Voronoi cell 在 8-連通意義下閉合
    k = np.ones((2, 2), dtype=np.uint8)
    mask = cv2.dilate(mask, k, iterations=1)
    return (mask > 0).astype(np.uint8)
