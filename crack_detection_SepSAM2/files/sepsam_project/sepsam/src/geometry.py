"""
geometry.py — 沿中軸（medial axis）取提示點 + 估算裂紋寬度（CMC 1st round）。
對應論文 Eq. 9, 10。
"""
import numpy as np
from skimage.morphology import medial_axis


def mask_to_points_and_width(mask_bin, n_points):
    """
    從 Agent 草稿的中軸均勻取點，作為 SAM 的正樣本提示點。

    Args:
        mask_bin: HxW，bool / 0-1（Agent 草稿，已二值化）
        n_points: 取樣點數，建議 = max(H, W) // POINTS_DIVISOR

    Returns:
        pts:    (K, 2) float32，每列 (x, y)，給 SAM 的 point_coords
        widths: (K,)   float32，各點的裂紋寬度估計 = 2 * 距邊界距離
    """
    mask_bin = np.asarray(mask_bin).astype(bool)
    if mask_bin.sum() == 0:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    # medial_axis 同時回傳骨架與距離轉換（到邊界的距離）
    skel, dist = medial_axis(mask_bin, return_distance=True)
    ys, xs = np.where(skel)
    if xs.size == 0:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    # [需推斷] 論文寫「沿中軸均勻取樣」。此處為簡單版：對骨架像素索引均勻取樣。
    # 更忠實的版本（建議日後實作為可選）：先把骨架像素依曲線排序（最近鄰串成路徑），
    #   再依弧長等距取樣，可避免分支密集處取樣不均。
    k = int(min(max(n_points, 1), xs.size))
    sel = np.linspace(0, xs.size - 1, num=k).astype(int)
    pts = np.stack([xs[sel], ys[sel]], axis=1).astype(np.float32)   # SAM 用 (x, y)
    widths = (2.0 * dist[ys[sel], xs[sel]]).astype(np.float32)
    return pts, widths


def mean_crack_width(mask_bin):
    """估計整張 mask 的平均裂紋寬度（沿骨架取 2*距離 的平均），供統計/分桶參考。"""
    mask_bin = np.asarray(mask_bin).astype(bool)
    if mask_bin.sum() == 0:
        return 0.0
    skel, dist = medial_axis(mask_bin, return_distance=True)
    vals = dist[skel]
    return float(2.0 * vals.mean()) if vals.size else 0.0
