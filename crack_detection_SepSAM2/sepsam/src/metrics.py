"""
metrics.py — 像素級分割指標（論文 Eq. 12–16）。
prf_iou(pred, gt) -> (Precision, Recall, F1, IoU)
"""
import numpy as np


def prf_iou(pred, gt):
    """
    Args:
        pred, gt: HxW，任何非零值視為前景（裂紋）。形狀不同會將 pred 最近鄰縮放到 gt。
    Returns:
        (precision, recall, f1, iou) 皆為 float
    """
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    if p.shape != g.shape:
        import cv2
        p = cv2.resize(
            p.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0
    return float(precision), float(recall), float(f1), float(iou)


def aggregate(records):
    """records: list of (P,R,F1,IoU) → 回傳各項平均的 dict。"""
    if not records:
        return {"P": 0.0, "R": 0.0, "F1": 0.0, "IoU": 0.0, "n": 0}
    arr = np.asarray(records, dtype=float)
    return {
        "P": float(arr[:, 0].mean()),
        "R": float(arr[:, 1].mean()),
        "F1": float(arr[:, 2].mean()),
        "IoU": float(arr[:, 3].mean()),
        "n": len(records),
    }

# 註：論文亦做「依裂紋寬度分桶」（如 width>3、width>20）的比較。
# 寬度分桶需先用距離轉換估計 GT 的逐像素寬度再篩選，較繁瑣，此處未實作。
# 若需要，可用 geometry.mean_crack_width 或對 GT 做距離轉換後依寬度遮罩再呼叫 prf_iou。
