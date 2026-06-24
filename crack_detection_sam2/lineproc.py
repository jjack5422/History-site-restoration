"""lineproc.py — 線狀預測後處理 + 對細線公平的指標(對齊 Notion:clDice / tolerant)。"""
import numpy as np
from scipy.ndimage import label, binary_dilation, generate_binary_structure, iterate_structure
from skimage.morphology import skeletonize, dilation, disk


def cc_filter(mask, min_area):
    """移除像素數 < min_area 的連通元件。回傳 bool。"""
    mask = np.asarray(mask).astype(bool)
    lab, n = label(mask)
    if n == 0:
        return mask
    out = np.zeros_like(mask)
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() >= min_area:
            out |= comp
    return out


def skeleton_centerline(mask, width=3):
    """skeletonize 後膨脹到 ~width px。回傳 bool。"""
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return mask
    sk = skeletonize(mask)
    r = max(1, width // 2)
    return dilation(sk, disk(r))


def _dilate(mask, tol):
    st = iterate_structure(generate_binary_structure(2, 1), tol)
    return binary_dilation(mask, structure=st)


def cldice(pred, gt):
    pred = np.asarray(pred).astype(bool); gt = np.asarray(gt).astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    sp = skeletonize(pred); sg = skeletonize(gt)
    tprec = (sp & gt).sum() / max(sp.sum(), 1)
    tsens = (sg & pred).sum() / max(sg.sum(), 1)
    if (tprec + tsens) == 0:
        return 0.0
    return float(2 * tprec * tsens / (tprec + tsens))


def tolerant_f1(pred, gt, tol=3):
    pred = np.asarray(pred).astype(bool); gt = np.asarray(gt).astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    gt_d = _dilate(gt, tol); pred_d = _dilate(pred, tol)
    prec = (pred & gt_d).sum() / max(pred.sum(), 1)
    rec = (gt & pred_d).sum() / max(gt.sum(), 1)
    if (prec + rec) == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))
