"""Point sampling for SAM-style interactive click training/eval.

Pure numpy/scipy (no torch / model deps) so it is unit-testable in isolation.
Coordinates are (row, col) i.e. (y, x) in image space; conversion to the
prompt encoder's (x, y) order happens in the trainer via build helpers.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def _largest_component_point(mask: np.ndarray):
    """A pixel inside the largest connected component of `mask` (boolean), chosen as the
    in-component pixel closest to that component's centroid. Returns (row, col) or None."""
    if mask.sum() == 0:
        return None
    lab, n = ndimage.label(mask)
    if n == 0:  # defensive: unreachable given the sum() checks above
        return None
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    big = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(lab == big)
    cy, cx = ys.mean(), xs.mean()
    i = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
    return int(ys[i]), int(xs[i])


def sample_initial_point(gt: np.ndarray, rng):
    """First click. Positive at the center of GT's largest component; if GT is empty,
    a negative click at image center. Returns ((row, col), label).

    Args:
        gt:  Boolean ground-truth mask.
        rng: Reserved for future stochastic tie-breaking; currently unused.
    """
    gt = gt.astype(bool)
    if gt.sum() == 0:
        h, w = gt.shape
        return (h // 2, w // 2), 0
    return _largest_component_point(gt), 1


def sample_correction_point(pred: np.ndarray, gt: np.ndarray, rng):
    """Correction click from the larger error region. False-negative (missed GT) -> positive(1);
    false-positive -> negative(0). Returns ((row, col), label) or None if pred == gt.

    Args:
        pred: Boolean predicted mask.
        gt:   Boolean ground-truth mask.
        rng:  Reserved for future stochastic tie-breaking; currently unused.
    """
    pred = pred.astype(bool); gt = gt.astype(bool)
    fn = np.logical_and(gt, ~pred)
    fp = np.logical_and(pred, ~gt)
    if fn.sum() == 0 and fp.sum() == 0:
        return None
    if fn.sum() >= fp.sum():
        p = _largest_component_point(fn); lbl = 1
    else:
        p = _largest_component_point(fp); lbl = 0
    if p is None:  # defensive: unreachable given the sum() checks above
        return None
    return p, lbl
