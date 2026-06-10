"""Sample positive/negative point prompts from a ResUNet craquelure probability map.

Returns image-space coordinates in (row=y, col=x) order. The SAM2 prompt encoder
expects (x, y) order, so the caller must flip before feeding the model.
"""
from __future__ import annotations

import numpy as np


def sample_points_from_prob(prob, n_pos=3, n_neg=3, thr=0.5, size=512, seed=0):
    """prob: (H,W) craquelure foreground probability.

    Returns coords (N,2) image coords (y,x) and labels (N,) with 1=positive, 0=negative.
    Positives are drawn from prob>=thr, negatives from prob<thr; if a pool is too small,
    it is topped up with uniform-random pixels.
    """
    rng = np.random.default_rng(seed)
    H, W = prob.shape
    pos_yx = np.argwhere(prob >= thr)
    neg_yx = np.argwhere(prob < thr)

    def pick(pool, k):
        if len(pool) >= k:
            idx = rng.choice(len(pool), size=k, replace=False)
            return pool[idx]
        extra = rng.integers(0, [H, W], size=(k - len(pool), 2))
        return np.concatenate([pool, extra], axis=0) if len(pool) else extra

    pos = pick(pos_yx, n_pos).astype(np.float32)
    neg = pick(neg_yx, n_neg).astype(np.float32)
    coords = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)
    return coords, labels
