import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from craq_prompt_sampling import sample_points_from_prob


def test_returns_pos_and_neg_in_bounds():
    prob = np.zeros((512, 512), np.float32)
    prob[100:150, 100:150] = 0.9
    coords, labels = sample_points_from_prob(prob, n_pos=3, n_neg=3, thr=0.5, size=512, seed=0)
    assert coords.shape == (6, 2) and labels.shape == (6,)
    assert ((coords >= 0) & (coords < 512)).all()
    assert (labels == 1).sum() == 3 and (labels == 0).sum() == 3
    for (y, x), l in zip(coords.astype(int), labels):
        if l == 1:
            assert prob[y, x] >= 0.5


def test_topup_when_no_positive():
    prob = np.zeros((64, 64), np.float32)  # no pixel >= thr
    coords, labels = sample_points_from_prob(prob, n_pos=2, n_neg=2, thr=0.5, size=64, seed=1)
    assert coords.shape == (4, 2)
    assert ((coords >= 0) & (coords < 64)).all()
