import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from click_sampling import mask_iou, sample_initial_point, sample_correction_point


def test_mask_iou():
    a = np.zeros((10, 10), bool); a[2:5, 2:5] = True
    b = np.zeros((10, 10), bool)
    assert mask_iou(a, a) == 1.0
    assert mask_iou(a, b) == 0.0
    assert mask_iou(b, b) == 1.0   # both empty -> defined as 1.0


def test_initial_positive_inside_gt():
    gt = np.zeros((20, 20), bool); gt[5:10, 5:10] = True
    (r, c), lbl = sample_initial_point(gt, np.random.default_rng(0))
    assert lbl == 1 and gt[r, c]


def test_initial_negative_when_empty():
    gt = np.zeros((20, 20), bool)
    (r, c), lbl = sample_initial_point(gt, np.random.default_rng(0))
    assert lbl == 0


def test_correction_false_negative_is_positive():
    gt = np.zeros((20, 20), bool); gt[5:15, 5:15] = True
    pred = np.zeros((20, 20), bool)                 # all FN
    (r, c), lbl = sample_correction_point(pred, gt, np.random.default_rng(0))
    assert lbl == 1 and gt[r, c]


def test_correction_false_positive_is_negative():
    gt = np.zeros((20, 20), bool)
    pred = np.zeros((20, 20), bool); pred[5:15, 5:15] = True   # all FP
    (r, c), lbl = sample_correction_point(pred, gt, np.random.default_rng(0))
    assert lbl == 0 and pred[r, c] and not gt[r, c]


def test_correction_none_when_perfect():
    gt = np.zeros((20, 20), bool); gt[5:10, 5:10] = True
    assert sample_correction_point(gt, gt, np.random.default_rng(0)) is None


def test_correction_prefers_fn_when_tied():
    gt = np.zeros((20, 20), bool); gt[5:10, 5:10] = True      # 25 px
    pred = np.zeros((20, 20), bool); pred[8:13, 8:13] = True  # overlap 4, fn 21, fp 21
    (r, c), lbl = sample_correction_point(pred, gt, np.random.default_rng(0))
    assert lbl == 1 and gt[r, c]   # fn == fp -> fn branch wins -> positive click inside gt


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("OK test_click_sampling")
