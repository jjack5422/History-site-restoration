import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train_click import batch_point_tensors, aggregate_click_metrics


def test_batch_point_tensors_xy_order():
    pts = [[(3, 7), (1, 2)]]      # (row, col)
    labs = [[1, 0]]
    coords, labels = batch_point_tensors(pts, labs, "cpu")
    assert coords.shape == (1, 2, 2) and labels.shape == (1, 2)
    # (row=3,col=7) -> (x=7, y=3)
    assert coords[0, 0].tolist() == [7.0, 3.0]
    assert labels[0].tolist() == [1, 0]


def test_aggregate_click_metrics():
    # 2 samples, n_clicks=3. per_click_iou[k] holds the IoU of each sample after click k+1.
    per_click_iou = [[0.5, 0.2], [0.85, 0.4], [0.9, 0.82]]
    noc = [2, 3]   # sample A reached 0.8 at click 2, sample B at click 3
    m = aggregate_click_metrics(per_click_iou, noc, n_clicks=3, iou_target=0.8)
    assert abs(m["iou@1"] - 0.35) < 1e-6
    assert abs(m["iou@3"] - 0.86) < 1e-6
    assert abs(m["noc@0.8"] - 2.5) < 1e-6
    print("OK test_click_metrics")


if __name__ == "__main__":
    test_batch_point_tensors_xy_order()
    test_aggregate_click_metrics()
    print("OK test_click_metrics")
