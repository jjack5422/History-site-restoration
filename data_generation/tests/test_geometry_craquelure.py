import numpy as np
import cv2
from synthgen.geometry.craquelure import generate

PARAMS = {"cell_px": [32, 60], "jitter": 0.3, "edge_w": 1, "break_p": 0.1}


def test_shape_dtype_binary():
    m = generate(256, PARAMS, np.random.default_rng(0))
    assert m.shape == (256, 256)
    assert m.dtype == np.uint8
    assert set(np.unique(m)).issubset({0, 1})


def test_forms_cells():
    # 背景(非裂縫)應被裂縫網切成多個 island(連通元件)
    m = generate(512, PARAMS, np.random.default_rng(2))
    bg = (m == 0).astype(np.uint8)
    n_components, _ = cv2.connectedComponents(bg)
    assert n_components - 1 >= 20, f"cell 數應 >=20, got {n_components - 1}"


def test_fg_in_craquelure_range():
    m = generate(512, PARAMS, np.random.default_rng(3))
    fg = m.mean() * 100
    assert 0.5 <= fg <= 12.0, f"craq fg% 應落在真實範圍, got {fg:.2f}%"
