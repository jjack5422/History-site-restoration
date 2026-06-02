import numpy as np
from synthgen.geometry.crack import generate

PARAMS = {"n_curves": [2, 6], "taper_alpha": 2.0, "taper_sigma": 0.5, "branch_p": [0.3, 0.5]}


def test_shape_dtype_binary():
    m = generate(256, PARAMS, np.random.default_rng(0))
    assert m.shape == (256, 256)
    assert m.dtype == np.uint8
    assert set(np.unique(m)).issubset({0, 1})


def test_sparse_nonempty():
    m = generate(512, PARAMS, np.random.default_rng(1))
    fg = m.mean()
    assert fg > 0.0, "crack mask 不可全空"
    assert fg < 0.05, f"crack 應稀疏 (<5%), got {fg:.4f}"


def test_deterministic_with_same_seed():
    a = generate(256, PARAMS, np.random.default_rng(7))
    b = generate(256, PARAMS, np.random.default_rng(7))
    assert np.array_equal(a, b)
