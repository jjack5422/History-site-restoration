import numpy as np
from PIL import Image
from synthgen.validate import mask_stats, compare


def test_mask_stats(tmp_path):
    md = tmp_path / "masks"; md.mkdir()
    m = np.zeros((100, 100), np.uint8); m[:, :2] = 1  # 2% fg
    Image.fromarray(m).save(md / "a.png")
    st = mask_stats(str(md))
    assert st["n"] == 1
    assert abs(st["fg_mean_pct"] - 2.0) < 0.5


def test_compare_in_range():
    real = {"fg_mean_pct": 2.86, "fg_pcts": [0.0, 9.0]}
    synth = {"fg_mean_pct": 3.0, "fg_pcts": [0.5, 8.0]}
    res = compare(real, synth)
    assert res["synth_mean_within_real_minmax"] is True
