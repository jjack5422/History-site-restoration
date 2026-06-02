import numpy as np
from synthgen.appearance import fit_profile, save_profile, load_profile, render


def test_fit_profile_detects_dark_cracks(tmp_path):
    import cv2
    from PIL import Image
    idir = tmp_path / "img"; mdir = tmp_path / "msk"
    idir.mkdir(); mdir.mkdir()
    for i in range(3):
        img = np.full((64, 64, 3), 200, np.uint8)
        m = np.zeros((64, 64), np.uint8)
        cv2.line(img, (0, 32), (63, 32), (40, 40, 40), 2)  # 暗裂縫
        cv2.line(m, (0, 32), (63, 32), 1, 2)
        Image.fromarray(img).save(idir / f"{i}.png")
        Image.fromarray(m).save(mdir / f"{i}.png")
    prof = fit_profile(str(idir), str(mdir))
    assert prof["n"] > 0
    assert prof["dL"][0] < 0, "裂縫應比鄰域暗 (dL<0)"


def test_save_load_roundtrip(tmp_path):
    prof = {"dL": [-20.0, 5.0], "da": [0.0, 1.0], "db": [0.0, 1.0], "n": 10}
    p = tmp_path / "p.json"
    save_profile(prof, str(p))
    assert load_profile(str(p)) == prof


def test_render_darkens_cracks_preserves_elsewhere():
    base = np.full((64, 64, 3), 200, np.uint8)
    geo = np.zeros((64, 64), np.uint8)
    geo[30:34, :] = 1
    prof = {"dL": [-30.0, 0.0], "da": [0.0, 0.0], "db": [0.0, 0.0], "n": 100}
    cfg = {"min_contrast": 12, "erosion": 0, "blur_sigma": 0}
    out = render(base, geo, prof, cfg, np.random.default_rng(0))
    assert out.shape == base.shape and out.dtype == np.uint8
    assert out[32].mean() < base[32].mean() - 10, "裂縫處應變暗"
    assert np.array_equal(out[0], base[0]), "非裂縫列不應改變"
