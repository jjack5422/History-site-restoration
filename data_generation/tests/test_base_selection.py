import json
import numpy as np
import cv2
from PIL import Image
from synthgen.base_selection import score, build_manifest


def test_clean_lower_than_cracky():
    clean = np.full((256, 256), 180, np.uint8)
    cracky = clean.copy()
    for y in range(0, 256, 16):
        cv2.line(cracky, (0, y), (255, y), 30, 1)
    assert score(clean) < score(cracky)
    assert score(clean) < 0.01


def test_build_manifest(tmp_path):
    sd = tmp_path / "slices"
    sd.mkdir()
    Image.fromarray(np.full((128, 128, 3), 200, np.uint8)).save(sd / "clean.jpg")
    cracky = np.full((128, 128, 3), 200, np.uint8)
    for y in range(0, 128, 6):
        cv2.line(cracky, (0, y), (127, y), 20, 1)
    Image.fromarray(cracky).save(sd / "dirty.jpg")
    out = tmp_path / "manifest.json"
    man = build_manifest(str(sd), thresh=0.015, out_path=str(out))
    assert out.exists()
    assert "clean.jpg" in man["clean"]
    assert "dirty.jpg" not in man["clean"]
    assert json.load(open(out))["thresh"] == 0.015
