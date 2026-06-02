import json
import numpy as np
from PIL import Image
from synthgen.tiling import tile_and_write, finalize_index


def test_tile_and_write(tmp_path):
    img = np.full((1024, 1024, 3), 180, np.uint8)
    mask = np.zeros((1024, 1024), np.uint8)
    mask[500:520, :] = 1  # 橫跨, 保證部分 tile 有前景
    items = tile_and_write(img, mask, "samp", str(tmp_path), 512, 256, 0.1,
                           np.random.default_rng(0))
    assert len(items) > 0
    # 檔案存在且命名合規
    for it in items:
        assert (tmp_path / "images" / it["tile"]).exists()
        assert (tmp_path / "masks" / it["tile"]).exists()
        assert it["tile"].startswith("samp__y")
    # 至少一個前景 tile, 且其 mask 為 {0,1}
    fg_items = [it for it in items if it["has_fg"]]
    assert fg_items
    m = np.array(Image.open(tmp_path / "masks" / fg_items[0]["tile"]))
    assert set(np.unique(m)).issubset({0, 1})


def test_finalize_index(tmp_path):
    (tmp_path / "images").mkdir(); (tmp_path / "masks").mkdir()
    items = [{"tile": "a__y00000_x00000.png", "stem": "a", "y": 0, "x": 0,
              "has_fg": True, "tile_std": 10.0, "fg_pixels": 50}]
    finalize_index(str(tmp_path), items, {"target_class": "crack"}, seed=42)
    idx = json.load(open(tmp_path / "tile_index.json"))
    assert idx["summary"]["target_class"] == "crack"
    assert idx["items"] == items
    split = json.load(open(tmp_path / "nofold_all_train.json"))
    assert split["group_by"] == "stem"
    assert "a" in split["groups"]
