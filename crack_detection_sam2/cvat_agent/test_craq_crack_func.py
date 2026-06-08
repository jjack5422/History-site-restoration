"""Offline test: the native function loads and returns valid CVAT mask shapes on a real tile."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import craq_crack_func as F  # noqa: E402
from cvat_sdk.masks import decode_mask  # noqa: E402


class _Ctx:
    conf_threshold = None


def _a_craq_tile():
    # pick a tile that the prior pre-label run found craquelure in (binary_craq > 0)
    binc = Path("/home/zzz90/research/crack_detection_sam2/runs/2026-06-08-prelabel-selected/merged/binary_craq")
    for p in sorted(binc.glob("*.png")):
        if np.asarray(Image.open(p)).any():
            for root in ("batch_1", "batch_2", "batch_3"):
                jpg = Path("/home/zzz90/research/_data/selected_slices") / root / f"{p.stem}.jpg"
                if jpg.exists():
                    return jpg
    raise AssertionError("no craq-positive tile found")


def test_detect_returns_valid_mask_shapes():
    fn = F.create()  # defaults: both final ckpts, tile512/stride256, thr0.5
    spec_names = {l.name for l in fn.spec.labels}
    assert spec_names == {"craquelure", "crack"}

    jpg = _a_craq_tile()
    image = Image.open(jpg).convert("RGB")
    shapes = fn.detect(_Ctx(), image)

    assert isinstance(shapes, list)
    assert len(shapes) >= 1, "expected at least one mask on a craq-positive tile"
    label_ids = {s.label_id for s in shapes}
    assert label_ids <= {0, 1}
    for s in shapes:
        assert str(s.type) == "mask"  # s.type is a ShapeType enum, not a bare str
        m = decode_mask(s.points, image_width=image.width, image_height=image.height)
        assert m.shape == (image.height, image.width) and m.any()
