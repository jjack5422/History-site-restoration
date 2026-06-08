"""Prove cvat_agent_env can load both experts and run one tile each. No CVAT involved."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SAM2_ROOT = Path("/home/zzz90/research/crack_detection_sam2")
UNET_SRC = Path("/home/zzz90/research/crack_detection_unet/src")
CRAQ_CKPT = SAM2_ROOT / "runs/expert_craq_v3_final_small/last.pt"
CRACK_CKPT = Path("/home/zzz90/research/crack_detection_unet/runs/expert_crack_v3_final_resnet50/last.pt")
TILE_DIR = Path("/home/zzz90/research/_data/selected_slices/batch_1")

for p in (str(SAM2_ROOT), str(UNET_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    sam2_pf = load_module("_sam2_pf", SAM2_ROOT / "predict_full.py")
    unet_pf = load_module("_unet_pf", UNET_SRC / "predict_full.py")

    craq_model, _ = sam2_pf.load_model_from_ckpt(str(CRAQ_CKPT), "cuda")
    crack_model, _ = unet_pf.load_model_from_ckpt(str(CRACK_CKPT), "cuda")

    tile = sorted(TILE_DIR.glob("*.jpg"))[0]
    img = np.asarray(Image.open(tile).convert("RGB"))

    pc = sam2_pf.predict_full(craq_model, img, "cuda", tile=512, stride=256)
    pk = unet_pf.predict_full(crack_model, img, "cuda", tile=512, stride=256)
    assert pc.shape[0] == 2 and pc.shape[1:] == img.shape[:2], pc.shape
    assert pk.shape[0] == 2 and pk.shape[1:] == img.shape[:2], pk.shape
    assert np.isfinite(pc).all() and np.isfinite(pk).all()
    print("ENV SMOKE OK:", tile.name, "craq", pc.shape, "crack", pk.shape)


if __name__ == "__main__":
    main()
