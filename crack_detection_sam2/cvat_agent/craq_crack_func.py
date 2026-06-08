"""CVAT native function: auto-annotate craquelure (SAM2 dense-seg) + crack (ResUNet) masks.

Loaded by cvat-cli (create-native / run-agent) via the create() factory. Runs both experts with
sliding-window inference, applies craq-over-crack priority, and returns one CVAT mask shape per
connected component.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import cvat_sdk.auto_annotation as cvataa
from cvat_sdk.masks import encode_mask
from scipy.ndimage import label as cc_label, find_objects

SAM2_ROOT = Path("/home/zzz90/research/crack_detection_sam2")
UNET_SRC = Path("/home/zzz90/research/crack_detection_unet/src")
DEFAULT_CRAQ_CKPT = str(SAM2_ROOT / "runs/expert_craq_v3_final_small/last.pt")
DEFAULT_CRACK_CKPT = "/home/zzz90/research/crack_detection_unet/runs/expert_crack_v3_final_resnet50/last.pt"

for _p in (str(SAM2_ROOT), str(UNET_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _CraqCrackFunction:
    def __init__(self, craq_model, crack_model, sam2_pf, unet_pf,
                 device, tile, stride, thresh_craq, thresh_crack, min_blob, priority):
        self._craq_model = craq_model
        self._crack_model = crack_model
        self._sam2_pf = sam2_pf
        self._unet_pf = unet_pf
        self._device = device
        self._tile = tile
        self._stride = stride
        self._thresh_craq = thresh_craq
        self._thresh_crack = thresh_crack
        self._min_blob = min_blob
        self._priority = priority

    @property
    def spec(self) -> cvataa.DetectionFunctionSpec:
        return cvataa.DetectionFunctionSpec(labels=[
            cvataa.label_spec("craquelure", 0, type="mask"),
            cvataa.label_spec("crack", 1, type="mask"),
        ])

    def detect(self, context, image: PIL.Image.Image):
        img = np.asarray(image.convert("RGB"))
        override = getattr(context, "conf_threshold", None)
        thr_craq = override if override is not None else self._thresh_craq
        thr_crack = override if override is not None else self._thresh_crack

        pc = self._sam2_pf.predict_full(self._craq_model, img, self._device,
                                        tile=self._tile, stride=self._stride)
        pk = self._unet_pf.predict_full(self._crack_model, img, self._device,
                                        tile=self._tile, stride=self._stride)
        craq = pc[1] > thr_craq
        crack = pk[1] > thr_crack
        if self._priority == "craq_over_crack":
            crack = crack & (~craq)

        shapes = []
        shapes += self._masks_for(craq, label_id=0)
        shapes += self._masks_for(crack, label_id=1)
        return shapes

    def _masks_for(self, mask, label_id):
        out = []
        lab, n = cc_label(mask)
        if n == 0:
            return out
        for i, sl in enumerate(find_objects(lab), start=1):
            if sl is None:
                continue
            sub = (lab[sl] == i)
            if int(sub.sum()) < self._min_blob:
                continue
            y1, x1 = sl[0].start, sl[1].start
            y2, x2 = sl[0].stop, sl[1].stop  # slice stop == exclusive upper, matches encode_mask
            comp = np.zeros(mask.shape, dtype=bool)
            comp[sl] = sub
            out.append(cvataa.mask(label_id, encode_mask(comp, [x1, y1, x2, y2])))
        return out


def create(craq_ckpt: str = DEFAULT_CRAQ_CKPT,
           crack_ckpt: str = DEFAULT_CRACK_CKPT,
           device: str = "cuda",
           tile: int = 512,
           stride: int = 256,
           thresh_craq: float = 0.5,
           thresh_crack: float = 0.5,
           min_blob: int = 64,
           priority: str = "craq_over_crack") -> _CraqCrackFunction:
    sam2_pf = _load_module("_sam2_pf", SAM2_ROOT / "predict_full.py")
    unet_pf = _load_module("_unet_pf", UNET_SRC / "predict_full.py")
    craq_model, _ = sam2_pf.load_model_from_ckpt(craq_ckpt, device)
    crack_model, _ = unet_pf.load_model_from_ckpt(crack_ckpt, device)
    return _CraqCrackFunction(craq_model, crack_model, sam2_pf, unet_pf,
                              device, tile, stride, thresh_craq, thresh_crack,
                              int(min_blob), priority)
