#!/usr/bin/env python3
"""
verify_sea.py — confirm C2f_SEA is wired into the YOLOv8-Seg model.
Run AFTER sea_setup.py, in a fresh process.

Passing here means: ultralytics recognises C2f_SEA, the custom YAML builds, every
C2f_SEA carries an SE block, and a forward pass runs. If it fails, re-run sea_setup.py
and check the printed tasks.py path.
"""
import os
import torch
from ultralytics import YOLO

try:
    from ultralytics.nn.tasks import C2f_SEA
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"C2f_SEA not importable from ultralytics — run `python sea_setup.py` first. ({e})")

CANDIDATES = ["configs/yolov8n-seg-sea.yaml", "yolov8n-seg-sea.yaml"]
cfg = next((p for p in CANDIDATES if os.path.exists(p)), None)
if cfg is None:
    raise SystemExit("yolov8n-seg-sea.yaml not found (looked in ./configs and current dir).")

print("config:", cfg)
model = YOLO(cfg)

n_sea = sum(isinstance(m, C2f_SEA) for m in model.model.modules())
print("C2f_SEA modules:", n_sea)
assert n_sea > 0, "No C2f_SEA found — parse_model did not pick up the custom module."
assert all(hasattr(m, "se") for m in model.model.modules() if isinstance(m, C2f_SEA)), \
    "A C2f_SEA is missing its .se block."

print("forward test (1x3x416x416)...")
model.model.eval()
with torch.no_grad():
    _ = model.model(torch.zeros(1, 3, 416, 416))

print("VERIFICATION PASSED \u2713  (C2f_SEA active, forward OK)")
