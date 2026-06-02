#!/usr/bin/env python3
"""
sea_setup.py — Make ultralytics aware of C2f_SEA (Squeeze-and-Excitation C2f) for SepSAM.

WHY THIS IS NEEDED
------------------
ultralytics rebuilds the model from the YAML at train time (the trainer does NOT reuse
a model object you modified in Python). So the SEA module must be recognised by
ultralytics' own `parse_model`. This script patches the installed
`ultralytics/nn/tasks.py` to:
  (1) define C2f_SEA + SE at module level, and
  (2) add C2f_SEA to parse_model's module-recognition sets so the channel-args and
      repeat-count handling fire correctly (same treatment as the stock C2f).

PROPERTIES
----------
- Idempotent: safe to run multiple times.
- Creates a one-time backup next to the file: tasks.py.sepsam.bak
- Re-run this after every `pip install -U ultralytics` (a reinstall reverts the patch).

USAGE
-----
    python sea_setup.py
    python verify_sea.py        # then verify it worked (fresh process)
"""
import os
import re
import sys
import shutil

import ultralytics.nn.tasks as T

MARKER = "# === SepSAM C2f_SEA injected ==="

CLASS_SRC = '''# === SepSAM C2f_SEA injected ===
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
from ultralytics.nn.modules import C2f as _C2f


class SE(_nn.Module):
    """Squeeze-and-Excitation block (Hu et al., CVPR 2018). r=16 by default."""

    def __init__(self, c, r=16):
        super().__init__()
        self.fc1 = _nn.Conv2d(c, max(c // r, 1), 1)
        self.fc2 = _nn.Conv2d(max(c // r, 1), c, 1)

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)      # squeeze: global average pooling
        s = _F.relu(self.fc1(s))
        s = _torch.sigmoid(self.fc2(s))        # excitation
        return x * s                            # channel-wise recalibration


class C2f_SEA(_C2f):
    """C2f followed by an SE block on its output (SepSAM, paper Eq. 8)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.se = SE(c2)

    def forward(self, x):
        return self.se(super().forward(x))

    def forward_split(self, x):
        return self.se(super().forward_split(x))
# === end SepSAM injection ==='''


def main():
    path = T.__file__
    print("ultralytics tasks.py:", path)

    src = open(path, encoding="utf-8").read()
    original = src
    changed = False

    bak = path + ".sepsam.bak"
    if not os.path.exists(bak):
        shutil.copyfile(path, bak)
        print("backup created:", bak)

    # locate parse_model
    m = re.search(r"\ndef parse_model\(", src)
    if not m:
        print("ERROR: could not find 'def parse_model(' in tasks.py — aborting, no changes made.")
        sys.exit(1)
    def_pos = m.start() + 1  # index of 'd' in 'def'

    # --- step 1: add C2f_SEA to recognition sets (operate ONLY inside parse_model body) ---
    rest = src[def_pos + 1:]
    m2 = re.search(r"\n(def |class )", rest)
    func_end = (def_pos + 1 + m2.start()) if m2 else len(src)
    func_src = src[def_pos:func_end]
    if "C2f_SEA" not in func_src:
        # Inside parse_model, a standalone 'C2f' token only appears inside the
        # module-recognition sets; the import lives at module top-level (outside this slice).
        func_patched = re.sub(r"(?<![A-Za-z0-9_])C2f(?![A-Za-z0-9_])", "C2f, C2f_SEA", func_src)
        if func_patched != func_src:
            src = src[:def_pos] + func_patched + src[func_end:]
            changed = True
            print("- added C2f_SEA to parse_model recognition set(s)")
    else:
        print("- parse_model already references C2f_SEA, skip")

    # --- step 2: insert class definition just before parse_model ---
    if MARKER not in src:
        m = re.search(r"\ndef parse_model\(", src)  # recompute (src may have shifted)
        def_pos = m.start() + 1
        src = src[:def_pos] + CLASS_SRC.strip("\n") + "\n\n\n" + src[def_pos:]
        changed = True
        print("- inserted C2f_SEA / SE class definitions")
    else:
        print("- class definition already present, skip")

    if changed and src != original:
        open(path, "w", encoding="utf-8").write(src)
        print("\nPATCH WRITTEN \u2713")
        print("Next:  python verify_sea.py")
    else:
        print("\nNothing to change — ultralytics already patched.")


if __name__ == "__main__":
    main()
