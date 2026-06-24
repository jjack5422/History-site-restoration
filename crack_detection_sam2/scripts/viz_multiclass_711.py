"""Visualize multiclass_512_dataset_711 masks: image | colored mask | overlay.

Picks tiles that together cover every class (prioritizing rare/multi-class ones)
and writes panels + a class-coverage montage + legend into <dataset>/viz/.

Run:
    /home/zzz90/research/sam2_env/bin/python \
        crack_detection_sam2/scripts/viz_multiclass_711.py
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DS = "/home/zzz90/research/_data/multiclass_512_dataset_711"
VIZ = os.path.join(DS, "viz")
os.makedirs(VIZ, exist_ok=True)

NAMES = {0: "background", 1: "crack", 2: "loss", 3: "shrinkage",
         4: "craquelure", 5: "flaking", 255: "ignore"}
# distinct display colors (RGB 0-255)
COLORS = {
    0: (0, 0, 0),
    1: (255, 40, 40),       # crack - red
    2: (40, 120, 255),      # loss - blue
    3: (170, 0, 230),       # shrinkage - purple
    4: (60, 220, 60),       # craquelure - green
    5: (255, 230, 0),       # flaking - yellow
    255: (255, 130, 90),    # ignore - salmon
}


def colorize(idx: np.ndarray) -> np.ndarray:
    h, w = idx.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c, col in COLORS.items():
        out[idx == c] = col
    return out


def overlay(img: np.ndarray, idx: np.ndarray, alpha=0.55) -> np.ndarray:
    col = colorize(idx)
    fg = idx != 0
    out = img.copy()
    out[fg] = (alpha * col[fg] + (1 - alpha) * img[fg]).astype(np.uint8)
    return out


def load(tile):
    img = np.array(Image.open(os.path.join(DS, "images", tile)).convert("RGB"))
    idx = np.array(Image.open(os.path.join(DS, "masks", tile)))
    return img, idx


def main():
    masks_dir = os.path.join(DS, "masks")
    files = sorted(os.listdir(masks_dir))

    # index: which classes each tile has, and #non-bg classes
    info = {}
    for f in files:
        idx = np.array(Image.open(os.path.join(masks_dir, f)))
        classes = set(int(c) for c in np.unique(idx))
        info[f] = classes

    # greedy pick: cover every non-bg class, prefer multi-class tiles
    targets = [1, 2, 3, 4, 5, 255]
    chosen = []
    covered = set()
    # rare-first so shrinkage/flaking/ignore definitely appear
    for t in [3, 5, 255, 2, 1, 4]:
        cands = [f for f in files if t in info[f] and f not in chosen]
        if not cands:
            continue
        # prefer the most multi-class example for this target
        cands.sort(key=lambda f: -len([c for c in info[f] if c not in (0, 255)]))
        f = cands[0]
        chosen.append(f)
        covered |= info[f]
    # add a few extra rich multi-class tiles
    extra = sorted(files, key=lambda f: -len([c for c in info[f] if c not in (0, 255)]))
    for f in extra:
        if len(chosen) >= 9:
            break
        if f not in chosen and len([c for c in info[f] if c not in (0, 255)]) >= 2:
            chosen.append(f)

    n = len(chosen)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.4 * n))
    if n == 1:
        axes = axes[None, :]
    for r, tile in enumerate(chosen):
        img, idx = load(tile)
        present = sorted(c for c in info[tile] if c != 0)
        title = ", ".join(NAMES[c] for c in present) or "background"
        for c, (im, t) in enumerate([(img, "image"),
                                     (colorize(idx), "mask"),
                                     (overlay(img, idx), "overlay")]):
            axes[r, c].imshow(im)
            axes[r, c].axis("off")
            axes[r, c].set_title(t if r else {0: "image", 1: "mask", 2: "overlay"}[c],
                                 fontsize=11)
        axes[r, 0].set_ylabel(tile.replace(".png", ""), fontsize=7)
        axes[r, 1].set_title(title, fontsize=9)
    handles = [Patch(facecolor=np.array(COLORS[c]) / 255, edgecolor="k", label=NAMES[c])
               for c in [1, 2, 3, 4, 5, 255]]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("multiclass_512_dataset_711 — sample tiles (image | mask | overlay)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    out = os.path.join(VIZ, "samples_image_mask_overlay.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)
    print("tiles:", chosen)

    # standalone legend
    figl, axl = plt.subplots(figsize=(6, 1.4))
    axl.axis("off")
    axl.legend(handles=handles, loc="center", ncol=3, fontsize=11, frameon=False)
    figl.savefig(os.path.join(VIZ, "legend.png"), dpi=120, bbox_inches="tight")
    plt.close(figl)
    print("wrote", os.path.join(VIZ, "legend.png"))


if __name__ == "__main__":
    main()
