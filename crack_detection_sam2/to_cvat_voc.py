"""Pack binary craquelure masks into a CVAT "Segmentation mask 1.1" import folder.

CVAT import matches SegmentationClass/<stem>.png to the task image <stem>.* by
name and decodes pixel colours via labelmap.txt. Craquelure = (102,255,102),
matching the existing _data/0-94 labelmap so it lands on the right class.

Usage (one folder holding all images you put in the CVAT task):
  sam2_env/bin/python to_cvat_voc.py \
      --out_dir .../cvat_import \
      --mask KJTHT-SC-L-1RB1-1=.../draft/draft_mask.png \
      --mask KJTHT-SC-R-A4-3=.../draft/gold_R-A4-3/draft_mask.png

Then in CVAT: create a task with the matching images, Actions -> Upload
annotations -> "Segmentation mask 1.1" -> zip this folder.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
CRAQ_RGB = (102, 255, 102)
LABELMAP = (
    "# label:color_rgb:parts:actions\n"
    "background:0,0,0::\n"
    "craquelure:102,255,102::\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mask", action="append", required=True,
                    help="stem=path/to/binary_mask.png ; repeatable")
    ap.add_argument("--zip", action="store_true", help="also write <out_dir>.zip")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "SegmentationClass").mkdir(parents=True, exist_ok=True)
    (out / "ImageSets" / "Segmentation").mkdir(parents=True, exist_ok=True)
    (out / "labelmap.txt").write_text(LABELMAP)

    stems = []
    for spec in args.mask:
        stem, path = spec.split("=", 1)
        m = np.array(Image.open(path).convert("L")) > 127
        rgb = np.zeros((*m.shape, 3), dtype=np.uint8)
        rgb[m] = CRAQ_RGB
        Image.fromarray(rgb).save(out / "SegmentationClass" / f"{stem}.png")
        stems.append(stem)
        print(f"{stem}: {m.shape[1]}x{m.shape[0]}  craq%={100*m.mean():.2f}")

    (out / "ImageSets" / "Segmentation" / "default.txt").write_text("\n".join(stems) + "\n")
    print(f"wrote CVAT Segmentation-mask folder -> {out}")

    if args.zip:
        zpath = str(out) + ".zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in out.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(out))
        print(f"wrote {zpath}")


if __name__ == "__main__":
    main()
