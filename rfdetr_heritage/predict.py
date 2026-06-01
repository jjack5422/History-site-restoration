import argparse
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PIL import Image

from rfdetr import RFDETRSegSmall


CLASSES = {1: "crack", 2: "loss", 3: "shrinkage", 4: "craquelure"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="path to fine-tuned .pth")
    parser.add_argument("--source", required=True, help="image file or directory")
    parser.add_argument("--out", default="/home/zzz90/research/rfdetr_heritage/runs/predict")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=640)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = RFDETRSegSmall(pretrain_weights=args.weights, resolution=args.resolution)

    src = Path(args.source)
    if src.is_dir():
        paths = sorted([p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    else:
        paths = [src]

    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    for p in paths:
        image = Image.open(p).convert("RGB")
        detections = model.predict(image, threshold=args.threshold)

        labels = [
            f"{CLASSES.get(int(c), str(int(c)))} {conf:.2f}"
            for c, conf in zip(detections.class_id, detections.confidence)
        ]

        annotated = np.array(image)
        annotated = mask_annotator.annotate(annotated, detections)
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections, labels=labels)

        cv2.imwrite(str(out_dir / p.name), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"{p.name}: {len(detections)} dets")


if __name__ == "__main__":
    main()
