import argparse
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PIL import Image

from rfdetr import RFDETRSegSmall


CLASSES = {1: "crack", 2: "loss", 3: "shrinkage", 4: "craquelure"}
COLORS = {
    1: (0, 0, 255),
    2: (0, 255, 0),
    3: (255, 0, 0),
    4: (0, 255, 255),
}


def tile_coords(W, H, tile, overlap):
    stride = tile - overlap
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    if xs[-1] + tile < W:
        xs.append(W - tile)
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if W <= tile:
        xs = [0]
    if H <= tile:
        ys = [0]
    coords = [(x, y) for y in ys for x in xs]
    return coords


def pad_tile(arr, tile):
    h, w = arr.shape[:2]
    if h == tile and w == tile:
        return arr, (h, w)
    out = np.zeros((tile, tile, 3), dtype=arr.dtype)
    out[:h, :w] = arr
    return out, (h, w)


def merge_masks_to_canvas(canvas_masks, canvas_classes, canvas_scores, dets, ox, oy, valid_h, valid_w, H, W):
    if len(dets) == 0:
        return canvas_masks, canvas_classes, canvas_scores
    if dets.mask is None:
        return canvas_masks, canvas_classes, canvas_scores
    new_masks, new_cls, new_sc = [], [], []
    for m, cid, s in zip(dets.mask, dets.class_id, dets.confidence):
        m = m[:valid_h, :valid_w]
        full = np.zeros((H, W), dtype=bool)
        full[oy:oy + valid_h, ox:ox + valid_w] = m
        new_masks.append(full)
        new_cls.append(int(cid))
        new_sc.append(float(s))
    canvas_masks.extend(new_masks)
    canvas_classes.extend(new_cls)
    canvas_scores.extend(new_sc)
    return canvas_masks, canvas_classes, canvas_scores


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return inter / union


def mask_nms(masks, classes, scores, iou_thr=0.5):
    keep = []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    suppressed = [False] * len(scores)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if j == i or suppressed[j]:
                continue
            if classes[i] != classes[j]:
                continue
            if mask_iou(masks[i], masks[j]) > iou_thr:
                suppressed[j] = True
    return keep


def draw(image_bgr, masks, classes, scores, alpha=0.5):
    out = image_bgr.copy()
    overlay = out.copy()
    for m, cid in zip(masks, classes):
        color = COLORS.get(cid, (255, 255, 255))
        overlay[m] = color
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    for m, cid, s in zip(masks, classes, scores):
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        color = COLORS.get(cid, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASSES.get(cid, cid)} {s:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", default="/home/zzz90/research/rfdetr_heritage/runs/predict_tiled")
    parser.add_argument("--tile", type=int, default=384)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = RFDETRSegSmall(pretrain_weights=args.weights, resolution=args.tile)

    img = np.array(Image.open(args.source).convert("RGB"))
    H, W = img.shape[:2]
    print(f"image: {W}x{H}, tile={args.tile}, overlap={args.overlap}")

    coords = tile_coords(W, H, args.tile, args.overlap)
    print(f"tiles: {len(coords)}")

    canvas_masks, canvas_cls, canvas_sc = [], [], []

    for i, (ox, oy) in enumerate(coords):
        crop = img[oy:oy + args.tile, ox:ox + args.tile]
        crop_pad, (vh, vw) = pad_tile(crop, args.tile)
        pil = Image.fromarray(crop_pad)
        dets = model.predict(pil, threshold=args.threshold)
        n = len(dets) if dets is not None else 0
        print(f"  tile {i+1}/{len(coords)} @({ox},{oy}) -> {n} dets")
        canvas_masks, canvas_cls, canvas_sc = merge_masks_to_canvas(
            canvas_masks, canvas_cls, canvas_sc, dets, ox, oy, vh, vw, H, W
        )

    print(f"raw merged dets: {len(canvas_masks)}")
    keep = mask_nms(canvas_masks, canvas_cls, canvas_sc, iou_thr=args.nms_iou)
    masks = [canvas_masks[i] for i in keep]
    classes = [canvas_cls[i] for i in keep]
    scores = [canvas_sc[i] for i in keep]
    print(f"after NMS: {len(masks)}")

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    annotated = draw(bgr, masks, classes, scores)
    src_name = Path(args.source).stem
    out_path = out_dir / f"{src_name}_tiled.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"saved: {out_path}")

    if masks:
        cls_arr = np.zeros((H, W), dtype=np.uint8)
        for m, cid in zip(masks, classes):
            cls_arr[m] = cid
        cv2.imwrite(str(out_dir / f"{src_name}_classmap.png"), cls_arr)
        print(f"saved classmap: {out_dir / (src_name + '_classmap.png')}")


if __name__ == "__main__":
    main()
