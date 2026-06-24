import argparse
import json
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from crackseg_common.data_utils import tile_image


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_pairs(image_dir, mask_dir):
    pairs = []
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(IMG_EXTS):
            continue
        stem, _ = os.path.splitext(fname)
        mpath = os.path.join(mask_dir, stem + ".png")
        if not os.path.isfile(mpath):
            print(f"[skip] mask 不存在: {mpath}")
            continue
        pairs.append((stem, os.path.join(image_dir, fname), mpath))
    return pairs


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def load_label(path):
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default=os.path.join(
        PROJECT_ROOT, "merged_4class_mask_semantic/images"))
    parser.add_argument("--mask_dir", default=os.path.join(
        PROJECT_ROOT, "merged_4class_mask_semantic/masks"))
    parser.add_argument("--out_dir", default=os.path.join(
        os.path.dirname(PROJECT_ROOT), "_data/labeled32_craq_v3/tiles_512"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=5,
                        help="包含 background, label 範圍 [0, num_classes-1]")
    parser.add_argument("--bg_keep_ratio", type=float, default=0.15,
                        help="純背景 tile 隨機保留比例")
    parser.add_argument("--bg_std_threshold", type=float, default=5.0,
                        help="若 tile 灰階 std 低於此值且為純背景則丟棄")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    img_out = os.path.join(args.out_dir, "images")
    msk_out = os.path.join(args.out_dir, "masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(msk_out, exist_ok=True)

    pairs = list_pairs(args.image_dir, args.mask_dir)
    if not pairs:
        raise SystemExit("找不到任何 image/mask 配對")

    index = []
    n_total = 0
    n_kept_fg = 0
    n_kept_bg = 0
    n_drop_blank = 0
    n_drop_bg_subsample = 0

    class_counts = np.zeros(args.num_classes, dtype=np.int64)

    for stem, img_path, mask_path in tqdm(pairs):
        img = load_rgb(img_path)
        msk = load_label(mask_path)
        if img.shape[:2] != msk.shape[:2]:
            print(f"[warn] shape 不一致 skip: {stem} img={img.shape} msk={msk.shape}")
            continue

        img_tiles, coords, _ = tile_image(img, tile_size=args.size, stride=args.stride, pad_value=0)
        msk_tiles, _, _ = tile_image(msk, tile_size=args.size, stride=args.stride, pad_value=0)

        for img_t, msk_t, (y, x) in zip(img_tiles, msk_tiles, coords):
            n_total += 1
            has_fg = bool((msk_t > 0).any())
            tile_std = float(img_t.astype(np.float32).std())

            if not has_fg:
                if tile_std < args.bg_std_threshold:
                    n_drop_blank += 1
                    continue
                if rng.random() > args.bg_keep_ratio:
                    n_drop_bg_subsample += 1
                    continue

            tile_name = f"{stem}__y{y:05d}_x{x:05d}.png"
            Image.fromarray(img_t).save(os.path.join(img_out, tile_name))
            Image.fromarray(msk_t).save(os.path.join(msk_out, tile_name))

            for c in range(args.num_classes):
                class_counts[c] += int((msk_t == c).sum())

            index.append({
                "tile": tile_name,
                "stem": stem,
                "y": int(y),
                "x": int(x),
                "has_fg": has_fg,
                "tile_std": tile_std,
            })
            if has_fg:
                n_kept_fg += 1
            else:
                n_kept_bg += 1

    summary = {
        "args": vars(args),
        "total_tiles": n_total,
        "kept_foreground": n_kept_fg,
        "kept_background_sampled": n_kept_bg,
        "dropped_blank": n_drop_blank,
        "dropped_background_subsample": n_drop_bg_subsample,
        "class_pixel_counts": {str(c): int(class_counts[c]) for c in range(args.num_classes)},
    }
    with open(os.path.join(args.out_dir, "tile_index.json"), "w") as f:
        json.dump({"summary": summary, "items": index}, f, indent=2)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"輸出: {args.out_dir}")


if __name__ == "__main__":
    main()
