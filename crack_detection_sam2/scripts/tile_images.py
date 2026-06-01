import argparse
import os
import sys
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_utils import list_images, load_image, tile_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs/default.yaml"))
    parser.add_argument("--src", default=None)
    parser.add_argument("--dst", default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--drop_blank", type=float, default=None,
                        help="若 tile 灰階標準差低於此值則丟棄(全黑/全白)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    src = args.src or cfg["project"]["raw_image_dir"]
    dst = args.dst or cfg["project"]["tiled_dir"]
    size = args.size or cfg["tile"]["size"]
    stride = args.stride or cfg["tile"]["stride"]
    drop_blank = args.drop_blank if args.drop_blank is not None else cfg["tile"]["drop_blank_threshold"]

    os.makedirs(dst, exist_ok=True)
    paths = list_images(src)
    print(f"來源: {src} ({len(paths)} 張), 輸出: {dst}, tile={size}, stride={stride}")

    n_total, n_kept = 0, 0
    for path in tqdm(paths):
        stem = os.path.splitext(os.path.basename(path))[0]
        img = load_image(path)
        tiles, coords, _ = tile_image(img, tile_size=size, stride=stride)
        for tile, (y, x) in zip(tiles, coords):
            n_total += 1
            if drop_blank > 0 and tile.astype(np.float32).std() < drop_blank:
                continue
            out = os.path.join(dst, f"{stem}__y{y:05d}_x{x:05d}.png")
            Image.fromarray(tile).save(out)
            n_kept += 1

    print(f"完成: 產出 {n_kept}/{n_total} tiles")


if __name__ == "__main__":
    main()
