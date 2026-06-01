"""把 v2 5-class semantic mask 重映射成 3-class (bg / crack / craquelure)。

- 來源 mask 已用 priority crack -> loss -> shrinkage -> craquelure (後者覆寫前者),
  所以 craquelure 在重疊處已優先於 crack。本腳本只是 drop loss 與 shrinkage。
- 輸出: {out_dir}/{images,masks,classes.csv}
  - images: symlink 至原圖
  - masks:  remap 後的單通道 png (0/1/2)
"""
import argparse
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm


REMAP = {0: 0, 1: 1, 2: 0, 3: 0, 4: 2}


def remap_mask(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.uint8)
    for src, dst in REMAP.items():
        if dst != 0:
            out[arr == src] = dst
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", default="merged_4class_mask_semantic_v2")
    parser.add_argument("--out_dir", default="merged_2class_crack_craquelure_v2")
    args = parser.parse_args()

    src_img = os.path.join(args.src_dir, "images")
    src_msk = os.path.join(args.src_dir, "masks")
    out_img = os.path.join(args.out_dir, "images")
    out_msk = os.path.join(args.out_dir, "masks")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_msk, exist_ok=True)

    counts = np.zeros(3, dtype=np.int64)
    n = 0
    for fname in tqdm(sorted(os.listdir(src_msk))):
        if not fname.lower().endswith(".png"):
            continue
        m = np.array(Image.open(os.path.join(src_msk, fname)))
        if m.ndim == 3:
            m = m[..., 0]
        m2 = remap_mask(m.astype(np.uint8))
        Image.fromarray(m2).save(os.path.join(out_msk, fname))
        for c in range(3):
            counts[c] += int((m2 == c).sum())
        n += 1

    for fname in sorted(os.listdir(src_img)):
        src = os.path.join(src_img, fname)
        dst = os.path.join(out_img, fname)
        if os.path.lexists(dst):
            continue
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError:
            shutil.copy2(src, dst)

    with open(os.path.join(args.out_dir, "classes.csv"), "w") as f:
        f.write("Pixel Value,Class,Visible RGB\n")
        f.write("0,background,\"0,0,0\"\n")
        f.write("1,crack,\"255,0,0\"\n")
        f.write("2,craquelure,\"255,0,255\"\n")

    print(f"masks remapped: {n}")
    print(f"class pixel counts: bg={counts[0]} crack={counts[1]} craquelure={counts[2]}")
    print(f"out: {args.out_dir}")


if __name__ == "__main__":
    main()
