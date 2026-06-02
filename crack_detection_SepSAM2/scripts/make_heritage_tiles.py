"""
make_heritage_tiles.py — 把 heritage_ft（1024x1024 全圖）切成 tile，產生
YOLO-seg 訓練資料集，解決 (a) 資料稀缺、(b) 全圖訓練→薄裂紋尺度過小 兩個問題。

作法（rasterize 法，比多邊形裁剪穩健）：
  對每張影像與其既有二值 mask，以 tile/stride 滑窗切片；
  每個 tile 對裁切後的 mask 跑 cv2.findContours 重新生成多邊形 →
  寫成 ultralytics YOLO-seg 標籤（class x1 y1 x2 y2 ... 皆 normalized 到 tile）。
  邊界切斷／多片／孔洞由重新 contour 自然處理。

預設 tile=512 stride=256：1024 → 每軸起點 {0,256,512} = 9 tiles/圖。
  train 24 → 216 tiles，val 8 → 72 tiles。

範例：
  python scripts/make_heritage_tiles.py \
      --src datasets/heritage_ft --dst datasets/heritage_ft_tiles \
      --tile 512 --stride 256
"""
import argparse
import glob
import os

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def tile_origins(length, tile, stride):
    if length <= tile:
        return [0]
    xs = list(range(0, length - tile + 1, stride))
    if xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


def index_by_stem(folder):
    out = {}
    for p in glob.glob(os.path.join(folder, "*")):
        if os.path.splitext(p)[1].lower() in IMG_EXTS:
            out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


def mask_to_yolo_lines(mask_tile, tile, min_area, epsilon):
    """二值 tile mask → YOLO-seg 標籤行 list（class=0）。"""
    contours, _ = cv2.findContours(mask_tile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        if epsilon > 0:
            c = cv2.approxPolyDP(c, epsilon, closed=True)
        c = c.reshape(-1, 2)
        if c.shape[0] < 3:
            continue
        norm = (c.astype(np.float64) / float(tile)).clip(0.0, 1.0)
        coords = " ".join(f"{v:.6f}" for v in norm.reshape(-1))
        lines.append(f"0 {coords}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/heritage_ft",
                    help="來源資料集根目錄，內含 {train,val}/images 與 {train,val}/masks")
    ap.add_argument("--dst", default="datasets/heritage_ft_tiles")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--min-area", type=float, default=8.0,
                    help="丟棄面積小於此（像素）的裁切碎片多邊形")
    ap.add_argument("--epsilon", type=float, default=1.0,
                    help="approxPolyDP 簡化容差（像素），0 表示不簡化")
    ap.add_argument("--keep-empty", action="store_true", default=True,
                    help="保留無裂紋的 tile 作為背景負樣本（預設保留，利於精度）")
    ap.add_argument("--drop-empty", dest="keep_empty", action="store_false")
    args = ap.parse_args()

    for sp in args.splits:
        idir = os.path.join(args.src, sp, "images")
        mdir = os.path.join(args.src, sp, "masks")
        out_img = os.path.join(args.dst, sp, "images")
        out_lbl = os.path.join(args.dst, sp, "labels")
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)

        imgs = index_by_stem(idir)
        masks = index_by_stem(mdir)
        stems = sorted(set(imgs) & set(masks))

        n_tiles = n_pos = n_empty = n_inst = 0
        for stem in stems:
            img = cv2.imread(imgs[stem])
            msk = cv2.imread(masks[stem], cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                continue
            if msk.shape != img.shape[:2]:
                msk = cv2.resize(msk, (img.shape[1], img.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            msk = (msk > 127).astype(np.uint8) * 255
            H, W = img.shape[:2]

            for y0 in tile_origins(H, args.tile, args.stride):
                for x0 in tile_origins(W, args.tile, args.stride):
                    it = img[y0:y0 + args.tile, x0:x0 + args.tile]
                    mt = msk[y0:y0 + args.tile, x0:x0 + args.tile]
                    lines = mask_to_yolo_lines(mt, args.tile, args.min_area, args.epsilon)
                    has = len(lines) > 0
                    if not has and not args.keep_empty:
                        continue
                    name = f"{stem}_y{y0}_x{x0}"
                    cv2.imwrite(os.path.join(out_img, name + ".jpg"), it,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    with open(os.path.join(out_lbl, name + ".txt"), "w", encoding="utf-8") as f:
                        f.write("\n".join(lines))
                    n_tiles += 1
                    n_inst += len(lines)
                    n_pos += int(has)
                    n_empty += int(not has)

        print(f"[{sp}] src_imgs={len(stems)}  tiles={n_tiles} "
              f"(pos={n_pos} empty={n_empty})  instances={n_inst}")
    print(f"done → {args.dst}")


if __name__ == "__main__":
    main()
