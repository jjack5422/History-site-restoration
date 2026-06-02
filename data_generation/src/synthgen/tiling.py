import json
import os
import numpy as np
from PIL import Image
from crackseg_common.data_utils import tile_image


def tile_and_write(img, mask, stem, out_root, tile_size, stride,
                   keep_negative_ratio, rng):
    """切 img/mask 成 tile 並寫檔, 回傳保留 tile 的 item dict list。"""
    img_dir = os.path.join(out_root, "images")
    msk_dir = os.path.join(out_root, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    img_tiles, coords, _ = tile_image(img, tile_size=tile_size, stride=stride)
    msk_tiles, _, _ = tile_image(mask, tile_size=tile_size, stride=stride)

    items = []
    for (it_img, (y, x), it_msk) in zip(img_tiles, coords, msk_tiles):
        fg_px = int((it_msk > 0).sum())
        has_fg = fg_px > 0
        if not has_fg and rng.uniform() >= keep_negative_ratio:
            continue  # 零前景 tile 依比例抽樣保留
        name = f"{stem}__y{y:05d}_x{x:05d}.png"
        Image.fromarray(it_img).save(os.path.join(img_dir, name))
        Image.fromarray((it_msk > 0).astype(np.uint8)).save(os.path.join(msk_dir, name))
        items.append({
            "tile": name, "stem": stem, "y": int(y), "x": int(x),
            "has_fg": has_fg, "tile_std": float(it_img.astype(np.float32).std()),
            "fg_pixels": fg_px,
        })
    return items


def finalize_index(out_root, items, summary_extra, seed):
    """寫 tile_index.json 與 nofold_all_train.json。"""
    n_fg = sum(1 for it in items if it["has_fg"])
    summary = {
        "tile_size": None, "stride": None, "seed": seed,
        "total_tiles": len(items), "kept_foreground": n_fg,
        "kept_background": len(items) - n_fg,
    }
    summary.update(summary_extra)
    with open(os.path.join(out_root, "tile_index.json"), "w") as fh:
        json.dump({"summary": summary, "items": items}, fh, ensure_ascii=False, indent=2)

    groups = sorted({it["stem"] for it in items})
    split = {
        "tiles_root": os.path.abspath(out_root),
        "group_by": "stem", "n_splits": 1, "seed": seed,
        "groups": groups,
        "folds": [{
            "fold": 0, "val_groups": [], "train_groups": groups,
            "n_train_tiles": len(items), "n_val_tiles": 0,
        }],
    }
    with open(os.path.join(out_root, "nofold_all_train.json"), "w") as fh:
        json.dump(split, fh, ensure_ascii=False, indent=2)
