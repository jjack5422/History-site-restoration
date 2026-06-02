import json
import os
import glob
import numpy as np
import cv2


def score(gray, kernel=11, blackhat_t=18):
    """black-hat 突顯暗細結構, 回傳候選像素佔比 (0-1)。gray 為單通道 uint8。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    return float((bh > blackhat_t).mean())


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def build_manifest_from_dir(base_dir, out_path):
    """手動挑底圖: 把 base_dir 內所有影像全列為 clean(不做 black-hat 篩選)。

    給使用者自行把選好的切片複製到一個資料夾時用; slices_dir 指向該資料夾。
    """
    files = sorted(
        f for f in os.listdir(base_dir)
        if f.lower().endswith(IMG_EXTS)
    )
    manifest = {
        "slices_dir": base_dir,
        "mode": "manual",
        "n_total": len(files),
        "n_clean": len(files),
        "clean": files,
        "scores": [],
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return manifest


def build_manifest(slices_dir, thresh, out_path, kernel=11, blackhat_t=18):
    """掃描 slices_dir 所有影像, 依 score<thresh 篩乾淨底圖, 寫 manifest json。"""
    files = sorted(glob.glob(os.path.join(slices_dir, "*.jpg")))
    scores = []
    for f in files:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        scores.append({"name": os.path.basename(f), "score": score(g, kernel, blackhat_t)})
    clean = [s["name"] for s in scores if s["score"] < thresh]
    manifest = {
        "slices_dir": slices_dir,
        "thresh": thresh,
        "kernel": kernel,
        "blackhat_t": blackhat_t,
        "n_total": len(scores),
        "n_clean": len(clean),
        "clean": clean,
        "scores": scores,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return manifest
