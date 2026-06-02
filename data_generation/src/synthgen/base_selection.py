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
