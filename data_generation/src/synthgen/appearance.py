import json
import os
import glob
import numpy as np
import cv2


def _to_lab(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _to_rgb(lab):
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def fit_profile(img_dir, mask_dir):
    """統計裂縫前景相對其周邊鄰域(dilate ring)的 Lab 差, 聚合成分佈。"""
    dL, da, db = [], [], []
    for ip in sorted(glob.glob(os.path.join(img_dir, "*"))):
        stem = os.path.splitext(os.path.basename(ip))[0]
        mp = None
        for ext in (".png", ".jpg", ".jpeg"):
            cand = os.path.join(mask_dir, stem + ext)
            if os.path.exists(cand):
                mp = cand
                break
        if mp is None:
            continue
        rgb = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if rgb is None or m is None:
            continue
        fg = m > 0
        if fg.sum() == 0:
            continue
        ring = cv2.dilate(fg.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        ring &= ~fg
        if ring.sum() == 0:
            continue
        lab = _to_lab(rgb)
        for ch, acc in zip(range(3), (dL, da, db)):
            acc.append(float(lab[..., ch][fg].mean() - lab[..., ch][ring].mean()))
    def ms(x):
        return [float(np.mean(x)), float(np.std(x))] if x else [0.0, 0.0]
    return {"dL": ms(dL), "da": ms(da), "db": ms(db), "n": len(dL)}


def save_profile(profile, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(profile, fh, ensure_ascii=False, indent=2)


def load_profile(path):
    with open(path) as fh:
        return json.load(fh)


def render(base_rgb, geo_mask, profile, cfg, rng):
    """把 geo_mask 以資料驅動 Lab offset 渲染到 base_rgb, 回傳 RGB uint8。"""
    geo = (geo_mask > 0).astype(np.uint8)
    if cfg.get("erosion", 0) > 0:
        k = np.ones((cfg["erosion"], cfg["erosion"]), np.uint8)
        geo = cv2.erode(geo, k)
    alpha = geo.astype(np.float32)
    if cfg.get("blur_sigma", 0) > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), cfg["blur_sigma"])
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    dL = rng.normal(*profile["dL"])
    mc = cfg.get("min_contrast", 0)
    if mc and abs(dL) < mc:
        dL = -mc if dL <= 0 else mc  # 維持原方向但拉到最小對比
    da = rng.normal(*profile["da"])
    db = rng.normal(*profile["db"])
    delta = np.array([dL, da, db], dtype=np.float32)

    lab = _to_lab(base_rgb)
    lab = lab + alpha * delta
    return _to_rgb(lab)
