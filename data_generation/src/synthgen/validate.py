import os
import glob
import numpy as np
from PIL import Image


def mask_stats(mask_dir):
    """統計一個 masks 目錄的前景佔比分佈。"""
    pcts = []
    for mp in sorted(glob.glob(os.path.join(mask_dir, "*.png"))):
        m = np.array(Image.open(mp))
        if m.ndim == 3:
            m = m[..., 0]
        pcts.append(float((m > 0).mean() * 100))
    pcts = pcts or [0.0]
    return {"n": len([p for p in pcts]), "fg_mean_pct": float(np.mean(pcts)),
            "fg_min_pct": float(np.min(pcts)), "fg_max_pct": float(np.max(pcts)),
            "fg_pcts": pcts}


def compare(real_stats, synth_stats):
    """合成 mean 是否落在真實 [min,max] 範圍內。"""
    rlo, rhi = min(real_stats["fg_pcts"]), max(real_stats["fg_pcts"])
    sm = synth_stats["fg_mean_pct"]
    return {
        "real_mean_pct": real_stats["fg_mean_pct"],
        "synth_mean_pct": sm,
        "real_range_pct": [rlo, rhi],
        "synth_mean_within_real_minmax": bool(rlo <= sm <= rhi),
    }
