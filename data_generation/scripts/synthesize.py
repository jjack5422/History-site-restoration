"""主合成流程: 生成 crack/craq 分型別 binary tile 資料集到 _data/synth_*_v0。

用法: python scripts/synthesize.py --config configs/synth_v0.yaml [--overwrite]
"""
import argparse
import os
import sys
import json
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from synthgen.config import load_config
from synthgen.appearance import load_profile
from synthgen.compose import compose
from synthgen.tiling import tile_and_write, finalize_index
from synthgen.geometry import crack as crack_eng
from synthgen.geometry import craquelure as craq_eng

ENGINES = {"crack": crack_eng.generate, "craq": craq_eng.generate}


def _load_base(slices_dir, name):
    bgr = cv2.imread(os.path.join(slices_dir, name))
    if bgr is None:
        raise FileNotFoundError(os.path.join(slices_dir, name))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def synth_type(typ, cfg, rng):
    out_root = cfg["output"][typ]
    if os.path.exists(out_root):
        raise FileExistsError(f"{out_root} 已存在, 加 --overwrite 或先移除")
    os.makedirs(out_root)

    manifest = json.load(open(os.path.join(ROOT, cfg["base"]["manifest"])))
    bases = manifest["clean"] or [s["name"] for s in
                                   sorted(manifest["scores"], key=lambda s: s["score"])[:30]]
    if not bases:
        raise ValueError(f"{typ}: manifest 無可用底圖 (clean 與 scores 皆空): "
                         f"{cfg['base']['manifest']}")
    profile = load_profile(os.path.join(ROOT, cfg["appearance"]["profile_dir"],
                                        f"appearance_profile_{typ}.json"))
    # 底圖目錄以 manifest 的 slices_dir 為準(手動挑選模式指向使用者資料夾),
    # 沒有才退回 config 設定。
    slices_dir = manifest.get("slices_dir") or cfg["base"]["slices_dir"]
    size = cfg["slice_size"]
    fg_lo, fg_hi = cfg[typ]["target_fg"]
    gen = ENGINES[typ]

    items, n_target, reuse = [], cfg["target_tiles"][typ], 0
    base_idx = 0
    sample_id = 0
    attempts = 0
    max_attempts = cfg.get("max_attempts", n_target * 50)
    while sum(1 for it in items if it["has_fg"]) < n_target:
        attempts += 1
        if attempts > max_attempts:
            n_fg = sum(1 for it in items if it["has_fg"])
            raise RuntimeError(
                f"{typ}: 無法達到 target_fg,已嘗試 {attempts} 次,"
                f"目前 fg tiles={n_fg}/{n_target};請檢查 {typ}.target_fg 或 geometry 參數")
        name = bases[base_idx % len(bases)]
        if base_idx >= len(bases):
            reuse = base_idx // len(bases)
            if reuse > cfg["base"]["max_reuse"]:
                print(f"[warn] {typ}: base 重用倍數 {reuse} 超過上限 {cfg['base']['max_reuse']}")
        base_idx += 1
        base = _load_base(slices_dir, name)

        # 生成幾何, 退化(空/超範圍)就重抽
        geo = None
        for _ in range(cfg["max_regen_fail"]):
            cand = gen(size, cfg[typ], rng)
            fg = cand.mean() * 100
            if fg_lo <= fg <= fg_hi:
                geo = cand
                break
        if geo is None:
            continue

        img, msk = compose(base, geo, profile, cfg["appearance"], rng)
        stem = f"{os.path.splitext(name)[0]}__s{sample_id:04d}"
        sample_id += 1
        items += tile_and_write(img, msk, stem, out_root, cfg["tile_size"],
                                cfg["stride"], cfg["keep_negative_ratio"], rng)

    summary = {"target_class": typ, "tile_size": cfg["tile_size"],
               "stride": cfg["stride"], "n_bases": len(bases),
               "base_reuse_factor": reuse, "n_samples": sample_id}
    finalize_index(out_root, items, summary, cfg["seed"])
    return {"type": typ, "out": out_root, "n_tiles": len(items),
            "n_fg": sum(1 for it in items if it["has_fg"]),
            "n_samples": sample_id, "reuse": reuse}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth_v0.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(os.path.join(ROOT, args.config))

    if args.overwrite:
        import shutil
        for typ in ("crack", "craq"):
            if os.path.exists(cfg["output"][typ]):
                shutil.rmtree(cfg["output"][typ])

    ss = np.random.SeedSequence(cfg["seed"])
    child = dict(zip(("crack", "craq"), ss.spawn(2)))
    results = []
    for typ in ("crack", "craq"):
        results.append(synth_type(typ, cfg, np.random.default_rng(child[typ])))
    for r in results:
        print(f"{r['type']}: tiles={r['n_tiles']} (fg={r['n_fg']}) "
              f"samples={r['n_samples']} reuse={r['reuse']} -> {r['out']}")


if __name__ == "__main__":
    main()
