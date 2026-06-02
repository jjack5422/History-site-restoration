"""快篩 _data/image_1024_slices 的「乾淨區」比例。

無標註情境下,用 black-hat 形態學當「暗細裂縫/劣化」的代理指標:
black-hat = morphological_closing(gray) - gray，突顯比鄰域暗的細結構(裂縫、龜裂)。
對每張切片算 crack-candidate 像素佔比 (proxy_ratio)，輸出分佈與分級計數，
並挑最乾淨/最髒各數張輸出對照圖供人眼校準門檻。

這是一次性探索腳本（RULES §1: 一次性腳本放 scripts/）。
用法: /tmp/pdfenv/bin/python scripts/scan_clean_slices.py
"""
import os, glob, json
import numpy as np
import cv2

SLICES = "/home/zzz90/research/_data/image_1024_slices"
OUT = "/tmp/clean_scan"
os.makedirs(OUT, exist_ok=True)

# 裂縫在 1024px 下寬約數 px；closing kernel 取略大於裂縫寬度以填起裂縫再相減
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
BLACKHAT_T = 18          # black-hat 強度門檻 (0-255)，>此值視為候選暗結構
# proxy_ratio 分級（對照已標 GT: crack 均 0.226%、craq 均 2.858%）
BINS = [("clean", 0.005), ("light", 0.015), ("moderate", 0.04), ("heavy", 1.01)]


def proxy_ratio(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    bh = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, KERNEL)
    cand = bh > BLACKHAT_T
    return float(cand.mean()), img, (cand.astype(np.uint8) * 255)


def grade(r):
    for name, hi in BINS:
        if r < hi:
            return name
    return "heavy"


def main():
    files = sorted(glob.glob(os.path.join(SLICES, "*.jpg")))
    rows = []
    for f in files:
        res = proxy_ratio(f)
        if res is None:
            continue
        r, _, _ = res
        rows.append((os.path.basename(f), r))
    rows.sort(key=lambda x: x[1])
    ratios = np.array([r for _, r in rows])

    counts = {name: 0 for name, _ in BINS}
    for _, r in rows:
        counts[grade(r)] += 1

    print(f"# 掃描 {len(rows)} 張切片 (proxy=black-hat>{BLACKHAT_T}, kernel=11)")
    print(f"proxy_ratio: min={ratios.min()*100:.3f}% p25={np.percentile(ratios,25)*100:.3f}% "
          f"median={np.median(ratios)*100:.3f}% p75={np.percentile(ratios,75)*100:.3f}% max={ratios.max()*100:.3f}%")
    print("分級計數 (cum 門檻):")
    cum = 0
    for name, hi in BINS:
        cum += counts[name]
        print(f"  {name:9s} < {hi*100:5.2f}% : {counts[name]:4d}  (累計 {cum})")

    # 存最乾淨/最髒各 4 張的 [影像|候選圖] 對照
    def dump(sel, tag):
        for i, (name, r) in enumerate(sel):
            res = proxy_ratio(os.path.join(SLICES, name))
            _, img, cand = res
            comp = np.concatenate([cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                                   cv2.cvtColor(cand, cv2.COLOR_GRAY2BGR)], axis=1)
            comp = cv2.resize(comp, (1024, 512))
            cv2.imwrite(os.path.join(OUT, f"{tag}_{i}_{r*100:.2f}pct_{name}.png"), comp)
    dump(rows[:4], "cleanest")
    dump(rows[-4:], "dirtiest")

    with open(os.path.join(OUT, "scan.json"), "w") as fh:
        json.dump({"counts": counts, "rows": rows}, fh, ensure_ascii=False, indent=2)
    print(f"# 對照圖與 scan.json 已存到 {OUT}")


if __name__ == "__main__":
    main()
