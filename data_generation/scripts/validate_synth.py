"""比對合成 vs 真實 mask 統計, 輸出 report 與 preview grid。

用法: python scripts/validate_synth.py --type crack --synth_root <dir> --out <dir>
"""
import argparse
import os
import sys
import glob
import json
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from synthgen.validate import mask_stats, compare

REAL = {"crack": "/home/zzz90/research/_data/labeled32_crack_v3/tiles_512/masks",
        "craq": "/home/zzz90/research/_data/labeled32_craq_v3/tiles_512/masks"}


def preview_grid(synth_root, real_img_dir, out_path, n=4):
    """各取合成/真實有前景 tile 拼 overlay grid。"""
    def pick(root, imgs_glob, masks_dir_from_img):
        rows = []
        for ip in sorted(glob.glob(imgs_glob)):
            mp = masks_dir_from_img(ip)
            if not os.path.exists(mp):
                continue
            m = np.array(Image.open(mp))
            if m.ndim == 3:
                m = m[..., 0]
            if (m > 0).sum() == 0:
                continue
            img = np.array(Image.open(ip).convert("RGB"))
            ov = img.copy(); ov[m > 0] = [255, 0, 0]
            rows.append(np.concatenate([img, ov], axis=1))
            if len(rows) >= n:
                break
        return rows
    synth_rows = pick(synth_root, f"{synth_root}/images/*.png",
                      lambda ip: ip.replace("/images/", "/masks/"))
    real_rows = pick(real_img_dir, f"{os.path.dirname(real_img_dir)}/images/*.png",
                     lambda ip: ip.replace("/images/", "/masks/"))
    rows = synth_rows + real_rows
    if rows:
        h = min(r.shape[0] for r in rows)
        grid = np.concatenate([r[:h] for r in rows], axis=0)
        Image.fromarray(grid).save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["crack", "craq"], required=True)
    ap.add_argument("--synth_root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    real = mask_stats(REAL[args.type])
    synth = mask_stats(os.path.join(args.synth_root, "masks"))
    cmp = compare(real, synth)
    report = {"type": args.type, "real": real, "synth": synth, "compare": cmp}
    # 移除冗長 list 再寫 report
    for d in (report["real"], report["synth"]):
        d.pop("fg_pcts", None)
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    preview_grid(args.synth_root, REAL[args.type].replace("/masks", "/images"),
                 os.path.join(args.out, f"preview_{args.type}.png"))
    print(json.dumps(cmp, ensure_ascii=False, indent=2))
    print(f"report + preview -> {args.out}")


if __name__ == "__main__":
    main()
