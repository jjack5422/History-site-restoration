"""建立底圖 manifest -> data/base_manifest.json。

兩種模式:
- 自動篩選(預設): 掃描 --slices 目錄, 用 black-hat proxy 篩 score<thresh 的乾淨切片。
  用法: python scripts/select_bases.py [--slices DIR] [--thresh 0.015] [--out PATH]
- 手動挑選: 把你自己選好的切片複製到一個資料夾, 用 --from-dir 指向它, 裡面全部當底圖。
  用法: python scripts/select_bases.py --from-dir /home/zzz90/research/_data/base_selected
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from synthgen.base_selection import build_manifest, build_manifest_from_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", default=None,
                    help="手動模式: 此資料夾內所有影像全當底圖(略過 black-hat 篩選)")
    ap.add_argument("--slices", default="/home/zzz90/research/_data/image_1024_slices")
    ap.add_argument("--thresh", type=float, default=0.015)
    ap.add_argument("--out", default="data/base_manifest.json")
    args = ap.parse_args()

    if args.from_dir:
        man = build_manifest_from_dir(args.from_dir, args.out)
        print(f"手動模式: {args.from_dir} 內 {man['n_clean']} 張全列為底圖 -> {args.out}")
    else:
        man = build_manifest(args.slices, args.thresh, args.out)
        print(f"自動篩選: 掃描 {man['n_total']} 張, 乾淨 {man['n_clean']} 張 "
              f"(thresh={args.thresh}) -> {args.out}")


if __name__ == "__main__":
    main()
