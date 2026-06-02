"""掃描 _data/image_1024_slices 篩乾淨底圖, 輸出 data/base_manifest.json。

用法: python scripts/select_bases.py [--slices DIR] [--thresh 0.015] [--out PATH]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from synthgen.base_selection import build_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="/home/zzz90/research/_data/image_1024_slices")
    ap.add_argument("--thresh", type=float, default=0.015)
    ap.add_argument("--out", default="data/base_manifest.json")
    args = ap.parse_args()
    man = build_manifest(args.slices, args.thresh, args.out)
    print(f"掃描 {man['n_total']} 張, 乾淨 {man['n_clean']} 張 (thresh={args.thresh}) -> {args.out}")


if __name__ == "__main__":
    main()
