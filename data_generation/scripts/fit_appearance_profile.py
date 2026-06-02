"""由 labeled32_{crack,craq}_v3 擬合外觀 profile, 存到 data/appearance/。

用法: python scripts/fit_appearance_profile.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from synthgen.appearance import fit_profile, save_profile

DATA = "/home/zzz90/research/_data"
JOBS = {
    "crack": (f"{DATA}/labeled32_crack_v3/images", f"{DATA}/labeled32_crack_v3/masks"),
    "craq": (f"{DATA}/labeled32_craq_v3/images", f"{DATA}/labeled32_craq_v3/masks"),
}


def main():
    for typ, (idir, mdir) in JOBS.items():
        prof = fit_profile(idir, mdir)
        out = f"data/appearance/appearance_profile_{typ}.json"
        save_profile(prof, out)
        print(f"{typ}: n={prof['n']} dL={prof['dL']} -> {out}")


if __name__ == "__main__":
    main()
