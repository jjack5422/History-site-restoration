"""
infer.py — 對單張影像或資料夾跑完整 CMC，輸出 mask 與 overlay。
--dump-steps 會額外輸出四欄中間結果（draft / SAM raw / SAM filtered / final），
重現論文 Fig. 18 那種逐階段可視化。

範例：
    python scripts/infer.py --source path/to/img.jpg --dump-steps
    python scripts/infer.py --source path/to/folder --out outputs_infer
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import yaml

# 讓本腳本可從 repo 根目錄執行並 import src.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace  # noqa: E402

from src.agent import Agent  # noqa: E402
from src.cmc import cmc_predict, cmc_predict_sliding  # noqa: E402
from src.large_model import build_large_model  # noqa: E402


def overlay(image_rgb, mask, color=(255, 0, 0), alpha=0.5):
    out = image_rgb.copy()
    m = np.asarray(mask).astype(bool)
    out[m] = (out[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def draw_points(image_rgb, pts):
    out = image_rgb.copy()
    for x, y in np.asarray(pts).astype(int):
        cv2.circle(out, (int(x), int(y)), 2, (0, 255, 0), -1)
    return out


def panel(image_rgb, info, final):
    """組四欄：draft+points / SAM raw / SAM filtered / final（皆為 overlay）。"""
    cells = [
        ("draft+pts", draw_points(overlay(image_rgb, info["draft"]), info["points"])),
        ("SAM raw", overlay(image_rgb, info["sam_raw"])),
        ("SAM filtered", overlay(image_rgb, info["sam_filtered"])),
        (f"final ({info['decision']})", overlay(image_rgb, final)),
    ]
    h = image_rgb.shape[0]
    labeled = []
    for name, img in cells:
        img = img.copy()
        cv2.putText(img, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        labeled.append(img)
    return np.concatenate(labeled, axis=1), h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cmc.yaml")
    ap.add_argument("--source", required=True, help="影像檔或資料夾")
    ap.add_argument("--out", default="outputs_infer")
    ap.add_argument("--dump-steps", action="store_true")
    ap.add_argument("--tile", type=int, default=0,
                    help=">0 時改用滑動窗口 CMC（512/stride256 風格）。"
                         "註：對全圖訓練的 Agent 實測會降低效能。")
    ap.add_argument("--stride", type=int, default=256)
    args = ap.parse_args()

    def run_cmc(rgb, return_intermediates=False):
        if args.tile > 0:
            return cmc_predict_sliding(rgb, agent, predictor, prompt_fn, hp,
                                       tile=args.tile, stride=args.stride,
                                       return_intermediates=return_intermediates)
        return cmc_predict(rgb, agent, predictor, prompt_fn, hp,
                           return_intermediates=return_intermediates)

    hp = SimpleNamespace(**yaml.safe_load(open(args.config, encoding="utf-8")))

    try:
        from ultralytics.nn.tasks import C2f_SEA  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"C2f_SEA 未註冊 — 請先執行 `python sea_setup.py`。({e})")

    agent = Agent(hp.agent_ckpt, device=hp.device)
    predictor, prompt_fn = build_large_model(hp)

    os.makedirs(args.out, exist_ok=True)
    if os.path.isfile(args.source):
        paths = [args.source]
    else:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        paths = sorted(p for e in exts for p in glob.glob(os.path.join(args.source, e)))

    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            print("skip (unreadable):", p)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        stem = os.path.splitext(os.path.basename(p))[0]

        if args.dump_steps:
            final, info = run_cmc(rgb, return_intermediates=True)
            grid, _ = panel(rgb, info, final)
            cv2.imwrite(os.path.join(args.out, f"{stem}_steps.png"),
                        cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"{stem}: decision={info['decision']} "
                  f"sam_score={info['sam_score']:.3f} conflict={info['conflict']:.3f}")
        else:
            final = run_cmc(rgb)

        cv2.imwrite(os.path.join(args.out, f"{stem}_mask.png"), final)
        cv2.imwrite(os.path.join(args.out, f"{stem}_overlay.png"),
                    cv2.cvtColor(overlay(rgb, final), cv2.COLOR_RGB2BGR))

    print("done →", args.out)


if __name__ == "__main__":
    main()
