"""High-recall craquelure prelabel draft: ResUNet craq prob UNION multiscale ridge.

Stage-0 of the semi-auto annotation workflow (see
docs/superpowers/specs/2026-06-14-craquelure-semiauto-annotation-design.md).

Idea: the trained ResUNet (channel 1 of its --save_prob .npy) gives a learned
craquelure prob; a multiscale dark-ridge filter (Sato/Meijering) is a dumb but
high-recall detector of thin dark net lines. We UNION a low-threshold model mask
with the ridge mask to get a high-recall draft. False positives outside the
painted craquelure area are meant to be clipped later by a human-drawn region
polygon in CVAT, so here we deliberately favour recall.

Outputs (full-res PNG + a downscaled preview of each):
  draft_mask.png        binary union mask (255=craq)
  overlay_model.png     red = model-only mask
  overlay_ridge.png     red = ridge-only mask
  overlay_compare.png   red = model mask, green = extra net lines ridge adds
  overlay_draft.png     red = final union draft

Run with sam2_env (has skimage + cv2):
  sam2_env/bin/python prelabel_draft.py \
      --image /home/zzz90/research/_data/image/KJTHT-SC-L-1RB1-1.jpg \
      --prob  /home/zzz90/research/_data/prelabel_demo_2026-06-14/resunet/prob/KJTHT-SC-L-1RB1-1.npy \
      --out_dir /home/zzz90/research/_data/prelabel_demo_2026-06-14/draft \
      --p_thr 0.15 --ridge_pct 96 --min_area 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.filters import sato
from skimage.morphology import remove_small_objects, binary_closing, disk

Image.MAX_IMAGE_PIXELS = None


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def ridge_response(gray_f, sigmas):
    """Multiscale dark-ridge (Sato) response, normalised to [0,1]."""
    r = sato(gray_f, sigmas=sigmas, black_ridges=True)
    r = r - r.min()
    if r.max() > 0:
        r = r / r.max()
    return r.astype(np.float32)


def red_overlay(rgb, mask, color=(255, 0, 0), alpha=0.55):
    out = rgb.copy()
    out[mask] = ((1 - alpha) * out[mask] + alpha * np.array(color)).astype(np.uint8)
    return out


def two_color_overlay(rgb, red_mask, green_mask, alpha=0.6):
    out = rgb.copy()
    out[red_mask] = ((1 - alpha) * out[red_mask] + alpha * np.array((255, 0, 0))).astype(np.uint8)
    out[green_mask] = ((1 - alpha) * out[green_mask] + alpha * np.array((0, 230, 0))).astype(np.uint8)
    return out


def save_with_preview(arr, path, preview_w=2000):
    Image.fromarray(arr).save(path)
    h, w = arr.shape[:2]
    if w > preview_w:
        scale = preview_w / w
        prev = cv2.resize(arr, (preview_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        Image.fromarray(prev).save(str(path).replace(".png", "_preview.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prob", required=True, help="ResUNet --save_prob .npy, shape (C,H,W), channel 1=craq")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--p_thr", type=float, default=0.15, help="low model threshold -> high recall")
    ap.add_argument("--ridge_pct", type=float, default=96.0, help="percentile on ridge response for mask")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--ridge_scale", type=float, default=1.0,
                    help="<1 downsamples before the (slow) ridge filter then upsamples back; "
                         "use ~0.5 on >50MP images to cut ridge time ~4x")
    ap.add_argument("--min_area", type=int, default=12, help="drop connected comps smaller than this (px)")
    ap.add_argument("--close", type=int, default=1, help="binary closing radius to link fragments")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rgb = load_rgb(args.image)
    H, W = rgb.shape[:2]
    print(f"image {W}x{H}")

    prob = np.load(args.prob)
    craq = prob[1] if prob.ndim == 3 else prob
    if craq.shape != (H, W):
        craq = cv2.resize(craq, (W, H), interpolation=cv2.INTER_LINEAR)
    model_mask = craq >= args.p_thr

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    print("computing multiscale ridge ...")
    if args.ridge_scale < 1.0:
        gs = cv2.resize(gray, None, fx=args.ridge_scale, fy=args.ridge_scale,
                        interpolation=cv2.INTER_AREA)
        ridge = ridge_response(gs, args.sigmas)
        ridge = cv2.resize(ridge, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        ridge = ridge_response(gray, args.sigmas)
    # persist the normalised ridge response so region_clip.py need not recompute
    Image.fromarray((ridge * 255).astype(np.uint8)).save(str(out / "ridge_u8.png"))
    r_thr = np.percentile(ridge, args.ridge_pct)
    ridge_mask = ridge >= r_thr

    draft = model_mask | ridge_mask
    if args.min_area > 0:
        draft = remove_small_objects(draft, min_size=args.min_area)
    if args.close > 0:
        draft = binary_closing(draft, disk(args.close))

    # also clean the individual masks for display
    model_disp = remove_small_objects(model_mask, min_size=args.min_area) if args.min_area > 0 else model_mask
    ridge_disp = remove_small_objects(ridge_mask, min_size=args.min_area) if args.min_area > 0 else ridge_mask
    extra = ridge_disp & ~model_disp  # what ridge adds on top of the model

    cov = lambda m: 100.0 * m.sum() / (H * W)
    print(f"coverage%%  model={cov(model_disp):.2f}  ridge={cov(ridge_disp):.2f}  "
          f"extra(ridge-only)={cov(extra):.2f}  draft={cov(draft):.2f}")

    save_with_preview((draft.astype(np.uint8) * 255), str(out / "draft_mask.png"))
    save_with_preview(red_overlay(rgb, model_disp), str(out / "overlay_model.png"))
    save_with_preview(red_overlay(rgb, ridge_disp), str(out / "overlay_ridge.png"))
    save_with_preview(two_color_overlay(rgb, model_disp, extra), str(out / "overlay_compare.png"))
    save_with_preview(red_overlay(rgb, draft), str(out / "overlay_draft.png"))
    print(f"wrote overlays to {out}")


if __name__ == "__main__":
    main()
