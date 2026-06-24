from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TILES = ROOT / "_data" / "craq_0-94_v1" / "tiles_512"
DEFAULT_PROB = DEFAULT_TILES / "resunet_prob" / "prob"
DEFAULT_OUT = ROOT / "crack_detection_sam2" / "results" / "resunet_prob_viz"


def overlay_prob(image: np.ndarray, prob: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    cmap = plt.get_cmap("magma")
    heat = (cmap(prob)[..., :3] * 255).astype(np.uint8)
    return np.clip((1 - alpha) * image + alpha * heat, 0, 255).astype(np.uint8)


def find_image(images_dir: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No matching image for stem: {stem}")


def make_panel(image_path: Path, prob_path: Path, out_path: Path, threshold: float) -> None:
    image = np.array(Image.open(image_path).convert("RGB"))
    prob_arr = np.load(prob_path)
    if prob_arr.ndim != 3 or prob_arr.shape[0] < 2:
        raise ValueError(f"Expected prob shape [2,H,W], got {prob_arr.shape}: {prob_path}")
    prob = prob_arr[1].astype(np.float32)
    mask = prob >= threshold
    overlay = overlay_prob(image, prob)
    mask_overlay = image.copy()
    mask_overlay[mask] = (255, 40, 80)
    mask_overlay = np.clip(0.55 * image + 0.45 * mask_overlay, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=180)
    titles = [
        "512x512 slice",
        "ResUNet prob map",
        f"mask prob>={threshold:g}",
        "prob overlay",
    ]
    axes[0].imshow(image)
    im = axes[1].imshow(prob, cmap="magma", vmin=0, vmax=1)
    axes[2].imshow(mask_overlay)
    axes[3].imshow(overlay)
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(prob_path.stem, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize ResUNet foreground probability maps.")
    p.add_argument("--tiles_root", type=Path, default=DEFAULT_TILES)
    p.add_argument("--prob_dir", type=Path, default=DEFAULT_PROB)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--n", type=int, default=8, help="Number of random maps to visualize.")
    p.add_argument("--seed", type=int, default=512)
    p.add_argument("--stems", nargs="*", default=None, help="Specific tile stems without extension.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = args.tiles_root / "images"
    prob_files = sorted(args.prob_dir.glob("*.npy"))
    if not prob_files:
        raise FileNotFoundError(f"No .npy prob files under {args.prob_dir}")

    if args.stems:
        selected = [args.prob_dir / f"{stem}.npy" for stem in args.stems]
    else:
        rng = random.Random(args.seed)
        selected = rng.sample(prob_files, min(args.n, len(prob_files)))

    for prob_path in selected:
        if not prob_path.exists():
            raise FileNotFoundError(prob_path)
        image_path = find_image(images_dir, prob_path.stem)
        out_path = args.out_dir / f"{prob_path.stem}_resunet_prob.png"
        make_panel(image_path, prob_path, out_path, args.threshold)
        print(out_path)


if __name__ == "__main__":
    main()
