"""Interactive craquelure refinement UI for the trained PromptedSAM2Seg model.

Features:
  1. Upload a 512x512 RGB image.
  2. Click positive / negative points to prompt the refine SAM2 model.
  3. Edit the generated mask with brush / eraser and export a binary PNG mask.

Run:
  /home/zzz90/research/sam2_env/bin/python -m pip install gradio
  NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 \
  /home/zzz90/research/sam2_env/bin/python crack_detection_sam2/interactive_refine_sam2.py

The default checkpoint is the latest canonical 0-94 GT refine model.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import gradio as gr
except ModuleNotFoundError as exc:  # fail with a useful message before loading SAM2
    raise SystemExit(
        "Missing dependency: gradio. Install it in sam2_env with:\n"
        "  /home/zzz90/research/sam2_env/bin/python -m pip install gradio"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
SAM2_DIR = ROOT / "crack_detection_sam2"
sys.path.insert(0, str(SAM2_DIR))

from model_prompted_sam2 import PromptedSAM2Seg  # noqa: E402

DEFAULT_CKPT = (
    SAM2_DIR
    / "runs/craq-refine-tversky28-aug-0-94gt-2026-06-22/best.pt"
)
DEFAULT_UNET_CKPT = (
    ROOT
    / "crack_detection_unet/runs/craq-resunet50-alldata-0-94gt-2026-06-22/last.pt"
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
POINT_COLORS = {1: np.array([30, 180, 70], np.uint8), 0: np.array([40, 90, 255], np.uint8)}


def _np_rgb(image: Any) -> np.ndarray | None:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
    return arr.astype(np.uint8)


def _normalize(img: np.ndarray, device: str) -> torch.Tensor:
    x = torch.from_numpy(img).float().div_(255.0).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0).to(device)


def _overlay(img: np.ndarray, mask: np.ndarray | None, alpha: float = 0.45) -> np.ndarray:
    out = img.copy()
    if mask is not None:
        m = mask.astype(bool)
        out[m] = (alpha * np.array([255, 0, 0]) + (1 - alpha) * out[m]).astype(np.uint8)
    return out


def _draw_points(img: np.ndarray, points: list[dict[str, int]], mask: np.ndarray | None = None) -> np.ndarray:
    canvas = _overlay(img, mask)
    for p in points:
        x, y, label = int(p["x"]), int(p["y"]), int(p["label"])
        color = tuple(int(v) for v in POINT_COLORS[label])
        cv2.circle(canvas, (x, y), 7, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 9, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        if label == 0:
            cv2.line(canvas, (x - 5, y - 5), (x + 5, y + 5), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (x + 5, y - 5), (x - 5, y + 5), (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _soft_prompt_from_points(
    points: list[dict[str, int]], size: int, radius: int, neg_strength: float
) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    score = np.full((size, size), -4.0, np.float32)
    # Craquelure annotations are typically only 2-3 px wide. Keep point prompts
    # local: radius is the visible influence radius, while sigma is tighter.
    sigma = max(float(radius) / 3.0, 0.75)
    for p in points:
        d2 = (xx - int(p["x"])) ** 2 + (yy - int(p["y"])) ** 2
        blob = np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)
        if int(p["label"]) == 1:
            score = np.maximum(score, -4.0 + 10.0 * blob)
        else:
            score -= neg_strength * 10.0 * blob
    return np.clip(score, -8.0, 8.0)


def _editor_payload(img: np.ndarray, mask: np.ndarray | None) -> dict[str, Any]:
    h, w = img.shape[:2]
    layer = np.zeros((h, w, 4), np.uint8)
    if mask is not None:
        layer[mask.astype(bool)] = np.array([255, 0, 0, 150], np.uint8)
    return {"background": img, "layers": [layer], "composite": _overlay(img, mask)}


def _mask_from_editor(value: Any, fallback_shape: tuple[int, int]) -> np.ndarray:
    if isinstance(value, dict):
        layers = value.get("layers") or []
        if layers:
            arr = np.asarray(layers[0])
            if arr.ndim == 3 and arr.shape[-1] == 4:
                return arr[..., 3] > 20
            if arr.ndim == 3:
                return arr[..., 0] > 20
            if arr.ndim == 2:
                return arr > 20
        comp = value.get("composite")
        if comp is not None:
            arr = np.asarray(comp)
            if arr.ndim == 3:
                return (arr[..., 0] > arr[..., 1] + 40) & (arr[..., 0] > arr[..., 2] + 40)
    return np.zeros(fallback_shape, bool)


class RefineApp:
    def __init__(self, ckpt: Path, unet_ckpt: Path, variant: str, image_size: int, device: str):
        self.ckpt = ckpt
        self.unet_ckpt = unet_ckpt
        self.variant = variant
        self.image_size = image_size
        self.device = device
        self.vanilla_predictor = None
        self.unet_model = None
        self.unet_predict_full = None
        self.model = PromptedSAM2Seg(
            variant=variant, image_size=image_size, device=device
        ).to(device)
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        self.model.load_state_dict(payload["model"], strict=False)
        self.model.eval()
        self.mask_hw = tuple(self.model.sam_prompt_encoder.mask_input_size)
        self.loaded = {
            "ckpt": str(ckpt),
            "epoch": payload.get("epoch"),
            "val_iou": (payload.get("val") or {}).get("craq_iou"),
            "mask_input_size": self.mask_hw,
            "device": device,
        }

    def _get_vanilla_predictor(self):
        if self.vanilla_predictor is None:
            from model import build_image_predictor

            self.vanilla_predictor = build_image_predictor(self.variant, device=self.device)
        return self.vanilla_predictor

    def _get_unet(self):
        if self.unet_model is None:
            import importlib.util

            unet_src = str(ROOT / "crack_detection_unet/src")
            if unet_src not in sys.path:
                sys.path.insert(0, unet_src)
            spec = importlib.util.spec_from_file_location(
                "crack_detection_unet_predict_full",
                ROOT / "crack_detection_unet/src/predict_full.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Could not load crack_detection_unet/src/predict_full.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            self.unet_model, _ = mod.load_model_from_ckpt(str(self.unet_ckpt), self.device)
            self.unet_predict_full = mod.predict_full
        return self.unet_model, self.unet_predict_full

    def load_image(self, image: Any):
        img = _np_rgb(image)
        if img is None:
            return None, None, [], None, "No image loaded."
        if img.shape[:2] != (self.image_size, self.image_size):
            pil = Image.fromarray(img)
            img = np.array(pil.resize((self.image_size, self.image_size), Image.BILINEAR))
            status = f"Loaded and resized to {self.image_size}x{self.image_size}."
        else:
            status = f"Loaded {self.image_size}x{self.image_size}."
        return img, _draw_points(img, []), [], _editor_payload(img, None), status

    def add_point(self, img: np.ndarray, points: list[dict[str, int]], mode: str, evt: gr.SelectData):
        if img is None:
            return None, points, "Load an image first."
        points = list(points or [])
        idx = evt.index
        if isinstance(idx, (list, tuple)) and len(idx) >= 2:
            x, y = int(idx[0]), int(idx[1])
        else:
            return _draw_points(img, points), points, "Could not read click coordinates."
        x = int(np.clip(x, 0, self.image_size - 1))
        y = int(np.clip(y, 0, self.image_size - 1))
        label = 1 if mode == "positive" else 0
        points.append({"x": x, "y": y, "label": label})
        return _draw_points(img, points), points, f"points: {len(points)}"

    def undo_point(self, img: np.ndarray, points: list[dict[str, int]], mask: np.ndarray | None):
        points = list(points or [])
        if points:
            points.pop()
        return _draw_points(img, points, mask), points, f"points: {len(points)}"

    def clear_points(self, img: np.ndarray):
        if img is None:
            return None, [], "No image loaded."
        return _draw_points(img, []), [], "points: 0"

    @torch.no_grad()
    def run_refine_points(self, img: np.ndarray, points: list[dict[str, int]], thr: float, radius: int, neg_strength: float):
        if img is None:
            return None, None, None, "Load an image first."
        if not points:
            return _draw_points(img, []), None, _editor_payload(img, None), "Add at least one point."
        prompt = _soft_prompt_from_points(points, self.image_size, radius, neg_strength)
        pm = torch.from_numpy(prompt)[None, None]
        pm = F.interpolate(pm, size=self.mask_hw, mode="bilinear", align_corners=False).to(self.device)
        coords = torch.tensor([[[p["x"], p["y"]] for p in points]], dtype=torch.float32, device=self.device)
        labels = torch.tensor([[p["label"] for p in points]], dtype=torch.long, device=self.device)
        x = _normalize(img, self.device)
        use_amp = self.device == "cuda"
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = self.model(x, coords, labels, pm)
        else:
            logits = self.model(x, coords, labels, pm)
        prob = torch.sigmoid(logits.float().squeeze()).cpu().numpy()
        mask = prob >= float(thr)
        return _draw_points(img, points, mask), mask, _editor_payload(img, mask), (
            f"mask pixels: {int(mask.sum())} | threshold={thr:.2f}"
        )

    @torch.no_grad()
    def run_vanilla_sam2(self, img: np.ndarray, points: list[dict[str, int]], thr: float):
        if img is None:
            return None, None, None, "Load an image first."
        if not points:
            return _draw_points(img, []), None, _editor_payload(img, None), "Add at least one point."
        predictor = self._get_vanilla_predictor()
        predictor.set_image(img)
        coords = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
        labels = np.array([p["label"] for p in points], dtype=np.int32)
        masks, scores, logits = predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=True,
            return_logits=True,
        )
        best = int(np.argmax(scores))
        logit = logits[best].astype(np.float32)
        prob = 1.0 / (1.0 + np.exp(-logit))
        mask = prob >= float(thr)
        return _draw_points(img, points, mask), mask, _editor_payload(img, mask), (
            f"vanilla SAM2 score={float(scores[best]):.3f} | mask pixels={int(mask.sum())}"
        )

    @torch.no_grad()
    def run_refine_from_prob(self, img: np.ndarray, prob_craq: np.ndarray, thr: float):
        pc = np.clip(prob_craq.astype(np.float32), 1e-4, 1 - 1e-4)
        logit = np.log(pc / (1.0 - pc)).astype(np.float32)
        pm = torch.from_numpy(logit)[None, None]
        pm = F.interpolate(pm, size=self.mask_hw, mode="bilinear", align_corners=False).to(self.device)
        coords = torch.zeros(1, 1, 2, dtype=torch.float32, device=self.device)
        labels = -torch.ones(1, 1, dtype=torch.long, device=self.device)
        x = _normalize(img, self.device)
        if self.device == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = self.model(x, coords, labels, pm)
        else:
            logits = self.model(x, coords, labels, pm)
        prob = torch.sigmoid(logits.float().squeeze()).cpu().numpy()
        return prob >= float(thr), prob

    @torch.no_grad()
    def run_unet_refine(self, img: np.ndarray, thr: float):
        if img is None:
            return None, None, None, "Load an image first."
        unet, predict_full = self._get_unet()
        probs = predict_full(
            unet,
            img,
            self.device,
            tile=self.image_size,
            stride=self.image_size,
            batch_size=1,
            tta_flip=False,
            use_amp=True,
        )
        if probs.shape[0] < 2:
            return _draw_points(img, []), None, _editor_payload(img, None), "UNet output has no craquelure channel."
        unet_prob = probs[1].astype(np.float32)
        mask, refine_prob = self.run_refine_from_prob(img, unet_prob, thr)
        return _overlay(img, mask), mask, _editor_payload(img, mask), (
            f"UNet->refine done | UNet max={float(unet_prob.max()):.3f} "
            f"refine max={float(refine_prob.max()):.3f} | mask pixels={int(mask.sum())}"
        )

    def set_zoom(self, zoom: float):
        h = int(round(self.image_size * float(zoom)))
        return gr.update(height=h), gr.update(height=h)

    def export_mask(self, editor_value: Any, img: np.ndarray):
        if img is None:
            return None, "No image loaded."
        mask = _mask_from_editor(editor_value, img.shape[:2])
        out = (mask.astype(np.uint8) * 255)
        fd, path = tempfile.mkstemp(prefix="craq_mask_", suffix=".png")
        os.close(fd)
        Image.fromarray(out).save(path)
        return path, f"exported binary mask: {path}"


def build_ui(app: RefineApp):
    with gr.Blocks(title="Craquelure Refine SAM2") as demo:
        gr.Markdown("# Craquelure Refine SAM2")
        gr.Markdown(
            f"Loaded `{app.ckpt}` | device `{app.device}` | "
            f"epoch `{app.loaded['epoch']}` | val_iou `{app.loaded['val_iou']}`"
        )
        gr.Markdown(f"UNet prompt model `{app.unet_ckpt}`")
        img_state = gr.State(None)
        points_state = gr.State([])
        mask_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.Image(label="512x512 image", type="numpy", image_mode="RGB")
                mode = gr.Radio(["positive", "negative"], value="positive", label="Point mode")
                with gr.Row():
                    undo = gr.Button("Undo point")
                    clear = gr.Button("Clear points")
                thr = gr.Slider(0.05, 0.95, value=0.25, step=0.01, label="Mask threshold")
                zoom = gr.Slider(0.75, 3.0, value=1.1, step=0.05, label="Editor zoom")
                vanilla_btn = gr.Button("Vanilla SAM2 points", variant="primary")
                auto_btn = gr.Button("UNet -> refine SAM2")
                export = gr.Button("Export edited mask")
                download = gr.File(label="Binary mask PNG")
                status = gr.Textbox(label="Status", elem_classes=["status"], interactive=False)

            with gr.Column(scale=2):
                click_img = gr.Image(
                    label="Click positive/negative points here",
                    type="numpy",
                    image_mode="RGB",
                    elem_id="click_img",
                    interactive=False,
                    height=563,
                )
                editor_kwargs = {
                    "label": "Mask brush / eraser",
                    "type": "numpy",
                    "height": 563,
                    "elem_id": "mask_editor",
                    "interactive": True,
                    "canvas_size": (512, 512),
                    "fixed_canvas": True,
                }
                if hasattr(gr, "Brush"):
                    editor_kwargs["brush"] = gr.Brush(
                        default_size=6,
                        colors=["#ff0000"],
                        default_color="#ff0000",
                        color_mode="fixed",
                    )
                if hasattr(gr, "Eraser"):
                    editor_kwargs["eraser"] = gr.Eraser(default_size=10)
                mask_editor = gr.ImageEditor(**editor_kwargs)

        upload.change(
            app.load_image,
            inputs=upload,
            outputs=[img_state, click_img, points_state, mask_editor, status],
        )
        click_img.select(
            app.add_point,
            inputs=[img_state, points_state, mode],
            outputs=[click_img, points_state, status],
        )
        undo.click(
            app.undo_point,
            inputs=[img_state, points_state, mask_state],
            outputs=[click_img, points_state, status],
        )
        clear.click(app.clear_points, inputs=img_state, outputs=[click_img, points_state, status])
        zoom.change(app.set_zoom, inputs=zoom, outputs=[click_img, mask_editor])
        vanilla_btn.click(
            app.run_vanilla_sam2,
            inputs=[img_state, points_state, thr],
            outputs=[click_img, mask_state, mask_editor, status],
        )
        auto_btn.click(
            app.run_unet_refine,
            inputs=[img_state, thr],
            outputs=[click_img, mask_state, mask_editor, status],
        )
        export.click(app.export_mask, inputs=[mask_editor, img_state], outputs=[download, status])
    return demo


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--unet_ckpt", default=str(DEFAULT_UNET_CKPT))
    ap.add_argument("--variant", default="small")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app = RefineApp(Path(args.ckpt), Path(args.unet_ckpt), args.variant, args.image_size, device)
    demo = build_ui(app)
    css = """
    #click_img img, #mask_editor canvas { image-rendering: auto; }
    #click_img, #mask_editor { overflow: auto; }
    .status { font-family: monospace; }
    """
    js = """
    () => {
      const base = 512;
      let zoom = 1.1;
      const ids = ["click_img", "mask_editor"];

      function resizeMedia(root) {
        if (!root) return;
        const size = `${Math.round(base * zoom)}px`;
        root.style.height = size;
        root.style.maxHeight = "78vh";
        root.style.overflow = "auto";
        root.querySelectorAll("img, canvas").forEach((el) => {
          el.style.width = size;
          el.style.height = size;
          el.style.maxWidth = "none";
          el.style.maxHeight = "none";
        });
      }

      function applyZoom() {
        ids.forEach((id) => resizeMedia(document.getElementById(id)));
      }

      function attach() {
        ids.forEach((id) => {
          const root = document.getElementById(id);
          if (!root || root.dataset.wheelZoomAttached === "1") return;
          root.dataset.wheelZoomAttached = "1";
          root.addEventListener("wheel", (event) => {
            event.preventDefault();
            const direction = event.deltaY < 0 ? 1 : -1;
            zoom = Math.max(0.75, Math.min(3.0, zoom + direction * 0.1));
            applyZoom();
          }, { passive: false });
        });
      }

      applyZoom();
      attach();
      new MutationObserver(() => { applyZoom(); attach(); })
        .observe(document.body, { childList: true, subtree: true });
    }
    """
    queued = demo.queue(default_concurrency_limit=1)
    last_err = None
    for port in range(args.port, args.port + 20):
        try:
            queued.launch(server_name=args.host, server_port=port, share=args.share, css=css, js=js)
            return
        except OSError as exc:
            last_err = exc
            if "Cannot find empty port" not in str(exc):
                raise
            print(f"port {port} is busy; trying {port + 1}", flush=True)
    raise last_err


if __name__ == "__main__":
    main()
