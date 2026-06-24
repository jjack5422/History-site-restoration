"""Render polished architecture diagrams for model_seg.py and model_prompt_seg.py.

Outputs three PNGs into docs/arch/:
  - arch_sam2semseg.png       (model_seg.py)
  - arch_sam2promptseg.png    (model_prompt_seg.py)
  - arch_compare.png          (side-by-side overview)

Shapes assume variant="small", image_size=512.
Run: sam2_env/bin/python crack_detection_sam2/scripts/draw_arch.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ----- palette -------------------------------------------------------------
BG       = "#0f1117"
PANEL    = "#161a23"
INK      = "#e6e6e6"
MUTE     = "#9aa3b2"
FROZEN   = "#3a4a63"   # frozen / encoder (cool slate)
FROZEN_E = "#6f86ad"
TRAIN    = "#1f6f54"   # trainable head/neck (green)
TRAIN_E  = "#3fcf9a"
PROMPT   = "#7a4f1f"   # learnable prompt (amber)
PROMPT_E = "#e0a44d"
DECODE   = "#5a2f6b"   # mask decoder (violet)
DECODE_E = "#bd7fd6"
TENSOR   = "#1b2330"   # tensor blob
TENSOR_E = "#39455c"
ARROW    = "#7f8aa0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
})


def box(ax, x, y, w, h, label, sub=None, fc=PANEL, ec=MUTE, fs=11, tc=INK, lw=1.6, round=0.02):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={round}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, mutation_aspect=1)
    ax.add_patch(p)
    cy = y + h / 2 + (0.10 * h if sub else 0)
    ax.text(x + w / 2, cy, label, ha="center", va="center", color=tc,
            fontsize=fs, fontweight="bold", zorder=5)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.26 * h, sub, ha="center", va="center",
                color=MUTE, fontsize=fs - 2.5, zorder=5)
    return (x + w / 2, y, x + w / 2, y + h)  # bottom-mid, top-mid helpers


def tensor(ax, x, y, w, h, label, fs=8.5):
    box(ax, x, y, w, h, label, fc=TENSOR, ec=TENSOR_E, fs=fs, tc="#cfe3ff", lw=1.2, round=0.015)


def arrow(ax, x1, y1, x2, y2, color=ARROW, lw=1.8, style="-|>", rad=0.0, ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                        linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}",
                        linestyle=ls, zorder=3)
    ax.add_patch(a)


def stage(ax, fig):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")


# ===========================================================================
# Figure 1: SAM2SemSeg  (model_seg.py)
# ===========================================================================
def draw_semseg(path):
    fig, ax = plt.subplots(figsize=(9, 13))
    stage(ax, fig)

    ax.text(5, 13.5, "SAM2SemSeg", ha="center", color=INK, fontsize=20, fontweight="bold")
    ax.text(5, 13.05, "model_seg.py  -  multi-class semantic segmentation",
            ha="center", color=MUTE, fontsize=10.5)

    # input
    tensor(ax, 3.7, 12.1, 2.6, 0.55, "input  x  [B, 3, 512, 512]")
    arrow(ax, 5, 12.1, 5, 11.75)

    # image encoder panel
    box(ax, 1.5, 9.3, 7.0, 2.35, "SAM2 Image Encoder", fc="#10131b", ec=FROZEN_E, fs=13)
    box(ax, 1.85, 10.55, 3.0, 0.85, "Hiera trunk", "frozen  (freeze_trunk)", fc=FROZEN, ec=FROZEN_E, fs=10.5)
    box(ax, 5.15, 10.55, 3.0, 0.85, "FpnNeck", "trainable", fc=TRAIN, ec=TRAIN_E, fs=10.5)
    ax.text(2.0, 10.1, "patch_embed + 4 Hiera stages  ->  strides 4/8/16/32",
            ha="left", color=MUTE, fontsize=8.3)
    ax.text(2.0, 9.78, "1x1 lateral -> d_model 256, top-down fuse, scalp drops stride-32",
            ha="left", color=MUTE, fontsize=8.3)
    arrow(ax, 3.35, 10.55, 3.35, 9.9, color=MUTE, lw=1.2)
    arrow(ax, 6.65, 9.9, 6.65, 10.55, color=MUTE, lw=1.2)

    arrow(ax, 5, 9.3, 5, 8.95)
    ax.text(5.15, 9.12, "backbone_fpn  (3 levels, 256 ch)", ha="left", color=MUTE, fontsize=8.5)

    # three fpn tensors
    fx = [1.7, 4.05, 6.4]
    flab = ["f0  s4\n[256,128,128]", "f1  s8\n[256, 64, 64]", "f2  s16\n[256, 32, 32]"]
    for x, l in zip(fx, flab):
        tensor(ax, x, 8.3, 1.9, 0.62, l, fs=8)
        arrow(ax, 5, 8.95, x + 0.95, 8.92, color=MUTE, lw=1.1, rad=0.0)

    # head panel
    box(ax, 1.3, 4.05, 7.4, 3.7, "", fc="#0e150f", ec=TRAIN_E, fs=12)
    ax.text(5, 7.5, "FPNSegHead   (trainable)", ha="center", color=TRAIN_E,
            fontsize=13, fontweight="bold")

    # lateral
    for x in fx:
        tensor(ax, x, 6.65, 1.9, 0.5, "1x1 conv -> 128", fs=8)
        arrow(ax, x + 0.95, 8.3, x + 0.95, 7.15, color=ARROW, lw=1.3)
    ax.text(8.62, 7.75, "f1,f2 upsample\nto 128x128", ha="right", color=MUTE, fontsize=7.6)

    # concat
    tensor(ax, 3.0, 5.85, 4.0, 0.5, "concat  ->  [384, 128, 128]", fs=9)
    for x in fx:
        arrow(ax, x + 0.95, 6.65, 5.0, 6.36, color=ARROW, lw=1.2, rad=0.0)

    tensor(ax, 3.0, 5.05, 4.0, 0.5, "3x3 conv + BN + ReLU + Dropout  ->  [128,128,128]", fs=8)
    arrow(ax, 5, 5.85, 5, 5.55)
    tensor(ax, 3.0, 4.3, 4.0, 0.5, "1x1 classifier  ->  [5, 128, 128]   (stride 4)", fs=8.5)
    arrow(ax, 5, 5.05, 5, 4.8)

    arrow(ax, 5, 4.05, 5, 3.6)
    tensor(ax, 3.2, 2.95, 3.6, 0.55, "bilinear upsample  ->  512x512", fs=9)
    arrow(ax, 5, 2.95, 5, 2.6)

    # output
    box(ax, 3.0, 1.85, 4.0, 0.7, "output  logits", "[B, 5, 512, 512]  per-pixel class scores",
        fc="#241016", ec="#d06b7e", fs=12, tc="#ffd9e0")

    # legend
    legend(ax, 0.55, [("Hiera trunk (frozen)", FROZEN, FROZEN_E),
                      ("FpnNeck / Head (trainable)", TRAIN, TRAIN_E),
                      ("tensor", TENSOR, TENSOR_E)])

    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# ===========================================================================
# Figure 2: SAM2PromptSeg  (model_prompt_seg.py)
# ===========================================================================
def draw_promptseg(path):
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 14)
    ax.axis("off")

    ax.text(5.5, 13.5, "SAM2PromptSeg", ha="center", color=INK, fontsize=20, fontweight="bold")
    ax.text(5.5, 13.05, "model_prompt_seg.py  -  native SAM2 pipeline, learnable prompt points",
            ha="center", color=MUTE, fontsize=10.5)

    tensor(ax, 4.2, 12.1, 2.6, 0.55, "input  x  [B, 3, 512, 512]")
    arrow(ax, 5.5, 12.1, 5.5, 11.75)

    # image encoder (frozen)
    box(ax, 2.0, 10.7, 7.0, 1.05, "SAM2 Image Encoder", "frozen  (freeze_image_encoder)",
        fc=FROZEN, ec=FROZEN_E, fs=13)
    arrow(ax, 5.5, 10.7, 5.5, 10.4)
    ax.text(5.65, 10.5, "backbone_fpn", ha="left", color=MUTE, fontsize=8.5)

    # fpn tensors
    tensor(ax, 0.6, 9.55, 2.0, 0.6, "fpn0 s4\n[256,128,128]", fs=7.8)
    tensor(ax, 3.0, 9.55, 2.0, 0.6, "fpn1 s8\n[256,64,64]", fs=7.8)
    tensor(ax, 6.4, 9.55, 2.0, 0.6, "fpn2 s16 (= backbone_features)\n[256,32,32]", fs=7.3)
    for cx in (1.6, 4.0, 7.4):
        arrow(ax, 5.5, 10.4, cx, 10.18, color=MUTE, lw=1.0)

    # high-res features
    tensor(ax, 0.6, 8.7, 2.0, 0.5, "conv_s0 -> [32,128,128]", fs=7.2)
    tensor(ax, 3.0, 8.7, 2.0, 0.5, "conv_s1 -> [64,64,64]", fs=7.2)
    arrow(ax, 1.6, 9.55, 1.6, 9.2, color=ARROW, lw=1.1)
    arrow(ax, 4.0, 9.55, 4.0, 9.2, color=ARROW, lw=1.1)
    ax.text(1.6, 8.45, "high_res_features (detail skip)", ha="center", color=MUTE, fontsize=7.5)

    # learnable prompt block (left)
    box(ax, 0.4, 6.35, 3.3, 1.55, "", fc="#1a1206", ec=PROMPT_E)
    ax.text(2.05, 7.66, "Learnable Prompt", ha="center", color=PROMPT_E, fontsize=11.5, fontweight="bold")
    box(ax, 0.7, 6.95, 2.7, 0.45, "prompt_coords  [1,8,2]", "nn.Parameter (learns where to click)",
        fc=PROMPT, ec=PROMPT_E, fs=8.5)
    box(ax, 0.7, 6.5, 2.7, 0.38, "prompt_labels  [1,8]=1  (buffer, fixed +)", fc="#2a2010", ec=PROMPT_E, fs=7.8)

    # prompt encoder
    box(ax, 0.4, 4.7, 3.3, 1.05, "Prompt Encoder", "trainable", fc=TRAIN, ec=TRAIN_E, fs=12)
    arrow(ax, 2.05, 6.35, 2.05, 5.75, color=PROMPT_E, lw=1.8)

    tensor(ax, 0.55, 4.0, 1.55, 0.5, "sparse_emb\n[B, 9, 256]", fs=7.5)
    tensor(ax, 2.25, 4.0, 1.5, 0.5, "dense_emb\n[B,256,32,32]", fs=7.5)
    arrow(ax, 1.3, 4.7, 1.3, 4.5, color=ARROW, lw=1.2)
    arrow(ax, 3.0, 4.7, 3.0, 4.5, color=ARROW, lw=1.2)

    # mask decoder (right/center)
    box(ax, 5.0, 3.55, 5.4, 4.45, "", fc="#160a1c", ec=DECODE_E)
    ax.text(7.7, 7.72, "Mask Decoder", ha="center", color=DECODE_E, fontsize=13, fontweight="bold")
    ax.text(7.7, 7.42, "(native SAM2, trainable)", ha="center", color=MUTE, fontsize=8.5)

    box(ax, 5.3, 6.4, 4.8, 0.85, "Two-Way Transformer",
        "tokens(iou+mask+sparse) <-> image: cross+self-attn x N", fc=DECODE, ec=DECODE_E, fs=10.5)

    box(ax, 5.3, 5.3, 4.8, 0.75, "Output upscaling  (transpose-conv x2 -> 128x128)",
        "+ high_res_features added back", fc="#2a1733", ec=DECODE_E, fs=9)
    box(ax, 5.3, 4.35, 4.8, 0.65, "mask-token hypernetwork MLP  .  upscaled feat",
        fc="#2a1733", ec=DECODE_E, fs=9)

    arrow(ax, 7.7, 6.4, 7.7, 6.05, color=DECODE_E, lw=1.4)
    arrow(ax, 7.7, 5.3, 7.7, 5.0, color=DECODE_E, lw=1.4)

    # feeds into decoder
    arrow(ax, 2.1, 4.0, 5.3, 6.7, color=ARROW, lw=1.3, rad=-0.15)   # sparse
    arrow(ax, 3.0, 4.0, 5.3, 6.55, color=ARROW, lw=1.3, rad=-0.1)   # dense
    arrow(ax, 7.4, 9.55, 7.7, 7.25, color=MUTE, lw=1.3, rad=0.1)    # backbone_features
    arrow(ax, 2.1, 8.7, 5.0, 5.6, color=ARROW, lw=1.1, rad=-0.2, ls=(0, (3, 2)))  # high-res skip
    ax.text(4.55, 6.05, "image_pe\n+ image_emb", ha="center", color=MUTE, fontsize=7.3)

    arrow(ax, 7.7, 4.35, 7.7, 4.0)
    tensor(ax, 6.0, 3.35, 3.4, 0.55, "low_res_masks  [B, 1, 128, 128]", fs=9)
    arrow(ax, 7.7, 3.35, 7.7, 3.0)
    tensor(ax, 6.1, 2.35, 3.2, 0.55, "bilinear upsample -> 512x512", fs=9)
    arrow(ax, 7.7, 2.35, 7.7, 2.0)
    box(ax, 6.0, 1.25, 3.4, 0.7, "output  masks", "[B, 1, 512, 512]  binary logits",
        fc="#241016", ec="#d06b7e", fs=12, tc="#ffd9e0")

    legend(ax, 0.45, [("Image encoder (frozen)", FROZEN, FROZEN_E),
                      ("Prompt encoder (trainable)", TRAIN, TRAIN_E),
                      ("Learnable prompt", PROMPT, PROMPT_E),
                      ("Mask decoder", DECODE, DECODE_E)], x0=0.5, xstep=2.55)

    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# ===========================================================================
# Figure 3: side-by-side comparison
# ===========================================================================
def draw_compare(path):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.text(6, 7.6, "SAM2SemSeg  vs  SAM2PromptSeg", ha="center", color=INK,
            fontsize=18, fontweight="bold")
    ax.text(6, 7.2, "same SAM2 backbone, different segmentation route", ha="center",
            color=MUTE, fontsize=10.5)

    # left column
    def col(x0, title, sub, blocks, out, outc):
        ax.text(x0 + 1.9, 6.55, title, ha="center", color=INK, fontsize=13.5, fontweight="bold")
        ax.text(x0 + 1.9, 6.22, sub, ha="center", color=MUTE, fontsize=9)
        tensor(ax, x0 + 0.7, 5.55, 2.4, 0.5, "x [B,3,512,512]", fs=9)
        y = 5.55
        prev = x0 + 1.9
        for lbl, s, fc, ec in blocks:
            arrow(ax, prev, y, x0 + 1.9, y - 0.35, color=ARROW, lw=1.4)
            y -= 0.95
            box(ax, x0 + 0.4, y, 3.0, 0.6, lbl, s, fc=fc, ec=ec, fs=10)
            prev = x0 + 1.9
        arrow(ax, x0 + 1.9, y, x0 + 1.9, y - 0.32, color=ARROW, lw=1.4)
        box(ax, x0 + 0.5, y - 0.92, 2.8, 0.6, out, None, fc="#241016", ec=outc, fs=10.5, tc="#ffd9e0")

    col(0.2, "SAM2SemSeg", "model_seg.py",
        [("Image Encoder", "trunk frozen / neck train", FROZEN, FROZEN_E),
         ("backbone_fpn", "3 levels, 256 ch", TENSOR, TENSOR_E),
         ("FPNSegHead", "lateral->concat->fuse->cls", TRAIN, TRAIN_E)],
        "logits [B,5,512,512]", "#d06b7e")

    col(7.6, "SAM2PromptSeg", "model_prompt_seg.py",
        [("Image Encoder", "fully frozen", FROZEN, FROZEN_E),
         ("Prompt Encoder", "learnable 8 points", PROMPT, PROMPT_E),
         ("Mask Decoder", "native two-way transformer", DECODE, DECODE_E)],
        "masks [B,1,512,512]", "#d06b7e")

    # divider
    ax.add_line(Line2D([6, 6], [0.6, 6.4], color="#2a3040", lw=1.2, ls=(0, (4, 4))))

    # bottom note
    ax.text(2.1, 0.7, "drops SAM2 decoder; custom multi-class head",
            ha="center", color=MUTE, fontsize=8.5, style="italic")
    ax.text(9.5, 0.7, "keeps full SAM2; prompt auto-guides binary mask",
            ha="center", color=MUTE, fontsize=8.5, style="italic")

    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# ===========================================================================
# Hand-draw reference: horizontal, sketched lines, paper background
# ===========================================================================
PAPER = "#faf6ec"
INK2  = "#2b2b2b"
DIM2  = "#6b6b6b"
H_FROZ_F, H_FROZ_C = "#cfe0f0", "#3a6ea5"
H_HEAD_F, H_HEAD_C = "#cfe9d6", "#2e7d4f"
H_PROM_F, H_PROM_C = "#f6e2bd", "#c98a2b"
H_DEC_F,  H_DEC_C  = "#e6d3ef", "#7a4f96"
H_TEN_F,  H_TEN_C  = "#ffffff", "#9a9a9a"
H_OUT_F,  H_OUT_C  = "#f7d6d9", "#b5485a"

_HAND_RC = {
    "path.sketch": (1.4, 110, 16),
    "font.family": "DejaVu Sans",
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
}

# Traditional-Chinese font (WSL-mounted Windows font); None if unavailable.
import matplotlib.font_manager as _fm
_CJK_PATH = "/mnt/c/Windows/Fonts/msjh.ttc"
_CJK_NAME = None


def _ensure_cjk():
    global _CJK_NAME
    if _CJK_NAME is not None:
        return _CJK_NAME
    if os.path.exists(_CJK_PATH):
        _fm.fontManager.addfont(_CJK_PATH)
        _CJK_NAME = _fm.FontProperties(fname=_CJK_PATH).get_name()
    else:
        _CJK_NAME = ""  # cache miss so we don't retry
    return _CJK_NAME or None


def hbox(ax, x, y, w, h, label, sub=None, fc=H_TEN_F, ec=INK2, fs=11, tc=INK2, lw=2.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                lw=lw, edgecolor=ec, facecolor=fc, mutation_aspect=1))
    if sub:
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center", color=tc,
                fontsize=fs, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", color=DIM2, fontsize=fs - 3)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color=tc,
                fontsize=fs, fontweight="bold")


def harrow(ax, x1, y1, x2, y2, color="#555", lw=2.0, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}"))


def draw_hand_semseg(path):
    with plt.rc_context(_HAND_RC):
        fig, ax = plt.subplots(figsize=(17, 5.6))
        ax.set_xlim(0, 19.4)
        ax.set_ylim(0, 6.4)
        ax.axis("off")
        ax.text(9.7, 6.05, "SAM2SemSeg  (model_seg.py)", ha="center", color=INK2,
                fontsize=17, fontweight="bold")
        ax.text(9.7, 5.65, "multi-class semantic segmentation  -  hand-draw reference",
                ha="center", color=DIM2, fontsize=10)

        ym = 3.4
        # input
        hbox(ax, 0.3, ym - 0.6, 1.6, 1.2, "input x", "[B,3,512,512]", fc=H_TEN_F, fs=10)
        harrow(ax, 1.9, ym, 2.25, ym)
        # encoder (black box)
        hbox(ax, 2.25, ym - 0.85, 2.7, 1.7, "SAM2 Image\nEncoder",
             "Hiera + FpnNeck . frozen", fc=H_FROZ_F, ec=H_FROZ_C, fs=12)
        harrow(ax, 4.95, ym, 5.25, ym)
        # backbone_fpn (3 stacked maps)
        for i, (lbl, yy) in enumerate([("f0 256@128^2  s4", 4.25),
                                       ("f1 256@64^2   s8", 3.4),
                                       ("f2 256@32^2   s16", 2.55)]):
            hbox(ax, 5.3, yy - 0.32, 1.75, 0.64, lbl, fc=H_TEN_F, ec=H_TEN_C, fs=7.6)
        ax.text(6.18, 5.0, "backbone_fpn", ha="center", color=DIM2, fontsize=8.5, style="italic")
        # head group
        gx0, gx1 = 7.5, 15.7
        ax.add_patch(FancyBboxPatch((gx0, 1.9), gx1 - gx0, 3.0,
                     boxstyle="round,pad=0,rounding_size=0.06", lw=2.2,
                     edgecolor=H_HEAD_C, facecolor="none", linestyle=(0, (6, 4))))
        ax.text((gx0 + gx1) / 2, 4.62, "FPNSegHead   (the part you built)",
                ha="center", color=H_HEAD_C, fontsize=12, fontweight="bold")
        # 3 arrows from fpn maps into head
        for yy in (4.25, 3.4, 2.55):
            harrow(ax, 7.05, yy, 7.75, 3.4, color="#888", lw=1.4, rad=0.0)
        steps = [("1x1 conv", "each lvl ->128"),
                 ("upsample", "align to s4"),
                 ("concat", "->384 ch"),
                 ("3x3 conv", "BN+ReLU+Drop"),
                 ("1x1 conv", "->5 classes")]
        sx = 7.75
        for i, (a, b) in enumerate(steps):
            x = sx + i * 1.58
            hbox(ax, x, 2.75, 1.4, 1.2, a, b, fc=H_HEAD_F, ec=H_HEAD_C, fs=9.5)
            if i > 0:
                harrow(ax, x - 0.18, 3.35, x, 3.35, color="#555", lw=1.6)
        # out of head
        harrow(ax, 15.55, 3.35, 16.0, 3.4, color="#555")
        hbox(ax, 16.0, ym - 0.55, 1.5, 1.1, "bilinear", "up ->512", fc=H_TEN_F, ec=H_TEN_C, fs=9.5)
        harrow(ax, 17.5, ym, 17.85, ym)
        hbox(ax, 17.85, ym - 0.6, 1.45, 1.2, "logits", "[B,5,512,512]", fc=H_OUT_F, ec=H_OUT_C, fs=10)

        ax.text(0.3, 0.6, "tip:  draw the encoder as ONE box (frozen, not yours);  spend the detail on the green FPNSegHead.",
                ha="left", color=DIM2, fontsize=9, style="italic")
        fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)


def draw_hand_promptseg(path):
    rc = dict(_HAND_RC)
    rc.pop("path.sketch", None)          # clean lines for max legibility
    name = _ensure_cjk()
    if name:
        rc["font.family"] = name
    rc["axes.unicode_minus"] = False
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(16.5, 7.2))
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 8.0)
        ax.axis("off")
        ax.text(10, 7.6, "SAM2PromptSeg  提示式裂縫分割", ha="center", color=INK2,
                fontsize=17, fontweight="bold")
        ax.text(10, 7.2, "model_prompt_seg.py   .   用 SAM2 原生管線 + 可學習提示點",
                ha="center", color=DIM2, fontsize=10)

        yt = 5.4   # 上排:影像
        yb = 2.3   # 下排:提示
        ax.text(0.35, yt + 1.15, "影像這條路", color=H_FROZ_C, fontsize=10, fontweight="bold")
        ax.text(0.35, yb - 1.05, "提示這條路", color=H_PROM_C, fontsize=10, fontweight="bold")

        # ---- 影像 lane ----
        hbox(ax, 0.3, yt - 0.6, 1.9, 1.2, "輸入影像", "[B,3,512,512]", fc=H_TEN_F, fs=10.5)
        harrow(ax, 2.2, yt, 2.55, yt)
        hbox(ax, 2.55, yt - 0.85, 2.8, 1.7, "SAM2 影像編碼器", "凍結(用預訓練)",
             fc=H_FROZ_F, ec=H_FROZ_C, fs=12)
        harrow(ax, 5.35, yt, 5.7, yt)
        hbox(ax, 5.7, yt - 0.32, 2.2, 0.9, "影像特徵", "[256,32,32]", fc=H_TEN_F, ec=H_TEN_C, fs=9.5)
        hbox(ax, 5.7, yt - 1.65, 2.2, 0.85, "高解析特徵", "stride 4/8: 128²+64²", fc=H_TEN_F, ec=H_TEN_C, fs=9)
        harrow(ax, 5.2, yt - 0.2, 5.7, yt - 1.2, color="#999", lw=1.2, rad=-0.2)

        # ---- 提示 lane ----
        hbox(ax, 2.55, yb - 0.65, 2.8, 1.3, "可學習的 8 個提示點", "(模型自己學要點哪裡)",
             fc=H_PROM_F, ec=H_PROM_C, fs=11)
        harrow(ax, 5.35, yb, 5.7, yb)
        hbox(ax, 5.7, yb - 0.6, 2.0, 1.2, "提示編碼器", "把點變成向量",
             fc=H_HEAD_F, ec=H_HEAD_C, fs=11)
        harrow(ax, 7.7, yb, 8.05, yb)
        hbox(ax, 8.05, yb - 0.65, 2.3, 1.3, "提示向量", "稀疏 = 8 個點的編碼",
             fc=H_TEN_F, ec=H_TEN_C, fs=10)

        # ---- 解碼器 group ----
        gx0, gx1 = 10.8, 16.2
        ax.add_patch(FancyBboxPatch((gx0, 1.3), gx1 - gx0, 5.0,
                     boxstyle="round,pad=0,rounding_size=0.05", lw=2.2,
                     edgecolor=H_DEC_C, facecolor="none", linestyle=(0, (6, 4))))
        ax.text((gx0 + gx1) / 2, 6.05, "遮罩解碼器 (SAM2 原生)", ha="center",
                color=H_DEC_C, fontsize=12, fontweight="bold")
        ymid = 3.85
        dsteps = [("雙向注意力", "點 與 影像 互動"),
                  ("上採樣 ×2", "逐元素相加"),
                  ("產生遮罩", "(小型 MLP)")]
        dx = 11.0
        for i, (a, b) in enumerate(dsteps):
            x = dx + i * 1.78
            hbox(ax, x, ymid - 0.7, 1.55, 1.4, a, b, fc=H_DEC_F, ec=H_DEC_C, fs=10)
            if i > 0:
                harrow(ax, x - 0.23, ymid, x, ymid, color="#555", lw=1.6)
        # 餵進解碼器
        harrow(ax, 7.9, yt, 11.0, ymid + 0.5, color="#999", lw=1.5, rad=0.12)    # 影像特徵
        harrow(ax, 10.35, yb, 11.0, ymid - 0.45, color="#999", lw=1.5, rad=-0.12)  # 提示向量
        # 高解析特徵:細節跳接 → 上採樣那一步
        ax.add_patch(FancyArrowPatch((7.9, 4.1), (10.8, 3.55), arrowstyle="-|>",
                     mutation_scale=14, lw=1.6, color=H_DEC_C, linestyle=(0, (4, 3)),
                     connectionstyle="arc3,rad=0.05"))
        ax.text(9.35, 4.28, "細節跳接 = 相加 (非 concat)", ha="center", color=H_DEC_C, fontsize=8)

        # ---- 解碼器輸出 ----
        harrow(ax, 16.0, ymid, 16.4, ymid, color="#555")
        hbox(ax, 16.4, ymid - 0.5, 2.1, 1.0, "低解析遮罩", "128×128 (先出小張)",
             fc=H_TEN_F, ec=H_TEN_C, fs=9.5)
        harrow(ax, 18.5, ymid, 18.55, ymid + 0.9, color="#555", lw=1.6, rad=0.0)
        hbox(ax, 17.5, ymid + 0.95, 2.1, 0.85, "放大", "雙線性 → 512", fc=H_TEN_F, ec=H_TEN_C, fs=9.5)
        harrow(ax, 18.55, ymid + 1.8, 18.55, yt - 0.6, color="#555", lw=1.6)
        hbox(ax, 17.55, yt - 0.6, 2.0, 1.2, "輸出遮罩", "[B,1,512,512]",
             fc=H_OUT_F, ec=H_OUT_C, fs=10.5)

        # ---- 白話說明 ----
        ax.text(0.35, 0.95,
                "流程:8 個提示點 → 編碼成向量 → 解碼器結合影像特徵 → 先產生 128×128 小遮罩 → 放大回 512×512",
                ha="left", color=INK2, fontsize=9.5)
        ax.text(0.35, 0.58,
                "稀疏向量 = 那 8 個點各自編成一條向量;   低解析遮罩 = 解碼器原本只算到 128×128,最後再雙線性放大成原圖大小。",
                ha="left", color=DIM2, fontsize=9)
        ax.text(0.35, 0.22,
                "高解析特徵 = backbone 的 stride-4/8 細節層(128×128、64×64);  上採樣時逐元素「相加」回去(不是 concat),把遮罩邊緣補銳利。",
                ha="left", color=DIM2, fontsize=9)
        fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)


def draw_fusion_compare(path):
    """add vs concat 概念 + 專案裡三個實例,並排對照。"""
    rc = dict(_HAND_RC)
    rc.pop("path.sketch", None)
    name = _ensure_cjk()
    if name:
        rc["font.family"] = name
    rc["axes.unicode_minus"] = False
    ADD_F, ADD_C = H_HEAD_F, H_HEAD_C        # 相加 = 綠
    CAT_F, CAT_C = H_PROM_F, H_PROM_C        # 串接 = 琥珀

    def node(ax, cx, cy, txt, fc, ec, r=0.42, fs=13):
        ax.add_patch(FancyBboxPatch((cx - r, cy - r), 2 * r, 2 * r,
                     boxstyle="circle,pad=0", lw=2.0, edgecolor=ec, facecolor=fc))
        ax.text(cx, cy, txt, ha="center", va="center", color=ec, fontsize=fs, fontweight="bold")

    def chip(ax, x, y, txt, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), 1.0, 0.42, boxstyle="round,pad=0,rounding_size=0.08",
                     lw=1.6, edgecolor=ec, facecolor=fc))
        ax.text(x + 0.5, y + 0.21, txt, ha="center", va="center", color=ec, fontsize=9.5, fontweight="bold")

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(16, 9.6))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9.6)
        ax.axis("off")
        ax.text(8, 9.2, "融合方式:相加 (add)  vs  串接 (concat)", ha="center",
                color=INK2, fontsize=18, fontweight="bold")

        # ===== 概念區 =====
        ax.text(0.4, 8.55, "① 兩種融合怎麼運作", ha="left", color=DIM2, fontsize=12, fontweight="bold")

        # -- 相加 panel --
        ax.add_patch(FancyBboxPatch((0.4, 5.55), 7.2, 2.7, boxstyle="round,pad=0,rounding_size=0.04",
                     lw=2.2, edgecolor=ADD_C, facecolor="none"))
        chip(ax, 6.4, 7.7, "相加", ADD_F, ADD_C)
        ax.text(0.75, 7.95, "相加 (add)", ha="left", color=ADD_C, fontsize=13, fontweight="bold")
        hbox(ax, 0.8, 7.0, 1.7, 0.6, "a", "[C, H, W]", fc=H_TEN_F, ec=H_TEN_C, fs=10)
        hbox(ax, 0.8, 6.0, 1.7, 0.6, "b", "[C, H, W]", fc=H_TEN_F, ec=H_TEN_C, fs=10)
        node(ax, 3.55, 6.6, "+", ADD_F, ADD_C)
        hbox(ax, 4.7, 6.3, 1.9, 0.6, "out", "[C, H, W]", fc=ADD_F, ec=ADD_C, fs=10)
        harrow(ax, 2.5, 7.3, 3.2, 6.85, color="#777", lw=1.6)
        harrow(ax, 2.5, 6.3, 3.2, 6.35, color="#777", lw=1.6)
        harrow(ax, 3.97, 6.6, 4.7, 6.6, color="#555", lw=1.8)
        ax.text(0.75, 5.85, "channel 不變;兩邊必須同 channel;像殘差/修正,省記憶體",
                ha="left", color=DIM2, fontsize=9)

        # -- 串接 panel --
        ax.add_patch(FancyBboxPatch((8.0, 5.55), 7.6, 2.7, boxstyle="round,pad=0,rounding_size=0.04",
                     lw=2.2, edgecolor=CAT_C, facecolor="none"))
        chip(ax, 14.4, 7.7, "串接", CAT_F, CAT_C)
        ax.text(8.35, 7.95, "串接 (concat)", ha="left", color=CAT_C, fontsize=13, fontweight="bold")
        hbox(ax, 8.4, 7.0, 1.7, 0.6, "a", "[C1, H, W]", fc=H_TEN_F, ec=H_TEN_C, fs=10)
        hbox(ax, 8.4, 6.0, 1.7, 0.6, "b", "[C2, H, W]", fc=H_TEN_F, ec=H_TEN_C, fs=10)
        node(ax, 11.0, 6.6, "接", CAT_F, CAT_C, fs=12)
        hbox(ax, 12.1, 6.3, 1.9, 0.6, "拼接", "[C1+C2, H, W]", fc=CAT_F, ec=CAT_C, fs=9.5)
        hbox(ax, 14.2, 6.3, 1.3, 0.6, "conv", "→[C,H,W]", fc=H_TEN_F, ec=H_TEN_C, fs=8.5)
        harrow(ax, 10.1, 7.3, 10.65, 6.85, color="#777", lw=1.6)
        harrow(ax, 10.1, 6.3, 10.65, 6.35, color="#777", lw=1.6)
        harrow(ax, 11.45, 6.6, 12.1, 6.6, color="#555", lw=1.8)
        harrow(ax, 14.0, 6.6, 14.2, 6.6, color="#555", lw=1.8)
        ax.text(8.35, 5.85, "channel 變大 (C1+C2);可不同 channel;資訊全留,交給後面 conv 自己混",
                ha="left", color=DIM2, fontsize=9)

        # ===== 實例區 =====
        ax.text(0.4, 4.95, "② 你專案裡的三個實例", ha="left", color=DIM2, fontsize=12, fontweight="bold")

        def panel(x0, x1, title, ec, tag, tagf, tagc):
            ax.add_patch(FancyBboxPatch((x0, 0.5), x1 - x0, 4.1,
                         boxstyle="round,pad=0,rounding_size=0.04", lw=2.2,
                         edgecolor=ec, facecolor="none"))
            ax.text((x0 + x1) / 2, 4.25, title, ha="center", color=ec, fontsize=12, fontweight="bold")
            chip(ax, x1 - 1.25, 0.7, tag, tagf, tagc)

        # ① FpnNeck (相加)
        panel(0.3, 5.2, "① SAM2 FpnNeck", ADD_C, "相加", ADD_F, ADD_C)
        hbox(ax, 0.55, 3.15, 1.95, 0.68, "C4 (s16)", "256", fc=H_TEN_F, ec=H_TEN_C, fs=9.5)
        hbox(ax, 2.75, 3.15, 2.0, 0.68, "↑上採樣 C5", "(s32)", fc=H_TEN_F, ec=H_TEN_C, fs=9.5)
        node(ax, 2.75, 2.25, "+", ADD_F, ADD_C, r=0.36, fs=12)
        harrow(ax, 1.5, 3.15, 2.5, 2.5, color="#777", lw=1.5)
        harrow(ax, 3.75, 3.15, 3.0, 2.5, color="#777", lw=1.5)
        hbox(ax, 1.8, 1.35, 1.9, 0.68, "輸出 (s16)", "256", fc=ADD_F, ec=ADD_C, fs=9.5)
        harrow(ax, 2.75, 1.9, 2.75, 2.03, color="#555", lw=1.6)
        ax.text(2.75, 0.95, "只有 s16 收 top-down,逐元素相加", ha="center", color=DIM2, fontsize=8)

        # ② mask decoder 高解析 (相加)
        panel(5.5, 10.4, "② mask decoder 高解析", ADD_C, "相加", ADD_F, ADD_C)
        hbox(ax, 6.95, 3.3, 2.0, 0.56, "src  32²", "256", fc=H_TEN_F, ec=H_TEN_C, fs=9)
        hbox(ax, 6.95, 2.45, 2.0, 0.56, "64²  + feat_s1", "(上採樣後相加)", fc=ADD_F, ec=ADD_C, fs=8.5)
        hbox(ax, 6.95, 1.6, 2.0, 0.56, "128² + feat_s0", "(上採樣後相加)", fc=ADD_F, ec=ADD_C, fs=8.5)
        harrow(ax, 7.95, 3.3, 7.95, 3.01, color="#555", lw=1.5)
        harrow(ax, 7.95, 2.45, 7.95, 2.16, color="#555", lw=1.5)
        ax.text(9.05, 3.0, "dc1↑", ha="left", color=DIM2, fontsize=7.5)
        ax.text(9.05, 2.15, "dc2↑", ha="left", color=DIM2, fontsize=7.5)
        ax.text(7.95, 0.95, "一邊上採樣,一邊把細節層相加回來", ha="center", color=DIM2, fontsize=8)

        # ③ FPNSegHead (串接)
        panel(10.7, 15.6, "③ 你的 FPNSegHead", CAT_C, "串接", CAT_F, CAT_C)
        hbox(ax, 12.05, 3.3, 2.2, 0.56, "f0 / f1 / f2", "各 128 (對齊解析度)", fc=H_TEN_F, ec=H_TEN_C, fs=8.5)
        hbox(ax, 12.05, 2.45, 2.2, 0.56, "cat → 384", "(串接)", fc=CAT_F, ec=CAT_C, fs=9)
        hbox(ax, 12.05, 1.6, 2.2, 0.56, "→ 128 → 分類", "3×3 conv 壓回", fc=H_TEN_F, ec=H_TEN_C, fs=8.5)
        harrow(ax, 13.15, 3.3, 13.15, 3.01, color="#555", lw=1.5)
        harrow(ax, 13.15, 2.45, 13.15, 2.16, color="#555", lw=1.5)
        ax.text(13.15, 0.95, "串接後再 conv 壓回 channel", ha="center", color=DIM2, fontsize=8)

        fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)


def draw_semseg_encoder_fpn(path):
    """Detailed Hiera trunk + FpnNeck used by model_seg.py (variant=small, 512)."""
    fig, ax = plt.subplots(figsize=(14, 13.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 13.6)
    ax.axis("off")

    ax.text(6.5, 13.15, "SAM2 Image Encoder  -  Hiera trunk + FpnNeck (detailed)",
            ha="center", color=INK, fontsize=18, fontweight="bold")
    ax.text(6.5, 12.78, "model_seg.py backbone   .   variant=small   .   input 512x512   .   d_model=256",
            ha="center", color=MUTE, fontsize=10.5)

    # column header band
    for cx, t, c in [(1.65, "Hiera trunk", FROZEN_E), (4.3, "trunk taps", TENSOR_E),
                     (6.7, "FpnNeck", TRAIN_E), (10.6, "backbone_fpn", "#d06b7e")]:
        ax.text(cx, 12.25, t, ha="center", color=c, fontsize=11, fontweight="bold")

    rows = {"s4": 10.4, "s8": 8.3, "s16": 6.2, "s32": 4.1}

    # ---- top: input + patch embed ----
    tensor(ax, 0.5, 11.75, 2.4, 0.5, "x  [B, 3, 512, 512]", fs=9)
    box(ax, 0.4, 10.95, 2.5, 0.62, "PatchEmbed", "Conv2d 3->96, k7 s4 p3  + pos_embed",
        fc=FROZEN, ec=FROZEN_E, fs=10)
    arrow(ax, 1.65, 11.75, 1.65, 11.58, color=ARROW, lw=1.4)
    arrow(ax, 1.65, 10.95, 1.65, 10.9, color=ARROW, lw=1.4)

    # ---- trunk stages ----
    stages = [
        ("s4",  "Stage 1", "1x MSBlock  .  dim 96  .  win 8"),
        ("s8",  "Stage 2", "2x MSBlock  .  dim 192  .  win 4"),
        ("s16", "Stage 3", "11x MSBlock  .  dim 384  .  global-attn @7,10,13"),
        ("s32", "Stage 4", "2x MSBlock  .  dim 768  .  win 7"),
    ]
    for key, title, sub in stages:
        y = rows[key]
        box(ax, 0.4, y - 0.5, 2.5, 1.0, title, sub, fc=FROZEN, ec=FROZEN_E, fs=10.5)

    # q_pool transitions (down arrows between stages)
    qpool = [("s4", "s8", "q_pool blk1:  MaxPool2d 2x2 s2  +  proj 96->192"),
             ("s8", "s16", "q_pool blk3:  MaxPool2d 2x2 s2  +  proj 192->384"),
             ("s16", "s32", "q_pool blk14: MaxPool2d 2x2 s2  +  proj 384->768")]
    for a, b, lbl in qpool:
        arrow(ax, 1.65, rows[a] - 0.5, 1.65, rows[b] + 0.5, color=PROMPT_E, lw=1.6)
        ax.text(1.78, (rows[a] - 0.5 + rows[b] + 0.5) / 2, lbl, ha="left", va="center",
                color=PROMPT_E, fontsize=7.3)

    # ---- trunk taps (middle tensors) ----
    taps = [("s4", "xs0  [B, 96,128,128]\nstride 4"),
            ("s8", "xs1  [B,192, 64, 64]\nstride 8"),
            ("s16", "xs2  [B,384, 32, 32]\nstride 16"),
            ("s32", "xs3  [B,768, 16, 16]\nstride 32")]
    for key, lbl in taps:
        y = rows[key]
        tensor(ax, 3.45, y - 0.32, 1.85, 0.64, lbl, fs=7.8)
        arrow(ax, 2.9, y, 3.45, y, color=MUTE, lw=1.3)

    # ---- FpnNeck lateral convs ----
    lat = [("s4",  "convs[3]: 1x1  96->256", "lateral only"),
           ("s8",  "convs[2]: 1x1 192->256", "lateral only"),
           ("s16", "convs[1]: 1x1 384->256", "lateral + top-down"),
           ("s32", "convs[0]: 1x1 768->256", "top-down source")]
    for key, lbl, sub in lat:
        y = rows[key]
        box(ax, 5.7, y - 0.36, 2.05, 0.72, lbl, sub, fc=TRAIN, ec=TRAIN_E, fs=8.6)
        arrow(ax, 5.3, y, 5.7, y, color=ARROW, lw=1.3)

    # ---- top-down add for s16 only ----
    addx, addy = 8.35, rows["s16"]
    box(ax, addx - 0.32, addy - 0.32, 0.64, 0.64, "(+)", "sum", fc="#2a1733", ec=DECODE_E, fs=12)
    arrow(ax, 7.75, addy, addx - 0.32, addy, color=ARROW, lw=1.4)
    # upsample path: s32 lateral -> up -> add
    arrow(ax, 7.4, rows["s32"] + 0.36, addx, addy - 0.34, color=DECODE_E, lw=1.7, rad=-0.32)
    ax.text(8.95, 5.0, "F.interpolate x2\nnearest  (fp32)", ha="left", va="center",
            color=DECODE_E, fontsize=7.6)

    # ---- backbone_fpn outputs ----
    outs = [("s4",  "out0 [B,256,128,128]  s4", False),
            ("s8",  "out1 [B,256, 64, 64]  s8", False),
            ("s16", "out2 [B,256, 32, 32]  s16", False),
            ("s32", "out3 [B,256, 16, 16]  s32", True)]
    for key, lbl, dropped in outs:
        y = rows[key]
        if dropped:
            box(ax, 9.6, y - 0.3, 2.6, 0.6, "out3  (DROPPED)", "scalp=1 discards lowest-res",
                fc="#1a1a1a", ec="#6b3a3a", fs=8.5, tc="#9a6b6b")
            arrow(ax, 7.75, y, 9.6, y, color="#5a4040", lw=1.2, ls=(0, (3, 2)))
            ax.text(10.9, y - 0.55, "x", ha="center", color="#9a6b6b", fontsize=12, fontweight="bold")
        else:
            tensor(ax, 9.6, y - 0.3, 2.6, 0.6, lbl, fs=8.5)
            src_x = addx + 0.32 if key == "s16" else 7.75
            arrow(ax, src_x, y, 9.6, y, color=ARROW, lw=1.3)

    # ---- handoff to head (right-side bus, avoids the dropped out3 box) ----
    box(ax, 9.0, 2.35, 2.4, 0.62, "to FPNSegHead", "3 levels x 256ch", fc=TRAIN, ec=TRAIN_E, fs=10)
    busx = 12.55
    ax.add_line(Line2D([busx, busx], [2.66, rows["s4"]], color=ARROW, lw=1.3))
    for key in ("s4", "s8", "s16"):
        ax.add_line(Line2D([12.2, busx], [rows[key], rows[key]], color=ARROW, lw=1.3))
    arrow(ax, busx, 2.66, 11.4, 2.66, color=ARROW, lw=1.3)

    # ---- footnotes ----
    ax.text(0.5, 2.1,
            "fpn_top_down_levels=[2,3]:  only the s16 level receives top-down fusion.\n"
            "s4 / s8 are pure 1x1 lateral convs (no top-down);  s32 is the top-down source then dropped by scalp.",
            ha="left", va="top", color=MUTE, fontsize=8.6)
    ax.text(0.5, 1.15,
            "Each FPN level also -> PositionEmbeddingSine(256) = pos[i],  but SAM2SemSeg consumes only backbone_fpn (pos ignored).\n"
            "fuse_type=sum.  Whole encoder is frozen when freeze_trunk=True; FpnNeck stays trainable when freeze_neck=False.",
            ha="left", va="top", color=MUTE, fontsize=8.6)

    legend(ax, 0.2, [("Hiera trunk (frozen)", FROZEN, FROZEN_E),
                     ("q_pool downsample", "#2a2010", PROMPT_E),
                     ("FpnNeck lateral (trainable)", TRAIN, TRAIN_E),
                     ("top-down / upsample", "#2a1733", DECODE_E),
                     ("tensor", TENSOR, TENSOR_E)], x0=0.5, xstep=2.45)

    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def legend(ax, y, items, x0=0.6, xstep=3.0):
    for i, (lbl, fc, ec) in enumerate(items):
        x = x0 + i * xstep
        ax.add_patch(FancyBboxPatch((x, y), 0.28, 0.28,
                     boxstyle="round,pad=0,rounding_size=0.04", fc=fc, ec=ec, lw=1.3))
        ax.text(x + 0.38, y + 0.14, lbl, ha="left", va="center", color=MUTE, fontsize=8.2)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "arch")
    os.makedirs(out_dir, exist_ok=True)
    draw_semseg(os.path.join(out_dir, "arch_sam2semseg.png"))
    draw_promptseg(os.path.join(out_dir, "arch_sam2promptseg.png"))
    draw_compare(os.path.join(out_dir, "arch_compare.png"))
    draw_semseg_encoder_fpn(os.path.join(out_dir, "arch_semseg_encoder_fpn.png"))
    draw_hand_semseg(os.path.join(out_dir, "arch_hand_semseg.png"))
    draw_hand_promptseg(os.path.join(out_dir, "arch_hand_promptseg.png"))
    draw_fusion_compare(os.path.join(out_dir, "arch_fusion_add_vs_concat.png"))
    print("wrote:")
    for f in ("arch_sam2semseg.png", "arch_sam2promptseg.png", "arch_compare.png",
              "arch_semseg_encoder_fpn.png", "arch_hand_semseg.png", "arch_hand_promptseg.png",
              "arch_fusion_add_vs_concat.png"):
        print("  ", os.path.join(out_dir, f))
