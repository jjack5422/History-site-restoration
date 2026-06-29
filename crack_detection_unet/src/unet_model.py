"""Segmentation models via segmentation_models_pytorch.

Default ResUNet = smp.Unet + ResNet encoder. ``build_model`` generalises to the
mainstream baselines compared in 2026-06-29 craquelure study (DeepLabV3+,
SegFormer, R-FPANet); all return logits-NCHW.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# arch key -> (smp constructor, default encoder when caller passes the sentinel "auto")
_ARCH = {
    "unet": (smp.Unet, "resnet50"),
    "deeplabv3plus": (smp.DeepLabV3Plus, "resnet50"),
    "segformer": (smp.Segformer, "mit_b0"),
}


def build_model(arch: str = "unet",
                encoder: str = "auto",
                encoder_weights: str = "imagenet",
                num_classes: int = 2,
                in_channels: int = 3) -> nn.Module:
    arch = arch.lower()
    if arch == "rfpanet":
        from rfpanet import build_rfpanet
        return build_rfpanet(num_classes=num_classes, encoder_weights=encoder_weights,
                             in_channels=in_channels)
    if arch not in _ARCH:
        raise ValueError(f"unknown arch '{arch}', choose from {list(_ARCH) + ['rfpanet']}")
    ctor, default_enc = _ARCH[arch]
    enc = default_enc if encoder == "auto" else encoder
    return ctor(
        encoder_name=enc,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )


def build_resunet(encoder: str = "resnet50",
                  encoder_weights: str = "imagenet",
                  num_classes: int = 2,
                  in_channels: int = 3) -> nn.Module:
    """Backwards-compatible thin wrapper (smp.Unet)."""
    return build_model("unet", encoder, encoder_weights, num_classes, in_channels)


def param_groups(model: nn.Module, base_lr: float, encoder_lr_mult: float = 0.1):
    enc_params, dec_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("encoder."):
            enc_params.append(p)
        else:
            dec_params.append(p)
    groups = [{"params": dec_params, "lr": base_lr, "name": "decoder"}]
    if enc_params:
        groups.append({"params": enc_params, "lr": base_lr * encoder_lr_mult, "name": "encoder"})
    return groups


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    m = build_resunet("resnet50", num_classes=2)
    total, trainable = count_params(m)
    print(f"ResUNet resnet50 total={total/1e6:.2f}M trainable={trainable/1e6:.2f}M")
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        y = m(x)
    print("out", tuple(y.shape))
