"""Res-UNet via segmentation_models_pytorch (smp.Unet with ResNet encoder)."""
from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_resunet(encoder: str = "resnet50",
                  encoder_weights: str = "imagenet",
                  num_classes: int = 2,
                  in_channels: int = 3) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )


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
