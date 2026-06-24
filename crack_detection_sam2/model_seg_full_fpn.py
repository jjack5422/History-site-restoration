"""SAM2 + Full FPN Head (Semantic FPN style, Kirillov et al. 2019).

vs `model_seg.py` (simplified, single concat):
- Each level applies: 3x3 conv + BN + ReLU + upsample 2x, repeated until at stride 4
- All levels are SUMMED (add) instead of concatenated
- More processing per level, but no big concat fusion

vs SAM2 internal FpnNeck (which already does top-down add):
- This is the seg head AFTER FpnNeck, takes its already-fused outputs
- "Full" here refers to per-level conv chain + add fusion at the head, not at the neck
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import build_sam2_model


class FPNUpsampleBlock(nn.Module):
    """Single block: 3x3 conv + BN + ReLU, optionally upsample 2x."""
    def __init__(self, in_ch: int, out_ch: int, upsample: bool):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return x


class SemanticFPNHead(nn.Module):
    """Semantic FPN head: each level goes through K conv+upsample steps to reach stride 4, then SUM.

    Args:
        in_channels_list: channels for each input level (highest res first)
        strides: stride at each input level relative to input image (e.g. [4, 8, 16])
        out_channels: hidden channels (default 128)
        num_classes: output classes
    """
    def __init__(self, in_channels_list: List[int], strides: List[int],
                 out_channels: int = 128, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        target_stride = strides[0]  # finest
        self.branches = nn.ModuleList()
        for in_ch, s in zip(in_channels_list, strides):
            # number of 2x upsamples needed to go from stride s to target_stride
            n_ups = 0
            cur = s
            while cur > target_stride:
                cur //= 2
                n_ups += 1
            # build chain: [conv(in→out, no-up), conv(out→out, up)*n_ups]
            blocks = [FPNUpsampleBlock(in_ch, out_channels, upsample=(n_ups > 0 and 0 < 1))]
            # First block: in_ch → out_channels, no upsample
            blocks = [FPNUpsampleBlock(in_ch, out_channels, upsample=False)]
            # Then n_ups blocks of out→out with upsample
            for _ in range(n_ups):
                blocks.append(FPNUpsampleBlock(out_channels, out_channels, upsample=True))
            self.branches.append(nn.Sequential(*blocks))

        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # Process each level then SUM
        target_h, target_w = feats[0].shape[-2:]
        ys = []
        for x, branch in zip(feats, self.branches):
            y = branch(x)
            if y.shape[-2:] != (target_h, target_w):
                # safety: bilinear adjust if rounding mismatch
                y = F.interpolate(y, size=(target_h, target_w),
                                  mode="bilinear", align_corners=False)
            ys.append(y)
        fused = torch.stack(ys, dim=0).sum(dim=0)  # add fusion
        fused = self.dropout(fused)
        return self.classifier(fused)


class SAM2SemSegFullFPN(nn.Module):
    def __init__(self,
                 variant: str = "small",
                 num_classes: int = 5,
                 hidden: int = 128,
                 dropout: float = 0.1,
                 freeze_trunk: bool = True,
                 freeze_neck: bool = False,
                 device: Optional[str] = None):
        super().__init__()
        sam2 = build_sam2_model(variant=variant, device=device, mode="train")
        self.image_encoder = sam2.image_encoder
        del sam2

        n_levels = len(self.image_encoder.neck.backbone_channel_list) - self.image_encoder.scalp
        d_model = self.image_encoder.neck.d_model
        in_channels_list = [d_model] * n_levels
        # FpnNeck output order: index 0 is finest (stride 4 for SAM2 small)
        strides = [4 * (2 ** i) for i in range(n_levels)]  # [4, 8, 16, ...]

        self.head = SemanticFPNHead(
            in_channels_list=in_channels_list, strides=strides,
            out_channels=hidden, num_classes=num_classes, dropout=dropout,
        )
        self.num_classes = num_classes

        if freeze_trunk:
            for p in self.image_encoder.trunk.parameters():
                p.requires_grad = False
        if freeze_neck:
            for p in self.image_encoder.neck.parameters():
                p.requires_grad = False

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = self.image_encoder(x)
        return out["backbone_fpn"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        feats = self.encode(x)
        logits = self.head(feats)
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_groups(self, base_lr: float, encoder_lr_mult: float = 0.1):
        head_params, enc_params = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("head."):
                head_params.append(p)
            else:
                enc_params.append(p)
        groups = [{"params": head_params, "lr": base_lr, "name": "head"}]
        if enc_params:
            groups.append({"params": enc_params, "lr": base_lr * encoder_lr_mult, "name": "encoder"})
        return groups


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SAM2SemSegFullFPN(variant="small", num_classes=2, device=device).to(device)
    total, trainable = count_params(model)
    print(f"total={total/1e6:.1f}M  trainable={trainable/1e6:.2f}M")
    x = torch.randn(2, 3, 512, 512, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"output shape={tuple(y.shape)} dtype={y.dtype}")
