"""Multi-class semantic segmentation 模型: SAM2 Hiera image encoder + 輕量 FPN seg head。

- 用 SAM2 的 image encoder 當 backbone (含 FpnNeck)
- 取 backbone_fpn 的 3 個 level (strides 4/8/16, 各 256ch)
- 投影 + 上採樣對齊到 stride 4 → concat → 3x3 conv → 1x1 conv 出 num_classes
- 最後 bilinear 上採樣回到原始輸入解析度

凍結策略:
- freeze_trunk=True: 凍結 trunk (Hiera blocks 與 patch_embed)
- freeze_neck=True:  同時凍結 FpnNeck
- 兩者皆 False 表示全部可訓練

Note: SAM2 預訓練於 1024x1024 normalize=ImageNet, 但此 head 對輸入大小不敏感 (全 conv);
512 輸入時 pos embed 會 bilinear interpolate, 仍可訓練, 但若品質不夠可改 1024。
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import build_sam2_model


class FPNSegHead(nn.Module):
    def __init__(self, in_channels_list: List[int], hidden: int = 128,
                 num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, hidden, kernel_size=1) for c in in_channels_list
        ])
        fused = hidden * len(in_channels_list)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(hidden, num_classes, kernel_size=1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # feats[0] 為最高解析度 (stride 4); 全部對齊到 feats[0] 大小再 concat
        target_h, target_w = feats[0].shape[-2:]
        ys = []
        for x, lat in zip(feats, self.lateral):
            y = lat(x)
            if y.shape[-2:] != (target_h, target_w):
                y = F.interpolate(y, size=(target_h, target_w),
                                  mode="bilinear", align_corners=False)
            ys.append(y)
        x = torch.cat(ys, dim=1)
        x = self.fuse(x)
        return self.classifier(x)


class SAM2SemSeg(nn.Module):
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
        # 只保留 image_encoder, 釋放 SAM2 其餘 (mask decoder/memory) 記憶體
        self.image_encoder = sam2.image_encoder
        del sam2

        n_levels = len(self.image_encoder.neck.backbone_channel_list) - self.image_encoder.scalp
        d_model = self.image_encoder.neck.d_model
        in_channels_list = [d_model] * n_levels

        self.head = FPNSegHead(in_channels_list=in_channels_list,
                               hidden=hidden, num_classes=num_classes, dropout=dropout)
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
        """回傳 optimizer param groups: head 用 base_lr, encoder unfrozen 部分用 base_lr * encoder_lr_mult。"""
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="small")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--freeze_trunk", action="store_true", default=True)
    parser.add_argument("--no_freeze_trunk", dest="freeze_trunk", action="store_false")
    args = parser.parse_args()

    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SAM2SemSeg(variant=args.variant, num_classes=5,
                       freeze_trunk=args.freeze_trunk, device=device).to(device)
    total, trainable = count_params(model)
    print(f"variant={args.variant} total={total/1e6:.1f}M trainable={trainable/1e6:.2f}M")

    x = torch.randn(2, 3, args.size, args.size, device=device)
    with torch.no_grad():
        feats = model.encode(x)
        for i, f in enumerate(feats):
            print(f"feat[{i}] shape={tuple(f.shape)}")
        y = model(x)
    print(f"logits={tuple(y.shape)} dtype={y.dtype}")
