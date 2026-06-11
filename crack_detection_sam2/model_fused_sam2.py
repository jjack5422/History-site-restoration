"""SAM2 prompt-refine + 凍結 DINOv2 語意注入。

FeatFusionAdapter 把離線 cache 的 DINOv2 token map 殘差注入 SAM2 的 feat (fpn[-1]);
zero-init 使初始 feat'==feat,訓練起點等於 baseline,只學增量。high_res 不經此模組。
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_prompted_sam2 import PromptedSAM2Seg


class FeatFusionAdapter(nn.Module):
    def __init__(self, sam_dim: int = 256, dino_dim: int = 384, hidden: int = 256):
        super().__init__()
        self.dino_proj = nn.Conv2d(dino_dim, hidden, kernel_size=1)
        self.fuse = nn.Conv2d(sam_dim + hidden, sam_dim, kernel_size=1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, feat: torch.Tensor, dino: torch.Tensor) -> torch.Tensor:
        # feat [B,sam_dim,h,w]; dino [B,dino_dim,gh,gw]
        d = self.dino_proj(dino)
        if d.shape[-2:] != feat.shape[-2:]:
            d = F.interpolate(d, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        delta = self.fuse(torch.cat([feat, d], dim=1))
        return feat + delta
