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


class FusedPromptedSAM2Seg(PromptedSAM2Seg):
    """PromptedSAM2Seg + DINOv2 殘差注入 feat。forward 多收一個 dino_feat。
    DINOv2 本身不在此建構(離線 cache),只持有 adapter。"""

    def __init__(self, variant="small", image_size=512, dino_dim: int = 384,
                 mask_prompt_size=None, device: Optional[str] = None):
        super().__init__(variant=variant, image_size=image_size,
                         mask_prompt_size=mask_prompt_size, device=device)
        sam_dim = self.image_encoder.neck.d_model  # = fpn channel = 256
        self.fusion = FeatFusionAdapter(sam_dim=sam_dim, dino_dim=dino_dim)
        if device:
            self.fusion = self.fusion.to(device)

    def forward(self, x, dino_feat, point_coords, point_labels, prev_mask=None):
        enc = self.encode_image(x)
        enc["feat"] = self.fusion(enc["feat"], dino_feat)
        masks, _ = self.decode(enc, point_coords, point_labels, prev_mask)
        return masks

    def param_groups(self, base_lr, encoder_lr_mult=0.1):
        dec, enc = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "fusion" in n or "sam_mask_decoder" in n or "sam_prompt_encoder" in n:
                dec.append(p)
            else:
                enc.append(p)
        groups = [{"params": dec, "lr": base_lr, "name": "decoder"}]
        if enc:
            groups.append({"params": enc, "lr": base_lr * encoder_lr_mult, "name": "encoder"})
        return [g for g in groups if g["params"]]
