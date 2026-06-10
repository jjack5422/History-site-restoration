"""DADNet (Wu et al., Heritage Science 2024) reimplementation.

ConvNeXt-T (timm) encoder + U-Net decoder, with a Dual-Attention module on each
skip connection:
  - Neighborhood Attention (NA): local k x k self-attention. Pure-PyTorch unfold
    implementation (no `natten` CUDA kernel needed -> works on any GPU incl. sm_120).
  - Biaxial Attention Block (BA-B): axial (row+col) attention wrapped in an
    RFB-like multi-branch dilated-conv block.

Paper hyper-params: neighborhood k=7, BA-B dilation=7, input 224x224.
No official code released; this is a faithful-as-possible reimplementation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborhoodAttention(nn.Module):
    """Single-head local self-attention over a k x k neighborhood (unfold version)."""

    def __init__(self, dim: int, k: int = 7):
        super().__init__()
        assert k % 2 == 1, "k must be odd"
        self.k = k
        self.pad = k // 2
        self.scale = dim ** -0.5
        self.q = nn.Conv2d(dim, dim, 1)
        self.kv = nn.Conv2d(dim, 2 * dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x)                                   # (B,C,H,W)
        k, v = self.kv(x).chunk(2, dim=1)               # each (B,C,H,W)
        kk = self.k * self.k
        # unfold neighbors -> (B, C, kk, H*W)
        k_n = F.unfold(k, self.k, padding=self.pad).view(B, C, kk, H * W)
        v_n = F.unfold(v, self.k, padding=self.pad).view(B, C, kk, H * W)
        q_f = q.view(B, C, 1, H * W)
        # attention scores over neighborhood: (B, kk, H*W)
        attn = (q_f * k_n).sum(dim=1) * self.scale
        attn = attn.softmax(dim=1)
        out = (attn.unsqueeze(1) * v_n).sum(dim=2)      # (B, C, H*W)
        out = out.view(B, C, H, W)
        return self.proj(out)


class AxialAttention(nn.Module):
    """Axial self-attention along one axis. axis='W' attends within each row."""

    def __init__(self, dim: int, heads: int = 4, axis: str = "W"):
        super().__init__()
        self.axis = axis
        self.heads = heads
        self.dim = dim
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.axis == "W":
            seq = x.permute(0, 2, 3, 1).reshape(B * H, W, C)   # attend along W
            L = W
        else:
            seq = x.permute(0, 3, 2, 1).reshape(B * W, H, C)   # attend along H
            L = H
        qkv = self.qkv(seq).reshape(seq.shape[0], L, 3, self.heads, C // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)                       # (3, N, heads, L, dh)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(seq.shape[0], L, C)
        out = self.proj(out)
        if self.axis == "W":
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            out = out.reshape(B, W, H, C).permute(0, 3, 2, 1)
        return out


class BiAxialBlock(nn.Module):
    """RFB-like multi-branch dilated conv + biaxial (row+col) attention."""

    def __init__(self, dim: int, dilation: int = 7):
        super().__init__()
        b = max(8, dim // 4)
        self.branch1 = nn.Conv2d(dim, b, 1)
        self.branch2 = nn.Conv2d(dim, b, 3, padding=1)
        self.branch3 = nn.Conv2d(dim, b, 3, padding=dilation, dilation=dilation)
        self.fuse = nn.Conv2d(3 * b, dim, 1)
        self.row = AxialAttention(dim, axis="W")
        self.col = AxialAttention(dim, axis="H")
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        m = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        m = self.fuse(m)
        m = self.norm(m)
        return m + self.row(m) + self.col(m)


class DualAttention(nn.Module):
    """Skip-connection refinement: x + NA(x) + BA-B(x)."""

    def __init__(self, dim: int, k: int = 7, dilation: int = 7):
        super().__init__()
        self.na = NeighborhoodAttention(dim, k=k)
        self.bab = BiAxialBlock(dim, dilation=dilation)

    def forward(self, x):
        return x + self.na(x) + self.bab(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch), nn.GELU(),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class DADNet(nn.Module):
    def __init__(self, num_classes: int = 2, k: int = 7, dilation: int = 7,
                 backbone: str = "convnext_tiny", pretrained: bool = True):
        super().__init__()
        import timm
        self.encoder = timm.create_model(
            backbone, features_only=True, pretrained=pretrained, in_chans=3)
        chs = self.encoder.feature_info.channels()   # e.g. [96,192,384,768]
        self.chs = chs
        # dual attention on the three shallower skips (not the bottleneck)
        self.da = nn.ModuleList([DualAttention(c, k=k, dilation=dilation) for c in chs[:-1]])
        self.dec3 = DecoderBlock(chs[3], chs[2], chs[2])
        self.dec2 = DecoderBlock(chs[2], chs[1], chs[1])
        self.dec1 = DecoderBlock(chs[1], chs[0], chs[0])
        self.up_head = nn.Sequential(
            nn.Conv2d(chs[0], chs[0] // 2, 3, padding=1),
            nn.GroupNorm(1, chs[0] // 2), nn.GELU(),
        )
        self.head = nn.Conv2d(chs[0] // 2, num_classes, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        f0, f1, f2, f3 = self.encoder(x)
        s0 = self.da[0](f0)
        s1 = self.da[1](f1)
        s2 = self.da[2](f2)
        d = self.dec3(f3, s2)
        d = self.dec2(d, s1)
        d = self.dec1(d, s0)
        d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
        d = self.up_head(d)
        return self.head(d)
