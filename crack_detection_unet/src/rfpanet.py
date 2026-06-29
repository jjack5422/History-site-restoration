"""R-FPANet (Yuan et al. 2024, Structures 105780) — faithful distillation.

Design distilled from the paper:
  - Backbone: ResNet-50 with stage-4/5 dilated (stride->1, dilation 2/4) => output
    stride 8 (paper: 224 -> 28x28). torchvision replace_stride_with_dilation.
  - Dense Block on the deepest feature before FPN (DenseNet-style feature reuse).
  - DANet dual self-attention (PAM position + CAM channel) on the deepest feature.
  - FPN: lateral 1x1 -> top-down element-wise sum -> per-level predict head ->
    upsample to input -> sum all level predicts = final logits.

Returns logits (B, num_classes, H, W) so it plugs into the existing 2-class
CEDiceLoss training pipeline (train.py --arch rfpanet). encoder params are named
``encoder.*`` so param_groups() applies encoder_lr_mult unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class PAM(nn.Module):
    """Position attention (DANet). energy is N x N over spatial positions."""
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, max(in_dim // 8, 1), 1)
        self.key = nn.Conv2d(in_dim, max(in_dim // 8, 1), 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # B,N,C'
        k = self.key(x).view(B, -1, H * W)                     # B,C',N
        att = self.softmax(torch.bmm(q, k))                    # B,N,N
        v = self.value(x).view(B, -1, H * W)                   # B,C,N
        out = torch.bmm(v, att.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class CAM(nn.Module):
    """Channel attention (DANet). energy is C x C over channels (no conv)."""
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = x.view(B, C, -1)                       # B,C,N
        k = x.view(B, C, -1).permute(0, 2, 1)      # B,N,C
        energy = torch.bmm(q, k)                    # B,C,C
        energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        att = self.softmax(energy)
        v = x.view(B, C, -1)                        # B,C,N
        out = torch.bmm(att, v).view(B, C, H, W)
        return self.gamma * out + x


class DANetHead(nn.Module):
    """Reduce -> PAM + CAM -> sum -> conv (DANet fusion)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        inter = out_dim
        self.conv_p_in = nn.Sequential(nn.Conv2d(in_dim, inter, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(inter), nn.ReLU(inplace=True))
        self.conv_c_in = nn.Sequential(nn.Conv2d(in_dim, inter, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(inter), nn.ReLU(inplace=True))
        self.pam = PAM(inter)
        self.cam = CAM()
        self.conv_p_out = nn.Sequential(nn.Conv2d(inter, inter, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(inter), nn.ReLU(inplace=True))
        self.conv_c_out = nn.Sequential(nn.Conv2d(inter, inter, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(inter), nn.ReLU(inplace=True))

    def forward(self, x):
        p = self.conv_p_out(self.pam(self.conv_p_in(x)))
        c = self.conv_c_out(self.cam(self.conv_c_in(x)))
        return p + c


class DenseBlock(nn.Module):
    """Small DenseNet-style block (feature reuse) then 1x1 compress."""
    def __init__(self, in_dim, growth=128, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_dim
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                nn.Conv2d(ch, growth, 3, padding=1, bias=False)))
            ch += growth
        self.compress = nn.Conv2d(ch, in_dim, 1)

    def forward(self, x):
        feats = x
        for layer in self.layers:
            new = layer(feats)
            feats = torch.cat([feats, new], 1)
        return self.compress(feats)


class RFPANet(nn.Module):
    def __init__(self, num_classes=2, fpn_dim=256, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        # stage-4 (layer3) and stage-5 (layer4) dilated -> output stride 8
        self.encoder = resnet50(weights=weights,
                                replace_stride_with_dilation=[False, True, True])
        chans = [256, 512, 1024, 2048]   # layer1..layer4 output channels
        self.danet = DANetHead(chans[3], fpn_dim)       # on deepest feature
        self.dense = DenseBlock(fpn_dim)                # before FPN lateral
        # FPN lateral 1x1 to fpn_dim for layer1..layer3 (layer4 handled by danet/dense)
        self.lat = nn.ModuleList([nn.Conv2d(chans[i], fpn_dim, 1) for i in range(3)])
        self.smooth = nn.ModuleList([nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1) for _ in range(4)])
        self.heads = nn.ModuleList([nn.Conv2d(fpn_dim, num_classes, 1) for _ in range(4)])

    def _backbone(self, x):
        e = self.encoder
        x = e.relu(e.bn1(e.conv1(x)))
        x = e.maxpool(x)
        c1 = e.layer1(x)   # stride 4
        c2 = e.layer2(c1)  # stride 8
        c3 = e.layer3(c2)  # stride 8 (dilated)
        c4 = e.layer4(c3)  # stride 8 (dilated)
        return c1, c2, c3, c4

    def forward(self, x):
        H, W = x.shape[-2:]
        c1, c2, c3, c4 = self._backbone(x)
        p4 = self.dense(self.danet(c4))                 # top of pyramid
        p3 = self.lat[2](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat[1](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p1 = self.lat[0](c1) + F.interpolate(p2, size=c1.shape[-2:], mode="nearest")
        feats = [self.smooth[i](p) for i, p in enumerate([p1, p2, p3, p4])]
        # per-level predict -> upsample to input -> sum
        out = 0
        for i, f in enumerate(feats):
            logit = F.interpolate(self.heads[i](f), size=(H, W), mode="bilinear",
                                  align_corners=False)
            out = out + logit
        return out


def build_rfpanet(num_classes=2, encoder_weights="imagenet", in_channels=3):
    assert in_channels == 3, "RFPANet uses 3-channel ImageNet backbone"
    return RFPANet(num_classes=num_classes, pretrained=(encoder_weights == "imagenet"))
