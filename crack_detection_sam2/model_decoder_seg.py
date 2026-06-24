"""model_decoder_seg.py — decoder-only SAM2:捨棄 prompt_encoder,只借其 pe_layer
(位置編碼,無可訓練參數),沿用 SAM2 mask_decoder 出 dense mask。

與 model_prompt_seg.py 的差別:沒有任何 prompt 點。mask_decoder 靠自帶的
iou_token + mask_tokens 當 query,對 image_encoder 特徵做 two-way cross-attention。
等於把 SAM2 預訓練的 transformer mask decoder 當成 segmentation head。

prompt 三件輸入改由本模組自行供給:
  image_pe   ← 借 prompt_encoder.pe_layer((embed,embed))  (deterministic, 凍結等價)
  sparse     ← 空 tensor [B,0,256](變體 A) 或自帶 learnable query tokens(變體 B)
  dense      ← 0 / learnable 全域 param / image-conditioned conv head(變體 C)

輸出 [B,1,H,W] binary logits(與 model_prompt_seg / model_prompted_sam2 同口徑)。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import build_sam2_model


class SAM2DecoderSeg(nn.Module):
    def __init__(
        self,
        variant: str = "small",
        image_size: int = 512,
        num_queries: int = 0,          # 變體 B:>0 加 learnable query tokens
        dense_mode: str = "learnable", # "zero" | "learnable" | "image"(變體 C)
        freeze_image_encoder: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()
        if dense_mode not in ("zero", "learnable", "image"):
            raise ValueError("dense_mode 必須是 zero / learnable / image")
        sam2 = build_sam2_model(variant=variant, device=device, mode="train")

        embed = image_size // sam2.backbone_stride  # 16 -> 32 @ 512

        self.image_encoder = sam2.image_encoder
        self.sam_mask_decoder = sam2.sam_mask_decoder
        # 只借位置編碼(PositionEmbeddingRandom,只有 buffer,無 nn.Parameter)
        self.pe_layer = sam2.sam_prompt_encoder.pe_layer
        self.image_size = image_size
        self.embed_size = embed
        self.use_high_res = sam2.use_high_res_features_in_sam

        del sam2

        # 變體 B:自帶 query tokens 當 sparse_prompt_embeddings;0 則退化成變體 A
        self.query_tokens = (
            nn.Parameter(torch.randn(1, num_queries, 256) * 0.02)
            if num_queries > 0 else None
        )

        # dense_prompt_embeddings 來源
        self.dense_mode = dense_mode
        if dense_mode == "learnable":
            self.dense_embed = nn.Parameter(torch.zeros(1, 256, embed, embed))
        elif dense_mode == "image":
            self.dense_head = nn.Conv2d(256, 256, kernel_size=1)

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        # pe_layer 無可訓練參數,不需處理

    def _image_pe(self) -> torch.Tensor:
        # [256, embed, embed] -> [1, 256, embed, embed]
        return self.pe_layer((self.embed_size, self.embed_size)).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape

        backbone_out = self.image_encoder(x)
        fpn = backbone_out["backbone_fpn"]

        if self.use_high_res:
            high_res_features = [
                self.sam_mask_decoder.conv_s0(fpn[0]),
                self.sam_mask_decoder.conv_s1(fpn[1]),
            ]
        else:
            high_res_features = None

        backbone_features = fpn[-1]  # [B, 256, embed, embed]

        # sparse:空(變體 A) 或 learnable query tokens(變體 B)
        if self.query_tokens is not None:
            sparse_emb = self.query_tokens.expand(B, -1, -1)
        else:
            sparse_emb = backbone_features.new_zeros(B, 0, 256)

        # dense:0 / learnable 全域 / image-conditioned
        if self.dense_mode == "learnable":
            dense_emb = self.dense_embed.expand(B, -1, -1, -1)
        elif self.dense_mode == "image":
            dense_emb = self.dense_head(backbone_features)
        else:  # zero
            dense_emb = backbone_features.new_zeros(B, 256, self.embed_size, self.embed_size)

        low_res_masks, ious, _, _ = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self._image_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        # low_res_masks: [B, 1, embed*4, embed*4]

        masks = F.interpolate(
            low_res_masks.float(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        return masks  # [B, 1, H, W] binary logits

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_groups(self, base_lr: float, decoder_lr_mult: float = 1.0):
        query_params, decoder_params, dense_params, encoder_params = [], [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "query_tokens" in n:
                query_params.append(p)
            elif "dense_embed" in n or "dense_head" in n:
                dense_params.append(p)
            elif "sam_mask_decoder" in n:
                decoder_params.append(p)
            else:
                encoder_params.append(p)

        groups = [
            {"params": query_params, "lr": base_lr, "name": "query"},
            {"params": dense_params, "lr": base_lr, "name": "dense"},
            {"params": decoder_params, "lr": base_lr * decoder_lr_mult, "name": "decoder"},
        ]
        if encoder_params:
            groups.append({"params": encoder_params, "lr": base_lr * 0.01, "name": "encoder"})
        return [g for g in groups if g["params"]]


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for nq, dm in [(0, "zero"), (0, "learnable"), (4, "learnable"), (0, "image")]:
        model = SAM2DecoderSeg(
            variant="small", image_size=512,
            num_queries=nq, dense_mode=dm, device=device,
        ).to(device)
        total, trainable = count_params(model)
        x = torch.randn(2, 3, 512, 512, device=device)
        with torch.no_grad():
            y = model(x)
        print(f"num_queries={nq} dense={dm:9s} | "
              f"total={total/1e6:.1f}M trainable={trainable/1e6:.2f}M | "
              f"out={tuple(y.shape)}")
        del model
