"""model_prompted_sam2.py — SAM2(image_encoder 凍結)+ prompt_encoder + mask_decoder,
prompt 點為 forward 的外部輸入(非 learnable)。輸出 [B,1,H,W] binary logits。"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import build_sam2_model


class PromptedSAM2Seg(nn.Module):
    def __init__(self, variant="small", image_size=512,
                 freeze_image_encoder=True, freeze_prompt_encoder=False,
                 device: Optional[str] = None):
        super().__init__()
        sam2 = build_sam2_model(variant=variant, device=device, mode="train")
        embed = image_size // sam2.backbone_stride
        sam2.image_size = image_size
        sam2.sam_image_embedding_size = embed
        sam2.sam_prompt_encoder.input_image_size = (image_size, image_size)
        sam2.sam_prompt_encoder.image_embedding_size = (embed, embed)
        sam2.sam_prompt_encoder.mask_input_size = (4 * embed, 4 * embed)
        self.image_encoder = sam2.image_encoder
        self.sam_prompt_encoder = sam2.sam_prompt_encoder
        self.sam_mask_decoder = sam2.sam_mask_decoder
        self.use_high_res = sam2.use_high_res_features_in_sam
        del sam2
        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        if freeze_prompt_encoder:
            for p in self.sam_prompt_encoder.parameters():
                p.requires_grad = False

    def forward(self, x, point_coords, point_labels):
        B, _, H, W = x.shape
        bb = self.image_encoder(x)
        fpn = bb["backbone_fpn"]
        high_res = ([self.sam_mask_decoder.conv_s0(fpn[0]),
                     self.sam_mask_decoder.conv_s1(fpn[1])] if self.use_high_res else None)
        feat = fpn[-1]
        sparse, dense = self.sam_prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=None)
        low, _, _, _ = self.sam_mask_decoder(
            image_embeddings=feat,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False, repeat_image=False, high_res_features=high_res)
        return F.interpolate(low.float(), size=(H, W), mode="bilinear", align_corners=False)

    def param_groups(self, base_lr, encoder_lr_mult=0.01):
        dec, enc = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (dec if ("sam_mask_decoder" in n or "sam_prompt_encoder" in n) else enc).append(p)
        groups = [{"params": dec, "lr": base_lr, "name": "decoder"}]
        if enc:
            groups.append({"params": enc, "lr": base_lr * encoder_lr_mult, "name": "encoder"})
        return [g for g in groups if g["params"]]
