"""SAM2 with learnable prompt tokens + native mask decoder for binary segmentation.

Instead of stripping SAM2 to backbone-only (model_seg.py),
uses the full pipeline: image_encoder -> prompt_encoder -> mask_decoder.

Learnable point prompts replace manual point/box annotations.
Points init randomly (num_points) or on a regular grid (grid_size, mode B:
grid-initialised but still learnable). Mask decoder outputs [B, 1, H, W] logits.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import build_sam2_model


class SAM2PromptSeg(nn.Module):
    def __init__(
        self,
        variant: str = "small",
        image_size: int = 512,
        num_points: int = 8,
        grid_size: Optional[int] = None,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = False,
        device: Optional[str] = None,
    ):
        super().__init__()
        sam2 = build_sam2_model(variant=variant, device=device, mode="train")

        # Override sizes for non-1024 input
        backbone_stride = sam2.backbone_stride  # 16
        embed_size = image_size // backbone_stride

        sam2.image_size = image_size
        sam2.sam_image_embedding_size = embed_size
        sam2.sam_prompt_encoder.input_image_size = (image_size, image_size)
        sam2.sam_prompt_encoder.image_embedding_size = (embed_size, embed_size)
        sam2.sam_prompt_encoder.mask_input_size = (4 * embed_size, 4 * embed_size)

        self.image_encoder = sam2.image_encoder
        self.sam_prompt_encoder = sam2.sam_prompt_encoder
        self.sam_mask_decoder = sam2.sam_mask_decoder
        self.image_size = image_size
        self.embed_size = embed_size
        self.hidden_dim = sam2.hidden_dim
        self.use_high_res = sam2.use_high_res_features_in_sam

        del sam2

        # Learnable prompt points: coords in [0, image_size] space.
        # grid_size set  -> init on a regular grid_size x grid_size grid
        #                   (num_points = grid_size**2) for even coverage,
        #                   then let the points move during training.
        # grid_size None -> random init with num_points (original behaviour).
        if grid_size is not None and grid_size > 0:
            init_coords = self._grid_coords(grid_size, image_size)  # [1, g*g, 2]
            num_points = grid_size * grid_size
        else:
            init_coords = torch.rand(1, num_points, 2) * image_size
        self.grid_size = grid_size
        self.num_points = num_points
        self.prompt_coords = nn.Parameter(init_coords)
        # Labels: 1=positive (not learnable -- discrete)
        self.register_buffer(
            "prompt_labels",
            torch.ones(1, num_points, dtype=torch.int32),
        )

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        if freeze_prompt_encoder:
            for p in self.sam_prompt_encoder.parameters():
                p.requires_grad = False

    @staticmethod
    def _grid_coords(grid_size: int, image_size: int) -> torch.Tensor:
        """Regular grid_size x grid_size points at cell centres, (x, y) order.

        Returns [1, grid_size**2, 2] in [0, image_size] coordinate space.
        """
        centers = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size * image_size
        yy, xx = torch.meshgrid(centers, centers, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # [g*g, 2]
        return coords.unsqueeze(0)  # [1, g*g, 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape

        # 1. Image encoding
        backbone_out = self.image_encoder(x)
        fpn = backbone_out["backbone_fpn"]

        # 2. Prepare features for mask decoder
        if self.use_high_res:
            high_res_features = [
                self.sam_mask_decoder.conv_s0(fpn[0]),
                self.sam_mask_decoder.conv_s1(fpn[1]),
            ]
        else:
            high_res_features = None

        backbone_features = fpn[-1]  # [B, 256, embed_size, embed_size]

        # 3. Prompt encoding (learnable points)
        coords = self.prompt_coords.expand(B, -1, -1)
        labels = self.prompt_labels.expand(B, -1)

        sparse_emb, dense_emb = self.sam_prompt_encoder(
            points=(coords, labels),
            boxes=None,
            masks=None,
        )

        # 4. Mask decoding
        low_res_masks, ious, _, _ = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        # low_res_masks: [B, 1, embed_size*4, embed_size*4]

        # 5. Upsample to input resolution
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
        prompt_params, decoder_params, encoder_params = [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "prompt_coords" in n:
                prompt_params.append(p)
            elif "sam_mask_decoder" in n:
                decoder_params.append(p)
            elif "sam_prompt_encoder" in n:
                decoder_params.append(p)
            else:
                encoder_params.append(p)

        groups = [
            {"params": prompt_params, "lr": base_lr, "name": "prompt"},
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
    # grid-initialised (mode B): grid_size=4 -> 16 learnable points on a 4x4 grid
    model = SAM2PromptSeg(variant="small", image_size=512, grid_size=4, device=device).to(device)
    total, trainable = count_params(model)
    print(f"total={total/1e6:.1f}M  trainable={trainable/1e6:.2f}M")
    print(f"grid_size={model.grid_size} num_points={model.num_points}")

    x = torch.randn(2, 3, 512, 512, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"output shape={tuple(y.shape)} dtype={y.dtype}")
    print(f"prompt_coords shape={tuple(model.prompt_coords.shape)} "
          f"requires_grad={model.prompt_coords.requires_grad}")
    print(f"prompt_coords range: {model.prompt_coords.min().item():.1f} ~ {model.prompt_coords.max().item():.1f}")
