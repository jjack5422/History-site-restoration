import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model_fused_sam2 import FeatFusionAdapter


def test_adapter_zero_init_is_identity():
    a = FeatFusionAdapter(sam_dim=256, dino_dim=384).eval()
    feat = torch.randn(2, 256, 32, 32)
    dino = torch.randn(2, 384, 37, 37)
    out = a(feat, dino)
    assert out.shape == feat.shape
    # zero-init 融合分支 -> delta=0 -> 輸出等於 feat
    assert torch.allclose(out, feat, atol=1e-6)


def test_fused_forward_shape_and_init_identity():
    from model_fused_sam2 import FusedPromptedSAM2Seg
    model = FusedPromptedSAM2Seg(variant="small", image_size=512, device="cpu").eval()
    img = torch.randn(1, 3, 512, 512)
    dino = torch.randn(1, 384, 37, 37)
    coords = torch.zeros(1, 1, 2)
    labels = -torch.ones(1, 1, dtype=torch.long)
    out = model(img, dino, coords, labels, None)
    assert out.shape == (1, 1, 512, 512)
    # zero-init fusion: feat' == feat
    enc = model.encode_image(img)
    assert torch.allclose(model.fusion(enc["feat"], dino), enc["feat"], atol=1e-6)
