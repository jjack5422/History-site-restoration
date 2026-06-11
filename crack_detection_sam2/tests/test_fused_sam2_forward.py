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
