import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model_prompted_sam2 import PromptedSAM2Seg


def test_forward_shape_and_grad():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = PromptedSAM2Seg(variant="small", image_size=512, device=dev).to(dev)
    x = torch.randn(2, 3, 512, 512, device=dev)
    coords = torch.tensor([[[100., 100.], [200., 200.]],
                           [[150., 150.], [0., 0.]]], device=dev)   # 第2張第2點為 pad
    labels = torch.tensor([[1, 1], [1, -1]], device=dev)            # -1 = padding
    y = m(x, coords, labels)
    assert y.shape == (2, 1, 512, 512), y.shape
    tr = [n for n, p in m.named_parameters() if p.requires_grad]
    assert any("sam_mask_decoder" in n for n in tr)
    assert not any("image_encoder" in n for n in tr)


if __name__ == "__main__":
    test_forward_shape_and_grad()
    print("OK test_prompted_sam2_forward")
