import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model_prompted_sam2 import PromptedSAM2Seg


def test_encode_decode_prev_mask_and_backcompat():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = PromptedSAM2Seg(variant="small", image_size=512, device=dev).to(dev)
    x = torch.randn(2, 3, 512, 512, device=dev)
    coords = torch.tensor([[[100., 100.]], [[150., 150.]]], device=dev)   # [B,1,2] (x,y)
    labels = torch.tensor([[1], [1]], dtype=torch.int32, device=dev)      # [B,1]

    enc = m.encode_image(x)
    assert enc["hw"] == (512, 512)
    masks, low = m.decode(enc, coords, labels, prev_mask=None)
    assert masks.shape == (2, 1, 512, 512), masks.shape
    assert low.shape == (2, 1, 128, 128), low.shape   # 4 * (512//16)

    # second click + prev-mask refinement; shapes stay stable
    coords2 = torch.cat([coords, coords + 10], dim=1)
    labels2 = torch.cat([labels, torch.zeros_like(labels)], dim=1)        # add a negative click
    masks2, low2 = m.decode(enc, coords2, labels2, prev_mask=low)
    assert masks2.shape == (2, 1, 512, 512)

    # gradients must flow through the refinement (prev_mask) path into the mask decoder
    low2.sum().backward()
    assert any(p.grad is not None for p in m.sam_mask_decoder.parameters())

    # backward-compatible forward (existing test_prompted_sam2_forward relies on this)
    y = m(x, coords, labels)
    assert y.shape == (2, 1, 512, 512)
    print("OK test_click_forward")


if __name__ == "__main__":
    test_encode_decode_prev_mask_and_backcompat()
