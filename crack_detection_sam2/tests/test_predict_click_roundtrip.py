import os, sys, tempfile
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model_prompted_sam2 import PromptedSAM2Seg
from predict_click import load_model_from_ckpt, predict_click


def test_roundtrip():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = PromptedSAM2Seg(variant="small", image_size=512, device=dev).to(dev)
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
    torch.save({"model": m.state_dict(), "args": {"variant": "small", "image_size": 512}}, path)

    model, payload = load_model_from_ckpt(path, dev)
    img = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    mask, low = predict_click(model, img, pos_points=[(256, 256)], neg_points=[],
                              prev_mask=None, device=dev)
    assert mask.shape == (512, 512) and mask.dtype == bool, (mask.shape, mask.dtype)
    assert tuple(low.shape[-2:]) == (128, 128), low.shape

    # refinement click reusing prev low-res logits
    mask2, low2 = predict_click(model, img, pos_points=[(256, 256)], neg_points=[(10, 10)],
                                prev_mask=low, device=dev)
    assert mask2.shape == (512, 512)
    os.remove(path)
    print("OK test_predict_click_roundtrip")


if __name__ == "__main__":
    test_roundtrip()
