import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dadnet_model import DADNet, NeighborhoodAttention, BiAxialBlock


def test_neighborhood_attention_preserves_shape():
    x = torch.randn(2, 16, 28, 28)
    out = NeighborhoodAttention(16, k=7)(x)
    assert out.shape == x.shape


def test_biaxial_block_preserves_shape():
    x = torch.randn(2, 32, 14, 14)
    out = BiAxialBlock(32, dilation=7)(x)
    assert out.shape == x.shape


def test_dadnet_forward_shape():
    m = DADNet(num_classes=2, k=7, dilation=7, pretrained=False).eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, 224, 224))
    assert y.shape == (2, 2, 224, 224), y.shape
