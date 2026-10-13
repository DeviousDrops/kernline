"""Unfused attention launchers vs a float64 reference of softmax(Q·Kᵀ/sqrt(d))·V."""
import ctypes
import math

import pytest
import torch

from harness import launch, lib

pytestmark = pytest.mark.gpu

VARIANTS = ["v0", "v1", "v2"]

# (heads, N, d): degenerate, sizes no GEMM tile or warp divides, GPT-2 head size 64, a d that
# disables v3's float4 path (d = 50), and N past 2048 so v1 switches softmax v2 -> v3.
SHAPES = [(1, 1, 1), (1, 7, 3), (2, 33, 8), (3, 64, 64), (12, 128, 64), (2, 257, 64),
          (1, 1000, 50), (2, 2049, 64)]


def attention(variant, Q, K, V):
    heads, N, d = Q.shape
    fn = lib().attention_workspace_bytes
    fn.restype = ctypes.c_size_t
    fn.argtypes = [ctypes.c_int] * 3
    work = torch.full((fn(heads, N, d) // 4,), float("nan"), device="cuda")
    O = torch.full_like(Q, float("nan"))
    launch(f"attention_{variant}", Q, K, V, O, work, heads, N, d)
    return O


def reference(Q, K, V):
    Q, K, V = Q.double(), K.double(), V.double()
    P = torch.softmax(Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1]), -1)
    return (P @ V).float()


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("heads,N,d", SHAPES)
def test_attention_matches_reference(variant, heads, N, d):
    torch.manual_seed(0)
    Q, K, V = (torch.randn(heads, N, d, device="cuda") for _ in range(3))
    torch.testing.assert_close(attention(variant, Q, K, V), reference(Q, K, V), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("variant", VARIANTS)
def test_attention_sharp_scores(variant):
    # Large-magnitude Q·Kᵀ (scores up to ~±300 after scaling): softmax must subtract the row max.
    torch.manual_seed(0)
    Q, K = (torch.randn(4, 300, 64, device="cuda") * 6 for _ in range(2))
    V = torch.randn(4, 300, 64, device="cuda")
    O = attention(variant, Q, K, V)
    assert torch.isfinite(O).all()
    torch.testing.assert_close(O, reference(Q, K, V), rtol=1e-3, atol=1e-4)
