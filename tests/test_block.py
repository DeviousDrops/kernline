"""Transformer block (src/api.cu, D4) and its elementwise kernels vs a float64 torch reference."""
import ctypes
import math

import pytest
import torch
import torch.nn.functional as F

from harness import launch, lib

pytestmark = pytest.mark.gpu

# (B, S, D, H): tiny, sizes no GEMM tile divides, and GPT-2 small's D = 768, H = 12.
SHAPES = [(1, 1, 8, 2), (2, 33, 96, 3), (1, 129, 256, 4), (2, 64, 768, 12)]


def workspace(B, S, D, H):
    fn = lib().transformer_block_workspace_bytes
    fn.restype = ctypes.c_size_t
    fn.argtypes = [ctypes.c_int] * 4
    return torch.full((fn(B, S, D, H) // 4,), float("nan"), device="cuda")


def weights(D):
    g = lambda *shape: torch.randn(*shape, device="cuda") * 0.02  # GPT-2 init scale
    return dict(ln1_g=1 + g(D), ln1_b=g(D), Wqkv=g(D, 3 * D), Wo=g(D, D), ln2_g=1 + g(D),
                ln2_b=g(D), W1=g(D, 4 * D), W2=g(4 * D, D))


def reference(x, w, B, S, D, H):
    x, w = x.double(), {k: v.double() for k, v in w.items()}
    d = D // H
    h = F.layer_norm(x, (D,), w["ln1_g"], w["ln1_b"], eps=1e-5)
    q, k, v = (h @ w["Wqkv"]).view(B, S, 3, H, d).permute(2, 0, 3, 1, 4)
    o = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d), -1) @ v
    x = x + o.permute(0, 2, 1, 3).reshape(B * S, D) @ w["Wo"]
    h = F.layer_norm(x, (D,), w["ln2_g"], w["ln2_b"], eps=1e-5)
    return (x + F.gelu(h @ w["W1"], approximate="tanh") @ w["W2"]).float()


@pytest.mark.parametrize("B,S,D,H", SHAPES)
def test_block_matches_reference(B, S, D, H):
    torch.manual_seed(0)
    x = torch.randn(B * S, D, device="cuda")
    w = weights(D)
    ref = reference(x, w, B, S, D, H)
    launch("transformer_block", x, w["ln1_g"], w["ln1_b"], w["Wqkv"], w["Wo"], w["ln2_g"],
           w["ln2_b"], w["W1"], w["W2"], workspace(B, S, D, H), B, S, D, H)
    torch.testing.assert_close(x, ref, rtol=1e-4, atol=1e-4)


def test_steps_compose_to_block():
    # Running the steps one by one is the same computation as transformer_block.
    B, S, D, H = 2, 40, 64, 4
    torch.manual_seed(0)
    x0 = torch.randn(B * S, D, device="cuda")
    w = weights(D)
    args = [w[k] for k in ("ln1_g", "ln1_b", "Wqkv", "Wo", "ln2_g", "ln2_b", "W1", "W2")]
    x_steps, x_block = x0.clone(), x0.clone()
    work = workspace(B, S, D, H)
    n = lib().transformer_block_num_steps()
    for step in range(n):
        launch("transformer_block_step", step, x_steps, *args, work, B, S, D, H)
    launch("transformer_block", x_block, *args, workspace(B, S, D, H), B, S, D, H)
    assert n == 12
    torch.testing.assert_close(x_steps, x_block, rtol=0, atol=0)
