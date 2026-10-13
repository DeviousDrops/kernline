"""Softmax and LayerNorm launchers vs torch, row-wise over the last dim."""
import pytest
import torch
import torch.nn.functional as F

from harness import launch

pytestmark = pytest.mark.gpu

SOFTMAX_VARIANTS = ["v0", "v1", "v2"]
LAYERNORM_VARIANTS = ["v0", "v1", "v2"]

# (rows, cols): degenerate, odd sizes no warp/vector width divides, and transformer shapes
# (attention-score rows of length 1024, hidden size 768). The tail exercises every v2 path:
# register caches of 1, 2, 4 and 16 float4 per lane (cols 4, 132, 500, 2048), rows that leave
# some lanes empty, the float4 streamed path past 2048 (2052, 16384), and scalar streaming (4099).
SHAPES = [(1, 1), (1, 37), (33, 1), (7, 33), (257, 129), (100, 300), (384, 1024), (512, 768),
          (9, 4), (9, 132), (9, 500), (17, 2048), (9, 2052), (8, 16384), (5, 4099)]

# Outputs are O(1/cols) for softmax, so an absolute tolerance of 1e-3 would accept garbage.
SOFTMAX_TOL = dict(rtol=1e-5, atol=1e-7)
LAYERNORM_TOL = dict(rtol=1e-4, atol=1e-4)


def softmax(variant, x):
    y = torch.full_like(x, float("nan"))
    launch(f"softmax_{variant}", x, y, *x.shape)
    return y


def layernorm(variant, x, gamma, beta, eps=1e-5):
    y = torch.full_like(x, float("nan"))
    launch(f"layernorm_{variant}", x, gamma, beta, y, *x.shape, eps)
    return y


@pytest.mark.parametrize("variant", SOFTMAX_VARIANTS)
@pytest.mark.parametrize("rows,cols", SHAPES)
def test_softmax_matches_torch(variant, rows, cols):
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda")
    torch.testing.assert_close(softmax(variant, x), torch.softmax(x, -1), **SOFTMAX_TOL)


def misaligned(rows, cols):
    """A contiguous rows×cols tensor whose data starts 4 bytes past a 16-byte boundary, so a
    float4-capable shape (cols % 4 == 0) must still take the scalar path."""
    buf = torch.randn(rows * cols + 1, device="cuda")
    t = buf[1:].view(rows, cols)
    assert t.data_ptr() % 16 != 0
    return t


@pytest.mark.parametrize("variant", SOFTMAX_VARIANTS)
def test_softmax_misaligned_rows(variant):
    torch.manual_seed(0)
    x = misaligned(96, 1024)
    torch.testing.assert_close(softmax(variant, x), torch.softmax(x, -1), **SOFTMAX_TOL)


@pytest.mark.parametrize("variant", LAYERNORM_VARIANTS)
def test_layernorm_misaligned_rows(variant):
    torch.manual_seed(0)
    x = misaligned(96, 768)
    gamma, beta = torch.randn(768, device="cuda"), torch.randn(768, device="cuda")
    ref = F.layer_norm(x, (768,), gamma, beta, eps=1e-5)
    torch.testing.assert_close(layernorm(variant, x, gamma, beta), ref, **LAYERNORM_TOL)


@pytest.mark.parametrize("variant", SOFTMAX_VARIANTS)
def test_softmax_is_stable_for_large_logits(variant):
    # exp(90) overflows FP32: only a max-subtracting implementation survives this.
    torch.manual_seed(0)
    x = torch.randn(64, 1000, device="cuda") * 30 + 60
    y = softmax(variant, x)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, torch.softmax(x, -1), **SOFTMAX_TOL)


@pytest.mark.parametrize("variant", LAYERNORM_VARIANTS)
@pytest.mark.parametrize("rows,cols", SHAPES)
def test_layernorm_matches_torch(variant, rows, cols):
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda")
    gamma = torch.randn(cols, device="cuda")
    beta = torch.randn(cols, device="cuda")
    ref = F.layer_norm(x, (cols,), gamma, beta, eps=1e-5)
    torch.testing.assert_close(layernorm(variant, x, gamma, beta), ref, **LAYERNORM_TOL)


@pytest.mark.parametrize("variant", [
    pytest.param(v, marks=pytest.mark.xfail(strict=True, reason=(
        "v0 sums the row sequentially in FP32: at a running sum of ~7.7e6 every add rounds to "
        "0.5, so the mean is off by ~0.01 (row std 1). A tree reduction cuts that ~12x.")))
    if v == "v0" else v
    for v in LAYERNORM_VARIANTS])
def test_layernorm_survives_large_mean(variant):
    # Rows of N(1e4, 1). A one-pass E[x^2] - E[x]^2 loses every significant digit here, and a
    # sequential FP32 sum loses the mean. v1's tree reduction and v2's Welford must pass.
    torch.manual_seed(0)
    x = torch.randn(64, 768, device="cuda") + 1e4
    gamma = torch.ones(768, device="cuda")
    beta = torch.zeros(768, device="cuda")
    ref = F.layer_norm(x, (768,), gamma, beta, eps=1e-5)
    torch.testing.assert_close(layernorm(variant, x, gamma, beta), ref, rtol=1e-3, atol=2e-3)
