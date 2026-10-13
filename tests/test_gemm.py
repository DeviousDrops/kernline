"""GEMM launchers in libkernels.so vs torch.matmul, called through ctypes on device pointers."""
import pytest
import torch

from harness import TOL, launch

pytestmark = pytest.mark.gpu

VARIANTS = ["v0", "v1", "v2", "v3", "cublas"]
# (M, N, K): C is M×N. Includes degenerate, non-square, and sizes no tile divides.
DIMS = [(1, 1, 1), (1, 37, 5), (33, 1, 7), (16, 16, 16), (257, 129, 65), (100, 300, 200),
        (1024, 1024, 1024)]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("M,N,K", DIMS)
def test_gemm_matches_torch(variant, M, N, K):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.full((M, N), float("nan"), device="cuda")
    launch(f"gemm_{variant}", A, B, C, M, N, K)
    torch.testing.assert_close(C, A @ B, **TOL[torch.float32])
