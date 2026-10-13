"""GEMM launchers in libkernels.so vs torch.matmul, called through ctypes on device pointers."""
import ctypes
from pathlib import Path

import pytest
import torch

LIB = ctypes.CDLL(str(Path(__file__).resolve().parents[1] / "build" / "libkernels.so"))

VARIANTS = ["v0", "cublas"]
# (M, N, K): C is M×N. Includes degenerate, non-square, and sizes no tile divides.
DIMS = [(1, 1, 1), (1, 37, 5), (33, 1, 7), (16, 16, 16), (257, 129, 65), (100, 300, 200),
        (1024, 1024, 1024)]


def gemm(variant, A, B):
    M, K = A.shape
    N = B.shape[1]
    C = torch.full((M, N), float("nan"), device="cuda")
    fn = getattr(LIB, f"gemm_{variant}")
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    err = fn(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K, None)
    assert err == 0, f"gemm_{variant} returned cudaError {err}"
    torch.cuda.synchronize()
    return C


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("M,N,K", DIMS)
def test_gemm_matches_torch(variant, M, N, K):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    torch.testing.assert_close(gemm(variant, A, B), A @ B, rtol=1e-4, atol=1e-3)
