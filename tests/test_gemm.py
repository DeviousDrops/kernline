"""GEMM launchers in libkernels.so vs torch.matmul, called through ctypes on device pointers."""
import ctypes

import pytest
import torch

from harness import TOL, launch, lib

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


# (batch, M, N, K): batched over gridDim.z. The last case has odd M·K so the per-batch strides
# break 16-byte alignment and the launcher must fall back to the scalar kernel.
@pytest.mark.parametrize("batch,M,N,K", [(1, 64, 64, 64), (12, 256, 256, 64), (5, 257, 64, 129),
                                         (3, 33, 37, 7)])
def test_gemm_v3_batched(batch, M, N, K):
    torch.manual_seed(0)
    A = torch.randn(batch, M, K, device="cuda")
    B = torch.randn(batch, K, N, device="cuda")
    C = torch.full((batch, M, N), float("nan"), device="cuda")
    fn = lib().gemm_v3_batched
    fn.restype = ctypes.c_int
    err = fn(ctypes.c_void_p(A.data_ptr()), ctypes.c_void_p(B.data_ptr()),
             ctypes.c_void_p(C.data_ptr()), M, N, K, batch, ctypes.c_longlong(M * K),
             ctypes.c_longlong(K * N), ctypes.c_longlong(M * N), ctypes.c_void_p(None))
    assert err == 0
    torch.cuda.synchronize()
    torch.testing.assert_close(C, A @ B, **TOL[torch.float32])


@pytest.mark.parametrize("M,N,K", DIMS + [(136, 264, 40)])  # last: 16-byte loads, clipped tiles
def test_gemm_v4_fp16_in_fp32_acc(M, N, K):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.full((M, N), float("nan"), device="cuda")
    launch("gemm_v4", A, B, C, M, N, K)
    # A product of two halves is exact in FP32, so only the FP32 summation order differs.
    torch.testing.assert_close(C, A.float() @ B.float(), **TOL[torch.float32])
