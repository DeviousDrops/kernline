"""Shared ctypes bridge to libkernels.so (docs/DESIGN.md, D6).

    from harness import launch
    launch("gemm_v0", A, B, C, M, N, K)   # tensors -> device pointers, ints -> int, stream -> 0

Every launcher returns a cudaError_t. launch() asserts it is 0 and synchronizes, so a kernel
fault surfaces in the test that caused it.
"""
import ctypes
from functools import lru_cache
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "build" / "libkernels.so"

# Tolerances against torch, by dtype. FP32 kernels reorder sums, so they are not bit-exact.
TOL = {
    torch.float32: dict(rtol=1e-4, atol=1e-3),
    torch.float16: dict(rtol=1e-2, atol=1e-2),
}


@lru_cache(maxsize=None)
def lib() -> ctypes.CDLL:
    if not LIB_PATH.exists():
        raise FileNotFoundError(f"{LIB_PATH} not found; run `cmake --build build` first")
    return ctypes.CDLL(str(LIB_PATH))


def _arg(x):
    if isinstance(x, torch.Tensor):
        assert x.is_cuda and x.is_contiguous(), "launchers take contiguous device tensors"
        return ctypes.c_void_p(x.data_ptr())
    if isinstance(x, bool):
        return ctypes.c_int(int(x))
    if isinstance(x, int):
        return ctypes.c_int(x)
    if isinstance(x, float):
        return ctypes.c_float(x)
    raise TypeError(f"unsupported launcher argument {type(x).__name__}")


def launch(name: str, *args) -> None:
    """Call extern "C" launcher `name(args..., cudaStream_t=0)` and wait for it."""
    fn = getattr(lib(), name)
    fn.restype = ctypes.c_int
    err = fn(*(_arg(a) for a in args), ctypes.c_void_p(None))
    assert err == 0, f"{name} returned cudaError {err}"
    torch.cuda.synchronize()
