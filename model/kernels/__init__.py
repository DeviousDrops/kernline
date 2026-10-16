"""Kernel descriptors (docs/DESIGN.md, "Model kernel descriptor").

Each module model/kernels/<kernel>.py describes every variant of one kernel. The core function is

    launches(dims, variant, gpu, params) -> list[Launch]

and the frozen interface derives from it (see `frozen` below): flops(dims), bytes(dims, variant),
launch_cfg(dims, variant), resources(variant). `dims` is the results CSV's dims string split on
"x", as ints. A variant without a descriptor (library baselines: cublas, torch) is not modeled.
"""
import importlib
from types import SimpleNamespace

from model import params as _params
from model import resources as _resources
from model import spec as _spec

KERNELS = ("gemm", "softmax", "layernorm", "attention", "elementwise", "block")


def get(kernel: str):
    if kernel not in KERNELS:
        raise KeyError(f"no descriptor for kernel {kernel!r}")
    return importlib.import_module(f"model.kernels.{kernel}")


def parse_dims(dims) -> tuple[int, ...]:
    return tuple(int(x) for x in dims.split("x")) if isinstance(dims, str) else tuple(dims)


def frozen(module, default_gpu: str = "rtx4070_laptop") -> SimpleNamespace:
    """The DESIGN.md interface over module.launches, at the default GPU and the tuned params."""

    def ls(dims, variant):
        return module.launches(parse_dims(dims), variant, _spec.load(default_gpu), _params.load())

    return SimpleNamespace(
        flops=lambda dims, variant=module.VARIANTS[-1]: sum(L.flops for L in ls(dims, variant)),
        bytes=lambda dims, variant: sum(L.dram for L in ls(dims, variant)),
        launch_cfg=lambda dims, variant: [(L.kernel, L.blocks, L.threads, L.dyn_smem)
                                          for L in ls(dims, variant)],
        resources=lambda variant: {k: _resources.get(k) for k in module.KERNEL_NAMES[variant]},
    )
