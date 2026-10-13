import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import LIB_PATH  # noqa: E402


def pytest_collection_modifyitems(config, items):
    reason = None
    if not torch.cuda.is_available():
        reason = "no CUDA device visible to torch"
    elif not LIB_PATH.exists():
        reason = f"{LIB_PATH.name} not built; run `cmake --build build`"
    if reason:
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device and build/libkernels.so")
