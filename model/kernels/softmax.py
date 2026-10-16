"""Softmax v0-v3 along the last dim, dims rowsxcols. The model is shared with the other row-wise
kernel: see model/kernels/rowwise.py."""
from model.kernels import rowwise

VARIANTS = rowwise.VARIANTS
KERNEL_NAMES = rowwise.kernel_names("softmax")


def launches(dims, variant, gpu, params):
    rows, cols = dims
    return [rowwise.launch("softmax", rows, cols, variant, gpu, params)]
