// Small kernels used by unfused attention: K transpose (with the 1/sqrt(d) scale folded in for
// v1) and v0's elementwise scale pass over the scores.
#include "kernels/attention/attention.h"

__global__ void transpose_scale_kernel(const float* in, float* out, int rows, int cols,
                                       float scale) {
    __shared__ float tile[kTransposeTile][kTransposeTile + 1];
    const size_t head = (size_t)blockIdx.z * rows * cols;
    in += head;
    out += head;
    const int c0 = blockIdx.x * kTransposeTile, r0 = blockIdx.y * kTransposeTile;
    for (int i = threadIdx.y; i < kTransposeTile; i += blockDim.y) {
        const int r = r0 + i, c = c0 + threadIdx.x;
        if (r < rows && c < cols) tile[i][threadIdx.x] = in[(size_t)r * cols + c];
    }
    __syncthreads();
    for (int i = threadIdx.y; i < kTransposeTile; i += blockDim.y) {
        const int c = c0 + i, r = r0 + threadIdx.x;  // out row = in column
        if (c < cols && r < rows) out[(size_t)c * rows + r] = tile[threadIdx.x][i] * scale;
    }
}

__global__ void scale_kernel(float* x, long long n, float scale) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x)
        x[i] *= scale;
}
