// Day-1 go/no-go: does the toolchain build, does the GPU run it, can ncu read counters?
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                        \
    do {                                                                     \
        cudaError_t err_ = (x);                                              \
        if (err_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,          \
                         cudaGetErrorString(err_));                          \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

__global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 24;
    const size_t bytes = n * sizeof(float);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);

    float *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));
    for (int i = 0; i < n; ++i) { a[i] = 1.0f; b[i] = 2.0f; }

    vadd<<<(n + 255) / 256, 256>>>(a, b, c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < n; ++i) {
        if (c[i] != 3.0f) { std::printf("FAIL at %d\n", i); return 1; }
    }
    std::printf("PASS\n");
    return 0;
}
