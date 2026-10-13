#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                        \
    do {                                                                     \
        cudaError_t err_ = (cudaError_t)(x);                                 \
        if (err_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,          \
                         cudaGetErrorString(err_));                          \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)
