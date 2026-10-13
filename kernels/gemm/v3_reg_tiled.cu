// GEMM v3: register tiling. A block of 256 threads computes a 128×128 tile of C; each thread owns
// an 8×8 sub-tile held in 64 registers. Per step of BK = 8 along K the block loads a 128×8 tile of
// A and an 8×128 tile of B into shared memory with float4 loads, then every thread does, for each
// k, 4 shared float4 loads (8 A values, 8 B values) and 64 FMAs: 16 FMAs per shared load, against
// 0.5 in v2.
//
// Shared-memory layout, chosen so no access has avoidable bank conflicts:
//  - As is transposed (As[k][row]) so a thread's 8 rows are two contiguous float4s, and padded to
//    a row stride of 132 words: while filling, a warp writes rows r..r+15 at k and k+4, and
//    4·132 mod 32 = 16 puts those two groups in disjoint banks.
//  - A thread's 8 columns are two float4s at 4·tx and 64 + 4·tx, not 8 contiguous floats: the 16
//    distinct tx in a warp then read 64 consecutive words of Bs instead of 16 float4s at stride 8,
//    which would pile onto half the banks.
//
// Occupancy is register-limited: 117 registers per thread fits 2 blocks per SM (65536 / 256 / 2 =
// 128). Rewrites that push past 128 registers halve occupancy and cost about 60% in runtime.
//
// float4 global loads need K and N divisible by 4 and 16-byte-aligned buffers; the launcher picks
// the scalar-load kernel otherwise. Edge tiles are zero-padded.
#include "kernels/gemm/gemm.h"

namespace {

const int BM = kGemmV3Tile, BN = kGemmV3Tile, BK = 8, TM = 8;
const int AS = BM + 4;  // padded row stride of the transposed A tile

// Four consecutive floats starting at p[idx], of which the first `avail` exist (rest are zero).
// With kVec, avail is always <= 0 or >= 4 and p + idx is 16-byte aligned.
template <bool kVec>
__device__ __forceinline__ float4 load4(const float* p, int idx, int avail) {
    if (kVec) return avail > 0 ? *reinterpret_cast<const float4*>(p + idx) : make_float4(0, 0, 0, 0);
    return make_float4(avail > 0 ? p[idx] : 0.0f, avail > 1 ? p[idx + 1] : 0.0f,
                       avail > 2 ? p[idx + 2] : 0.0f, avail > 3 ? p[idx + 3] : 0.0f);
}

template <bool kVec>
__device__ __forceinline__ void gemm_v3_body(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
    __shared__ __align__(16) float As[BK][AS];
    __shared__ __align__(16) float Bs[BK][BN];
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    // Tile-fill assignment: one float4 of A and one of B per thread.
    const int a_row = tid / 2, a_k = (tid % 2) * 4;
    const int b_k = tid / 32, b_col = (tid % 32) * 4;

    float acc[TM][8] = {};
    for (int k0 = 0; k0 < K; k0 += BK) {
        const float4 a = load4<kVec>(A, (row0 + a_row) * K + k0 + a_k,
                                     row0 + a_row < M ? K - k0 - a_k : 0);
        As[a_k][a_row] = a.x;
        As[a_k + 1][a_row] = a.y;
        As[a_k + 2][a_row] = a.z;
        As[a_k + 3][a_row] = a.w;
        *reinterpret_cast<float4*>(&Bs[b_k][b_col]) =
            load4<kVec>(B, (k0 + b_k) * N + col0 + b_col, k0 + b_k < K ? N - col0 - b_col : 0);
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            const float4 a0 = *reinterpret_cast<const float4*>(&As[k][ty * TM]);
            const float4 a1 = *reinterpret_cast<const float4*>(&As[k][ty * TM + 4]);
            const float4 b0 = *reinterpret_cast<const float4*>(&Bs[k][tx * 4]);
            const float4 b1 = *reinterpret_cast<const float4*>(&Bs[k][BN / 2 + tx * 4]);
            const float av[TM] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] += av[i] * bv[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = row0 + ty * TM + i;
        if (row >= M) break;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = col0 + h * (BN / 2) + tx * 4;
            const float* v = &acc[i][4 * h];
            if (kVec) {
                if (col < N)
                    *reinterpret_cast<float4*>(&C[row * N + col]) = make_float4(v[0], v[1], v[2], v[3]);
            } else {
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    if (col + e < N) C[row * N + col + e] = v[e];
            }
        }
    }
}

}  // namespace

// blockIdx.z selects one matrix of a batch, `stride*` floats apart (0 for a single GEMM).
__global__ void gemm_v3_kernel(const float* A, const float* B, float* C, int M, int N, int K,
                               long long strideA, long long strideB, long long strideC) {
    const long long z = blockIdx.z;
    gemm_v3_body<true>(A + z * strideA, B + z * strideB, C + z * strideC, M, N, K);
}

__global__ void gemm_v3_scalar_kernel(const float* A, const float* B, float* C, int M, int N,
                                      int K, long long strideA, long long strideB,
                                      long long strideC) {
    const long long z = blockIdx.z;
    gemm_v3_body<false>(A + z * strideA, B + z * strideB, C + z * strideC, M, N, K);
}
