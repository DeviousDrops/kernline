// GEMM v4: tensor cores through WMMA, FP16 inputs with FP32 accumulation (D10). A block of 8 warps
// computes a 128×128 tile of C; each warp owns a 32×64 sub-tile as 2×4 accumulator fragments
// (16×16 each). Per step of BK = 64 along K the block copies a 128×64 tile of A and a 64×128 tile
// of B into shared memory in 16-byte chunks (8 halves), then each warp runs four k-slices of
// 2 A-fragment loads, 4 B-fragment loads and 8 mma_sync. BK = 64 beat 32 and 16 at most shapes:
// the kernel waits on the tile copy (long_scoreboard), and a deeper tile means fewer barriers.
//
// The 16-byte global loads need K and N divisible by 8, 16-byte-aligned A and B, and (for direct
// fragment stores) 32-byte-aligned C; the launcher picks the scalar-load kernel otherwise. Edge
// tiles are zero-padded on load. Fragments that cross the edge of C, and every fragment in the
// scalar kernel, are stored through shared memory and copied out with bounds checks.
#include <mma.h>
#include "kernels/gemm/gemm.h"

using namespace nvcuda;

namespace {

const int BM = kGemmV4Tile, BN = kGemmV4Tile, BK = 64;
const int WM = 32, WN = 64;          // tile of C per warp: 2×4 fragments
const int AS = BK + 8, BS = BN + 8;  // shared row strides in halves, padded by one 16-byte chunk

// Eight consecutive halves starting at p[idx] as one 16-byte value; the first `avail` exist (rest
// are zero). With kVec, avail is always <= 0 or >= 8 and p + idx is 16-byte aligned.
template <bool kVec>
__device__ __forceinline__ int4 load8(const unsigned short* p, int idx, int avail) {
    if (kVec) {
        return avail > 0 ? *reinterpret_cast<const int4*>(p + idx) : make_int4(0, 0, 0, 0);
    } else {
        unsigned h[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) h[e] = e < avail ? p[idx + e] : 0u;
        return make_int4(h[0] | h[1] << 16, h[2] | h[3] << 16, h[4] | h[5] << 16, h[6] | h[7] << 16);
    }
}

template <bool kVec>
__device__ __forceinline__ void gemm_v4_body(const __half* Ah, const __half* Bh, float* C, int M,
                                             int N, int K) {
    // Raw 16-bit storage: __half has constructors, which shared arrays may not.
    __shared__ __align__(32) unsigned short As[BM * AS];
    __shared__ __align__(32) unsigned short Bs[BK * BS];
    const unsigned short* A = reinterpret_cast<const unsigned short*>(Ah);
    const unsigned short* B = reinterpret_cast<const unsigned short*>(Bh);
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int wr = (warp / 2) * WM, wc = (warp % 2) * WN;  // this warp's origin inside the tile
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int l = tid; l < BM * BK / 8; l += kGemmV4Threads) {
            const int r = l / (BK / 8), c = (l % (BK / 8)) * 8;
            *reinterpret_cast<int4*>(&As[r * AS + c]) =
                load8<kVec>(A, (row0 + r) * K + k0 + c, row0 + r < M ? K - k0 - c : 0);
        }
#pragma unroll
        for (int l = tid; l < BK * BN / 8; l += kGemmV4Threads) {
            const int kr = l / (BN / 8), c = (l % (BN / 8)) * 8;
            *reinterpret_cast<int4*>(&Bs[kr * BS + c]) =
                load8<kVec>(B, (k0 + kr) * N + col0 + c, k0 + kr < K ? N - col0 - c : 0);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
#pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(
                    a[i], reinterpret_cast<const half*>(&As[(wr + i * 16) * AS + kk]), AS);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                wmma::load_matrix_sync(
                    b[j], reinterpret_cast<const half*>(&Bs[kk * BS + wc + j * 16]), BS);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    // The A tile is free now: each warp uses 1 KiB of it to stage a fragment that needs clipping.
    float* stage = reinterpret_cast<float*>(As) + warp * 256;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = row0 + wr + i * 16, c = col0 + wc + j * 16;
            if (r >= M || c >= N) continue;
            if (kVec && r + 16 <= M && c + 16 <= N) {
                wmma::store_matrix_sync(C + r * N + c, acc[i][j], N, wmma::mem_row_major);
                continue;
            }
            wmma::store_matrix_sync(stage, acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32)
                if (r + e / 16 < M && c + e % 16 < N) C[(r + e / 16) * N + c + e % 16] = stage[e];
            __syncwarp();
        }
}

}  // namespace

__global__ void gemm_v4_kernel(const __half* A, const __half* B, float* C, int M, int N, int K) {
    gemm_v4_body<true>(A, B, C, M, N, K);
}

__global__ void gemm_v4_scalar_kernel(const __half* A, const __half* B, float* C, int M, int N,
                                      int K) {
    gemm_v4_body<false>(A, B, C, M, N, K);
}
