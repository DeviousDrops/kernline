// GEMM generators: kernels/gemm/v0_naive.cu, v2_smem_tiled.cu, v3_reg_tiled.cu (float4 path),
// global loads and stores in program order. Operands: 0 = A, 1 = B, 2 = C.
#include <stdexcept>
#include "sim/generators/generators.h"

namespace sim {
namespace {

// v0: block (16, 16), row = blockIdx.x*16 + threadIdx.x, col = blockIdx.y*16 + threadIdx.y.
// Warp w holds threadIdx.y = 2w and 2w+1. Per k: load A[row][k], load B[k][col]; then store C.
class GemmV0 : public Generator {
  public:
    GemmV0(uint64_t A, uint64_t B, uint64_t C, int M, int N, int K)
        : A_(A), B_(B), C_(C), M_(M), N_(N), K_(K) {
        name = "gemm_v0_kernel";
        threads = 256;
        smem_per_block = 1024;
        gx_ = (M + 15) / 16;
        blocks = gx_ * ((N + 15) / 16);
        ops_per_warp = 2LL * K + 1;
    }
    void op(int64_t block, int warp, int64_t i, WarpOp& o) const override {
        const int64_t bx = block % gx_, by = block / gx_, k = i / 2;
        const bool st = i == 2 * K_;
        o.mask = 0;
        o.bytes = 4;
        o.store = st;
        o.operand = st ? 2 : int(i % 2);
        for (int l = 0; l < 32; ++l) {
            const int64_t row = bx * 16 + l % 16, col = by * 16 + 2 * warp + l / 16;
            if (row >= M_ || col >= N_) continue;
            o.mask |= 1u << l;
            o.addr[l] = st ? C_ + 4 * uint64_t(row * N_ + col)
                        : i % 2 == 0 ? A_ + 4 * uint64_t(row * K_ + k)
                                     : B_ + 4 * uint64_t(k * N_ + col);
        }
    }

  private:
    uint64_t A_, B_, C_;
    int64_t M_, N_, K_, gx_;
};

// v2: block (T, T), tile T x T. Per k-step: load A[row][k0 + tx], load B[k0 + ty][col]; then
// store C. Edge loads are predicated off (zero-padded tiles).
class GemmV2 : public Generator {
  public:
    GemmV2(uint64_t A, uint64_t B, uint64_t C, int M, int N, int K, int T)
        : A_(A), B_(B), C_(C), M_(M), N_(N), K_(K), T_(T) {
        name = "gemm_v2_kernel";
        threads = T * T;
        smem_per_block = 2 * T * T * 4 + 1024;
        gx_ = (N + T - 1) / T;
        blocks = gx_ * ((M + T - 1) / T);
        steps_ = (K + T - 1) / T;
        ops_per_warp = 2 * steps_ + 1;
    }
    void op(int64_t block, int warp, int64_t i, WarpOp& o) const override {
        const int64_t bx = block % gx_, by = block / gx_, k0 = (i / 2) * T_;
        const bool st = i == 2 * steps_;
        o.mask = 0;
        o.bytes = 4;
        o.store = st;
        o.operand = st ? 2 : int(i % 2);
        for (int l = 0; l < 32; ++l) {
            const int64_t tid = int64_t(warp) * 32 + l;
            if (tid >= threads) continue;
            const int64_t tx = tid % T_, ty = tid / T_, row = by * T_ + ty, col = bx * T_ + tx;
            bool on;
            uint64_t a;
            if (st) {
                on = row < M_ && col < N_;
                a = C_ + 4 * uint64_t(row * N_ + col);
            } else if (i % 2 == 0) {
                on = row < M_ && k0 + tx < K_;
                a = A_ + 4 * uint64_t(row * K_ + k0 + tx);
            } else {
                on = k0 + ty < K_ && col < N_;
                a = B_ + 4 * uint64_t((k0 + ty) * N_ + col);
            }
            if (on) {
                o.mask |= 1u << l;
                o.addr[l] = a;
            }
        }
    }

  private:
    uint64_t A_, B_, C_;
    int64_t M_, N_, K_, T_, gx_, steps_;
};

// v3, float4 path. Per k-step of BK: the A tile (BM x BK) then the B tile (BK x BN) in float4s,
// float4 f = tid + j*threads of each; then 2*TM float4 stores: row r of the thread's TM, halves
// h = 0, 1 (columns h*BN/2 + 4*tx). blockIdx.z selects the batch matrix, stride* floats apart.
class GemmV3 : public Generator {
  public:
    GemmV3(uint64_t A, uint64_t B, uint64_t C, int M, int N, int K, V3Tile t, int batch,
           int64_t sA, int64_t sB, int64_t sC)
        : A_(A), B_(B), C_(C), M_(M), N_(N), K_(K), sA_(sA), sB_(sB), sC_(sC), t_(t) {
        if (K % 4 || N % 4 || sA % 4 || sB % 4 || sC % 4)
            throw std::invalid_argument("gemm_v3: only the float4 path is modeled (K, N, strides % 4 == 0)");
        name = "gemm_v3_kernel";
        threads = (t.BM / t.TM) * (t.BN / 8);
        if (threads % 32 || (t.BM * t.BK / 4) % threads || (t.BK * t.BN / 4) % threads)
            throw std::invalid_argument("gemm_v3: tile needs whole warps and whole float4 rounds");
        smem_per_block = (t.BK * (t.BM + 4) + t.BK * t.BN) * 4 + 1024;
        gx_ = (N + t.BN - 1) / t.BN;
        gy_ = (M + t.BM - 1) / t.BM;
        blocks = gx_ * gy_ * batch;
        na_ = t.BM * t.BK / 4 / threads;
        nb_ = t.BK * t.BN / 4 / threads;
        steps_ = (K + t.BK - 1) / t.BK;
        ops_per_warp = steps_ * (na_ + nb_) + 2 * t.TM;
    }
    void op(int64_t block, int warp, int64_t i, WarpOp& o) const override {
        const int64_t bx = block % gx_, by = block / gx_ % gy_, bz = block / (gx_ * gy_);
        const uint64_t A = A_ + 4 * uint64_t(bz * sA_), B = B_ + 4 * uint64_t(bz * sB_),
                       C = C_ + 4 * uint64_t(bz * sC_);
        const int64_t row0 = by * t_.BM, col0 = bx * t_.BN, per = na_ + nb_;
        o.mask = 0;
        o.bytes = 16;
        if (i < steps_ * per) {
            const int64_t k0 = i / per * t_.BK, j = i % per;
            o.store = false;
            o.operand = j < na_ ? 0 : 1;
            for (int l = 0; l < 32; ++l) {
                const int64_t tid = int64_t(warp) * 32 + l;
                bool on;
                uint64_t a;
                if (j < na_) {
                    const int64_t f = tid + j * threads, r = f / (t_.BK / 4), k = f % (t_.BK / 4) * 4;
                    on = row0 + r < M_ && k0 + k < K_;
                    a = A + 4 * uint64_t((row0 + r) * K_ + k0 + k);
                } else {
                    const int64_t f = tid + (j - na_) * threads, k = f / (t_.BN / 4),
                                  c = f % (t_.BN / 4) * 4;
                    on = k0 + k < K_ && col0 + c < N_;
                    a = B + 4 * uint64_t((k0 + k) * N_ + col0 + c);
                }
                if (on) {
                    o.mask |= 1u << l;
                    o.addr[l] = a;
                }
            }
            return;
        }
        const int64_t s = i - steps_ * per, r = s / 2, h = s % 2;
        o.store = true;
        o.operand = 2;
        for (int l = 0; l < 32; ++l) {
            const int64_t tid = int64_t(warp) * 32 + l, tx = tid % (t_.BN / 8), ty = tid / (t_.BN / 8);
            const int64_t row = row0 + ty * t_.TM + r, col = col0 + h * (t_.BN / 2) + tx * 4;
            if (row < M_ && col < N_) {
                o.mask |= 1u << l;
                o.addr[l] = C + 4 * uint64_t(row * N_ + col);
            }
        }
    }

  private:
    uint64_t A_, B_, C_;
    int64_t M_, N_, K_, sA_, sB_, sC_, gx_ = 0, gy_ = 0, na_ = 0, nb_ = 0, steps_ = 0;
    V3Tile t_;
};

}  // namespace

std::unique_ptr<Generator> gemm_v0(uint64_t A, uint64_t B, uint64_t C, int M, int N, int K) {
    return std::make_unique<GemmV0>(A, B, C, M, N, K);
}
std::unique_ptr<Generator> gemm_v2(uint64_t A, uint64_t B, uint64_t C, int M, int N, int K, int T) {
    return std::make_unique<GemmV2>(A, B, C, M, N, K, T);
}
std::unique_ptr<Generator> gemm_v3(uint64_t A, uint64_t B, uint64_t C, int M, int N, int K,
                                   V3Tile t, int batch, int64_t sA, int64_t sB, int64_t sC) {
    return std::make_unique<GemmV3>(A, B, C, M, N, K, t, batch, sA, sB, sC);
}

}  // namespace sim
