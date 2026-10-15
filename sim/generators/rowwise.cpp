// Row-wise and helper kernels: kernels/attention/helpers.cu (transpose_scale, scale) and
// kernels/softmax v0, v2 (register-cached rows) and v3 (float4 path). Program order per warp;
// the compiler's unrolling reorders a thread's independent loads but not how many there are.
#include <algorithm>
#include <stdexcept>
#include <string>
#include "sim/generators/generators.h"

namespace sim {
namespace {

// Block (32, 8): warp w is threadIdx.y = w. Four loads (rows r0 + w + 8i, column c0 + lane),
// then four stores (out row c0 + w + 8i, column r0 + lane). blockIdx.z is the head.
class TransposeScale : public Generator {
  public:
    TransposeScale(uint64_t in, uint64_t out, int heads, int rows, int cols)
        : in_(in), out_(out), rows_(rows), cols_(cols) {
        name = "transpose_scale_kernel";
        threads = 256;
        smem_per_block = 32 * 33 * 4 + 1024;
        gx_ = (cols + 31) / 32;
        gy_ = (rows + 31) / 32;
        blocks = gx_ * gy_ * heads;
        ops_per_warp = 8;
    }
    void op(int64_t block, int warp, int64_t i, WarpOp& o) const override {
        const int64_t bx = block % gx_, by = block / gx_ % gy_, bz = block / (gx_ * gy_);
        const int64_t head = bz * rows_ * cols_, c0 = bx * 32, r0 = by * 32;
        o.mask = 0;
        o.bytes = 4;
        o.store = i >= 4;
        o.operand = o.store ? 2 : 0;
        for (int l = 0; l < 32; ++l) {
            int64_t r, c;
            uint64_t a;
            if (!o.store) {
                r = r0 + warp + 8 * i;
                c = c0 + l;
                a = in_ + 4 * uint64_t(head + r * cols_ + c);
            } else {
                c = c0 + warp + 8 * (i - 4);
                r = r0 + l;
                a = out_ + 4 * uint64_t(head + c * rows_ + r);
            }
            if (r < rows_ && c < cols_) {
                o.mask |= 1u << l;
                o.addr[l] = a;
            }
        }
    }

  private:
    uint64_t in_, out_;
    int64_t rows_, cols_, gx_, gy_;
};

// Grid 1024 x 256, grid-stride: load x[i], store x[i] for i = tid, tid + 262144, ...
class Scale : public Generator {
  public:
    Scale(uint64_t x, int64_t n) : x_(x), n_(n) {
        name = "scale_kernel";
        threads = 256;
        smem_per_block = 1024;
        blocks = 1024;
        ops_per_warp = 2 * ((n + kStride - 1) / kStride);
    }
    void op(int64_t block, int warp, int64_t i, WarpOp& o) const override {
        o.mask = 0;
        o.bytes = 4;
        o.store = i % 2;
        o.operand = o.store ? 2 : 0;
        for (int l = 0; l < 32; ++l) {
            const int64_t idx = block * 256 + warp * 32 + l + (i / 2) * kStride;
            if (idx < n_) {
                o.mask |= 1u << l;
                o.addr[l] = x_ + 4 * uint64_t(idx);
            }
        }
    }

  private:
    static constexpr int64_t kStride = 1024 * 256;
    uint64_t x_;
    int64_t n_;
};

// One thread per row (128 per block). Passes: max (cols loads), sum (cols loads), then per column
// load and store.
class SoftmaxV0 : public Generator {
  public:
    SoftmaxV0(uint64_t x, uint64_t y, int64_t rows, int64_t cols) : x_(x), y_(y), rows_(rows), cols_(cols) {
        name = "softmax_v0_kernel";
        threads = 128;
        smem_per_block = 1024;
        blocks = (rows + 127) / 128;
        ops_per_warp = 4 * cols;
    }
    void op(int64_t block, int warp, int64_t i, WarpOp& o) const override {
        int64_t j;
        if (i < 2 * cols_) {
            j = i % cols_;
            o.store = false;
        } else {
            j = (i - 2 * cols_) / 2;
            o.store = (i - 2 * cols_) % 2;
        }
        o.mask = 0;
        o.bytes = 4;
        o.operand = o.store ? 2 : 0;
        for (int l = 0; l < 32; ++l) {
            const int64_t row = block * 128 + warp * 32 + l;
            if (row < rows_) {
                o.mask |= 1u << l;
                o.addr[l] = (o.store ? y_ : x_) + 4 * uint64_t(row * cols_ + j);
            }
        }
    }

  private:
    uint64_t x_, y_;
    int64_t rows_, cols_;
};

// One warp per row, 8 rows per block. k float4 loads (column 4*lane + 128*i) into registers,
// then k float4 stores to the same columns.
class SoftmaxV2Cached : public Generator {
  public:
    SoftmaxV2Cached(uint64_t x, uint64_t y, int64_t rows, int64_t cols, int k)
        : x_(x), y_(y), rows_(rows), cols_(cols), k_(k) {
        if (cols % 4 || cols > 128 * k) throw std::invalid_argument("softmax v2 cached: cols % 4 == 0, cols <= 128 k");
        name = "softmax_v2_kernel<" + std::to_string(k) + ">";
        threads = 256;
        smem_per_block = 1024;
        blocks = (rows + 7) / 8;
        ops_per_warp = 2 * k;
    }
    void op(int64_t block, int warp, int64_t i, WarpOp& o) const override {
        const int64_t row = block * 8 + warp;
        o.mask = 0;
        o.bytes = 16;
        o.store = i >= k_;
        o.operand = o.store ? 2 : 0;
        if (row >= rows_) return;  // the whole warp returns
        for (int l = 0; l < 32; ++l) {
            const int64_t j = l * 4 + (i % k_) * 128;
            if (j < cols_) {
                o.mask |= 1u << l;
                o.addr[l] = (o.store ? y_ : x_) + 4 * uint64_t(row * cols_ + j);
            }
        }
    }

  private:
    uint64_t x_, y_;
    int64_t rows_, cols_, k_;
};

// One block per row, the row in dynamic shared memory. Threads per src/api.cu v3_shape; float4
// j = t, t + threads, ... below cols/4: the load pass, then the store pass.
class SoftmaxV3 : public Generator {
  public:
    SoftmaxV3(uint64_t x, uint64_t y, int64_t rows, int64_t cols) : x_(x), y_(y), cols_(cols) {
        if (cols % 4) throw std::invalid_argument("softmax v3: only the float4 path is modeled");
        name = "softmax_v3_kernel<1>";
        const int64_t want = ((cols + 15) / 16 + 31) / 32 * 32;
        threads = int(std::min<int64_t>(std::max<int64_t>(want, 128), 1024));
        smem_per_block = int(cols * 4 + 128 + 1024);
        blocks = rows;
        iters_ = (cols / 4 + threads - 1) / threads;
        ops_per_warp = 2 * iters_;
    }
    void op(int64_t block, int warp, int64_t i, WarpOp& o) const override {
        o.mask = 0;
        o.bytes = 16;
        o.store = i >= iters_;
        o.operand = o.store ? 2 : 0;
        for (int l = 0; l < 32; ++l) {
            const int64_t j = int64_t(warp) * 32 + l + (i % iters_) * threads;
            if (j < cols_ / 4) {
                o.mask |= 1u << l;
                o.addr[l] = (o.store ? y_ : x_) + 4 * uint64_t(block * cols_) + 16 * uint64_t(j);
            }
        }
    }

  private:
    uint64_t x_, y_;
    int64_t cols_, iters_;
};

}  // namespace

std::unique_ptr<Generator> transpose_scale(uint64_t in, uint64_t out, int heads, int rows, int cols) {
    return std::make_unique<TransposeScale>(in, out, heads, rows, cols);
}
std::unique_ptr<Generator> scale(uint64_t x, int64_t n) { return std::make_unique<Scale>(x, n); }
std::unique_ptr<Generator> softmax_v0(uint64_t x, uint64_t y, int64_t rows, int64_t cols) {
    return std::make_unique<SoftmaxV0>(x, y, rows, cols);
}
std::unique_ptr<Generator> softmax_v2_cached(uint64_t x, uint64_t y, int64_t rows, int64_t cols, int k) {
    return std::make_unique<SoftmaxV2Cached>(x, y, rows, cols, k);
}
std::unique_ptr<Generator> softmax_v3(uint64_t x, uint64_t y, int64_t rows, int64_t cols) {
    return std::make_unique<SoftmaxV3>(x, y, rows, cols);
}

}  // namespace sim
