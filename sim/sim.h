#pragma once
// Cache simulator (docs/SIM.md): each kernel launch is replayed from its index math as warp-level
// global memory instructions, scheduled as waves of resident blocks, and run through one sectored
// L1 per SM and a shared L2.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sim {

constexpr uint64_t kLineBytes = 128, kSectorBytes = 32;
constexpr int kOperands = 3;  // load counters by operand: 0 = first input (GEMM A), 1 = B, 2 = other

// One warp-level global memory instruction. Lanes outside `mask` don't access memory. mask == 0
// means no lane executes it (a branch nobody takes, or a predicate false everywhere); it is
// skipped and makes no request, as on the GPU.
struct WarpOp {
    uint64_t addr[32] = {};
    uint32_t mask = 0;
    int bytes = 4;      // per lane: 4 (LDG/STG.E) or 16 (.E.128)
    bool store = false;
    int operand = 0;    // which input a load reads, for the per-operand counters
};

// One kernel launch, replayed from its index math. Blocks are numbered x fastest, then y, then z.
class Generator {
  public:
    virtual ~Generator() = default;
    std::string name;           // as Nsight Compute prints it, without parameters
    int64_t blocks = 0;
    int threads = 0;            // per block
    int smem_per_block = 0;     // bytes: static + dynamic + 1 KB driver reservation (ncu's value)
    int64_t ops_per_warp = 0;   // global memory instructions per warp, in program order
    int warps() const { return (threads + 31) / 32; }
    // The i-th global memory instruction of `warp` in `block`, 0 <= i < ops_per_warp. Must set
    // every field of `out`.
    virtual void op(int64_t block, int warp, int64_t i, WarpOp& out) const = 0;
};
using Launches = std::vector<std::unique_ptr<Generator>>;

// Device buffers placed as the bench's cudaMalloc calls place them: in allocation order, 2 MB
// aligned. Addresses are bytes.
class Buffers {
  public:
    uint64_t alloc(const std::string& name, uint64_t bytes);
    // "A+12": the buffer holding `addr` and the offset in floats; "?" outside every buffer.
    std::string describe(uint64_t addr) const;

  private:
    struct Buf {
        std::string name;
        uint64_t base, bytes;
    };
    std::vector<Buf> bufs_;
    uint64_t next_ = 1ull << 32;
};

// The distinct 32 B sectors (as addr / 32) that the active lanes of `op` touch, ascending.
void coalesce(const WarpOp& op, std::vector<uint64_t>& out);

}  // namespace sim
