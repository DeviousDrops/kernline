#include <algorithm>
#include "sim/sim.h"

namespace sim {

static const uint64_t kAlign = 2ull << 20;  // cudaMalloc hands out 2 MB-aligned blocks for these sizes

uint64_t Buffers::alloc(const std::string& name, uint64_t bytes) {
    const uint64_t base = next_;
    bufs_.push_back({name, base, bytes});
    next_ = (base + std::max<uint64_t>(bytes, 1) + kAlign - 1) / kAlign * kAlign;
    return base;
}

std::string Buffers::describe(uint64_t addr) const {
    for (const Buf& b : bufs_)
        if (addr >= b.base && addr < b.base + b.bytes)
            return b.name + "+" + std::to_string((addr - b.base) / 4);
    return "?";
}

}  // namespace sim
