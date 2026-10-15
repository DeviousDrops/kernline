#include <algorithm>
#include "sim/sim.h"

namespace sim {

void coalesce(const WarpOp& op, std::vector<uint64_t>& out) {
    out.clear();
    for (int l = 0; l < 32; ++l) {
        if (!(op.mask >> l & 1u)) continue;
        const uint64_t first = op.addr[l] / kSectorBytes;
        const uint64_t last = (op.addr[l] + op.bytes - 1) / kSectorBytes;
        for (uint64_t s = first; s <= last; ++s) out.push_back(s);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
}

}  // namespace sim
