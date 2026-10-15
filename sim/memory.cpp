#include "sim/memory.h"

namespace sim {

MemorySystem::MemorySystem(int sms, const CacheParams& p)
    : l1_(size_t(sms), Cache(p.l1_bytes, p.l1_ways, false)), l2_(p.l2_bytes, p.l2_ways, p.l2_xor) {}

void MemorySystem::issue(int sm, const WarpOp& op) {
    if (!op.mask) return;
    coalesce(op, sectors_);
    const uint64_t n = sectors_.size();
    if (op.store) {
        ++c_.st_requests;
        c_.st_sectors += n;
        for (uint64_t s : sectors_) {
            l1_[sm].invalidate(s);
            ++c_.l2_wr_sectors;
            if (l2_.write(s)) ++c_.l2_wr_hit_sectors;
        }
        return;
    }
    const int o = op.operand;
    ++c_.ld_requests;
    c_.ld_sectors += n;
    ++c_.op_ld_requests[o];
    c_.op_ld_sectors[o] += n;
    for (uint64_t s : sectors_) {
        if (l1_[sm].read(s)) {
            ++c_.l1_hit_sectors;
            ++c_.op_l1_hit_sectors[o];
            continue;
        }
        ++c_.l2_rd_sectors;
        if (l2_.read(s)) {
            ++c_.l2_rd_hit_sectors;
            ++c_.op_l2_rd_hit_sectors[o];
        } else {
            ++c_.dram_rd_sectors;
        }
    }
}

Counters MemorySystem::finish() {
    c_.dram_wr_sectors = l2_.dirty_evicted();
    return c_;
}

}  // namespace sim
