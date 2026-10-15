#pragma once
#include <cstdint>
#include <vector>
#include "sim/cache.h"
#include "sim/sim.h"

namespace sim {

// Cache sizes and assumptions for one launch (docs/SIM.md, "Cache model"). l1_bytes is per SM:
// 128 KB minus the launch's shared-memory carveout.
struct CacheParams {
    uint64_t l1_bytes = 112 << 10;
    int l1_ways = 4;
    uint64_t l2_bytes = 32ull << 20;
    int l2_ways = 16;
    bool l2_xor = true;
};

// Sector counts of one launch, named like the ncu counters they are compared with.
struct Counters {
    uint64_t ld_requests = 0, ld_sectors = 0, l1_hit_sectors = 0;
    uint64_t st_requests = 0, st_sectors = 0;
    uint64_t l2_rd_sectors = 0, l2_rd_hit_sectors = 0, l2_wr_sectors = 0, l2_wr_hit_sectors = 0;
    uint64_t dram_rd_sectors = 0, dram_wr_sectors = 0;
    uint64_t op_ld_requests[kOperands] = {}, op_ld_sectors[kOperands] = {},
             op_l1_hit_sectors[kOperands] = {}, op_l2_rd_hit_sectors[kOperands] = {};
};

// One L1 per SM and the shared L2. Global loads allocate in L1; global stores write through to
// L2 without allocating, and invalidate the sectors in their SM's L1. The L2 is write-back:
// DRAM writes are dirty sectors evicted during the launch, not what is still dirty at its end.
class MemorySystem {
  public:
    MemorySystem(int sms, const CacheParams& p);
    void issue(int sm, const WarpOp& op);
    Counters finish();

  private:
    std::vector<Cache> l1_;
    Cache l2_;
    Counters c_;
    std::vector<uint64_t> sectors_;
};

// Replays every block of `g` on `sms` SMs with `resident` blocks per SM (schedule.cpp).
Counters run(const Generator& g, int sms, int resident, const CacheParams& p);

}  // namespace sim
