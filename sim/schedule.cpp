// run(): waves of resident blocks (docs/SIM.md, "Launch order and concurrency"). Blocks go to
// SMs in blockIdx order, block b of the first wave to SM b % sms; a finished block's slot takes
// the next block. Each SM issues one memory instruction per turn, from its resident warps in
// round-robin order, and the SMs take turns. Time is counted in issued instructions, not cycles.
#include <vector>
#include "sim/memory.h"

namespace sim {

Counters run(const Generator& g, int sms, int resident, const CacheParams& p) {
    MemorySystem mem(sms, p);
    if (g.blocks == 0 || g.ops_per_warp == 0) return mem.finish();
    const int W = g.warps(), R = resident, per_sm = R * W;
    struct Slot {
        int64_t block = -1;
        int live = 0;  // warps of this block with ops left
        std::vector<int64_t> next;
    };
    // static_cast, not size_t(sms): that spelling parses as a function declaration.
    std::vector<std::vector<Slot>> sm(static_cast<size_t>(sms),
                                      std::vector<Slot>(static_cast<size_t>(R)));
    std::vector<int> rr(size_t(sms), 0);
    int64_t next_block = 0;
    auto start = [&](Slot& s) {
        if (next_block >= g.blocks) {
            s.block = -1;
            return;
        }
        s.block = next_block++;
        s.live = W;
        s.next.assign(size_t(W), 0);
    };
    for (int r = 0; r < R; ++r)
        for (int i = 0; i < sms; ++i) start(sm[i][r]);

    WarpOp op;
    for (bool issued = true; issued;) {
        issued = false;
        for (int i = 0; i < sms; ++i) {
            for (int tries = 0; tries < per_sm; ++tries) {
                const int k = rr[i];
                rr[i] = (k + 1) % per_sm;
                Slot& s = sm[i][k / W];
                const int w = k % W;
                if (s.block < 0 || s.next[w] >= g.ops_per_warp) continue;
                g.op(s.block, w, s.next[w]++, op);
                mem.issue(i, op);
                if (s.next[w] == g.ops_per_warp && --s.live == 0) start(s);
                issued = true;
                break;
            }
        }
    }
    return mem.finish();
}

}  // namespace sim
