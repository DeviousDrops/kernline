// build/cachesim_test: unit tests for the cache simulator (docs/SIM.md, "Testing"). Exit 0 when
// every check passes; each failure prints file:line and the check.
#include <cstdio>
#include <vector>
#include "sim/cache.h"
#include "sim/memory.h"
#include "sim/sim.h"

using namespace sim;

static int failures = 0;
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                                   \
        }                                                                                 \
    } while (0)

// 32 active lanes at base + lane * stride.
static WarpOp lanes(uint64_t base, uint64_t stride, int bytes) {
    WarpOp op;
    op.mask = 0xffffffffu;
    op.bytes = bytes;
    for (int l = 0; l < 32; ++l) op.addr[l] = base + l * stride;
    return op;
}

static void test_coalesce() {
    std::vector<uint64_t> s;
    coalesce(lanes(0, 4, 4), s);  // 32 consecutive floats: one 128 B line
    CHECK((s == std::vector<uint64_t>{0, 1, 2, 3}));
    coalesce(lanes(0, 16, 16), s);  // 32 float4s: 512 B
    CHECK(s.size() == 16 && s.front() == 0 && s.back() == 15);
    coalesce(lanes(0, 4096, 4), s);  // 1024 floats apart: a sector per lane
    CHECK(s.size() == 32);
    WarpOp half = lanes(0, 4, 4);
    half.mask = 0x0000ffffu;  // 16 lanes, 64 B
    coalesce(half, s);
    CHECK((s == std::vector<uint64_t>{0, 1}));
    coalesce(lanes(16, 4, 4), s);  // 16 B past a sector boundary: 5 sectors
    CHECK(s.size() == 5);
}

static void test_cache_lru() {
    Cache c(2 * kLineBytes, 2, false);  // one set, two ways
    const uint64_t L0 = 0, L1 = 4, L2 = 8;  // first sectors of lines 0, 1, 2
    CHECK(!c.read(L0));
    CHECK(!c.read(L1));
    CHECK(c.read(L0));   // line 1 is now least recently used
    CHECK(!c.read(L2));  // evicts line 1
    CHECK(c.read(L0));
    CHECK(!c.read(L1));  // evicts line 2
    CHECK(!c.read(L2));
}

static void test_cache_sectors_and_writeback() {
    Cache c(2 * kLineBytes, 2, false);
    CHECK(!c.read(0));
    CHECK(!c.read(1));  // same line, another sector: a miss, but no eviction
    CHECK(c.read(0) && c.read(1));
    CHECK(!c.write(8));  // write-allocate without a read
    CHECK(c.write(8));
    CHECK(c.dirty_evicted() == 0);
    c.read(12);  // line 3 evicts line 0, which is clean
    CHECK(c.dirty_evicted() == 0);
    c.read(16);  // line 4 evicts line 2: one dirty sector
    CHECK(c.dirty_evicted() == 1);
    c.invalidate(12);
    CHECK(!c.read(12));
}

static void test_xor_hash_spreads_power_of_two_strides() {
    // 64 lines 16 KB apart in a 256 KB, 16-way cache (128 sets): modulo indexing maps them all to
    // set 0, so a 64-line loop thrashes 16 ways. The XOR fold spreads them and they all stay.
    Cache mod(256 << 10, 16, false), hashed(256 << 10, 16, true);
    for (int pass = 0; pass < 2; ++pass)
        for (uint64_t i = 0; i < 64; ++i) {
            mod.read(i * 512);
            hashed.read(i * 512);
        }
    int mod_hits = 0, hashed_hits = 0;
    for (uint64_t i = 0; i < 64; ++i) {
        mod_hits += mod.read(i * 512);
        hashed_hits += hashed.read(i * 512);
    }
    CHECK(mod_hits == 0);
    CHECK(hashed_hits == 64);
}

static void test_store_invalidates_l1_and_writes_l2() {
    CacheParams p;
    p.l1_bytes = 4 * kLineBytes;
    p.l2_bytes = 64 * kLineBytes;
    p.l2_ways = 4;
    MemorySystem m(1, p);
    const WarpOp ld = lanes(0, 4, 4);
    WarpOp st = ld;
    st.store = true;
    m.issue(0, ld);  // 4 sectors: L1 miss, L2 miss, DRAM
    m.issue(0, ld);  // L1 hit
    m.issue(0, st);  // write-through: drops them from L1, L2 write hits
    m.issue(0, ld);  // L1 miss, L2 hit
    WarpOp off = ld;
    off.mask = 0;
    m.issue(0, off);  // no lane active: no request
    const Counters c = m.finish();
    CHECK(c.ld_requests == 3 && c.ld_sectors == 12 && c.l1_hit_sectors == 4);
    CHECK(c.l2_rd_sectors == 8 && c.l2_rd_hit_sectors == 4 && c.dram_rd_sectors == 4);
    CHECK(c.st_requests == 1 && c.st_sectors == 4);
    CHECK(c.l2_wr_sectors == 4 && c.l2_wr_hit_sectors == 4);
    CHECK(c.dram_wr_sectors == 0);  // still dirty in L2 when the launch ends: not counted
    CHECK(c.op_ld_requests[0] == 3 && c.op_ld_sectors[0] == 12);
}

int main() {
    test_coalesce();
    test_cache_lru();
    test_cache_sectors_and_writeback();
    test_xor_hash_spreads_power_of_two_strides();
    test_store_invalidates_l1_and_writes_l2();
    if (failures) std::fprintf(stderr, "%d check(s) failed\n", failures);
    else std::printf("cachesim_test: all checks passed\n");
    return failures ? 1 : 0;
}
