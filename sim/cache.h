#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "sim/sim.h"

namespace sim {

// Sectored set-associative cache: 128 B lines of four 32 B sectors, LRU replacement per set. A
// miss fills only the sector asked for. Addresses are sector indices (byte address / 32).
// Writes allocate their sector without reading it and mark it dirty; evicting a line counts its
// dirty sectors. The set index is the line address modulo the set count, or, with xor_hash, that
// of an XOR fold of the line address: a stand-in for the hardware's undisclosed address hash,
// so that power-of-two strides don't all land in one set.
class Cache {
  public:
    Cache(uint64_t bytes, int ways, bool xor_hash);
    bool read(uint64_t sector);        // true on hit; fills the sector on a miss
    bool write(uint64_t sector);       // true if the sector was already present; marks it dirty
    void invalidate(uint64_t sector);  // drops the sector if present (L1 under a store)
    uint64_t dirty_evicted() const { return dirty_evicted_; }
    int sets() const { return sets_; }

  private:
    struct Line {
        uint64_t tag = 0, lru = 0;
        uint8_t present = 0, valid = 0, dirty = 0;  // valid/dirty: one bit per sector
    };
    Line* find(uint64_t line);
    Line& allocate(uint64_t line);
    size_t set_of(uint64_t line) const;

    int ways_, sets_, bits_ = 0;
    bool xor_;
    std::vector<Line> lines_;
    uint64_t clock_ = 0, dirty_evicted_ = 0;
};

}  // namespace sim
