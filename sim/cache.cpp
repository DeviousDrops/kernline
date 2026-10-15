#include "sim/cache.h"
#include <stdexcept>

namespace sim {

Cache::Cache(uint64_t bytes, int ways, bool xor_hash) : ways_(ways), xor_(xor_hash) {
    const uint64_t lines = bytes / kLineBytes;
    if (ways < 1 || lines == 0 || lines % ways)
        throw std::invalid_argument("cache size must be a positive multiple of ways x 128 B");
    sets_ = int(lines / ways);
    while ((1ull << bits_) < uint64_t(sets_)) ++bits_;
    lines_.resize(lines);
}

size_t Cache::set_of(uint64_t line) const {
    const uint64_t h = xor_ && bits_ ? line ^ (line >> bits_) ^ (line >> (2 * bits_)) : line;
    return size_t(h % uint64_t(sets_));
}

Cache::Line* Cache::find(uint64_t line) {
    Line* set = &lines_[set_of(line) * ways_];
    for (int w = 0; w < ways_; ++w)
        if (set[w].present && set[w].tag == line) return &set[w];
    return nullptr;
}

Cache::Line& Cache::allocate(uint64_t line) {
    Line* set = &lines_[set_of(line) * ways_];
    Line* victim = nullptr;
    for (int w = 0; w < ways_ && !victim; ++w)
        if (!set[w].present) victim = &set[w];
    if (!victim) {
        victim = &set[0];
        for (int w = 1; w < ways_; ++w)
            if (set[w].lru < victim->lru) victim = &set[w];
        dirty_evicted_ += __builtin_popcount(victim->dirty);
    }
    *victim = Line{};
    victim->present = 1;
    victim->tag = line;
    return *victim;
}

bool Cache::read(uint64_t sector) {
    const uint64_t line = sector >> 2;
    const uint8_t bit = uint8_t(1u << (sector & 3));
    Line* l = find(line);
    const bool hit = l && (l->valid & bit);
    if (!l) l = &allocate(line);
    l->valid |= bit;
    l->lru = ++clock_;
    return hit;
}

bool Cache::write(uint64_t sector) {
    const uint64_t line = sector >> 2;
    const uint8_t bit = uint8_t(1u << (sector & 3));
    Line* l = find(line);
    const bool hit = l && (l->valid & bit);
    if (!l) l = &allocate(line);
    l->valid |= bit;
    l->dirty |= bit;
    l->lru = ++clock_;
    return hit;
}

void Cache::invalidate(uint64_t sector) {
    if (Line* l = find(sector >> 2)) {
        const uint8_t bit = uint8_t(1u << (sector & 3));
        l->valid &= uint8_t(~bit);
        l->dirty &= uint8_t(~bit);
    }
}

}  // namespace sim
