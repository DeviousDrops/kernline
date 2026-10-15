#include "sim/launch_config.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace sim {

LaunchConfig::LaunchConfig(const std::string& csv_path) {
    if (csv_path.empty()) return;
    std::ifstream f(csv_path);
    if (!f) throw std::runtime_error("cannot open " + csv_path);
    std::string line;
    std::getline(f, line);  // gpu,kernel_name,block_size,smem_per_block,resident_blocks,smem_config_bytes
    while (std::getline(f, line)) {
        std::vector<std::string> c;
        std::stringstream ss(line);
        for (std::string x; std::getline(ss, x, ',');) c.push_back(x);
        if (c.size() < 6) continue;
        rows_.push_back({c[1], std::stoi(c[2]), std::stoi(c[3]), std::stoi(c[4]), std::stoull(c[5])});
    }
}

LaunchShape LaunchConfig::shape(const Generator& g) const {
    for (const Row& r : rows_)
        if (r.name == g.name && r.threads == g.threads && r.smem == g.smem_per_block)
            return {r.resident, r.carveout, true};

    // Fallback. Ada per-SM limits: 24 blocks, 48 warps, 100 KB of shared memory.
    static const uint64_t kCarveouts[] = {0, 8 << 10, 16 << 10, 32 << 10, 64 << 10, 100 << 10};
    const int warps = g.warps();
    int resident = std::min(24, 48 / warps);
    if (g.smem_per_block) resident = std::min(resident, (100 << 10) / g.smem_per_block);
    uint64_t carveout = 0;
    bool real = false;
    for (const Row& r : rows_)
        if (r.name == g.name) {  // same kernel, other tile: keep its warps per SM
            resident = std::max(1, std::min(resident, r.resident * ((r.threads + 31) / 32) / warps));
            carveout = r.carveout;
            real = true;
            break;
        }
    const uint64_t need = uint64_t(resident) * uint64_t(g.smem_per_block);
    if (!real || need > carveout)
        for (uint64_t c : kCarveouts)
            if (c >= need) {
                carveout = c;
                break;
            }
    return {resident, carveout, false};
}

}  // namespace sim
