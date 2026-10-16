#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "sim/sim.h"

namespace sim {

struct LaunchShape {
    int resident = 1;             // blocks per SM
    uint64_t carveout_bytes = 0;  // shared-memory carveout; L1 gets 128 KB minus this
    bool measured = false;        // from ncu (real kernel config), not the fallback rule
};

// Resident blocks per SM and the shared-memory carveout of each launch, as ncu recorded them
// (results/sim/launch_config_<gpu>.csv, written by tools/sim_ncu.py). The driver's carveout
// choice isn't predictable (gemm_v3 needs 2 x 9.1 KB and gets 64 KB), so real configs are
// looked up, not computed. Launches the table doesn't have (simulated-only tiles) keep their
// kernel's real warps per SM and carveout where the shared memory allows, else take the next
// carveout size that fits (docs/SIM.md, "Cache model"). Simulated-only tiles are never compiled,
// so they have no register count: the real kernel's warps per SM stand in for its register
// limit. A shape that fits no block on an SM throws std::invalid_argument.
class LaunchConfig {
  public:
    explicit LaunchConfig(const std::string& csv_path);  // "" = empty table
    LaunchShape shape(const Generator& g) const;

  private:
    struct Row {
        std::string name;
        int threads, smem, resident;
        uint64_t carveout;
    };
    std::vector<Row> rows_;
};

}  // namespace sim
