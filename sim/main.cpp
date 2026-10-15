// build/cachesim <kernel> <variant> <dims> [options]: replay one call of a kernel through the
// simulated L1s and L2 (docs/SIM.md) and print a CSV header and one row per kernel launch.
//
//   gemm v0|v2|v3 MxNxK
//   --gpu NAME              gpu column (default rtx4060_laptop)
//   --sms N                 SM count (default 24; the 4070 Laptop has 36)
//   --launch-config PATH    ncu-measured resident blocks and carveouts
//                           (default results/sim/launch_config_rtx4060_laptop.csv, if present)
//   --tile T | BM,BN,BK,TM  GEMM v2 / v3 tile (default: the real kernel's)
//   --l1-ways N --l2-ways N --l2-hash xor|mod   cache assumptions (default 4, 16, xor)
//   --op L B W I            print op I of warp W of block B in launch L instead of simulating
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "sim/generators/generators.h"
#include "sim/launch_config.h"
#include "sim/memory.h"
#include "sim/sim.h"

using namespace sim;

static std::vector<int> split_ints(const std::string& s, char sep) {
    std::vector<int> out;
    std::stringstream ss(s);
    for (std::string x; std::getline(ss, x, sep);) out.push_back(std::stoi(x));
    return out;
}

static Launches build(const std::string& kernel, const std::string& variant,
                      const std::vector<int>& d, const std::vector<int>& tile, Buffers& buf) {
    Launches out;
    if (kernel == "gemm" && d.size() == 3) {
        const int M = d[0], N = d[1], K = d[2];
        // bench/bench.cu allocates A, B, C in this order.
        const uint64_t A = buf.alloc("A", 4ull * M * K), B = buf.alloc("B", 4ull * K * N),
                       C = buf.alloc("C", 4ull * M * N);
        if (variant == "v0") {
            out.push_back(gemm_v0(A, B, C, M, N, K));
        } else if (variant == "v2") {
            out.push_back(gemm_v2(A, B, C, M, N, K, tile.empty() ? 32 : tile.at(0)));
        } else if (variant == "v3") {
            V3Tile t;
            if (!tile.empty()) {
                if (tile.size() != 4) throw std::invalid_argument("gemm v3 --tile BM,BN,BK,TM");
                t = {tile[0], tile[1], tile[2], tile[3]};
            }
            out.push_back(gemm_v3(A, B, C, M, N, K, t));
        } else {
            throw std::invalid_argument("gemm variants: v0 v2 v3");
        }
        return out;
    }
    throw std::invalid_argument("usage: cachesim gemm v0|v2|v3 MxNxK [options]");
}

static unsigned long long u(uint64_t x) { return (unsigned long long)x; }

int main(int argc, char** argv) {
    try {
        if (argc < 4) throw std::invalid_argument("usage: cachesim <kernel> <variant> <dims> [options]");
        const std::string kernel = argv[1], variant = argv[2], dims = argv[3];
        std::string gpu = "rtx4060_laptop", config = "results/sim/launch_config_rtx4060_laptop.csv";
        std::string tile_arg;
        int sms = 24;
        CacheParams p;
        std::vector<long long> at;
        for (int i = 4; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
                return argv[++i];
            };
            if (a == "--gpu") gpu = next();
            else if (a == "--sms") sms = std::stoi(next());
            else if (a == "--launch-config") config = next();
            else if (a == "--tile") tile_arg = next();
            else if (a == "--l1-ways") p.l1_ways = std::stoi(next());
            else if (a == "--l2-ways") p.l2_ways = std::stoi(next());
            else if (a == "--l2-hash") p.l2_xor = next() == "xor";
            else if (a == "--op") for (int k = 0; k < 4; ++k) at.push_back(std::stoll(next()));
            else throw std::invalid_argument("unknown option " + a);
        }
        Buffers buf;
        const std::vector<int> tile = tile_arg.empty() ? std::vector<int>{} : split_ints(tile_arg, ',');
        const Launches launches = build(kernel, variant, split_ints(dims, 'x'), tile, buf);

        if (!at.empty()) {
            WarpOp o;
            launches.at(size_t(at[0]))->op(at[1], int(at[2]), at[3], o);
            std::printf("mask=%08x bytes=%d store=%d operand=%d", o.mask, o.bytes, int(o.store), o.operand);
            for (int l = 0; l < 32; ++l)
                std::printf(" %s", o.mask >> l & 1u ? buf.describe(o.addr[l]).c_str() : "-");
            std::printf("\n");
            return 0;
        }

        const LaunchConfig cfg(std::ifstream(config).good() ? config : "");
        std::string tile_col = tile_arg;
        for (char& c : tile_col)
            if (c == ',') c = 'x';
        std::printf("gpu,kernel,variant,dims,tile,launch,sim_kernel,simulated_only,resident,l1_kb,"
                    "ld_requests,ld_sectors,l1_hit_sectors,st_requests,st_sectors,l2_rd_sectors,"
                    "l2_rd_hit_sectors,l2_wr_sectors,l2_wr_hit_sectors,dram_read_bytes,"
                    "dram_write_bytes,a_ld_requests,a_ld_sectors,a_l1_hit_sectors,a_l2_rd_hit_sectors\n");
        for (size_t n = 0; n < launches.size(); ++n) {
            const Generator& g = *launches[n];
            const LaunchShape s = cfg.shape(g);
            CacheParams lp = p;
            lp.l1_bytes = (128u << 10) - s.carveout_bytes;
            const Counters c = run(g, sms, s.resident, lp);
            std::printf("%s,%s,%s,%s,%s,%zu,%s,%d,%d,%llu,", gpu.c_str(), kernel.c_str(),
                        variant.c_str(), dims.c_str(), tile_col.c_str(), n, g.name.c_str(),
                        int(!s.measured), s.resident, u(lp.l1_bytes >> 10));
            std::printf("%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
                        u(c.ld_requests), u(c.ld_sectors), u(c.l1_hit_sectors), u(c.st_requests),
                        u(c.st_sectors), u(c.l2_rd_sectors), u(c.l2_rd_hit_sectors),
                        u(c.l2_wr_sectors), u(c.l2_wr_hit_sectors), u(c.dram_rd_sectors * kSectorBytes),
                        u(c.dram_wr_sectors * kSectorBytes), u(c.op_ld_requests[0]),
                        u(c.op_ld_sectors[0]), u(c.op_l1_hit_sectors[0]), u(c.op_l2_rd_hit_sectors[0]));
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "cachesim: %s\n", e.what());
        return 2;
    }
}
