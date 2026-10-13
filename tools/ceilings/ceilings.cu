// Measured roofline ceilings (D3) + the device limits the occupancy model needs. Prints one JSON
// object; tools/ceilings/run.sh merges it into specs/<gpu>.json.
//
// Throughput tests report the best of kReps timed launches after a >= 500 ms warmup (ceilings
// are maxima, so best-of is the right statistic). Latency tests chase a random cyclic pointer
// chain with one thread, so every load waits for the previous one, and report the median.
#include <mma.h>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>
#include "common/cuda_check.h"
#include "common/gpu_info.h"
#include "common/timer.h"

using namespace nvcuda;

static const int kReps = 10;
static const float kWarmupMs = 500.0f;

// ---------------------------------------------------------------- bandwidth kernels

__global__ void dram_copy(const float4* __restrict__ a, float4* __restrict__ b, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        b[i] = a[i];
}

// Read-only sweep. `cg` loads bypass L1 (cache-global), so a buffer that fits in L2 measures L2.
template <bool kCacheGlobal>
__global__ void read_sweep(const float4* __restrict__ a, size_t n, int passes, float* out) {
    float acc = 0.0f;
    for (int p = 0; p < passes; ++p)
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
             i += (size_t)gridDim.x * blockDim.x) {
            float4 v = kCacheGlobal ? __ldcg(a + i) : a[i];
            acc += v.x + v.y + v.z + v.w;
        }
    if (acc == 1234.5f) out[0] = acc;  // data is zero, never true; keeps the loads alive
}

// 128-bit loads (a warp moves 512 B per instruction) at conflict-free addresses: consecutive
// lanes read consecutive float4s. Four accumulators keep the adds off the critical path.
__global__ void smem_read(int iters, float* out) {
    __shared__ float4 s[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) s[i] = make_float4(i, i, i, i);
    __syncthreads();
    float4 acc[4] = {};
    const int lane = threadIdx.x & 31;
    for (int i = 0; i < iters; i += 4)
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            float4 v = s[lane + 32 * ((i + u) & 31)];
            acc[u].x += v.x; acc[u].y += v.y; acc[u].z += v.z; acc[u].w += v.w;
        }
    float sum = 0.0f;
    for (auto& a : acc) sum += a.x + a.y + a.z + a.w;
    if (sum == 1234.5f) out[0] = sum;
}

// ---------------------------------------------------------------- compute kernels

// 8 independent FMA chains per thread hide the 4-cycle FMA latency; unrolling the outer loop
// amortizes the loop counter so nearly every issued instruction is an FMA.
__global__ void fp32_fma(int iters, float b, float c, float* out) {
    float a[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) a[j] = threadIdx.x + j;
#pragma unroll 8
    for (int i = 0; i < iters; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) a[j] = fmaf(a[j], b, c);
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) acc += a[j];
    if (acc == 1234.5f) out[0] = acc;
}

// FP16 inputs, FP32 accumulate (what WMMA GEMM v4 uses). 4 independent accumulators per warp.
__global__ void tc_fp16_fp32acc(int iters, float* out) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[4];
    wmma::fill_fragment(a, __float2half(0.001f));
    wmma::fill_fragment(b, __float2half(0.001f));
    for (auto& f : c) wmma::fill_fragment(f, 0.0f);
    for (int i = 0; i < iters; ++i)
#pragma unroll
        for (int k = 0; k < 4; ++k) wmma::mma_sync(c[k], a, b, c[k]);
    float acc = c[0].x[0] + c[1].x[0] + c[2].x[0] + c[3].x[0];
    if (acc == 1234.5f) out[0] = acc;
}

// ---------------------------------------------------------------- latency kernel

template <bool kCacheGlobal>
__global__ void chase(const unsigned* __restrict__ next, int steps, unsigned* out) {
    unsigned p = 0;
    for (int i = 0; i < steps; ++i) p = kCacheGlobal ? __ldcg(next + p) : next[p];
    out[0] = p;
}

// ---------------------------------------------------------------- harness

struct Ctx {
    GpuTimer timer;
    ClockSampler clocks;
    float* sink;
};

// Times kReps launches after warming up. Throughput ceilings take the best (minimum) time;
// latencies take the median, because a single fast outlier is not a property of the memory system.
template <class Launch>
static float time_ms(Ctx& ctx, Launch launch, bool median) {
    for (float warm = 0.0f; warm < kWarmupMs;) {
        ctx.timer.start();
        launch();
        warm += ctx.timer.stop_ms();
    }
    std::vector<float> t(kReps);
    for (float& x : t) {
        ctx.timer.start();
        launch();
        ctx.clocks.sample();
        x = ctx.timer.stop_ms();
    }
    CUDA_CHECK(cudaGetLastError());
    std::sort(t.begin(), t.end());
    return median ? t[kReps / 2] : t[0];
}

template <class Launch>
static float best_ms(Ctx& ctx, Launch launch) { return time_ms(ctx, launch, false); }

// One element per 128-byte line, linked in a random cycle (Sattolo), indices in floats.
static unsigned* make_chain(size_t bytes) {
    const size_t stride = 32, lines = bytes / 128;
    std::vector<unsigned> order(lines);
    std::iota(order.begin(), order.end(), 0u);
    std::mt19937 rng(42);
    for (size_t i = lines - 1; i > 0; --i)
        std::swap(order[i], order[std::uniform_int_distribution<size_t>(0, i - 1)(rng)]);
    std::vector<unsigned> next(lines * stride, 0);
    for (size_t i = 0; i < lines; ++i)
        next[order[i] * stride] = order[(i + 1) % lines] * stride;
    unsigned* d;
    CUDA_CHECK(cudaMalloc(&d, next.size() * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(d, next.data(), next.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
    return d;
}

template <bool kCacheGlobal>
static float latency_ns(Ctx& ctx, size_t bytes, int steps) {
    unsigned* next = make_chain(bytes);
    unsigned* out;
    CUDA_CHECK(cudaMalloc(&out, sizeof(unsigned)));
    const size_t lines = bytes / 128;
    chase<kCacheGlobal><<<1, 1>>>(next, (int)std::min<size_t>(lines * 2, 1 << 22), out);  // fill caches
    float ms = time_ms(ctx, [&] { chase<kCacheGlobal><<<1, 1>>>(next, steps, out); }, true);
    CUDA_CHECK(cudaFree(next));
    CUDA_CHECK(cudaFree(out));
    return ms * 1e6f / steps;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int sms = prop.multiProcessorCount;
    int sm_clk_khz = 0, mem_clk_khz = 0;
    cudaDeviceGetAttribute(&sm_clk_khz, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, 0);

    Ctx ctx;
    CUDA_CHECK(cudaMalloc(&ctx.sink, sizeof(float)));
    const int threads = 256, blocks = sms * 32;

    // DRAM: 2 x 256 MiB, far beyond the 32 MiB L2.
    const size_t dram_bytes = 256u << 20, n4 = dram_bytes / sizeof(float4);
    float4 *a, *b;
    CUDA_CHECK(cudaMalloc(&a, dram_bytes));
    CUDA_CHECK(cudaMalloc(&b, dram_bytes));
    CUDA_CHECK(cudaMemset(a, 0, dram_bytes));
    float t_copy = best_ms(ctx, [&] { dram_copy<<<blocks, threads>>>(a, b, n4); });
    float t_read = best_ms(ctx, [&] { read_sweep<false><<<blocks, threads>>>(a, n4, 1, ctx.sink); });

    // L2: 8 MiB read 32 times through cache-global loads (fits L2, far exceeds total L1).
    const size_t l2_bytes = 8u << 20, l2_n4 = l2_bytes / sizeof(float4);
    const int l2_passes = 32;
    float t_l2 = best_ms(ctx, [&] { read_sweep<true><<<blocks, threads>>>(a, l2_n4, l2_passes, ctx.sink); });
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(a));

    const int smem_iters = 1 << 16;
    float t_smem = best_ms(ctx, [&] { smem_read<<<sms * 8, threads>>>(smem_iters, ctx.sink); });

    const int fma_iters = 1 << 14;
    float t_fma = best_ms(ctx, [&] { fp32_fma<<<sms * 8, threads>>>(fma_iters, 0.999f, 0.001f, ctx.sink); });

    const int tc_iters = 1 << 12, tc_warps = sms * 8 * (threads / 32);
    float t_tc = best_ms(ctx, [&] { tc_fp16_fp32acc<<<sms * 8, threads>>>(tc_iters, ctx.sink); });

    float lat_l1 = latency_ns<false>(ctx, 16u << 10, 1 << 20);   // 16 KiB: L1-resident
    float lat_l2 = latency_ns<true>(ctx, 4u << 20, 1 << 20);     // 4 MiB via ld.cg: L2-resident
    float lat_dram = latency_ns<true>(ctx, 256u << 20, 1 << 18);  // 256 MiB: DRAM (incl. TLB misses)

    const double sm_mhz = ctx.clocks.median_sm();
    auto gbs = [](double bytes, float ms) { return bytes / (ms * 1e6); };
    auto cyc = [&](float ns) { return ns * sm_mhz / 1e3; };

    std::printf("{\n");
    std::printf("  \"gpu\": \"%s\",\n", gpu_slug(prop.name).c_str());
    std::printf("  \"device\": {\n");
    std::printf("    \"sm_count\": %d,\n", sms);
    std::printf("    \"l2_bytes\": %d,\n", prop.l2CacheSize);
    std::printf("    \"dram_bus_bits\": %d,\n", prop.memoryBusWidth);
    std::printf("    \"warp_size\": %d,\n", prop.warpSize);
    std::printf("    \"max_threads_per_sm\": %d,\n", prop.maxThreadsPerMultiProcessor);
    std::printf("    \"max_blocks_per_sm\": %d,\n", prop.maxBlocksPerMultiProcessor);
    std::printf("    \"max_threads_per_block\": %d,\n", prop.maxThreadsPerBlock);
    std::printf("    \"regs_per_sm\": %d,\n", prop.regsPerMultiprocessor);
    std::printf("    \"regs_per_block\": %d,\n", prop.regsPerBlock);
    std::printf("    \"smem_per_sm_bytes\": %zu,\n", prop.sharedMemPerMultiprocessor);
    std::printf("    \"smem_per_block_optin_bytes\": %zu,\n", prop.sharedMemPerBlockOptin);
    std::printf("    \"smem_reserved_per_block_bytes\": %zu,\n", prop.reservedSharedMemPerBlock);
    std::printf("    \"base_sm_clk_mhz\": %d,\n", sm_clk_khz / 1000);
    std::printf("    \"mem_clk_mhz\": %d\n", mem_clk_khz / 1000);
    std::printf("  },\n");
    std::printf("  \"ceilings\": {\n");
    std::printf("    \"sm_clk_mhz\": %d,\n", ctx.clocks.median_sm());
    std::printf("    \"mem_clk_mhz\": %d,\n", ctx.clocks.median_mem());
    std::printf("    \"dram_copy_gbs\": %.1f,\n", gbs(2.0 * dram_bytes, t_copy));
    std::printf("    \"dram_read_gbs\": %.1f,\n", gbs((double)dram_bytes, t_read));
    std::printf("    \"l2_read_gbs\": %.1f,\n", gbs((double)l2_bytes * l2_passes, t_l2));
    std::printf("    \"smem_read_gbs\": %.1f,\n",
                gbs(16.0 * sms * 8 * threads * (double)smem_iters, t_smem));
    std::printf("    \"fp32_gflops\": %.1f,\n",
                2.0 * 8 * fma_iters * sms * 8 * (double)threads / (t_fma * 1e6));
    std::printf("    \"fp16_tc_fp32acc_gflops\": %.1f,\n",
                2.0 * 16 * 16 * 16 * 4 * tc_iters * (double)tc_warps / (t_tc * 1e6));
    std::printf("    \"latency_ns\": {\"l1\": %.1f, \"l2\": %.1f, \"dram\": %.1f},\n", lat_l1, lat_l2,
                lat_dram);
    std::printf("    \"latency_cycles\": {\"l1\": %.0f, \"l2\": %.0f, \"dram\": %.0f}\n", cyc(lat_l1),
                cyc(lat_l2), cyc(lat_dram));
    std::printf("  }\n}\n");
    CUDA_CHECK(cudaFree(ctx.sink));
    return 0;
}
