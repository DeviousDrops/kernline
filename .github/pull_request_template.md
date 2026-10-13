## Phase
<!-- e.g. Phase 2 (Optimize), Part B: softmax v1–v2, LayerNorm v1–v2, attention v1 -->

## Evidence
- [ ] `pytest tests/` passes on my GPU (incl. sizes not divisible by the tile)
- [ ] CSV rows added under `results/<gpu>/` for every variant
- [ ] ncu reports under `results/ncu/<gpu>/` for every variant

| Kernel | Variant | GPU | dims | time (ms) | GFLOP/s or GB/s | % peak | % baseline | limiter | why it moved (one sentence, ncu metric) |
|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |

## Reviewer checklist
- [ ] Builds and tests pass on the reviewer's GPU
- [ ] Benchmarks reproduced on the reviewer's GPU
- [ ] Frozen interfaces unchanged (or `docs/DESIGN.md` updated)
