## What
<!-- kernel + variant, or tooling change -->

## Evidence
- [ ] `pytest tests/` passes on my GPU (incl. sizes not divisible by the tile)
- [ ] CSV row(s) added under `results/<gpu>/`
- [ ] ncu report under `results/ncu/<gpu>/`

| GPU | dims | before (ms) | after (ms) | % peak | limiter |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

**Why it moved:** <!-- "moved from X to Y because <compute|memory|latency>-bound — ncu showed <metric>" -->

## Reviewer checklist
- [ ] Builds and tests pass on the reviewer's GPU
- [ ] Benchmark reproduced on the reviewer's GPU
- [ ] Frozen interfaces unchanged (or `docs/DESIGN.md` updated)
